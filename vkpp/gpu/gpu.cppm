export module vkpp.gpu;

import std;
import vulkan;
import fiat;

import vkpp.error;
import vkpp.diagnostics;
import vkpp.capabilities;
import vkpp.instance;
import vkpp.device;
import vkpp.buffer;
import vkpp.buffer.upload;
import vkpp.buffer.download;
import vkpp.command;
import vkpp.command.record;
import vkpp.pipeline.compute;
import vkpp.pipeline.cache;
export import vkpp.cstring_view;

namespace vkpp::gpu
{

export class context;

export template<fiat::scalar Element>
class device_array
{
public:
  [[nodiscard]]
  auto
  size() const noexcept -> std::uint32_t
  { return count_; }

  [[nodiscard]]
  auto
  buffer() const noexcept -> const device_address_buffer&
  { return buffer_; }

  [[nodiscard]]
  auto
  addressed_view(const device_context& device) const -> std::expected<
    buffer_view<buffer_kind::device_address, Element, addressed>,
    error_t
  >
  {
    return buffer_.template addressed_view<Element>(
      device, 0UZ, vk::DeviceSize {count_} * sizeof(Element));
  }

private:
  friend class context;

  device_array(device_address_buffer&& buffer, std::uint32_t count)
  : buffer_ {std::move(buffer)}, count_ {count}
  {}

  device_address_buffer buffer_;
  std::uint32_t         count_ {};
};

export struct context_create_info
{
  cstring_view                      name {"vkpp-gpu"};
  bool                              enable_validation {false};
  std::optional<diagnostic_buffer&> diagnostics {};
  std::uint32_t                     group_size {256U};
  std::filesystem::path             pipeline_cache_path {};
};

template<fiat::scalar Element, class Operation>
  requires fiat::traceable<Operation, Element>
inline constexpr auto kernel_words =
  fiat::compile<Element>(fiat::transform_stage<Operation> {});

template<fiat::scalar Element, class Operation>
  requires fiat::traceable<Operation, Element>
inline constexpr auto kernel_fingerprint =
  fiat::fingerprint(kernel_words<Element, Operation>);

struct cached_kernel
{
  std::uint64_t                  fingerprint {};
  std::span<const std::uint32_t> words {};
  std::uint32_t                  group_size {};
  compute_pipeline               pipeline {};
};

export class context
{
public:
  [[nodiscard]]
  static
  auto
  create(const context_create_info& create_info)
    -> std::expected<context, error_t>
  {
    const auto extensions =
      create_info.enable_validation
        ? std::span {validation_extensions}
        : std::span<const char* const> {};
    const auto layers =
      create_info.enable_validation
        ? std::span {validation_layers}
        : std::span<const char* const> {};

    return instance_context::create(
      {
        .app_info =
          {
            .pApplicationName = create_info.name.c_str(),
            .apiVersion       = vk::ApiVersion14,
          },
        .extensions        = extensions,
        .layers            = layers,
        .enable_validation = create_info.enable_validation,
        .diagnostics       = create_info.diagnostics,
      })
      .and_then(
        [ & ](instance_context&& instance) -> std::expected<context, error_t> {
          return assemble(std::move(instance), create_info);
        });
  }

  template<fiat::scalar Element, class Operation>
    requires fiat::traceable<Operation, Element>
  [[nodiscard]]
  auto
  kernel()
    -> std::expected<std::reference_wrapper<const compute_pipeline>, error_t>
  {
    constexpr std::span<const std::uint32_t> words {
      kernel_words<Element, Operation>,
    };
    constexpr std::uint64_t fingerprint =
      kernel_fingerprint<Element, Operation>;

    const auto found = std::ranges::find_if(
      kernels_, [ & ](const cached_kernel& cached) -> bool {
        return cached.fingerprint == fingerprint &&
               cached.group_size == group_size_ &&
               std::ranges::equal(cached.words, words);
      });
    if (found != kernels_.end()) { return std::cref(found->pipeline); }

    const std::array specialization {
      specialization_constant {
        .id    = fiat::group_size_constant_id,
        .value = group_size_,
      },
    };
    return make_compute_pipeline(
      device_.device(),
      {
        .push_constant_size = sizeof(fiat::transform_push),
      },
      {
        .words          = words,
        .entry          = cstring_view {fiat::transform_entry},
        .specialization = specialization,
      },
      cache_.get(),
      diagnostics_)
      .transform(
        [ & ](compute_pipeline&& created)
          -> std::reference_wrapper<const compute_pipeline> {
          kernels_.push_back({
            .fingerprint = fingerprint,
            .words       = words,
            .group_size  = group_size_,
            .pipeline    = std::move(created),
          });
          return std::cref(kernels_.back().pipeline);
        });
  }

  [[nodiscard]]
  auto
  kernel_count() const noexcept -> std::size_t
  { return kernels_.size(); }

  template<fiat::scalar Element>
  [[nodiscard]]
  auto
  upload(std::span<const Element> elements)
    -> std::expected<device_array<Element>, error_t>
  {
    if (
      elements.empty() ||
      elements.size() > std::numeric_limits<std::uint32_t>::max())
    {
      return std::unexpected {
        make_app_error(app_error_code::invalid_data_size),
      };
    }
    return upload_device_local_buffer(
      buffer_upload_create_info_for<buffer_kind::device_address> {
        .device = device_,
        .pool   = pool_,
        .bytes  = std::as_bytes(elements),
      })
      .transform([ & ](device_address_buffer&& buffer) -> device_array<Element> {
        return device_array<Element> {
          std::move(buffer),
          static_cast<std::uint32_t>(elements.size()),
        };
      });
  }

  template<fiat::scalar Element>
  [[nodiscard]]
  auto
  allocate(std::uint32_t count) -> std::expected<device_array<Element>, error_t>
  {
    if (count == 0U)
    {
      return std::unexpected {
        make_app_error(app_error_code::invalid_data_size),
      };
    }
    return device_address_buffer::create<Element>(device_.allocator(), count)
      .transform(
        [ count ](device_address_buffer&& buffer) -> device_array<Element> {
          return device_array<Element> {std::move(buffer), count};
        });
  }

  template<fiat::scalar Element>
  [[nodiscard]]
  auto
  download(device_array<Element>& array)
    -> std::expected<std::vector<Element>, error_t>
  {
    return download_device_local_buffer_and_join(
      {
        .device = device_,
        .pool   = pool_,
        .source = array.buffer().buffer(),
        .size   = array.buffer().size(),
      })
      .and_then(
        [ & ](readback_buffer&& readback)
          -> std::expected<std::vector<Element>, error_t> {
          std::vector<Element> elements(array.size());
          return readback.copy_mapped_into(std::span {elements})
            .transform([ & ] -> std::vector<Element> {
              return std::move(elements);
            });
        });
  }

  [[nodiscard]]
  auto
  save_pipeline_cache() const -> std::expected<void, error_t>
  {
    if (pipeline_cache_path_.empty()) { return {}; }
    return cache_.data().and_then(
      [ & ](std::vector<std::uint8_t>&& bytes) -> std::expected<void, error_t> {
        return save_pipeline_cache_file(
          pipeline_cache_path_, std::as_bytes(std::span {bytes}));
      });
  }

  [[nodiscard]]
  auto
  device(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.device_); }

  [[nodiscard]]
  auto
  pool(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.pool_); }

  [[nodiscard]]
  auto
  group_size() const noexcept -> std::uint32_t
  { return group_size_; }

  [[nodiscard]]
  auto
  max_group_count() const noexcept -> std::uint32_t
  { return max_group_count_; }

private:
  static constexpr std::array validation_extensions {
    vk::EXTDebugUtilsExtensionName,
  };
  static constexpr std::array validation_layers {
    "VK_LAYER_KHRONOS_validation",
  };
  static constexpr std::array profiles {
    device_profile {
      .name     = "vkpp-gpu",
      .features = {
        .synchronization2      = capability_requirement::required,
        .buffer_device_address = capability_requirement::required,
        .maintenance4          = capability_requirement::required,
      },
    },
  };

  context(
    instance_context&&                instance,
    device_context&&                  device,
    command_pool&&                    pool,
    pipeline_cache&&                  cache,
    std::optional<diagnostic_buffer&> diagnostics,
    std::uint32_t                     group_size,
    std::uint32_t                     max_group_count,
    std::filesystem::path             pipeline_cache_path)
  : instance_ {std::move(instance)},
    device_ {std::move(device)},
    pool_ {std::move(pool)},
    cache_ {std::move(cache)},
    diagnostics_ {diagnostics},
    group_size_ {group_size},
    max_group_count_ {max_group_count},
    pipeline_cache_path_ {std::move(pipeline_cache_path)}
  {}

  [[nodiscard]]
  static
  auto
  create_cache(const device_context& device, const std::filesystem::path& path)
    -> std::expected<pipeline_cache, error_t>
  {
    if (path.empty()) { return pipeline_cache::create(device.device()); }
    return load_pipeline_cache_file(path).and_then(
      [ & ](std::vector<std::byte>&& bytes)
        -> std::expected<pipeline_cache, error_t> {
        const bool matches = pipeline_cache_header_matches(
          bytes, device.physical_device().getProperties());
        return pipeline_cache::create(
          device.device(),
          matches ? std::span {bytes} : std::span<const std::byte> {});
      });
  }

  [[nodiscard]]
  static
  auto
  assemble(instance_context&& instance, const context_create_info& create_info)
    -> std::expected<context, error_t>
  {
    auto device = device_context::create(
      instance,
      {
        .profiles        = profiles,
        .require_present = false,
      },
      create_info.diagnostics);
    if (!device) { return std::unexpected {device.error()}; }

    auto pool =
      command_pool::create(device->device(), device->graphics_qf_index());
    if (!pool) { return std::unexpected {pool.error()}; }

    auto cache = create_cache(*device, create_info.pipeline_cache_path);
    if (!cache) { return std::unexpected {cache.error()}; }

    const auto limits = device->physical_device().getProperties().limits;
    const auto clamped_group_size = std::clamp(
      create_info.group_size,
      1U,
      std::min(
        limits.maxComputeWorkGroupSize[ 0 ],
        limits.maxComputeWorkGroupInvocations));

    return context {
      std::move(instance),
      std::move(*device),
      std::move(*pool),
      std::move(*cache),
      create_info.diagnostics,
      clamped_group_size,
      limits.maxComputeWorkGroupCount[ 0 ],
      create_info.pipeline_cache_path,
    };
  }

  instance_context                  instance_;
  device_context                    device_;
  command_pool                      pool_;
  pipeline_cache                    cache_;
  std::deque<cached_kernel>         kernels_ {};
  std::optional<diagnostic_buffer&> diagnostics_ {};
  std::uint32_t                     group_size_ {};
  std::uint32_t                     max_group_count_ {};
  std::filesystem::path             pipeline_cache_path_ {};
};

export template<fiat::scalar Element, class Operation>
  requires fiat::traceable<Operation, Element>
class pipeline
{
public:
  using argument_tuple = decltype(fiat::transform_stage<Operation>::arguments);

  explicit pipeline(
    const device_array<Element>& source, argument_tuple arguments)
  : source_ {source}, arguments_ {std::move(arguments)}
  {}

  [[nodiscard]]
  auto
  source() const noexcept -> const device_array<Element>&
  { return source_; }

  [[nodiscard]]
  auto
  arguments() const noexcept -> const argument_tuple&
  { return arguments_; }

private:
  const device_array<Element>& source_;
  argument_tuple               arguments_;
};

inline constexpr std::string_view transform_mismatch {
  "this fiat::transform cannot run on this array: every runtime argument must "
  "have the array's element type, and the lambda must take (value, "
  "arguments...)",
};

export
template<fiat::scalar Element, class Operation>
[[nodiscard]]
auto
operator |(
  const device_array<Element>& source, fiat::transform_stage<Operation> stage)
{
  if constexpr (!fiat::traceable<Operation, Element>)
  {
    static_assert(false, transform_mismatch);
  }
  else
  {
    return pipeline<Element, Operation> {source, std::move(stage.arguments)};
  }
}

template<fiat::scalar Element, class Operation, class Fused>
[[nodiscard]]
auto
extend_pipeline(
  const pipeline<Element, Operation>& current,
  fiat::transform_stage<Fused>        fused)
{
  if constexpr (!fiat::traceable<Operation, Element>)
  {
    static_assert(false, transform_mismatch);
  }
  else
  {
    return pipeline<Element, Fused> {
      current.source(), std::move(fused.arguments)
    };
  }
}

export
template<fiat::scalar Element, class Operation, class Next>
[[nodiscard]]
auto
operator |(
  const pipeline<Element, Operation>& current, fiat::transform_stage<Next> next)
{
  return extend_pipeline(
    current,
    fiat::fuse(
      fiat::transform_stage<Operation> {.arguments = current.arguments()},
      fiat::transform_stage<Next> {}));
}

template<class... Arguments>
[[nodiscard]]
constexpr
auto
pack_arguments(const std::tuple<Arguments...>& arguments)
  -> std::array<std::uint32_t, fiat::argument_capacity>
{
  std::array<std::uint32_t, fiat::argument_capacity> words {};
  std::apply(
    [ & ](const Arguments&... argument) {
      const std::array<std::uint32_t, sizeof...(Arguments)> packed {
        std::bit_cast<std::uint32_t>(argument)...,
      };
      std::ranges::copy(packed, words.begin());
    },
    arguments);
  return words;
}

export
template<fiat::scalar Element, class Operation>
[[nodiscard]]
auto
record(
  vk::raii::CommandBuffer&            command_buffer,
  context&                            context,
  const pipeline<Element, Operation>& computation,
  device_array<Element>&              output) -> std::expected<void, error_t>
{
  const auto count = computation.source().size();
  if (output.size() != count)
  {
    return std::unexpected {make_app_error(app_error_code::invalid_data_size)};
  }

  auto kernel = context.kernel<Element, Operation>();
  if (!kernel) { return std::unexpected {kernel.error()}; }

  auto input_view  = computation.source().addressed_view(context.device());
  auto output_view = output.addressed_view(context.device());
  if (!input_view) { return std::unexpected {input_view.error()}; }
  if (!output_view) { return std::unexpected {output_view.error()}; }

  const compute_pipeline& compute = kernel->get();
  command_buffer.bindPipeline(
    vk::PipelineBindPoint::eCompute, *compute.pipeline());

  const std::uint32_t group_size = context.group_size();
  const auto          stride     = static_cast<std::uint32_t>(std::min(
    std::uint64_t {context.max_group_count()} * group_size,
    std::uint64_t {count}));

  const auto arguments = pack_arguments(computation.arguments());

  for (
    const auto first : std::views::indices(count) | std::views::stride(stride))
  {
    const auto elements = std::min(count - first, stride);
    push_compute<descriptor_table_backend::classic>(
      command_buffer,
      compute,
      fiat::transform_push {
        .input     = input_view->device_address(),
        .output    = output_view->device_address(),
        .count     = count,
        .first     = first,
        .arguments = arguments,
      });
    dispatch(command_buffer, (elements + group_size - 1U) / group_size);
  }
  return {};
}

export
template<fiat::scalar Element, class Operation>
[[nodiscard]]
auto
run(
  context&                            context,
  const pipeline<Element, Operation>& computation,
  device_array<Element>&              output) -> std::expected<void, error_t>
{
  single_time_submit submit {
    context.pool(),
    context.device().device(),
    context.device().graphics_queue(),
  };
  return submit.begin()
    .and_then([ & ] -> std::expected<void, error_t> {
      return record(submit.command_buffer(), context, computation, output);
    })
    .and_then([ & ] -> std::expected<void, error_t> {
      const vk::MemoryBarrier2 written {
        .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
      };
      submit.command_buffer().pipelineBarrier2({
        .memoryBarrierCount = 1U,
        .pMemoryBarriers    = &written,
      });
      return submit.end_and_submit(upload::wait_idle);
    });
}

export
template<fiat::scalar Element, class Operation>
[[nodiscard]]
auto
run(context& context, const pipeline<Element, Operation>& computation)
  -> std::expected<device_array<Element>, error_t>
{
  return context.allocate<Element>(computation.source().size())
    .and_then(
      [ & ](device_array<Element>&& output)
        -> std::expected<device_array<Element>, error_t> {
        return run(context, computation, output)
          .transform([ & ] -> device_array<Element> {
            return std::move(output);
          });
      });
}

export struct execute_stage
{
  gpu::context& context;
};

export [[nodiscard]]
inline
auto
execute(context& context) -> execute_stage
{ return execute_stage {context}; }

export
template<fiat::scalar Element, class Operation>
[[nodiscard]]
auto
operator |(const pipeline<Element, Operation>& computation, execute_stage stage)
  -> std::expected<device_array<Element>, error_t>
{ return run(stage.context, computation); }

} // namespace vkpp::gpu
