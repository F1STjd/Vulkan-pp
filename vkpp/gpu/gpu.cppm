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

export template<fiat::scalar Element>
class device_scalar
{
public:
  using element_type = Element;

  device_scalar(device_address_buffer&& buffer, std::uint64_t address)
  : buffer_ {std::move(buffer)}, address_ {address}
  {}

  [[nodiscard]]
  auto
  address() const noexcept -> std::uint64_t
  { return address_; }

  [[nodiscard]]
  auto
  buffer() const noexcept -> const device_address_buffer&
  { return buffer_; }

private:
  device_address_buffer buffer_;
  std::uint64_t         address_ {};
};

export struct context_create_info
{
  cstring_view                      name {"vkpp-gpu"};
  bool                              enable_validation {false};
  std::optional<diagnostic_buffer&> diagnostics {};
  std::uint32_t                     group_size {256U};
  std::filesystem::path             pipeline_cache_path {};
};

namespace detail
{

template<fiat::scalar Element, class Stage>
struct kernel_traits;

template<fiat::scalar Element, class Operation>
  requires fiat::traceable<Operation, Element>
struct kernel_traits<Element, fiat::transform_stage<Operation>>
{
  static constexpr auto words =
    fiat::compile<Element>(fiat::transform_stage<Operation> {});
  static constexpr std::string_view entry {fiat::transform_entry};
  static constexpr std::size_t push_size {sizeof(fiat::detail::transform_push)};
};

template<fiat::scalar Element, class Map, class Combine, class Identity>
  requires (
    fiat::traceable<Map, Element> &&
    fiat::combinable<Combine, Element> &&
    std::same_as<Identity, Element>)
struct kernel_traits<Element, fiat::reduce_stage<Map, Combine, Identity>>
{
  static constexpr auto words =
    fiat::compile<Element>(fiat::reduce_stage<Map, Combine, Identity> {});
  static constexpr std::string_view entry {fiat::reduce_entry};
  static constexpr std::size_t push_size {sizeof(fiat::detail::reduce_push)};
};

template<fiat::scalar Element, class Uniform, class Operation>
  requires fiat::uniform_traceable<Operation, Element, Uniform>
struct kernel_traits<Element, fiat::transform_with_stage<Uniform, Operation>>
{
  static constexpr auto words =
    fiat::compile<Element>(fiat::transform_with_stage<Uniform, Operation> {});
  static constexpr std::string_view entry {fiat::transform_entry};
  static constexpr std::size_t push_size {sizeof(fiat::detail::transform_push)};
};

} // namespace detail

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
  create(context_create_info create_info) -> std::expected<context, error_t>
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
          return assemble(std::move(instance), std::move(create_info));
        });
  }

  template<fiat::scalar Element, class Stage>
  [[nodiscard]]
  auto
  kernel()
    -> std::expected<std::reference_wrapper<const compute_pipeline>, error_t>
  {
    using traits = detail::kernel_traits<Element, Stage>;
    constexpr std::span<const std::uint32_t> words {traits::words};
    constexpr std::uint64_t fingerprint = fiat::fingerprint(traits::words);

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
        .push_constant_size = static_cast<std::uint32_t>(traits::push_size),
      },
      {
        .words          = words,
        .entry          = cstring_view {traits::entry},
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

  [[nodiscard]]
  auto
  reduce_scratch(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.reduce_scratch_); }

  [[nodiscard]]
  auto
  reduce_readback(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.reduce_readback_); }

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
        .vulkan_memory_model   = capability_requirement::required,
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
    std::filesystem::path             pipeline_cache_path,
    device_address_buffer&&           reduce_scratch,
    readback_buffer&&                 reduce_readback)
  : instance_ {std::move(instance)},
    device_ {std::move(device)},
    pool_ {std::move(pool)},
    cache_ {std::move(cache)},
    diagnostics_ {diagnostics},
    group_size_ {group_size},
    max_group_count_ {max_group_count},
    pipeline_cache_path_ {std::move(pipeline_cache_path)},
    reduce_scratch_ {std::move(reduce_scratch)},
    reduce_readback_ {std::move(reduce_readback)}
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
  assemble(instance_context&& instance, context_create_info create_info)
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
    const auto group_size = std::bit_floor(clamped_group_size);

    const std::array<std::byte, fiat::detail::reduce_scratch_bytes> zeros {};
    auto reduce_scratch = upload_device_local_buffer(
      buffer_upload_create_info_for<buffer_kind::device_address> {
        .device = *device,
        .pool   = *pool,
        .bytes  = std::as_bytes(std::span {zeros}),
      });
    if (!reduce_scratch) { return std::unexpected {reduce_scratch.error()}; }

    auto reduce_readback =
      readback_buffer::create(device->allocator(), sizeof(std::uint32_t));
    if (!reduce_readback) { return std::unexpected {reduce_readback.error()}; }

    return context {
      std::move(instance),
      std::move(*device),
      std::move(*pool),
      std::move(*cache),
      create_info.diagnostics,
      group_size,
      limits.maxComputeWorkGroupCount[ 0 ],
      std::move(create_info.pipeline_cache_path),
      std::move(*reduce_scratch),
      std::move(*reduce_readback),
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
  device_address_buffer             reduce_scratch_;
  readback_buffer                   reduce_readback_;
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
  if constexpr (!fiat::traceable<Fused, Element>)
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
      std::move(next)));
}

export template<fiat::scalar Element, fiat::scalar Uniform, class Operation>
  requires fiat::uniform_traceable<Operation, Element, Uniform>
class transform_with_pipeline
{
public:
  using argument_tuple =
    decltype(fiat::transform_with_stage<Uniform, Operation>::arguments);

  explicit transform_with_pipeline(
    const device_array<Element>& source,
    std::uint64_t                address,
    argument_tuple               arguments)
  : source_ {source}, address_ {address}, arguments_ {std::move(arguments)}
  {}

  [[nodiscard]]
  auto
  source() const noexcept -> const device_array<Element>&
  { return source_; }

  [[nodiscard]]
  auto
  address() const noexcept -> std::uint64_t
  { return address_; }

  [[nodiscard]]
  auto
  arguments() const noexcept -> const argument_tuple&
  { return arguments_; }

private:
  const device_array<Element>& source_ {};
  std::uint64_t                address_ {};
  argument_tuple               arguments_;
};

inline constexpr std::string_view transform_with_mismatch {
  "this fiat::transform_with cannot run on this array: the uniform element "
  "type must match the array element type, every further runtime argument must "
  "have that element type, and the lambda must take (value, uniform, "
  "arguments...)"
};

export
template<fiat::scalar Element, fiat::scalar Uniform, class Operation>
[[nodiscard]]
auto
operator |(
  const device_array<Element>&                   source,
  fiat::transform_with_stage<Uniform, Operation> stage)
{
  if constexpr (!fiat::uniform_traceable<Operation, Element, Uniform>)
  {
    static_assert(false, transform_with_mismatch);
  }
  else
  {
    return transform_with_pipeline<Element, Uniform, Operation> {
      source,
      stage.address,
      std::move(stage.arguments),
    };
  }
}

export template<fiat::scalar Element, class Map, class Combine>
  requires (fiat::traceable<Map, Element> && fiat::combinable<Combine, Element>)
class reduce_pipeline
{
public:
  using argument_tuple =
    decltype(fiat::reduce_stage<Map, Combine, Element>::arguments);

  explicit reduce_pipeline(
    const device_array<Element>& source,
    argument_tuple               arguments,
    Element                      identity)
  : source_ {source}, arguments_ {std::move(arguments)}, identity_ {identity}
  {}

  [[nodiscard]]
  auto
  source() const noexcept -> const device_array<Element>&
  { return source_; }

  [[nodiscard]]
  auto
  arguments() const noexcept -> const argument_tuple&
  { return arguments_; }

  [[nodiscard]]
  auto
  identity() const noexcept -> Element
  { return identity_; }

private:
  const device_array<Element>& source_;
  argument_tuple               arguments_;
  Element                      identity_;
};

inline constexpr std::string_view reduce_mismatch {
  "this fiat::reduce cannot run on this array: the map and the combine must "
  "trace on the array's element type, and the identity must have that type"
};

export
template<fiat::scalar Element, class Map, class Combine, class Identity>
[[nodiscard]]
auto
operator |(
  const device_array<Element>&               source,
  fiat::reduce_stage<Map, Combine, Identity> stage)
{
  if constexpr (
    !fiat::traceable<Map, Element> ||
    !fiat::combinable<Combine, Element> ||
    !std::same_as<Identity, Element>)
  {
    static_assert(false, reduce_mismatch);
  }
  else
  {
    return reduce_pipeline<Element, Map, Combine> {
      source,
      std::move(stage.arguments),
      stage.identity,
    };
  }
}

export
template<fiat::scalar Element, class Operation, class Combine, class Identity>
[[nodiscard]]
auto
operator |(
  const pipeline<Element, Operation>&                  current,
  fiat::reduce_stage<fiat::chain<>, Combine, Identity> next)
{
  if constexpr (
    !fiat::traceable<Operation, Element> ||
    !fiat::combinable<Combine, Element> ||
    !std::same_as<Identity, Element>)
  {
    static_assert(false, reduce_mismatch);
  }
  else
  {
    auto fused =
      fiat::transform_stage<Operation> {.arguments = current.arguments()} |
      std::move(next);
    return reduce_pipeline<Element, Operation, Combine> {
      current.source(),
      std::move(fused.arguments),
      fused.identity,
    };
  }
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

namespace detail
{

template<class... Arguments>
[[nodiscard]]
constexpr
auto
pack_uniform_arguments(
  std::uint64_t address, const std::tuple<Arguments...>& arguments)
  -> std::array<std::uint32_t, fiat::argument_capacity>
{
  std::array<std::uint32_t, fiat::argument_capacity> words {};
  words[ 0 ] = static_cast<std::uint32_t>(address);
  words[ 1 ] = static_cast<std::uint32_t>(address >> 32U);
  std::apply(
    [ & ](const Arguments&... argument) {
      const std::array<std::uint32_t, sizeof...(Arguments)> packed {
        std::bit_cast<std::uint32_t>(argument)...,
      };
      std::ranges::copy(packed, words.begin() + 2);
    },
    arguments);
  return words;
}

} // namespace detail

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

  auto kernel = context.kernel<Element, fiat::transform_stage<Operation>>();
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
      fiat::detail::transform_push {
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
{ return execute_stage {.context = context}; }

export
template<fiat::scalar Element, class Operation>
[[nodiscard]]
auto
operator |(const pipeline<Element, Operation>& computation, execute_stage stage)
  -> std::expected<device_array<Element>, error_t>
{ return run(stage.context, computation); }

export struct keep_stage
{
  gpu::context& context;
};

export [[nodiscard]]
inline
auto
keep(context& context) -> keep_stage
{ return keep_stage {.context = context}; }

export
template<fiat::scalar Element, class Map, class Combine>
[[nodiscard]]
auto
operator |(
  const reduce_pipeline<Element, Map, Combine>& computation, keep_stage stage)
  -> std::expected<device_scalar<Element>, error_t>
{
  auto&              context = stage.context;
  single_time_submit submit {
    context.pool(),
    context.device().device(),
    context.device().graphics_queue(),
  };

  auto scalar_buffer =
    device_address_buffer::create<Element>(context.device().allocator(), 1U);
  if (!scalar_buffer) { return std::unexpected {scalar_buffer.error()}; }

  auto scalar_view = scalar_buffer->template addressed_view<Element>(
    context.device(), 0UZ, sizeof(Element));
  if (!scalar_view) { return std::unexpected {scalar_view.error()}; }

  const auto address = scalar_view->device_address();

  return submit.begin()
    .and_then([ & ] -> std::expected<void, error_t> {
      return record(submit.command_buffer(), context, computation);
    })
    .and_then([ & ] -> std::expected<void, error_t> {
      const vk::MemoryBarrier2 to_copy {
        .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask  = vk::PipelineStageFlagBits2::eCopy,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
      };
      submit.command_buffer().pipelineBarrier2({
        .memoryBarrierCount = 1U,
        .pMemoryBarriers    = &to_copy,
      });
      const vk::BufferCopy region {
        .srcOffset = 4UZ,
        .dstOffset = 0UZ,
        .size      = sizeof(Element),
      };
      submit.command_buffer().copyBuffer(
        context.reduce_scratch().buffer(), scalar_buffer->buffer(), region);
      const vk::MemoryBarrier2 to_shader {
        .srcStageMask  = vk::PipelineStageFlagBits2::eCopy,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask =
          vk::AccessFlagBits2::eShaderStorageRead |
          vk::AccessFlagBits2::eShaderStorageWrite,
      };
      submit.command_buffer().pipelineBarrier2({
        .memoryBarrierCount = 1U,
        .pMemoryBarriers    = &to_shader,
      });
      return submit.end_and_submit(upload::wait_idle);
    })
    .transform([ & ] -> device_scalar<Element> {
      return device_scalar<Element> {
        std::move(*scalar_buffer),
        address,
      };
    });
}

export
template<fiat::scalar Element, fiat::scalar Uniform, class Operation>
[[nodiscard]]
auto
record(
  vk::raii::CommandBuffer&                                    command_buffer,
  context&                                                    context,
  const transform_with_pipeline<Element, Uniform, Operation>& computation,
  device_array<Element>& output) -> std::expected<void, error_t>
{
  const auto count = computation.source().size();
  if (output.size() != count)
  {
    return std::unexpected {make_app_error(app_error_code::invalid_data_size)};
  }

  auto kernel =
    context.kernel<Element, fiat::transform_with_stage<Uniform, Operation>>();
  if (!kernel) { return std::unexpected {kernel.error()}; }

  auto input_view  = computation.source().addressed_view(context.device());
  auto output_view = output.addressed_view(context.device());
  if (!input_view) { return std::unexpected {input_view.error()}; }
  if (!output_view) { return std::unexpected {output_view.error()}; }

  const auto& compute = kernel->get();
  command_buffer.bindPipeline(
    vk::PipelineBindPoint::eCompute, *compute.pipeline());

  const auto group_size = context.group_size();
  const auto stride     = static_cast<std::uint32_t>(std::min(
    std::uint64_t {context.max_group_count()} * group_size,
    std::uint64_t {count}));

  const auto arguments = detail::pack_uniform_arguments(
    computation.address(), computation.arguments());

  for (
    const auto first : std::views::indices(count) | std::views::stride(stride))
  {
    const auto elements = std::min(count - first, stride);
    push_compute<descriptor_table_backend::classic>(
      command_buffer,
      compute,
      fiat::detail::transform_push {
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
template<fiat::scalar Element, fiat::scalar Uniform, class Operation>
[[nodiscard]]
auto
run(
  context&                                                    context,
  const transform_with_pipeline<Element, Uniform, Operation>& computation,
  device_array<Element>& output) -> std::expected<void, error_t>
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
template<fiat::scalar Element, fiat::scalar Uniform, class Operation>
[[nodiscard]]
auto
run(
  context&                                                    context,
  const transform_with_pipeline<Element, Uniform, Operation>& computation)
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

export
template<fiat::scalar Element, fiat::scalar Uniform, class Operation>
[[nodiscard]]
auto
operator |(
  const transform_with_pipeline<Element, Uniform, Operation>& computation,
  execute_stage stage) -> std::expected<device_array<Element>, error_t>
{ return run(stage.context, computation); }

export
template<fiat::scalar Element, class Map, class Combine>
[[nodiscard]]
auto
record(
  vk::raii::CommandBuffer&                      command_buffer,
  context&                                      context,
  const reduce_pipeline<Element, Map, Combine>& computation)
  -> std::expected<void, error_t>
{
  const auto count      = computation.source().size();
  const auto group_size = context.group_size();
  if (
    count == 0U ||
    count > 0xFF'FF'FFU - fiat::detail::reduce_group_limit * group_size)
  {
    return std::unexpected {make_app_error(app_error_code::invalid_data_size)};
  }

  auto kernel =
    context.kernel<Element, fiat::reduce_stage<Map, Combine, Element>>();
  if (!kernel) { return std::unexpected {kernel.error()}; }

  auto input_view = computation.source().addressed_view(context.device());
  if (!input_view) { return std::unexpected {input_view.error()}; }

  auto scratch_view =
    context.reduce_scratch().template addressed_view<std::byte>(
      context.device(), 0UZ, fiat::detail::reduce_scratch_bytes);
  if (!scratch_view) { return std::unexpected {scratch_view.error()}; }

  const vk::MemoryBarrier2 reuse {
    .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
    .srcAccessMask =
      vk::AccessFlagBits2::eShaderStorageRead |
      vk::AccessFlagBits2::eShaderStorageWrite,
    .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
    .dstAccessMask =
      vk::AccessFlagBits2::eShaderStorageRead |
      vk::AccessFlagBits2::eShaderStorageWrite,
  };
  command_buffer.pipelineBarrier2({
    .memoryBarrierCount = 1U,
    .pMemoryBarriers    = &reuse,
  });

  const compute_pipeline& compute = kernel->get();
  command_buffer.bindPipeline(
    vk::PipelineBindPoint::eCompute, *compute.pipeline());

  const auto groups = std::min(
    (count + group_size - 1U) / group_size, fiat::detail::reduce_group_limit);
  const auto arguments = pack_arguments(computation.arguments());

  push_compute<descriptor_table_backend::classic>(
    command_buffer,
    compute,
    fiat::detail::reduce_push {
      .input     = input_view->device_address(),
      .scratch   = scratch_view->device_address(),
      .count     = count,
      .identity  = std::bit_cast<std::uint32_t>(computation.identity()),
      .arguments = arguments,
    });
  dispatch(command_buffer, groups);
  return {};
}

export
template<fiat::scalar Element, class Map, class Combine>
[[nodiscard]]
auto
run(context& context, const reduce_pipeline<Element, Map, Combine>& computation)
  -> std::expected<Element, error_t>
{
  single_time_submit submit {
    context.pool(),
    context.device().device(),
    context.device().graphics_queue(),
  };
  return submit.begin()
    .and_then([ & ] -> std::expected<void, error_t> {
      return record(submit.command_buffer(), context, computation);
    })
    .and_then([ & ] -> std::expected<void, error_t> {
      const vk::MemoryBarrier2 to_copy {
        .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask  = vk::PipelineStageFlagBits2::eCopy,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
      };
      submit.command_buffer().pipelineBarrier2({
        .memoryBarrierCount = 1U,
        .pMemoryBarriers    = &to_copy,
      });
      const vk::BufferCopy region {
        .srcOffset = 4UZ,
        .dstOffset = 0UZ,
        .size      = sizeof(Element),
      };
      submit.command_buffer().copyBuffer(
        context.reduce_scratch().buffer(),
        context.reduce_readback().buffer(),
        region);
      const vk::MemoryBarrier2 to_host {
        .srcStageMask  = vk::PipelineStageFlagBits2::eCopy,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask  = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
      };
      submit.command_buffer().pipelineBarrier2({
        .memoryBarrierCount = 1U,
        .pMemoryBarriers    = &to_host,
      });
      return submit.end_and_submit(upload::wait_idle);
    })
    .and_then([ & ] -> std::expected<Element, error_t> {
      Element value {};
      return context.reduce_readback()
        .copy_mapped_into(std::span {&value, 1UZ})
        .transform([ & ] -> Element { return value; });
    });
}

export
template<fiat::scalar Element, class Map, class Combine>
[[nodiscard]]
auto
operator |(
  const reduce_pipeline<Element, Map, Combine>& computation,
  execute_stage stage) -> std::expected<Element, error_t>
{ return run(stage.context, computation); }

} // namespace vkpp::gpu
