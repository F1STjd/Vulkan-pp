module;
#if defined(__cpp_impl_reflection)
  #include <meta>
#endif

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

export template<fiat::soa_element Element>
class soa_array
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
  base_address(const device_context& device) const
    -> std::expected<std::uint64_t, error_t>
  {
    return buffer_
      .template addressed_view<std::uint32_t>(device, 0UZ, buffer_.size())
      .transform([](const auto& view) -> std::uint64_t {
        return view.device_address();
      });
  }

private:
  friend class context;

  soa_array(device_address_buffer&& buffer, std::uint32_t count)
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

template<class Element, class Stage>
struct kernel_traits;

template<fiat::scalar Element, class Operation>
  requires fiat::traceable<Operation, Element>
struct kernel_traits<Element, fiat::transform_stage<Operation>>
{
  static constexpr auto words =
    fiat::compile<Element>(fiat::transform_stage<Operation> {});
  static constexpr std::array  entries {fiat::transform_entry};
  static constexpr std::size_t push_size {sizeof(fiat::detail::transform_push)};
};

template<
  fiat::scalar Element,
  class Map,
  class Combine,
  class Identity,
  class Epilogue
>
  requires (
    fiat::traceable<Map, Element> &&
    fiat::combinable<Combine, Element> &&
    std::same_as<Identity, Element> &&
    fiat::traceable<Epilogue, Element>)
struct kernel_traits<
  Element,
  fiat::reduce_stage<Map, Combine, Identity, Epilogue>
>
{
  static constexpr auto words = fiat::compile<Element>(
    fiat::reduce_stage<Map, Combine, Identity, Epilogue> {});
  static constexpr std::array  entries {fiat::reduce_entry};
  static constexpr std::size_t push_size {sizeof(fiat::detail::reduce_push)};
};

template<fiat::scalar Element, class Uniform, class Operation>
  requires fiat::uniform_traceable<Operation, Element, Uniform>
struct kernel_traits<Element, fiat::transform_with_stage<Uniform, Operation>>
{
  static constexpr auto words =
    fiat::compile<Element>(fiat::transform_with_stage<Uniform, Operation> {});
  static constexpr std::array  entries {fiat::transform_entry};
  static constexpr std::size_t push_size {sizeof(fiat::detail::transform_push)};
};

template<fiat::soa_element Element, class Operation>
struct kernel_traits<Element, fiat::foreach_stage<Operation>>
{
  static constexpr auto words =
    fiat::compile<Element>(fiat::foreach_stage<Operation> {});
  static constexpr std::array  entries {fiat::foreach_entry};
  static constexpr std::size_t push_size {sizeof(fiat::detail::foreach_push)};
};

template<
  fiat::scalar Element,
  class Map,
  class Combine,
  class Identity,
  fiat::scan_mode Mode,
  class Epilogue
>
  requires (
    fiat::traceable<Map, Element> &&
    fiat::combinable<Combine, Element> &&
    std::same_as<Identity, Element> &&
    fiat::traceable<Epilogue, Element>)
struct kernel_traits<
  Element,
  fiat::scan_stage<Map, Combine, Identity, Mode, Epilogue>
>
{
  static constexpr auto words = fiat::compile<Element>(
    fiat::scan_stage<Map, Combine, Identity, Mode, Epilogue> {});
  static constexpr std::array entries {
    fiat::scan_upsweep_entry,
    fiat::scan_downsweep_entry,
  };
  static constexpr std::size_t push_size {sizeof(fiat::detail::scan_push)};
};

inline constexpr std::size_t scan_upsweep_index {0UZ};
inline constexpr std::size_t scan_downsweep_index {1UZ};

#if defined(__cpp_impl_reflection)

template<fiat::soa_element Element>
void
transpose_to_soa_bytes(
  std::span<const Element> host, std::span<std::byte> staging)
{
  const auto            count = static_cast<std::uint32_t>(host.size());
  std::size_t           leaf_index {};
  static constexpr auto members = std::define_static_array(
    std::meta::nonstatic_data_members_of(
      ^^Element, std::meta::access_context::unchecked()));
  template for (constexpr auto member : members)
  {
    using leaf_type = [:std::meta::type_of(member):];
    static_assert(fiat::scalar<leaf_type>);
    const auto byte_offset = static_cast<std::size_t>(
      fiat::detail::region_offset<Element>(count, leaf_index));
    for (auto element_index : std::views::indices(count))
    {
      const leaf_type value = host[ element_index ].[:member:];
      std::memcpy(
        staging.data() +
          byte_offset +
          static_cast<std::size_t>(element_index) * sizeof(leaf_type),
        &value,
        sizeof(leaf_type));
    }
    ++leaf_index;
  }
}

template<fiat::soa_element Element>
void
transpose_from_soa_bytes(
  std::span<const std::byte> staging, std::span<Element> host)
{
  const auto            count = static_cast<std::uint32_t>(host.size());
  std::size_t           leaf_index {};
  static constexpr auto members = std::define_static_array(
    std::meta::nonstatic_data_members_of(
      ^^Element, std::meta::access_context::current()));
  template for (constexpr auto member : members)
  {
    using leaf_type = [:std::meta::type_of(member):];
    static_assert(fiat::scalar<leaf_type>);
    const auto byte_offset = static_cast<std::size_t>(
      fiat::detail::region_offset<Element>(count, leaf_index));
    for (auto element_index : std::views::indices(count))
    {
      leaf_type value {};
      std::memcpy(
        &value,
        staging.data() +
          byte_offset +
          static_cast<std::size_t>(element_index) * sizeof(leaf_type),
        sizeof(leaf_type));
      host[ element_index ].[:member:] = value;
    }
    ++leaf_index;
  }
}

#else

template<fiat::soa_element Element>
void
transpose_to_soa_bytes(std::span<const Element>, std::span<std::byte>)
{}

template<fiat::soa_element Element>
void
transpose_from_soa_bytes(std::span<const std::byte>, std::span<Element>)
{}

#endif

} // namespace detail

struct cached_kernel
{
  std::uint64_t                  fingerprint {};
  std::string_view               entry {};
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

  template<class Element, class Stage, std::size_t EntryIndex = 0UZ>
    requires fiat::scalar<Element> || fiat::soa_element<Element>
  [[nodiscard]]
  auto
  kernel()
    -> std::expected<std::reference_wrapper<const compute_pipeline>, error_t>
  {
    using traits = detail::kernel_traits<Element, Stage>;
    static_assert(
      EntryIndex < traits::entries.size(),
      "this kernel has no entry point with that index");
    constexpr std::span<const std::uint32_t> words {traits::words};
    constexpr std::uint64_t    fingerprint = fiat::fingerprint(traits::words);
    constexpr std::string_view entry       = traits::entries[ EntryIndex ];

    const auto found = std::ranges::find_if(
      kernels_, [ & ](const cached_kernel& cached) -> bool {
        return cached.fingerprint == fingerprint &&
               cached.entry == entry &&
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
        .entry          = cstring_view {entry},
        .specialization = specialization,
      },
      cache_.get(),
      diagnostics_)
      .transform(
        [ & ](compute_pipeline&& created)
          -> std::reference_wrapper<const compute_pipeline> {
          kernels_.push_back({
            .fingerprint = fingerprint,
            .entry       = entry,
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

  template<fiat::soa_element Element>
  [[nodiscard]]
  auto
  upload_soa([[maybe_unused]] std::span<const Element> host)
    -> std::expected<soa_array<Element>, error_t>
  {
#if !defined(__cpp_impl_reflection)
    return std::unexpected {make_app_error(app_error_code::invalid_data_size)};
#else
    if (host.empty() || host.size() > std::numeric_limits<std::uint32_t>::max())
    {
      return std::unexpected {
        make_app_error(app_error_code::invalid_data_size),
      };
    }
    const auto             count      = static_cast<std::uint32_t>(host.size());
    const auto             byte_count = fiat::detail::soa_bytes<Element>(count);
    std::vector<std::byte> staging(static_cast<std::size_t>(byte_count));
    detail::transpose_to_soa_bytes<Element>(host, staging);
    return upload_device_local_buffer(
      buffer_upload_create_info_for<buffer_kind::device_address> {
        .device = device_,
        .pool   = pool_,
        .bytes  = std::span {staging},
      })
      .transform(
        [ count ](device_address_buffer&& buffer) -> soa_array<Element> {
          return soa_array<Element> {std::move(buffer), count};
        });
#endif
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

  template<fiat::soa_element Element>
  [[nodiscard]]
  auto
  download_soa([[maybe_unused]] soa_array<Element>& array)
    -> std::expected<std::vector<Element>, error_t>
  {
#if !defined(__cpp_impl_reflection)
    return std::unexpected {make_app_error(app_error_code::invalid_data_size)};
#else
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
          std::vector<std::byte> staging(
            static_cast<std::size_t>(
              fiat::detail::soa_bytes<Element>(array.size())));
          return readback.copy_mapped_into(std::span {staging})
            .transform([ & ] -> std::vector<Element> {
              std::vector<Element> host(array.size());
              detail::transpose_from_soa_bytes<Element>(staging, host);
              return host;
            });
        });
#endif
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

namespace detail
{

inline constexpr vk::MemoryBarrier2 compute_write_to_any_read {
  .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
  .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
  .dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands,
  .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
};

inline constexpr vk::MemoryBarrier2 compute_reuse {
  .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
  .srcAccessMask =
    vk::AccessFlagBits2::eShaderStorageRead |
    vk::AccessFlagBits2::eShaderStorageWrite,
  .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
  .dstAccessMask =
    vk::AccessFlagBits2::eShaderStorageRead |
    vk::AccessFlagBits2::eShaderStorageWrite,
};

inline constexpr vk::MemoryBarrier2 compute_write_to_copy_read {
  .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
  .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
  .dstStageMask  = vk::PipelineStageFlagBits2::eCopy,
  .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
};

inline constexpr vk::MemoryBarrier2 copy_write_to_compute {
  .srcStageMask  = vk::PipelineStageFlagBits2::eCopy,
  .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
  .dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
  .dstAccessMask =
    vk::AccessFlagBits2::eShaderStorageRead |
    vk::AccessFlagBits2::eShaderStorageWrite,
};

inline constexpr vk::MemoryBarrier2 copy_write_to_host_read {
  .srcStageMask  = vk::PipelineStageFlagBits2::eCopy,
  .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
  .dstStageMask  = vk::PipelineStageFlagBits2::eHost,
  .dstAccessMask = vk::AccessFlagBits2::eHostRead,
};

inline
void
record_barrier_impl(
  vk::raii::CommandBuffer& command_buffer, const vk::MemoryBarrier2& barrier)
{
  command_buffer.pipelineBarrier2({
    .memoryBarrierCount = 1U,
    .pMemoryBarriers    = &barrier,
  });
}

template<std::invocable<vk::raii::CommandBuffer&> Record>
  requires std::same_as<
    std::invoke_result_t<Record&, vk::raii::CommandBuffer&>,
    std::expected<void, error_t>
  >
[[nodiscard]]
auto
submit_impl(context& context, Record&& record) -> std::expected<void, error_t>
{
  single_time_submit submit {
    context.pool(),
    context.device().device(),
    context.device().graphics_queue(),
  };
  return submit.begin()
    .and_then([ & ] -> std::expected<void, error_t> {
      return std::invoke(record, submit.command_buffer());
    })
    .and_then([ & ] -> std::expected<void, error_t> {
      return submit.end_and_submit(upload::wait_idle);
    });
}

inline constexpr std::string_view transform_mismatch {
  "this fiat::transform cannot run on this array: every runtime argument must "
  "have the array's element type, and the lambda must take (value, "
  "arguments...)",
};

inline constexpr std::string_view transform_with_mismatch {
  "this fiat::transform_with cannot run on this array: the uniform element "
  "type must match the array element type, every further runtime argument must "
  "have that element type, and the lambda must take (value, uniform, "
  "arguments...)"
};

inline constexpr std::string_view foreach_mismatch {
  "this fiat::for_each cannot run on this soa_array: the callable must take "
  "(traced_struct<Element>&, runtime scalar arguments...) and every runtime "
  "argument must be float, std::int32_t, or std::uint32_t"
};

inline constexpr std::string_view reduce_mismatch {
  "this fiat::reduce cannot run on this array: the map and the combine must "
  "trace on the array's element type, and the identity must have that type"
};

inline constexpr std::string_view scan_mismatch {
  "this fiat::scan cannot run on this array: the map and the combine must "
  "trace on the array's element type, and the identity must have that type"
};

template<std::size_t first_slot, class... Arguments>
[[nodiscard]]
constexpr
auto
pack_arguments_impl(const std::tuple<Arguments...>& arguments)
  -> std::array<std::uint32_t, fiat::argument_capacity>
{
  std::array<std::uint32_t, fiat::argument_capacity> words {};
  std::apply(
    [ & ](const Arguments&... argument) {
      std::array<std::uint32_t, sizeof...(Arguments)> packed {
        std::bit_cast<std::uint32_t>(argument)...,
      };
      std::ranges::copy(packed, words.begin() + first_slot);
    },
    arguments);
  return words;
}

template<class MapArguments, class EpilogueArguments>
[[nodiscard]]
constexpr
auto
pack_reduce_arguments_impl(
  const MapArguments&      map_arguments,
  const EpilogueArguments& epilogue_arguments)
  -> std::array<std::uint32_t, fiat::argument_capacity>
{
  static_assert(
    std::tuple_size_v<MapArguments> + std::tuple_size_v<EpilogueArguments> <=
      fiat::argument_capacity,
    "reduce map and epilogue arguments together take at most 26 slots");
  auto words = pack_arguments_impl<0UZ>(map_arguments);
  std::apply(
    [ & ]<class... Arguments>(const Arguments&... argument) {
      std::array<std::uint32_t, sizeof...(Arguments)> packed {
        std::bit_cast<std::uint32_t>(argument)...,
      };
      std::ranges::copy(
        packed, words.begin() + std::tuple_size_v<MapArguments>);
    },
    epilogue_arguments);
  return words;
}

template<class MapArguments, class EpilogueArguments>
[[nodiscard]]
constexpr
auto
pack_scan_arguments_impl(
  const MapArguments&      map_arguments,
  const EpilogueArguments& epilogue_arguments)
  -> std::array<std::uint32_t, fiat::scan_argument_capacity>
{
  static_assert(
    std::tuple_size_v<MapArguments> + std::tuple_size_v<EpilogueArguments> <=
      fiat::scan_argument_capacity,
    "scan map and epilogue arguments together take at most 24 slots");
  std::array<std::uint32_t, fiat::scan_argument_capacity> words {};
  std::apply(
    [ & ]<class... Arguments>(const Arguments&... argument) {
      std::array<std::uint32_t, sizeof...(Arguments)> packed {
        std::bit_cast<std::uint32_t>(argument)...,
      };
      std::ranges::copy(packed, words.begin());
    },
    map_arguments);
  std::apply(
    [ & ]<class... Arguments>(const Arguments&... argument) {
      std::array<std::uint32_t, sizeof...(Arguments)> packed {
        std::bit_cast<std::uint32_t>(argument)...,
      };
      std::ranges::copy(
        packed, words.begin() + std::tuple_size_v<MapArguments>);
    },
    epilogue_arguments);
  return words;
}

template<class... Arguments>
[[nodiscard]]
constexpr
auto
pack_uniform_arguments_impl(
  std::uint64_t address, const std::tuple<Arguments...>& arguments)
  -> std::array<std::uint32_t, fiat::argument_capacity>
{
  auto words = pack_arguments_impl<2UZ>(arguments);
  words[ 0 ] = static_cast<std::uint32_t>(address);
  words[ 1 ] = static_cast<std::uint32_t>(address >> 32U);
  return words;
}

template<std::invocable<std::uint32_t> MakePush>
void
record_chunks_impl(
  vk::raii::CommandBuffer& command_buffer,
  const context&           context,
  const compute_pipeline&  compute,
  std::uint32_t            count,
  MakePush&&               make_push)
{
  command_buffer.bindPipeline(
    vk::PipelineBindPoint::eCompute, *compute.pipeline());
  const std::uint32_t group_size = context.group_size();
  const auto          stride     = static_cast<std::uint32_t>(std::min(
    std::uint64_t {context.max_group_count()} * group_size,
    std::uint64_t {count}));
  for (
    const auto first : std::views::indices(count) | std::views::stride(stride))
  {
    const auto first_element = static_cast<std::uint32_t>(first);
    const auto elements      = std::min(count - first_element, stride);
    push_compute<descriptor_table_backend::classic>(
      command_buffer, compute, make_push(first_element));
    dispatch(command_buffer, (elements + group_size - 1U) / group_size);
  }
}

template<fiat::scalar Element, class Stage>
[[nodiscard]]
auto
record_transform_impl(
  vk::raii::CommandBuffer&                                  command_buffer,
  context&                                                  context,
  const device_array<Element>&                              source,
  device_array<Element>&                                    output,
  const std::array<std::uint32_t, fiat::argument_capacity>& arguments)
  -> std::expected<void, error_t>
{
  const auto count = source.size();
  if (output.size() != count)
  {
    return std::unexpected {make_app_error(app_error_code::invalid_data_size)};
  }

  auto kernel = context.kernel<Element, Stage>();
  if (!kernel) { return std::unexpected {kernel.error()}; }

  auto input_view  = source.addressed_view(context.device());
  auto output_view = output.addressed_view(context.device());
  if (!input_view) { return std::unexpected {input_view.error()}; }
  if (!output_view) { return std::unexpected {output_view.error()}; }

  const auto input_address  = input_view->device_address();
  const auto output_address = output_view->device_address();
  record_chunks_impl(
    command_buffer, context, kernel->get(), count, [ & ](std::uint32_t first) {
      return fiat::detail::transform_push {
        .input     = input_address,
        .output    = output_address,
        .count     = count,
        .first     = first,
        .arguments = arguments,
      };
    });
  return {};
}

template<fiat::scalar Element, class Computation>
[[nodiscard]]
auto
run_into_new_array_impl(context& context, const Computation& computation)
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

template<fiat::scalar Element>
[[nodiscard]]
auto
allocate_scalar_impl(context& context)
  -> std::expected<device_scalar<Element>, error_t>
{
  auto buffer =
    device_address_buffer::create<Element>(context.device().allocator(), 1U);
  if (!buffer) { return std::unexpected {buffer.error()}; }

  auto view = buffer->template addressed_view<Element>(
    context.device(), 0UZ, sizeof(Element));
  if (!view) { return std::unexpected {view.error()}; }

  const auto address = view->device_address();
  return device_scalar<Element> {std::move(*buffer), address};
}

template<fiat::scalar Element>
void
record_result_copy_impl(
  vk::raii::CommandBuffer&  command_buffer,
  vk::Buffer                source,
  vk::DeviceSize            source_offset,
  vk::Buffer                destination,
  const vk::MemoryBarrier2& after)
{
  record_barrier_impl(command_buffer, compute_write_to_copy_read);
  const vk::BufferCopy region {
    .srcOffset = source_offset,
    .dstOffset = 0UZ,
    .size      = sizeof(Element),
  };
  command_buffer.copyBuffer(source, destination, region);
  record_barrier_impl(command_buffer, after);
}

template<fiat::scalar Element>
[[nodiscard]]
auto
read_scalar_impl(context& context) -> std::expected<Element, error_t>
{
  Element value {};
  return context.reduce_readback()
    .copy_mapped_into(std::span {&value, 1UZ})
    .transform([ & ] -> Element { return value; });
}

[[nodiscard]]
auto
check_count_impl(std::uint32_t count, std::uint32_t group_size)
  -> std::expected<void, error_t>
{
  if (
    count == 0U ||
    count > 0xFF'FF'FFU - fiat::detail::reduce_group_limit * group_size)
  {
    return std::unexpected {make_app_error(app_error_code::invalid_data_size)};
  }
  return {};
}

struct scan_partition
{
  std::uint32_t tiles {};
  std::uint32_t tiles_per_group {};
  std::uint32_t groups {};
};

[[nodiscard]]
constexpr
auto
scan_partition_impl(std::uint32_t count, std::uint32_t group_size)
  -> scan_partition
{
  const std::uint32_t tiles = (count + group_size - 1U) / group_size;
  const std::uint32_t tiles_per_group =
    (tiles + fiat::detail::reduce_group_limit - 1U) /
    fiat::detail::reduce_group_limit;
  const std::uint32_t groups = (tiles + tiles_per_group - 1U) / tiles_per_group;
  return {
    .tiles           = tiles,
    .tiles_per_group = tiles_per_group,
    .groups          = groups,
  };
}

[[nodiscard]]
constexpr
auto
scan_partition_matches_shader_impl(
  std::uint32_t count, std::uint32_t group_size) -> bool
{
  const auto          partition = scan_partition_impl(count, group_size);
  const std::uint32_t shader_tiles_per_group =
    (partition.tiles + partition.groups - 1U) / partition.groups;
  return shader_tiles_per_group == partition.tiles_per_group &&
         partition.groups <= fiat::detail::reduce_group_limit &&
         (partition.groups - 1U) * partition.tiles_per_group < partition.tiles;
}

static_assert(scan_partition_matches_shader_impl(1U, 256U));
static_assert(scan_partition_matches_shader_impl(257U, 256U));
static_assert(scan_partition_matches_shader_impl(1'000'003U, 256U));
static_assert(
  scan_partition_matches_shader_impl(0xFF'FF'FFU - 1'024U * 256U, 256U));
static_assert(
  scan_partition_matches_shader_impl(0xFF'FF'FFU - 1'024U * 1'024U, 1'024U));
static_assert(scan_partition_matches_shader_impl(0xFF'FF'FFU - 1'024U, 1U));
static_assert(scan_partition_impl(1'000'003U, 256U).groups == 977U);
static_assert(scan_partition_impl(1'000'003U, 256U).tiles_per_group == 4U);

} // namespace detail

export template<fiat::scalar Element, class Operation>
  requires fiat::traceable<Operation, Element>
class transform_pipeline
{
public:
  using argument_tuple = decltype(fiat::transform_stage<Operation>::arguments);

  explicit transform_pipeline(
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

export
template<fiat::scalar Element, class Operation>
[[nodiscard]]
auto
operator |(
  const device_array<Element>& source, fiat::transform_stage<Operation> stage)
{
  if constexpr (!fiat::traceable<Operation, Element>)
  {
    static_assert(false, detail::transform_mismatch);
  }
  else
  {
    return transform_pipeline<Element, Operation> {
      source, std::move(stage.arguments)
    };
  }
}

namespace detail
{

template<fiat::scalar Element, class Operation, class Fused>
[[nodiscard]]
auto
extend_transform_pipeline_impl(
  const transform_pipeline<Element, Operation>& current,
  fiat::transform_stage<Fused>                  fused)
{
  if constexpr (!fiat::traceable<Fused, Element>)
  {
    static_assert(false, transform_mismatch);
  }
  else
  {
    return transform_pipeline<Element, Fused> {
      current.source(), std::move(fused.arguments)
    };
  }
}

} // namespace detail

export
template<fiat::scalar Element, class Operation, class Next>
[[nodiscard]]
auto
operator |(
  const transform_pipeline<Element, Operation>& current,
  fiat::transform_stage<Next>                   next)
{
  return detail::extend_transform_pipeline_impl(
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
    static_assert(false, detail::transform_with_mismatch);
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

export template<fiat::soa_element Element, class Operation>
class foreach_pipeline
{
public:
  using argument_tuple = decltype(fiat::foreach_stage<Operation>::arguments);

  explicit foreach_pipeline(
    soa_array<Element>& buffer, argument_tuple arguments)
  : buffer_ {buffer}, arguments_ {std::move(arguments)}
  {}

  [[nodiscard]]
  auto
  buffer() const noexcept -> soa_array<Element>&
  { return buffer_; }

  [[nodiscard]]
  auto
  arguments() const noexcept -> const argument_tuple&
  { return arguments_; }

private:
  soa_array<Element>& buffer_;
  argument_tuple      arguments_;
};

export
template<fiat::soa_element Element, class Operation>
[[nodiscard]]
auto
operator |(soa_array<Element>& buffer, fiat::foreach_stage<Operation> stage)
{
  if constexpr (!fiat::foreach_traceable<Operation, Element>)
  {
    static_assert(false, detail::foreach_mismatch);
  }
  else
  {
    return foreach_pipeline<Element, Operation> {
      buffer, std::move(stage.arguments)
    };
  }
}

export
template<fiat::soa_element Element, class Operation>
[[nodiscard]]
auto
record(
  vk::raii::CommandBuffer&                    command_buffer,
  context&                                    context,
  const foreach_pipeline<Element, Operation>& computation)
  -> std::expected<void, error_t>
{
  const auto count = computation.buffer().size();

  auto kernel = context.kernel<Element, fiat::foreach_stage<Operation>>();
  if (!kernel) { return std::unexpected {kernel.error()}; }

  auto base = computation.buffer().base_address(context.device());
  if (!base) { return std::unexpected {base.error()}; }

  const auto arguments =
    detail::pack_arguments_impl<0UZ>(computation.arguments());

  detail::record_chunks_impl(
    command_buffer, context, kernel->get(), count, [ & ](std::uint32_t first) {
      return fiat::detail::foreach_push {
        .base      = *base,
        .unused    = 0U,
        .count     = count,
        .first     = first,
        .arguments = arguments,
      };
    });
  return {};
}

export
template<fiat::soa_element Element, class Operation>
[[nodiscard]]
auto
run(context& context, const foreach_pipeline<Element, Operation>& computation)
  -> std::expected<void, error_t>
{
  return detail::submit_impl(
    context,
    [ & ](
      vk::raii::CommandBuffer& command_buffer) -> std::expected<void, error_t> {
      if (auto error = record(command_buffer, context, computation); !error)
      {
        return error;
      }
      detail::record_barrier_impl(
        command_buffer, detail::compute_write_to_any_read);
      return {};
    });
}

export template<
  fiat::scalar Element,
  class Map,
  class Combine,
  class Epilogue = fiat::identity
>
  requires (
    fiat::traceable<Map, Element> &&
    fiat::combinable<Combine, Element> &&
    fiat::traceable<Epilogue, Element>)
class reduce_pipeline
{
public:
  using argument_tuple =
    decltype(fiat::reduce_stage<Map, Combine, Element, Epilogue>::arguments);
  using epilogue_argument_tuple = decltype(fiat::reduce_stage<
    Map,
    Combine,
    Element,
    Epilogue
  >::epilogue_arguments);

  explicit reduce_pipeline(
    const device_array<Element>& source,
    argument_tuple               arguments,
    Element                      identity,
    epilogue_argument_tuple      epilogue_arguments = {})
  : source_ {source},
    arguments_ {std::move(arguments)},
    identity_ {identity},
    epilogue_arguments_ {std::move(epilogue_arguments)}
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

  [[nodiscard]]
  auto
  epilogue_arguments() const noexcept -> const epilogue_argument_tuple&
  { return epilogue_arguments_; }

private:
  const device_array<Element>& source_;
  argument_tuple               arguments_;
  Element                      identity_;
  epilogue_argument_tuple      epilogue_arguments_;
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
    static_assert(false, detail::reduce_mismatch);
  }
  else
  {
    return reduce_pipeline<Element, Map, Combine> {
      source,
      std::move(stage.arguments),
      stage.identity,
      {},
    };
  }
}

export
template<fiat::scalar Element, class Operation, class Combine, class Identity>
[[nodiscard]]
auto
operator |(
  const transform_pipeline<Element, Operation>&        current,
  fiat::reduce_stage<fiat::chain<>, Combine, Identity> next)
{
  if constexpr (
    !fiat::traceable<Operation, Element> ||
    !fiat::combinable<Combine, Element> ||
    !std::same_as<Identity, Element>)
  {
    static_assert(false, detail::reduce_mismatch);
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
      {},
    };
  }
}

export
template<fiat::scalar Element, class Map, class Combine, class Next>
[[nodiscard]]
auto
operator |(
  const reduce_pipeline<Element, Map, Combine, fiat::identity>& computation,
  fiat::transform_stage<Next>                                   stage)
{
  if constexpr (!fiat::traceable<Next, Element>)
  {
    static_assert(false, detail::transform_mismatch);
  }
  else
  {
    static_assert(
      std::tuple_size_v<
          typename reduce_pipeline<Element, Map, Combine, Next>::argument_tuple
      > +
          std::tuple_size_v<typename reduce_pipeline<
            Element,
            Map,
            Combine,
            Next
          >::epilogue_argument_tuple> <=
        fiat::argument_capacity,
      "reduce map and epilogue arguments together take at most 26 slots");
    return reduce_pipeline<Element, Map, Combine, Next> {
      computation.source(),
      computation.arguments(),
      computation.identity(),
      std::move(stage.arguments),
    };
  }
}

export
template<
  fiat::scalar Element,
  class Map,
  class Combine,
  class... Current,
  class... Next
>
[[nodiscard]]
auto
operator |(
  const reduce_pipeline<Element, Map, Combine, fiat::chain<Current...>>& current,
  fiat::transform_stage<fiat::chain<Next...>> next)
{
  auto fused = fiat::fuse(
    fiat::transform_stage<fiat::chain<Current...>> {
      .arguments = current.epilogue_arguments(),
    },
    std::move(next));
  if constexpr (!fiat::traceable<fiat::chain<Current..., Next...>, Element>)
  {
    static_assert(false, detail::transform_mismatch);
  }
  else
  {
    static_assert(
      std::tuple_size_v<typename reduce_pipeline<
            Element,
            Map,
            Combine,
            fiat::chain<Current..., Next...>
          >::argument_tuple> +
          std::tuple_size_v<typename reduce_pipeline<
            Element,
            Map,
            Combine,
            fiat::chain<Current..., Next...>
          >::epilogue_argument_tuple> <=
        fiat::argument_capacity,
      "reduce map and epilogue arguments together take at most 26 slots");
    return reduce_pipeline<
      Element,
      Map,
      Combine,
      fiat::chain<Current..., Next...>
    > {
      current.source(),
      current.arguments(),
      current.identity(),
      std::move(fused.arguments),
    };
  }
}

export
template<fiat::scalar Element, class Operation>
[[nodiscard]]
auto
record(
  vk::raii::CommandBuffer&                      command_buffer,
  context&                                      context,
  const transform_pipeline<Element, Operation>& computation,
  device_array<Element>& output) -> std::expected<void, error_t>
{
  return detail::record_transform_impl<
    Element,
    fiat::transform_stage<Operation>
  >(command_buffer,
    context,
    computation.source(),
    output,
    detail::pack_arguments_impl<0UZ>(computation.arguments()));
}

export
template<fiat::scalar Element, class Operation>
[[nodiscard]]
auto
run(
  context&                                      context,
  const transform_pipeline<Element, Operation>& computation,
  device_array<Element>& output) -> std::expected<void, error_t>
{
  return detail::submit_impl(
    context,
    [ & ](
      vk::raii::CommandBuffer& command_buffer) -> std::expected<void, error_t> {
      if (
        auto error = record(command_buffer, context, computation, output);
        !error)
      {
        return error;
      }
      detail::record_barrier_impl(
        command_buffer, detail::compute_write_to_any_read);
      return {};
    });
}

export
template<fiat::scalar Element, class Operation>
[[nodiscard]]
auto
run(context& context, const transform_pipeline<Element, Operation>& computation)
  -> std::expected<device_array<Element>, error_t>
{ return detail::run_into_new_array_impl<Element>(context, computation); }

export struct execute_stage
{
  gpu::context& context;
};

export [[nodiscard]]
inline
auto
execute(context& context) -> execute_stage
{ return execute_stage {.context = context}; }

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
template<
  fiat::scalar Element,
  class Map,
  class Combine,
  class Epilogue = fiat::identity
>
[[nodiscard]]
auto
operator |(
  const reduce_pipeline<Element, Map, Combine, Epilogue>& computation,
  keep_stage stage) -> std::expected<device_scalar<Element>, error_t>
{
  auto& context = stage.context;
  auto  slot    = detail::allocate_scalar_impl<Element>(context);
  if (!slot) { return std::unexpected {slot.error()}; }

  return detail::submit_impl(
    context,
    [ & ](
      vk::raii::CommandBuffer& command_buffer) -> std::expected<void, error_t> {
      if (auto error = record(command_buffer, context, computation); !error)
      {
        return error;
      }
      detail::record_result_copy_impl<Element>(
        command_buffer,
        context.reduce_scratch().buffer(),
        fiat::reduce_result_offset,
        slot->buffer().buffer(),
        detail::copy_write_to_compute);
      return {};
    })
    .transform([ & ] -> device_scalar<Element> { return std::move(*slot); });
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
  return detail::record_transform_impl<
    Element,
    fiat::transform_with_stage<Uniform, Operation>
  >(command_buffer,
    context,
    computation.source(),
    output,
    detail::pack_uniform_arguments_impl(
      computation.address(), computation.arguments()));
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
  return detail::submit_impl(
    context,
    [ & ](
      vk::raii::CommandBuffer& command_buffer) -> std::expected<void, error_t> {
      if (
        auto error = record(command_buffer, context, computation, output);
        !error)
      {
        return error;
      }
      detail::record_barrier_impl(
        command_buffer, detail::compute_write_to_any_read);
      return {};
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
{ return detail::run_into_new_array_impl<Element>(context, computation); }

export
template<
  fiat::scalar Element,
  class Map,
  class Combine,
  class Epilogue = fiat::identity
>
[[nodiscard]]
auto
record(
  vk::raii::CommandBuffer&                                command_buffer,
  context&                                                context,
  const reduce_pipeline<Element, Map, Combine, Epilogue>& computation)
  -> std::expected<void, error_t>
{
  const auto count      = computation.source().size();
  const auto group_size = context.group_size();
  if (auto error = detail::check_count_impl(count, group_size); !error)
  {
    return error;
  }

  auto kernel =
    context
      .kernel<Element, fiat::reduce_stage<Map, Combine, Element, Epilogue>>();
  if (!kernel) { return std::unexpected {kernel.error()}; }

  auto input_view = computation.source().addressed_view(context.device());
  if (!input_view) { return std::unexpected {input_view.error()}; }

  auto scratch_view =
    context.reduce_scratch().template addressed_view<std::byte>(
      context.device(), 0UZ, fiat::detail::reduce_scratch_bytes);
  if (!scratch_view) { return std::unexpected {scratch_view.error()}; }

  detail::record_barrier_impl(command_buffer, detail::compute_reuse);

  const compute_pipeline& compute = kernel->get();
  command_buffer.bindPipeline(
    vk::PipelineBindPoint::eCompute, *compute.pipeline());

  const auto groups = std::min(
    (count + group_size - 1U) / group_size, fiat::detail::reduce_group_limit);
  const auto arguments = detail::pack_reduce_arguments_impl(
    computation.arguments(), computation.epilogue_arguments());

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
template<
  fiat::scalar Element,
  class Map,
  class Combine,
  class Epilogue = fiat::identity
>
[[nodiscard]]
auto
run(
  context&                                                context,
  const reduce_pipeline<Element, Map, Combine, Epilogue>& computation)
  -> std::expected<Element, error_t>
{
  return detail::submit_impl(
    context,
    [ & ](
      vk::raii::CommandBuffer& command_buffer) -> std::expected<void, error_t> {
      if (auto error = record(command_buffer, context, computation); !error)
      {
        return error;
      }
      detail::record_result_copy_impl<Element>(
        command_buffer,
        context.reduce_scratch().buffer(),
        fiat::reduce_result_offset,
        context.reduce_readback().buffer(),
        detail::copy_write_to_host_read);
      return {};
    })
    .and_then([ & ] -> std::expected<Element, error_t> {
      return detail::read_scalar_impl<Element>(context);
    });
}

export template<
  fiat::scalar Element,
  class Map,
  class Combine,
  fiat::scan_mode Mode,
  class Epilogue = fiat::identity
>
  requires (
    fiat::traceable<Map, Element> &&
    fiat::combinable<Combine, Element> &&
    fiat::traceable<Epilogue, Element>)
class scan_pipeline
{
public:
  using argument_tuple =
    decltype(fiat::scan_stage<Map, Combine, Element, Mode, Epilogue>::arguments);
  using epilogue_argument_tuple = decltype(fiat::scan_stage<
    Map,
    Combine,
    Element,
    Mode,
    Epilogue
  >::epilogue_arguments);

  explicit scan_pipeline(
    const device_array<Element>& source,
    argument_tuple               arguments,
    Element                      identity,
    epilogue_argument_tuple      epilogue_arguments = {})
  : source_ {source},
    arguments_ {std::move(arguments)},
    identity_ {identity},
    epilogue_arguments_ {std::move(epilogue_arguments)}
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

  [[nodiscard]]
  auto
  epilogue_arguments() const noexcept -> const epilogue_argument_tuple&
  { return epilogue_arguments_; }

private:
  const device_array<Element>& source_;
  argument_tuple               arguments_;
  Element                      identity_;
  epilogue_argument_tuple      epilogue_arguments_;
};

export
template<
  fiat::scalar Element,
  class Map,
  class Combine,
  class Identity,
  fiat::scan_mode Mode
>
[[nodiscard]]
auto
operator |(
  const device_array<Element>&                   source,
  fiat::scan_stage<Map, Combine, Identity, Mode> stage)
{
  if constexpr (
    !fiat::traceable<Map, Element> ||
    !fiat::combinable<Combine, Element> ||
    !std::same_as<Identity, Element>)
  {
    static_assert(false, detail::scan_mismatch);
  }
  else
  {
    return scan_pipeline<Element, Map, Combine, Mode> {
      source,
      std::move(stage.arguments),
      stage.identity,
      {},
    };
  }
}

export
template<
  fiat::scalar Element,
  class Operation,
  class Combine,
  class Identity,
  fiat::scan_mode Mode
>
[[nodiscard]]
auto
operator |(
  const transform_pipeline<Element, Operation>&            current,
  fiat::scan_stage<fiat::chain<>, Combine, Identity, Mode> next)
{
  if constexpr (
    !fiat::traceable<Operation, Element> &&
    !fiat::combinable<Combine, Element> &&
    !fiat::traceable<Identity, Element>)
  {
    static_assert(false, detail::scan_mismatch);
  }
  else
  {
    auto fused =
      fiat::transform_stage<Operation> {.arguments = current.arguments()} |
      std::move(next);
    return scan_pipeline<Element, Operation, Combine, Mode> {
      current.source(),
      std::move(fused.arguments),
      fused.identity,
      {},
    };
  }
}

export
template<
  fiat::scalar Element,
  class Map,
  class Combine,
  fiat::scan_mode Mode,
  class Next
>
[[nodiscard]]
auto
operator |(
  const scan_pipeline<Element, Map, Combine, Mode, fiat::identity>& computation,
  fiat::transform_stage<Next>                                       stage)
{
  if constexpr (!fiat::traceable<Next, Element>)
  {
    static_assert(false, detail::transform_mismatch);
  }
  else
  {
    static_assert(
      std::tuple_size_v<
          typename scan_pipeline<Element, Map, Combine, Mode, Next>::argument_tuple
      > +
          std::tuple_size_v<typename scan_pipeline<
            Element,
            Map,
            Combine,
            Mode,
            Next
          >::epilogue_argument_tuple> <=
        fiat::scan_argument_capacity,
      "scan map and epilogue arguments together take at most 24 slots");
    return scan_pipeline<Element, Map, Combine, Mode, Next> {
      computation.source(),
      computation.arguments(),
      computation.identity(),
      std::move(stage.arguments),
    };
  }
}

export
template<
  fiat::scalar Element,
  class Map,
  class Combine,
  fiat::scan_mode Mode,
  class... Current,
  class... Next
>
[[nodiscard]]
auto
operator |(
  const scan_pipeline<Element, Map, Combine, Mode, fiat::chain<Current...>>&
                                              current,
  fiat::transform_stage<fiat::chain<Next...>> next)
{
  auto fused = fiat::fuse(
    fiat::transform_stage<fiat::chain<Current...>> {
      .arguments = current.epilogue_arguments(),
    },
    std::move(next));
  if constexpr (!fiat::traceable<fiat::chain<Current..., Next...>, Element>)
  {
    static_assert(false, detail::transform_mismatch);
  }
  else
  {
    static_assert(
      std::tuple_size_v<typename scan_pipeline<
            Element,
            Map,
            Combine,
            Mode,
            fiat::chain<Current..., Next...>
          >::argument_tuple> +
          std::tuple_size_v<typename scan_pipeline<
            Element,
            Map,
            Combine,
            Mode,
            fiat::chain<Current..., Next...>
          >::epilogue_argument_tuple> <=
        fiat::scan_argument_capacity,
      "scan map and epilogue arguments together take at most 24 slots");
    return scan_pipeline<
      Element,
      Map,
      Combine,
      Mode,
      fiat::chain<Current..., Next...>
    > {
      current.source(),
      current.arguments(),
      current.identity(),
      std::move(fused.arguments),
    };
  }
}

export
template<
  fiat::scalar Element,
  class Map,
  class Combine,
  fiat::scan_mode Mode,
  class Epilogue
>
[[nodiscard]]
auto
record(
  vk::raii::CommandBuffer&                                    command_buffer,
  context&                                                    context,
  const scan_pipeline<Element, Map, Combine, Mode, Epilogue>& computation,
  device_array<Element>& output) -> std::expected<void, error_t>
{
  const auto count      = computation.source().size();
  const auto group_size = context.group_size();
  if (auto error = detail::check_count_impl(count, group_size); !error)
  {
    return error;
  }
  if (output.size() != count)
  {
    return std::unexpected {make_app_error(app_error_code::invalid_data_size)};
  }

  using scan_kernel = fiat::scan_stage<Map, Combine, Element, Mode, Epilogue>;
  auto upsweep =
    context.kernel<Element, scan_kernel, detail::scan_upsweep_index>();
  if (!upsweep) { return std::unexpected {upsweep.error()}; }
  auto downsweep =
    context.kernel<Element, scan_kernel, detail::scan_downsweep_index>();
  if (!downsweep) { return std::unexpected {downsweep.error()}; }

  auto input_view  = computation.source().addressed_view(context.device());
  auto output_view = output.addressed_view(context.device());
  if (!input_view) { return std::unexpected {input_view.error()}; }
  if (!output_view) { return std::unexpected {output_view.error()}; }

  auto scratch_view =
    context.reduce_scratch().template addressed_view<std::byte>(
      context.device(), 0UZ, fiat::detail::reduce_scratch_bytes);
  if (!scratch_view) { return std::unexpected {scratch_view.error()}; }

  const detail::scan_partition partition =
    detail::scan_partition_impl(count, group_size);
  const fiat::detail::scan_push scan_constants {
    .input     = input_view->device_address(),
    .scratch   = scratch_view->device_address(),
    .count     = count,
    .identity  = std::bit_cast<std::uint32_t>(computation.identity()),
    .arguments = detail::pack_scan_arguments_impl(
      computation.arguments(), computation.epilogue_arguments()),
    .output = output_view->device_address(),
  };

  detail::record_barrier_impl(command_buffer, detail::compute_reuse);

  const auto& upsweep_compute = upsweep->get();
  command_buffer.bindPipeline(
    vk::PipelineBindPoint::eCompute, *upsweep_compute.pipeline());
  push_compute<descriptor_table_backend::classic>(
    command_buffer, upsweep_compute, scan_constants);
  dispatch(command_buffer, partition.groups);

  detail::record_barrier_impl(
    command_buffer, detail::compute_write_to_any_read);

  const auto& downsweep_compute = downsweep->get();
  command_buffer.bindPipeline(
    vk::PipelineBindPoint::eCompute, *downsweep_compute.pipeline());
  push_compute<descriptor_table_backend::classic>(
    command_buffer, downsweep_compute, scan_constants);
  dispatch(command_buffer, partition.groups);
  return {};
}

export
template<
  fiat::scalar Element,
  class Map,
  class Combine,
  fiat::scan_mode Mode,
  class Epilogue
>
[[nodiscard]]
auto
run(
  context&                                                    context,
  const scan_pipeline<Element, Map, Combine, Mode, Epilogue>& computation,
  device_array<Element>& output) -> std::expected<void, error_t>
{
  return detail::submit_impl(
    context,
    [ & ](
      vk::raii::CommandBuffer& command_buffer) -> std::expected<void, error_t> {
      if (
        auto error = record(command_buffer, context, computation, output);
        !error)
      {
        return error;
      }
      detail::record_barrier_impl(
        command_buffer, detail::compute_write_to_any_read);
      return {};
    });
}

export
template<
  fiat::scalar Element,
  class Map,
  class Combine,
  fiat::scan_mode Mode,
  class Epilogue
>
[[nodiscard]]
auto
run(
  context&                                                    context,
  const scan_pipeline<Element, Map, Combine, Mode, Epilogue>& computation)
  -> std::expected<device_array<Element>, error_t>
{ return detail::run_into_new_array_impl<Element>(context, computation); }

namespace detail
{

template<class Computation>
concept runnable = requires(context& context, const Computation& computation) {
  run(context, computation);
};

} // namespace detail

export
template<detail::runnable Computation>
[[nodiscard]]
auto
operator |(const Computation& computation, execute_stage stage)
  -> decltype(auto)
{ return run(stage.context, computation); }

} // namespace vkpp::gpu
