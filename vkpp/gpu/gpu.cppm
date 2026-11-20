module;
#if defined(__cpp_impl_reflection)
  #include <meta>
#endif

export module vkpp.gpu;

import std;
import vulkan;
import fiat;

import vkpp.error;
import vkpp.integer;
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
export import :recorder;
export import :workspace;

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
  size_bytes() const noexcept -> vk::DeviceSize
  { return vk::DeviceSize {count_} * sizeof(Element); }

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

template<fiat::scalar Element, class Stage>
  requires fiat::archetype_stage<Stage, Element>
struct kernel_traits<Element, Stage>
{
  static constexpr auto       words = fiat::compile<Element>(Stage {});
  static constexpr std::array entries =
    []<std::size_t... Index>(std::index_sequence<Index...>) {
      return std::array<std::string_view, sizeof...(Index)> {
        Stage::dispatches[ Index ].entry...,
      };
    }(std::make_index_sequence<Stage::dispatches.size()> {});
  static constexpr std::size_t push_size {
    fiat::skeleton_push_of<Stage, Element>.size(),
  };
};

template<fiat::soa_element Element, class Operation>
struct kernel_traits<Element, fiat::foreach_stage<Operation>>
{
  static constexpr auto words =
    fiat::compile<Element>(fiat::foreach_stage<Operation> {});
  static constexpr std::array  entries {fiat::foreach_entry};
  static constexpr std::size_t push_size {sizeof(fiat::detail::foreach_push)};
};

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

inline constexpr std::uint32_t shader_index_limit {
  std::numeric_limits<std::int32_t>::max(),
};
static_assert(
  std::uint64_t {shader_index_limit} + (1ULL << 21U) < (1ULL << 32U),
  "reduce and scan index arithmetic stays below 2^32 for every group size "
  "below 2^21");

static_assert(
  covers(
    {.scratch_bytes = fiat::detail::reduce_scratch_bytes},
    fiat::workspace_of<fiat::scan_stage<
      fiat::chain<>,
      fiat::plus,
      std::uint32_t,
      fiat::scan_mode::inclusive
    >>),
  "the default workspace covers every scan");

template<class Stage>
inline constexpr std::uint32_t buckets_of {};

template<class Stage>
  requires requires { Stage::buckets; }
inline constexpr std::uint32_t buckets_of<Stage> {Stage::buckets};

struct cached_kernel
{
  std::uint64_t                  fingerprint {};
  std::string_view               entry {};
  std::span<const std::uint32_t> words {};
  std::array<std::uint32_t, 2UZ> specialization {};
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
    const std::array<std::uint32_t, 2UZ> specialization_values {
      group_size_,
      buckets_of<Stage>,
    };

    const auto found = std::ranges::find_if(
      kernels_, [ & ](const cached_kernel& cached) -> bool {
        return cached.fingerprint == fingerprint &&
               cached.entry == entry &&
               cached.specialization == specialization_values &&
               std::ranges::equal(cached.words, words);
      });
    if (found != kernels_.end()) { return std::cref(found->pipeline); }

    const specialization_constant group_size_spec {
      .id    = fiat::group_size_constant_id,
      .value = specialization_values[ fiat::group_size_constant_id ],
    };
    const specialization_constant buckets_spec {
      .id    = fiat::buckets_constant_id,
      .value = specialization_values[ fiat::buckets_constant_id ],
    };
    const std::array one {group_size_spec};
    const std::array two {group_size_spec, buckets_spec};
    const auto       specialization =
      buckets_of<Stage> == 0U
        ? std::span<const specialization_constant> {one}
        : std::span<const specialization_constant> {two};

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
            .fingerprint    = fingerprint,
            .entry          = entry,
            .words          = words,
            .specialization = specialization_values,
            .pipeline       = std::move(created),
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
    const auto count = try_narrow<std::uint32_t>(elements.size());
    if (!count || *count == 0U || *count > maximum_element_count<Element>())
    {
      return std::unexpected {make_app_error(app_error_code::invalid_count)};
    }
    return upload_device_local_buffer(
      buffer_upload_create_info_for<buffer_kind::device_address> {
        .device = device_,
        .pool   = pool_,
        .bytes  = std::as_bytes(elements),
      })
      .transform([ & ](device_address_buffer&& buffer) -> device_array<Element> {
        return device_array<Element> {std::move(buffer), *count};
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
    const auto count = try_narrow<std::uint32_t>(host.size());
    if (
      !count ||
      *count == 0U ||
      *count > shader_index_limit ||
      fiat::detail::soa_bytes<Element>(*count) >
        capabilities().maximum_buffer_size)
    {
      return std::unexpected {make_app_error(app_error_code::invalid_count)};
    }
    const auto byte_count = fiat::detail::soa_bytes<Element>(*count);
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
          return soa_array<Element> {std::move(buffer), *count};
        });
#endif
  }

  template<fiat::scalar Element>
  [[nodiscard]]
  auto
  allocate(std::uint32_t count) -> std::expected<device_array<Element>, error_t>
  {
    if (count == 0U || count > maximum_element_count<Element>())
    {
      return std::unexpected {make_app_error(app_error_code::invalid_count)};
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
        .size   = array.size_bytes(),
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
  capabilities() const noexcept -> const compute_capabilities&
  { return device_.selected_capabilities().compute; }

  [[nodiscard]]
  auto
  max_group_count() const noexcept -> std::uint32_t
  { return max_group_count_; }

  template<fiat::scalar Element>
  [[nodiscard]]
  auto
  maximum_element_count() const noexcept -> std::uint32_t
  {
    return narrow<std::uint32_t>(std::min<vk::DeviceSize>(
      shader_index_limit,
      capabilities().maximum_buffer_size / sizeof(Element)));
  }

  [[nodiscard]]
  auto
  allocate_workspace(const workspace_sizes& sizes)
    -> std::expected<workspace, error_t>
  {
    const auto bytes = workspace_bytes(sizes);
    if (bytes == 0U || bytes > capabilities().maximum_buffer_size)
    {
      return std::unexpected {make_app_error(app_error_code::invalid_count)};
    }
    return create_workspace(device_, pool_, sizes);
  }

  [[nodiscard]]
  auto
  default_workspace(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.default_workspace_); }

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
        .synchronization2                 = capability_requirement::required,
        .buffer_device_address            = capability_requirement::required,
        .maintenance4                     = capability_requirement::required,
        .vulkan_memory_model              = capability_requirement::required,
        .zero_initialize_workgroup_memory = capability_requirement::required,
        .subgroup_size_control            = capability_requirement::preferred,
        .subgroup_extended_types          = capability_requirement::preferred,
        .subgroup_uniform_control_flow    = capability_requirement::preferred,
        .maximal_reconvergence            = capability_requirement::preferred,
        .subgroup_partitioned             = capability_requirement::preferred,
        .cooperative_matrix               = capability_requirement::preferred,
        .integer_dot_product              = capability_requirement::preferred,
        .int64_atomics                    = capability_requirement::preferred,
        .float_atomics                    = capability_requirement::preferred,
        .workgroup_memory_explicit_layout = capability_requirement::preferred,
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
    workspace&&                       default_workspace,
    readback_buffer&&                 reduce_readback)
  : instance_ {std::move(instance)},
    device_ {std::move(device)},
    pool_ {std::move(pool)},
    cache_ {std::move(cache)},
    diagnostics_ {diagnostics},
    group_size_ {group_size},
    max_group_count_ {max_group_count},
    pipeline_cache_path_ {std::move(pipeline_cache_path)},
    default_workspace_ {std::move(default_workspace)},
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
  create_workspace(
    device_context& device, command_pool& pool, const workspace_sizes& sizes)
    -> std::expected<workspace, error_t>
  {
    const std::vector<std::byte> zeros(
      static_cast<std::size_t>(workspace_bytes(sizes)));
    return upload_device_local_buffer(
      buffer_upload_create_info_for<buffer_kind::device_address> {
        .device = device,
        .pool   = pool,
        .bytes  = std::span {zeros},
      })
      .and_then(
        [ & ](
          device_address_buffer&& buffer) -> std::expected<workspace, error_t> {
          const auto view = buffer.addressed_view(device);
          if (!view) { return std::unexpected {view.error()}; }
          return workspace {std::move(buffer), sizes, view->device_address()};
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

    auto default_workspace = create_workspace(
      *device, *pool, {.scratch_bytes = fiat::detail::reduce_scratch_bytes});
    if (!default_workspace)
    {
      return std::unexpected {default_workspace.error()};
    }

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
      std::move(*default_workspace),
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
  workspace                         default_workspace_;
  readback_buffer                   reduce_readback_;
};

namespace detail
{

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

inline constexpr std::string_view select_mismatch {
  "this vkpp::gpu::select cannot run on this array: the predicate must take "
  "(value, arguments...) and return fiat::value<bool>, and every runtime "
  "argument must have the array's element type"
};

template<std::size_t first_slot, class... Arguments>
[[nodiscard]]
constexpr
auto
pack_arguments(const std::tuple<Arguments...>& arguments)
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

template<fiat::scalar Element, class Computation>
[[nodiscard]]
auto
run_into_new_array(context& context, const Computation& computation)
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
allocate_scalar(context& context)
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
record_result_copy(
  vk::raii::CommandBuffer&  command_buffer,
  vk::Buffer                source,
  vk::DeviceSize            source_offset,
  vk::Buffer                destination,
  const vk::MemoryBarrier2& after)
{
  record_barrier(command_buffer, compute_write_to_copy_read);
  const vk::BufferCopy region {
    .srcOffset = source_offset,
    .dstOffset = 0UZ,
    .size      = sizeof(Element),
  };
  command_buffer.copyBuffer(source, destination, region);
  record_barrier(command_buffer, after);
}

template<fiat::scalar Element>
[[nodiscard]]
auto
read_scalar(context& context) -> std::expected<Element, error_t>
{
  Element value {};
  return context.reduce_readback()
    .copy_mapped_into(std::span {&value, 1UZ})
    .transform([ & ] -> Element { return value; });
}

} // namespace detail

static_assert(
  fill_dispatch_push<float>(
    fiat::transform(fiat::multiplies {}, 3.0F) |
      fiat::transform(fiat::plus {}, 7.0F),
    {0x01'23'45'67'89'AB'CD'EFULL, 0xFE'DC'BA'98'76'54'32'10ULL},
    1'048'576U) ==
    dispatch_words {
      0x89'AB'CD'EFU,
      0x01'23'45'67U,
      0x76'54'32'10U,
      0xFE'DC'BA'98U,
      1'048'576U,
      0U,
      std::bit_cast<std::uint32_t>(3.0F),
      std::bit_cast<std::uint32_t>(7.0F),
    },
  "the generic dispatch push lays out the transform skeleton push block");

export template<fiat::scalar Element, fiat::archetype_stage<Element> Stage>
  requires (
    Stage::kind == fiat::stage_class::elementwise ||
    Stage::kind == fiat::stage_class::scan)
class array_pipeline
{
public:
  using argument_tuple = std::remove_cvref_t<decltype(Stage::arguments)>;

  explicit array_pipeline(const device_array<Element>& source, Stage stage)
  : source_ {source}, stage_ {std::move(stage)}
  {}

  [[nodiscard]]
  auto
  source() const noexcept -> const device_array<Element>&
  { return source_; }

  [[nodiscard]]
  auto
  stage() const noexcept -> const Stage&
  { return stage_; }

  [[nodiscard]]
  auto
  arguments() const noexcept -> const argument_tuple&
  { return stage_.arguments; }

private:
  const device_array<Element>& source_;
  Stage                        stage_;
};

export template<fiat::scalar Element, class Operation>
using transform_pipeline =
  array_pipeline<Element, fiat::transform_stage<Operation>>;

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
    return transform_pipeline<Element, Operation> {source, std::move(stage)};
  }
}

namespace detail
{

template<fiat::scalar Element, class Operation, class Fused>
[[nodiscard]]
auto
extend_transform_pipeline(
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
      current.source(), std::move(fused)
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
  return detail::extend_transform_pipeline(
    current,
    fiat::fuse(
      fiat::transform_stage<Operation> {.arguments = current.arguments()},
      std::move(next)));
}

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
    return array_pipeline<
      Element,
      fiat::transform_with_stage<Uniform, Operation>
    > {source, std::move(stage)};
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

  const auto arguments = detail::pack_arguments<0UZ>(computation.arguments());

  record_chunks(
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
  return submit_and_wait(
    context,
    [ & ](
      vk::raii::CommandBuffer& command_buffer) -> std::expected<void, error_t> {
      if (auto error = record(command_buffer, context, computation); !error)
      {
        return error;
      }
      record_barrier(command_buffer, compute_write_to_any_read);
      return {};
    });
}

inline constexpr std::string_view stage_mismatch {
  "this fiat sink or scan stage cannot run on this array: the map, the combine "
  "and every epilogue transform must trace on the array's element type, and "
  "the identity must have that type; a histogram needs a std::uint32_t array"
};

export template<fiat::scalar Element, fiat::archetype_stage<Element> Stage>
  requires (Stage::kind == fiat::stage_class::sink)
class sink_pipeline
{
public:
  explicit sink_pipeline(const device_array<Element>& source, Stage stage)
  : source_ {source}, stage_ {std::move(stage)}
  {}

  [[nodiscard]]
  auto
  source() const noexcept -> const device_array<Element>&
  { return source_; }

  [[nodiscard]]
  auto
  stage() const noexcept -> const Stage&
  { return stage_; }

private:
  const device_array<Element>& source_;
  Stage                        stage_;
};

export
template<fiat::scalar Element, class Stage>
  requires (
    Stage::kind == fiat::stage_class::sink ||
    Stage::kind == fiat::stage_class::scan)
[[nodiscard]]
auto
operator |(const device_array<Element>& source, Stage stage)
{
  if constexpr (!fiat::archetype_stage<Stage, Element>)
  {
    static_assert(false, stage_mismatch);
  }
  else if constexpr (Stage::kind == fiat::stage_class::sink)
  {
    return sink_pipeline<Element, Stage> {source, std::move(stage)};
  }
  else
  {
    return array_pipeline<Element, Stage> {source, std::move(stage)};
  }
}

export
template<fiat::scalar Element, class Operation, class Next>
  requires (
    Next::kind == fiat::stage_class::sink ||
    Next::kind == fiat::stage_class::scan)
[[nodiscard]]
auto
operator |(const transform_pipeline<Element, Operation>& current, Next next)
{ return current.source() | (current.stage() | std::move(next)); }

export
template<fiat::scalar Element, class Stage, class Next>
[[nodiscard]]
auto
operator |(
  const sink_pipeline<Element, Stage>& current,
  fiat::transform_stage<Next>          next)
{ return current.source() | (current.stage() | std::move(next)); }

export
template<fiat::scalar Element, class Stage, class Next>
  requires (Stage::kind == fiat::stage_class::scan)
[[nodiscard]]
auto
operator |(
  const array_pipeline<Element, Stage>& current,
  fiat::transform_stage<Next>           next)
{ return current.source() | (current.stage() | std::move(next)); }

export
template<fiat::scalar Element, class Stage>
  requires (fiat::workspace_of<Stage>.region_count() == 0UZ)
[[nodiscard]]
auto
record(
  vk::raii::CommandBuffer&              command_buffer,
  context&                              context,
  const array_pipeline<Element, Stage>& computation,
  device_array<Element>&                output) -> std::expected<void, error_t>
{
  return record_kernel<Element>(
    command_buffer,
    context,
    computation.stage(),
    {},
    computation.source(),
    output);
}

export
template<fiat::scalar Element, class Stage>
[[nodiscard]]
auto
record(
  vk::raii::CommandBuffer&              command_buffer,
  context&                              context,
  const array_pipeline<Element, Stage>& computation,
  device_array<Element>&                output,
  workspace& workspace) -> std::expected<void, error_t>
{
  return bind_workspace<Stage>(workspace).and_then(
    [ & ](
      const region_addresses<Stage>& regions) -> std::expected<void, error_t> {
      return record_kernel<Element>(
        command_buffer,
        context,
        computation.stage(),
        regions,
        computation.source(),
        output);
    });
}

export
template<fiat::scalar Element, class Stage>
[[nodiscard]]
auto
run(
  context&                              context,
  const array_pipeline<Element, Stage>& computation,
  device_array<Element>&                output,
  workspace& workspace) -> std::expected<void, error_t>
{
  return submit_and_wait(
    context,
    [ & ](
      vk::raii::CommandBuffer& command_buffer) -> std::expected<void, error_t> {
      if (
        auto error =
          record(command_buffer, context, computation, output, workspace);
        !error)
      {
        return error;
      }
      record_barrier(command_buffer, compute_write_to_any_read);
      return {};
    });
}

export
template<fiat::scalar Element, class Stage>
[[nodiscard]]
auto
run(
  context&                              context,
  const array_pipeline<Element, Stage>& computation,
  device_array<Element>&                output) -> std::expected<void, error_t>
{ return run(context, computation, output, context.default_workspace()); }

export
template<fiat::scalar Element, class Stage>
[[nodiscard]]
auto
run(context& context, const array_pipeline<Element, Stage>& computation)
  -> std::expected<device_array<Element>, error_t>
{ return detail::run_into_new_array<Element>(context, computation); }

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
template<class Computation>
  requires requires(const Computation& computation) { computation.stage(); }
[[nodiscard]]
constexpr auto
workspace_requirements(const context&, const Computation& computation)
  -> workspace_sizes
{
  return fiat::workspace_of<std::remove_cvref_t<decltype(computation.stage())>>;
}

export
template<fiat::scalar Element, class Stage>
[[nodiscard]]
auto
record(
  vk::raii::CommandBuffer&             command_buffer,
  context&                             context,
  const sink_pipeline<Element, Stage>& computation,
  workspace& workspace) -> std::expected<void, error_t>
{
  return bind_workspace<Stage>(workspace).and_then(
    [ & ](
      const region_addresses<Stage>& regions) -> std::expected<void, error_t> {
      return record_kernel<Element>(
        command_buffer,
        context,
        computation.stage(),
        regions,
        computation.source());
    });
}

template<fiat::scalar Element, class Stage>
[[nodiscard]]
auto
record_sink_result(
  vk::raii::CommandBuffer&             command_buffer,
  context&                             context,
  const sink_pipeline<Element, Stage>& computation,
  workspace&                           workspace,
  vk::Buffer                           destination,
  const vk::MemoryBarrier2&            after) -> std::expected<void, error_t>
{
  return record(command_buffer, context, computation, workspace)
    .transform([ & ] -> void {
      detail::record_result_copy<Element>(
        command_buffer,
        workspace.buffer().buffer(),
        workspace.offset(workspace_region::scratch) + fiat::reduce_result_offset,
        destination,
        after);
    });
}

export
template<fiat::scalar Element, class Stage>
[[nodiscard]]
auto
run(
  context&                             context,
  const sink_pipeline<Element, Stage>& computation,
  workspace& workspace) -> std::expected<Element, error_t>
{
  return submit_and_wait(
    context,
    [ & ](
      vk::raii::CommandBuffer& command_buffer) -> std::expected<void, error_t> {
      return record_sink_result(
        command_buffer,
        context,
        computation,
        workspace,
        context.reduce_readback().buffer(),
        copy_write_to_host_read);
    })
    .and_then([ & ] -> std::expected<Element, error_t> {
      return detail::read_scalar<Element>(context);
    });
}

export
template<fiat::scalar Element, class Stage>
[[nodiscard]]
auto
run(context& context, const sink_pipeline<Element, Stage>& computation)
  -> std::expected<Element, error_t>
{ return run(context, computation, context.default_workspace()); }

export
template<fiat::scalar Element, class Stage>
[[nodiscard]]
auto
operator |(const sink_pipeline<Element, Stage>& computation, keep_stage stage)
  -> std::expected<device_scalar<Element>, error_t>
{
  auto& context = stage.context;
  auto  slot    = detail::allocate_scalar<Element>(context);
  if (!slot) { return std::unexpected {slot.error()}; }

  return submit_and_wait(
    context,
    [ & ](
      vk::raii::CommandBuffer& command_buffer) -> std::expected<void, error_t> {
      return record_sink_result(
        command_buffer,
        context,
        computation,
        context.default_workspace(),
        slot->buffer().buffer(),
        copy_write_to_compute);
    })
    .transform([ & ] -> device_scalar<Element> { return std::move(*slot); });
}

template<fiat::scalar Element, class Stage>
inline constexpr std::uint32_t result_port_size {
  stage_ports<Element, Stage>::sizes.back(),
};

export
template<fiat::scalar Element, class Stage>
  requires (result_port_size<Element, Stage> != 0U)
[[nodiscard]]
auto
record(
  vk::raii::CommandBuffer&             command_buffer,
  context&                             context,
  const sink_pipeline<Element, Stage>& computation,
  device_array<Element>&               result) -> std::expected<void, error_t>
{
  command_buffer.fillBuffer(
    result.buffer().buffer(), 0UZ, result.size_bytes(), 0U);
  record_barrier(command_buffer, transfer_write_to_compute);
  return record_kernel<Element>(
    command_buffer,
    context,
    computation.stage(),
    {},
    computation.source(),
    result);
}

export
template<fiat::scalar Element, class Stage>
  requires (result_port_size<Element, Stage> != 0U)
[[nodiscard]]
auto
run(
  context&                             context,
  const sink_pipeline<Element, Stage>& computation,
  device_array<Element>&               result) -> std::expected<void, error_t>
{
  return submit_and_wait(
    context,
    [ & ](
      vk::raii::CommandBuffer& command_buffer) -> std::expected<void, error_t> {
      if (
        auto error = record(command_buffer, context, computation, result);
        !error)
      {
        return error;
      }
      record_barrier(command_buffer, compute_write_to_any_read);
      return {};
    });
}

export
template<fiat::scalar Element, class Stage>
  requires (result_port_size<Element, Stage> != 0U)
[[nodiscard]]
auto
run(context& context, const sink_pipeline<Element, Stage>& computation)
  -> std::expected<device_array<Element>, error_t>
{
  return context.allocate<Element>(result_port_size<Element, Stage>)
    .and_then(
      [ & ](device_array<Element>&& result)
        -> std::expected<device_array<Element>, error_t> {
        return run(context, computation, result)
          .transform([ & ] -> device_array<Element> {
            return std::move(result);
          });
      });
}

export template<fiat::scalar Element>
struct compacted
{
  device_array<Element>       values;
  device_array<std::uint32_t> count;
};

export template<class PredicateStage>
struct select_stage
{
  PredicateStage predicate;
};

export
template<class Predicate, class... Arguments>
[[nodiscard]]
constexpr auto
select(Predicate predicate, Arguments... arguments)
  -> select_stage<decltype(fiat::predicate(predicate, arguments...))>
{ return {fiat::predicate(predicate, arguments...)}; }

export template<fiat::scalar Element, class PredicateStage>
class select_pipeline
{
public:
  explicit select_pipeline(
    const device_array<Element>& source, PredicateStage stage)
  : source_ {source}, stage_ {std::move(stage)}
  {}

  [[nodiscard]]
  auto
  source() const noexcept -> const device_array<Element>&
  { return source_; }

  [[nodiscard]]
  auto
  stage() const noexcept -> const PredicateStage&
  { return stage_; }

private:
  const device_array<Element>& source_;
  PredicateStage               stage_;
};

export
template<fiat::scalar Element, class PredicateStage>
[[nodiscard]]
auto
operator |(
  const device_array<Element>& source, select_stage<PredicateStage> stage)
{
  if constexpr (!fiat::archetype_stage<PredicateStage, Element>)
  {
    static_assert(false, detail::select_mismatch);
  }
  else
  {
    return select_pipeline<Element, PredicateStage> {
      source, std::move(stage.predicate)
    };
  }
}

export
template<fiat::scalar Element, class PredicateStage>
[[nodiscard]]
auto
record(
  vk::raii::CommandBuffer&                        command_buffer,
  context&                                        context,
  const select_pipeline<Element, PredicateStage>& computation,
  device_array<std::uint32_t>&                    flags,
  device_array<std::uint32_t>&                    indices,
  compacted<Element>&                             result,
  workspace& workspace) -> std::expected<void, error_t>
{
  record_barrier(command_buffer, compute_reuse);
  if (
    auto error = record_kernel<Element>(
      command_buffer,
      context,
      computation.stage(),
      {},
      computation.source(),
      flags);
    !error)
  {
    return error;
  }
  if (
    auto error = record(
      command_buffer,
      context,
      flags | fiat::exclusive_scan(fiat::plus {}, 0U),
      indices,
      workspace);
    !error)
  {
    return error;
  }
  record_barrier(command_buffer, compute_write_to_any_read);
  return record_kernel<Element>(
    command_buffer,
    context,
    fiat::scatter(),
    {},
    computation.source(),
    result.values,
    flags,
    indices,
    result.count);
}

export
template<fiat::scalar Element, class PredicateStage>
[[nodiscard]]
auto
run(
  context& context, const select_pipeline<Element, PredicateStage>& computation)
  -> std::expected<compacted<Element>, error_t>
{
  const auto count = computation.source().size();

  auto flags = context.allocate<std::uint32_t>(count);
  if (!flags) { return std::unexpected {flags.error()}; }
  auto indices = context.allocate<std::uint32_t>(count);
  if (!indices) { return std::unexpected {indices.error()}; }
  auto values = context.allocate<Element>(count);
  if (!values) { return std::unexpected {values.error()}; }
  auto kept_count = context.allocate<std::uint32_t>(1U);
  if (!kept_count) { return std::unexpected {kept_count.error()}; }

  compacted<Element> result {
    .values = std::move(*values),
    .count  = std::move(*kept_count),
  };
  return submit_and_wait(
    context,
    [ & ](
      vk::raii::CommandBuffer& command_buffer) -> std::expected<void, error_t> {
      if (
        auto error = record(
          command_buffer,
          context,
          computation,
          *flags,
          *indices,
          result,
          context.default_workspace());
        !error)
      {
        return error;
      }
      record_barrier(command_buffer, compute_write_to_any_read);
      return {};
    })
    .transform([ & ] -> compacted<Element> { return std::move(result); });
}

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
