export module vkpp.buffer;

import std;
import vulkan;

import vkpp.memory;
import vkpp.memory.vma;
import vkpp.error;
import vkpp.device;
import vkpp.command;
import vkpp.barrier;
import vkpp.capabilities;

namespace vkpp
{
using namespace std::string_view_literals;

export enum class buffer_kind : std::uint8_t {
  uniform,
  storage,
  vertex,
  index,
  staging,
  readback,
  indirect,
  device_address,
};

export struct buffer_type_spec
{
  vk::BufferUsageFlags usage {};
  memory_intent intent { memory_intent::gpu_only };
};

export template<buffer_kind Kind>
struct buffer_traits;

template<>
struct buffer_traits<buffer_kind::uniform>
{
  static constexpr buffer_type_spec spec {
    .usage = vk::BufferUsageFlagBits::eUniformBuffer,
    .intent = memory_intent::cpu_to_gpu,
  };
};

template<>
struct buffer_traits<buffer_kind::storage>
{
  static constexpr buffer_type_spec spec {
    .usage = vk::BufferUsageFlagBits::eStorageBuffer |
      vk::BufferUsageFlagBits::eTransferDst |
      vk::BufferUsageFlagBits::eTransferSrc,
    .intent = memory_intent::gpu_only,
  };
};

template<>
struct buffer_traits<buffer_kind::vertex>
{
  static constexpr buffer_type_spec spec {
    .usage = vk::BufferUsageFlagBits::eVertexBuffer |
      vk::BufferUsageFlagBits::eTransferDst,
    .intent = memory_intent::gpu_only,
  };
};

template<>
struct buffer_traits<buffer_kind::index>
{
  static constexpr buffer_type_spec spec {
    .usage = vk::BufferUsageFlagBits::eIndexBuffer |
      vk::BufferUsageFlagBits::eTransferDst,
    .intent = memory_intent::gpu_only,
  };
};

template<>
struct buffer_traits<buffer_kind::staging>
{
  static constexpr buffer_type_spec spec {
    .usage = vk::BufferUsageFlagBits::eTransferSrc,
    .intent = memory_intent::staging,
  };
};

template<>
struct buffer_traits<buffer_kind::readback>
{
  static constexpr buffer_type_spec spec {
    .usage = vk::BufferUsageFlagBits::eTransferDst,
    .intent = memory_intent::gpu_to_cpu,
  };
};

template<>
struct buffer_traits<buffer_kind::indirect>
{
  static constexpr buffer_type_spec spec {
    .usage = vk::BufferUsageFlagBits::eIndirectBuffer |
      vk::BufferUsageFlagBits::eStorageBuffer |
      vk::BufferUsageFlagBits::eTransferDst,
    .intent = memory_intent::gpu_only,
  };
};

template<>
struct buffer_traits<buffer_kind::device_address>
{
  static constexpr buffer_type_spec spec {
    .usage = vk::BufferUsageFlagBits::eShaderDeviceAddress |
      vk::BufferUsageFlagBits::eStorageBuffer |
      vk::BufferUsageFlagBits::eTransferDst,
    .intent = memory_intent::gpu_only,
  };
};

export struct unaddressed
{};

export struct addressed
{
  vk::DeviceAddress value {};
};

export template<buffer_kind Kind, class T = std::byte,
  class Address = unaddressed>
class buffer_view
{
public:
  using element_type = T;
  using address_policy = Address;

  buffer_view() = default;
  buffer_view(const buffer_view&) = default;
  buffer_view(buffer_view&&) = default;
  auto
  operator=(const buffer_view&) -> buffer_view& = default;
  auto
  operator=(buffer_view&&) -> buffer_view& = default;

  buffer_view(const buffer_view<Kind, T, addressed>& other)
    requires std::same_as<Address, unaddressed>
  : buffer_ { other.buffer() }, offset_ { other.offset() },
    size_ { other.size() }
  {}

  [[nodiscard]] auto
  buffer() const -> vk::Buffer
  { return buffer_; }

  [[nodiscard]] auto
  offset() const -> vk::DeviceSize
  { return offset_; }

  [[nodiscard]] auto
  size() const -> vk::DeviceSize
  { return size_; }

  [[nodiscard]] auto
  device_address() const -> vk::DeviceAddress
    requires std::same_as<Address, addressed>
  { return address_.value; }

private:
  template<buffer_kind, device_allocator>
  friend class buffer_resource;

  buffer_view(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size,
    Address address = {})
  : buffer_ { buffer }, offset_ { offset }, size_ { size }, address_ { address }
  {}

  vk::Buffer buffer_ {};
  vk::DeviceSize offset_ {};
  vk::DeviceSize size_ {};
  [[no_unique_address]] Address address_ {};
};

export template<class V>
concept addressed_buffer_view = requires(const V& v) {
  { v.device_address() } -> std::same_as<vk::DeviceAddress>;
};

export enum class shader_address_policy : std::uint8_t {
  never,
  auto_for_session,
  required,
};

export struct resource_create_options
{
  std::optional<shader_address_policy> address_policy {};

  [[nodiscard]] constexpr auto
  shader_address(shader_address_policy policy) const -> resource_create_options
  {
    auto copy = *this;
    copy.address_policy = policy;
    return copy;
  }
};

export [[nodiscard]] constexpr auto
default_shader_address_policy_for(
  descriptor_table_backend backend, buffer_kind kind) -> shader_address_policy
{
  if (backend != descriptor_table_backend::heap)
  {
    return shader_address_policy::never;
  }
  if (kind == buffer_kind::uniform || kind == buffer_kind::storage)
  {
    return shader_address_policy::auto_for_session;
  }
  return shader_address_policy::never;
}

export [[nodiscard]] constexpr auto
resolve_shader_address_policy(descriptor_table_backend backend,
  buffer_kind kind, const resource_create_options& options)
  -> shader_address_policy
{
  return options.address_policy.value_or(
    default_shader_address_policy_for(backend, kind));
}

export [[nodiscard]] constexpr auto
buffer_usage_with_address_policy(vk::BufferUsageFlags base,
  shader_address_policy policy) -> vk::BufferUsageFlags
{
  if (policy == shader_address_policy::never) { return base; }
  return base | vk::BufferUsageFlagBits::eShaderDeviceAddress;
}

export [[nodiscard]] auto
get_buffer_device_address(const vk::raii::Device& device, vk::Buffer buffer)
  -> vk::DeviceAddress
{
  return device.getBufferAddress({
    .buffer = buffer,
  });
}

export template<buffer_kind Kind>
inline constexpr bool is_host_visible_buffer_kind_v =
  buffer_traits<Kind>::spec.intent == memory_intent::staging ||
  buffer_traits<Kind>::spec.intent == memory_intent::cpu_to_gpu ||
  buffer_traits<Kind>::spec.intent == memory_intent::gpu_to_cpu;

export template<buffer_kind Kind>
concept host_visible_buffer_kind = is_host_visible_buffer_kind_v<Kind>;

export template<buffer_kind Kind, device_allocator Alloc = vma_policy>
class buffer_resource
{
public:
  buffer_resource() = default;

  [[nodiscard]] static auto
  create(Alloc& allocator, vk::DeviceSize size,
    std::span<const std::uint32_t> sharing_families = {})
    -> std::expected<buffer_resource, error_t>
  {
    constexpr buffer_type_spec spec = buffer_traits<Kind>::spec;
    return create_with_flags(
      allocator, size, spec.usage, spec.intent, sharing_families);
  }

  [[nodiscard]] static auto
  create(Alloc& allocator, vk::DeviceSize size,
    const resource_create_options& options,
    descriptor_table_backend backend = descriptor_table_backend::classic,
    std::span<const std::uint32_t> sharing_families = {})
    -> std::expected<buffer_resource, error_t>
  {
    constexpr buffer_type_spec spec = buffer_traits<Kind>::spec;
    const auto policy = resolve_shader_address_policy(backend, Kind, options);
    const auto usage = buffer_usage_with_address_policy(spec.usage, policy);
    return create_with_flags(
      allocator, size, usage, spec.intent, sharing_families);
  }

  template<typename T>
  [[nodiscard]] static auto
  create(Alloc& allocator, std::uint32_t count,
    std::span<const std::uint32_t> sharing_families = {})
    -> std::expected<buffer_resource, error_t>
  {
    static_assert(std::is_trivially_copyable_v<T>);
    return create(allocator, static_cast<vk::DeviceSize>(sizeof(T)) * count,
      sharing_families);
  }

  template<typename T>
  [[nodiscard]] static auto
  create(Alloc& allocator, std::span<const T> elements,
    std::span<const std::uint32_t> sharing_families = {})
    -> std::expected<buffer_resource, error_t>
  { return create(allocator, elements.size_bytes(), sharing_families); }

  [[nodiscard]] auto
  buffer() const -> vk::Buffer
  { return handle_.get(); }

  [[nodiscard]] auto
  size() const -> vk::DeviceSize
  { return size_; }

  [[nodiscard]] auto
  mapped() const -> void*
  { return handle_.mapped(); }

  [[nodiscard]] auto
  usage() const -> vk::BufferUsageFlags
  { return usage_; }

  [[nodiscard]] auto
  has_device_address() const -> bool
  { return bool { usage_ & vk::BufferUsageFlagBits::eShaderDeviceAddress }; }

  template<typename T>
    requires host_visible_buffer_kind<Kind>
  [[nodiscard]] auto
  mapped_span() const -> std::expected<std::span<T>, error_t>
  {
    static_assert(std::is_trivially_copyable_v<T>);
    auto* const ptr = mapped();
    if (ptr == nullptr)
    {
      return std::unexpected { make_app_error(app_error_code::mapping_failed) };
    }
    if (size_ % sizeof(T) != 0UZ)
    {
      return std::unexpected {
        make_app_error(app_error_code::invalid_data_size),
      };
    }
    if (auto inv = handle_.invalidate_mapped(0UZ, size_); !inv)
    {
      return std::unexpected { std::move(inv).error() };
    }
    return std::span<T> {
      static_cast<T*>(ptr),
      static_cast<std::size_t>(size_ / sizeof(T)),
    };
  }

  template<typename T>
    requires host_visible_buffer_kind<Kind>
  [[nodiscard]] auto
  copy_mapped_into(std::span<T> destination) const
    -> std::expected<void, error_t>
  {
    static_assert(std::is_trivially_copyable_v<T>);
    return mapped_span<T>().and_then(
      [ destination ](std::span<T> source) -> std::expected<void, error_t>
      {
        if (destination.size() > source.size())
        {
          return std::unexpected {
            make_app_error(app_error_code::invalid_data_size),
          };
        }
        std::ranges::copy_n(
          source.data(), destination.size(), destination.data());
        return {};
      });
  }

  template<class T = std::byte>
  [[nodiscard]] auto
  view(vk::DeviceSize offset = 0UZ, vk::DeviceSize size = vk::WholeSize) const
    -> buffer_view<Kind, T, unaddressed>
  {
    const auto extent = (size == vk::WholeSize) ? (size_ - offset) : size;
    return buffer_view<Kind, T, unaddressed> { buffer(), offset, extent };
  }

  template<class T = std::byte>
  [[nodiscard]] auto
  addressed_view(const device_context& device, vk::DeviceSize offset = 0UZ,
    vk::DeviceSize size = vk::WholeSize) const
    -> std::expected<buffer_view<Kind, T, addressed>, error_t>
  {
    if (!device.selected_capabilities().enabled.buffer_device_address)
    {
      return std::unexpected {
        make_app_error(app_error_code::feature_not_enabled),
      };
    }
    if (!has_device_address())
    {
      return std::unexpected {
        make_app_error(app_error_code::feature_not_enabled),
      };
    }
    const auto extent = (size == vk::WholeSize) ? (size_ - offset) : size;
    const auto base = get_buffer_device_address(device.device(), buffer());
    return buffer_view<Kind, T, addressed> {
      buffer(),
      offset,
      extent,
      addressed { .value = base + offset },
    };
  }

  explicit buffer_resource(typename Alloc::buffer_handle&& handle,
    vk::DeviceSize size, vk::BufferUsageFlags usage)
  : handle_ { std::move(handle) }, size_ { size }, usage_ { usage }
  {}

private:
  [[nodiscard]] static auto
  create_with_flags(Alloc& allocator, vk::DeviceSize size,
    vk::BufferUsageFlags usage, memory_intent intent,
    std::span<const std::uint32_t> sharing_families)
    -> std::expected<buffer_resource, error_t>
  {
    const bool concurrent = sharing_families.size() >= 2UZ;
    const vk::BufferCreateInfo buffer_info {
      .size = size,
      .usage = usage,
      .sharingMode =
        concurrent ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive,
      .queueFamilyIndexCount =
        concurrent ? static_cast<std::uint32_t>(sharing_families.size()) : 0U,
      .pQueueFamilyIndices = sharing_families.data(),
    };
    return allocator.create_buffer(buffer_info, intent)
      .transform(
        [ & ](typename Alloc::buffer_handle&& handle) -> buffer_resource
        { return buffer_resource { std::move(handle), size, usage }; });
  }

  typename Alloc::buffer_handle handle_ {};
  vk::DeviceSize size_ {};
  vk::BufferUsageFlags usage_ {};
};

export template<buffer_kind Kind, device_allocator Alloc = vma_policy>
  requires host_visible_buffer_kind<Kind>
class mapped_buffer
{
public:
  mapped_buffer() = default;

  explicit mapped_buffer(buffer_resource<Kind, Alloc>&& resource)
  : resource_ { std::move(resource) }
  {}

  template<typename T>
    requires(Kind == buffer_kind::uniform)
  [[nodiscard]] static auto
  create(Alloc& allocator, vk::DeviceSize min_ubo_alignment = 1UZ,
    std::span<const std::uint32_t> sharing_families = {})
    -> std::expected<mapped_buffer, error_t>
  {
    static_assert(std::is_trivially_copyable_v<T>);
    const vk::DeviceSize raw = sizeof(T);
    const vk::DeviceSize aligned = min_ubo_alignment <= 1UZ
      ? raw
      : (raw + min_ubo_alignment - 1UZ) / min_ubo_alignment * min_ubo_alignment;
    return buffer_resource<Kind, Alloc>::create(
      allocator, aligned, sharing_families)
      .transform([](buffer_resource<Kind, Alloc>&& resource) -> mapped_buffer
        { return mapped_buffer { std::move(resource) }; });
  }

  [[nodiscard]] static auto
  create(Alloc& allocator, vk::DeviceSize size,
    std::span<const std::uint32_t> sharing_families = {})
    -> std::expected<mapped_buffer, error_t>
    requires(Kind == buffer_kind::uniform)
  {
    return buffer_resource<Kind, Alloc>::create(
      allocator, size, sharing_families)
      .transform([](buffer_resource<Kind, Alloc>&& resource) -> mapped_buffer
        { return mapped_buffer { std::move(resource) }; });
  }

  void
  write(const void* data, std::size_t bytes, std::size_t offset = 0UZ)
  {
    std::memcpy(
      static_cast<std::byte*>(resource_.mapped()) + offset, data, bytes);
  }

  [[nodiscard]] auto
  buffer() const -> vk::Buffer
  { return resource_.buffer(); }

  [[nodiscard]] auto
  size() const -> vk::DeviceSize
  { return resource_.size(); }

  [[nodiscard]] auto
  mapped() -> void*
  { return resource_.mapped(); }

  [[nodiscard]] auto
  resource() & -> buffer_resource<Kind, Alloc>&
  { return resource_; }

  [[nodiscard]] auto
  resource() && -> buffer_resource<Kind, Alloc>&&
  { return std::move(resource_); }

  [[nodiscard]] auto
  usage() const -> vk::BufferUsageFlags
  { return resource_.usage(); }

  [[nodiscard]] auto
  has_device_address() const -> bool
  { return resource_.has_device_address(); }

  template<typename T>
  [[nodiscard]] auto
  mapped_span() const -> std::expected<std::span<T>, error_t>
  { return resource_.template mapped_span<T>(); }

  template<typename T>
  [[nodiscard]] auto
  copy_mapped_into(std::span<T> destination) const
    -> std::expected<void, error_t>
  { return resource_.copy_mapped_into(destination); }

  template<class T = std::byte>
  [[nodiscard]] auto
  view(vk::DeviceSize offset = 0UZ, vk::DeviceSize size = vk::WholeSize) const
    -> buffer_view<Kind, T, unaddressed>
  { return resource_.template view<T>(offset, size); }

  template<class T = std::byte>
  [[nodiscard]] auto
  addressed_view(const device_context& device, vk::DeviceSize offset = 0UZ,
    vk::DeviceSize size = vk::WholeSize) const
    -> std::expected<buffer_view<Kind, T, addressed>, error_t>
  { return resource_.template addressed_view<T>(device, offset, size); }

private:
  buffer_resource<Kind, Alloc> resource_ {};
};

export using uniform_buffer = mapped_buffer<buffer_kind::uniform>;
export using storage_buffer = buffer_resource<buffer_kind::storage>;
export using vertex_buffer = buffer_resource<buffer_kind::vertex>;
export using index_buffer = buffer_resource<buffer_kind::index>;
export using staging_buffer = buffer_resource<buffer_kind::staging>;
export using readback_buffer = buffer_resource<buffer_kind::readback>;
export using indirect_buffer = buffer_resource<buffer_kind::indirect>;
export using device_address_buffer =
  buffer_resource<buffer_kind::device_address>;

export template<descriptor_table_backend Backend,
  device_allocator Alloc = vma_policy>
auto
create_storage_buffer(Alloc& allocator, vk::DeviceSize size,
  resource_create_options options = {},
  std::span<const std::uint32_t> sharing_families = {})
  -> std::expected<storage_buffer, error_t>
{
  return storage_buffer::create(
    allocator, size, options, Backend, sharing_families);
}

export template<descriptor_table_backend Backend,
  device_allocator Alloc = vma_policy>
auto
create_uniform_buffer(Alloc& allocator, vk::DeviceSize size,
  resource_create_options options = {},
  std::span<const std::uint32_t> sharing_families = {})
  -> std::expected<uniform_buffer, error_t>
{
  return buffer_resource<buffer_kind::uniform, Alloc>::create(
    allocator, size, options, Backend, sharing_families)
    .transform([](buffer_resource<buffer_kind::uniform, Alloc>&& resource)
                 -> uniform_buffer
      { return uniform_buffer { std::move(resource) }; });
}

export template<device_allocator Alloc = vma_policy>
class buffer_resource_custom
{
public:
  buffer_resource_custom() = default;

  explicit buffer_resource_custom(
    typename Alloc::buffer_handle&& handle, vk::DeviceSize size)
  : handle_ { std::move(handle) }, size_ { size }
  {}

  [[nodiscard]] static auto
  create(Alloc& allocator, vk::DeviceSize size, vk::BufferUsageFlags usage,
    memory_intent intent, std::span<const std::uint32_t> sharing_families = {})
    -> std::expected<buffer_resource_custom, error_t>
  {
    const bool concurrent = sharing_families.size() >= 2UZ;
    const vk::BufferCreateInfo buffer_info {
      .size = size,
      .usage = usage,
      .sharingMode =
        concurrent ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive,
      .queueFamilyIndexCount =
        concurrent ? static_cast<std::uint32_t>(sharing_families.size()) : 0U,
      .pQueueFamilyIndices = sharing_families.data(),
    };
    return allocator.create_buffer(buffer_info, intent)
      .transform(
        [ & ](typename Alloc::buffer_handle&& handle) -> buffer_resource_custom
        { return buffer_resource_custom { std::move(handle), size }; });
  }

  [[nodiscard]] auto
  buffer() const -> vk::Buffer
  { return handle_.get(); }

  [[nodiscard]] auto
  size() const -> vk::DeviceSize
  { return size_; }

  [[nodiscard]] auto
  mapped() const -> void*
  { return handle_.mapped(); }

private:
  typename Alloc::buffer_handle handle_ {};
  vk::DeviceSize size_ {};
};

export template<device_allocator Alloc = vma_policy>
auto
make_buffer_resource(Alloc& allocator, vk::DeviceSize size,
  vk::BufferUsageFlags usage, memory_intent intent,
  std::span<const std::uint32_t> sharing_families = {})
  -> std::expected<buffer_resource_custom<Alloc>, error_t>
{
  return buffer_resource_custom<Alloc>::create(
    allocator, size, usage, intent, sharing_families);
}

export template<buffer_kind Kind, device_allocator Alloc = vma_policy>
auto
make_buffer_resource(Alloc& allocator, vk::DeviceSize size,
  std::span<const std::uint32_t> sharing_families = {})
  -> std::expected<buffer_resource<Kind, Alloc>, error_t>
{
  return buffer_resource<Kind, Alloc>::create(
    allocator, size, sharing_families);
}

export template<buffer_kind Kind, typename T,
  device_allocator Alloc = vma_policy>
auto
make_buffer_resource(Alloc& allocator, std::uint32_t count,
  std::span<const std::uint32_t> sharing_families = {})
  -> std::expected<buffer_resource<Kind, Alloc>, error_t>
{
  return buffer_resource<Kind, Alloc>::template create<T>(
    allocator, count, sharing_families);
}

export template<typename T, device_allocator Alloc = vma_policy>
auto
make_uniform_buffer(Alloc& allocator, vk::DeviceSize min_ubo_alignment = 1UZ,
  std::span<const std::uint32_t> sharing_families = {})
  -> std::expected<uniform_buffer, error_t>
  requires std::same_as<Alloc, vma_policy>
{
  return uniform_buffer::template create<T>(
    allocator, min_ubo_alignment, sharing_families);
}

export template<device_allocator Alloc = vma_policy>
auto
make_storage_buffer(Alloc& allocator, vk::DeviceSize size,
  std::span<const std::uint32_t> sharing_families = {})
  -> std::expected<storage_buffer, error_t>
{ return storage_buffer::create(allocator, size, sharing_families); }

export template<device_allocator Alloc = vma_policy>
auto
make_staging_buffer(Alloc& allocator, vk::DeviceSize size,
  std::span<const std::uint32_t> sharing_families = {})
  -> std::expected<staging_buffer, error_t>
{ return staging_buffer::create(allocator, size, sharing_families); }

export template<device_allocator Alloc = vma_policy>
auto
make_readback_buffer(Alloc& allocator, vk::DeviceSize size,
  std::span<const std::uint32_t> sharing_families = {})
  -> std::expected<readback_buffer, error_t>
{ return readback_buffer::create(allocator, size, sharing_families); }

export template<device_allocator Alloc = vma_policy>
auto
make_indirect_buffer(Alloc& allocator, vk::DeviceSize size,
  std::span<const std::uint32_t> sharing_families = {})
  -> std::expected<indirect_buffer, error_t>
{ return indirect_buffer::create(allocator, size, sharing_families); }

export template<device_allocator Alloc = vma_policy>
auto
make_device_address_buffer(Alloc& allocator, vk::DeviceSize size,
  std::span<const std::uint32_t> sharing_families = {})
  -> std::expected<device_address_buffer, error_t>
{ return device_address_buffer::create(allocator, size, sharing_families); }

export [[nodiscard]] constexpr auto
align_up(vk::DeviceSize value, vk::DeviceSize alignment) -> vk::DeviceSize
{
  if (alignment <= 1UZ) { return value; }
  return (value + alignment - 1UZ) / alignment * alignment;
}

export [[nodiscard]] constexpr auto
uniform_slice_stride(vk::DeviceSize element_size, vk::DeviceSize min_alignment)
  -> vk::DeviceSize
{ return align_up(element_size, min_alignment); }

export [[nodiscard]] constexpr auto
uniform_slice_offset(std::uint32_t index, vk::DeviceSize stride)
  -> vk::DeviceSize
{ return static_cast<vk::DeviceSize>(index) * stride; }

}; // namespace vkpp
