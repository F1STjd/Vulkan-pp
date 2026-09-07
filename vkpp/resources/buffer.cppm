module;

#include "error/vk_error_config.hpp"

export module vkpp.buffer;

import std;
import vulkan;

import vkpp.memory;
import vkpp.memory.vma;
import vkpp.error;
import vkpp.device;
import vkpp.command;
import vkpp.barrier;

namespace vkpp
{
using namespace std::string_view_literals;

export enum class buffer_kind : std::uint8_t {
  uniform, // FIF UBO
  storage, // ssbo
  vertex,
  index,
  staging,
  readback, // gpu_to_cpu
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

  explicit buffer_resource(
    typename Alloc::buffer_handle&& handle, vk::DeviceSize size)
  : handle_ { std::move(handle) }, size_ { size }
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
        { return buffer_resource { std::move(handle), size }; });
  }

  typename Alloc::buffer_handle handle_ {};
  vk::DeviceSize size_ {};
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

private:
  buffer_resource<Kind, Alloc> resource_ {};
};

export using uniform_buffer = mapped_buffer<buffer_kind::uniform>;
export using storage_buffer = buffer_resource<buffer_kind::storage>;
export using vertex_buffer = buffer_resource<buffer_kind::vertex>;
export using index_buffer = buffer_resource<buffer_kind::index>;
export using staging_buffer = buffer_resource<buffer_kind::staging>;
export using readback_buffer = buffer_resource<buffer_kind::readback>;

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

}; // namespace vkpp
