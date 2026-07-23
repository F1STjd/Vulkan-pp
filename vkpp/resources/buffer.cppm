export module vkpp.buffer;

import std;
import vulkan;
import vkpp.memory;
import vkpp.memory.vma;
import vkpp.error;

namespace vkpp
{
export template<device_allocator Alloc = vma_policy>
class buffer_resource
{
public:
  buffer_resource() = default;
  buffer_resource(typename Alloc::buffer_handle&& handle, vk::DeviceSize size)
  : handle_ { std::move(handle) }, size_ { size }
  {}

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
class mapped_buffer
{
public:
  mapped_buffer() = default;
  explicit mapped_buffer(buffer_resource<Alloc>&& resource)
  : resource_ { std::move(resource) }
  {}

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

private:
  buffer_resource<Alloc> resource_ {};
};

export template<device_allocator Alloc = vma_policy>
auto
make_buffer_resource(Alloc& allocator, vk::DeviceSize size,
  vk::BufferUsageFlags usage, memory_intent intent)
  -> std::expected<buffer_resource<Alloc>, error_t>
{
  const vk::BufferCreateInfo buffer_info {
    .size = size,
    .usage = usage,
    .sharingMode = vk::SharingMode::eExclusive,
  };
  return allocator.create_buffer(buffer_info, intent)
    .transform(
      [ & ](typename Alloc::buffer_handle&& handle)
      {
        return buffer_resource<Alloc> {
          std::move(handle),
          size,
        };
      });
}
}; // namespace vkpp
