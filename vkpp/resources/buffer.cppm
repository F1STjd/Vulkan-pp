export module vkpp.buffer;

import std;
import vulkan;

import vkpp.memory;
import vkpp.memory.vma;
import vkpp.error;
import vkpp.device;
import vkpp.command;

namespace vkpp
{
using namespace std::string_view_literals;

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

export struct buffer_upload_create_info
{
  device_context& device;
  command_pool& pool;
  std::span<const std::byte> bytes {};
  vk::BufferUsageFlags gpu_usage {};
};

export auto
upload_device_local_buffer(const buffer_upload_create_info& create_info)
  -> std::expected<buffer_resource<>, error_t>
{
  const vk::DeviceSize byte_size = create_info.bytes.size_bytes();
  const vk::BufferUsageFlags destination_usage =
    create_info.gpu_usage | vk::BufferUsageFlagBits::eTransferDst;

  return make_buffer_resource(create_info.device.allocator(), byte_size,
    vk::BufferUsageFlagBits::eTransferSrc, memory_intent::staging)
    .and_then(
      [ & ](buffer_resource<>&& staging_buffer)
        -> std::expected<buffer_resource<>, error_t>
      {
        if (staging_buffer.mapped() == nullptr)
        {
          return std::unexpected {
            app_error {
              .kind = app_error_kind::mapping_failed,
              .detail = "Staging buffer map returned nullptr"sv,
            },
          };
        }

        std::memcpy(
          staging_buffer.mapped(), create_info.bytes.data(), byte_size);

        return make_buffer_resource(create_info.device.allocator(), byte_size,
          destination_usage, memory_intent::gpu_only)
          .and_then(
            [ &, staging_buffer = std::move(staging_buffer) ](
              buffer_resource<>&& device_local_buffer) mutable
              -> std::expected<buffer_resource<>, error_t>
            {
              single_time_submit single_time {
                create_info.pool,
                create_info.device.device(),
                create_info.device.graphics_queue(),
              };
              const vk::Buffer source_buffer = staging_buffer.buffer();
              const vk::Buffer destination_buffer =
                device_local_buffer.buffer();
              return single_time.begin()
                .and_then(
                  [ & ](vk::raii::CommandBuffer* command_buffer)
                    -> std::expected<void, error_t>
                  {
                    vk::BufferCopy region {
                      .srcOffset = 0UZ,
                      .dstOffset = 0UZ,
                      .size = byte_size,
                    };
                    command_buffer->copyBuffer(
                      source_buffer, destination_buffer, region);
                    return single_time.end_and_submit();
                  })
                .transform(
                  [ &,
                    device_local_buffer = std::move(
                      device_local_buffer) ] mutable -> buffer_resource<>
                  { return std::move(device_local_buffer); });
            });
      });
}
}; // namespace vkpp
