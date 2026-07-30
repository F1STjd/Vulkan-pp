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
  vk::BufferUsageFlags usage, memory_intent intent,
  std::span<const std::uint32_t> sharing_families = {})
  -> std::expected<buffer_resource<Alloc>, error_t>
{
  const bool concurrent = sharing_families.size() >= 2UZ;
  const vk::BufferCreateInfo buffer_info {
    .size = size,
    .usage = usage,
    .sharingMode =
      concurrent ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive,
    .queueFamilyIndexCount =
      concurrent ? static_cast<std::uint32_t>(sharing_families.size()) : 0U,
    .pQueueFamilyIndices = concurrent ? sharing_families.data() : nullptr,
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

export template<typename Resource>
class pending_upload
{
public:
  pending_upload() = default;
  pending_upload(buffer_resource<>&& staging, vk::raii::Semaphore&& copy_done,
    std::inplace_vector<submission, 2>&& submissions, Resource&& value)
  : staging_ { std::move(staging) }, copy_done_ { std::move(copy_done) },
    submissions_ { std::move(submissions) }, value_ { std::move(value) }
  {}

  [[nodiscard]] auto
  join() && -> std::expected<Resource, error_t>
  {
    for (submission& pending : submissions_)
    {
      if (auto done = pending.wait(); !done)
      {
        return std::unexpected { std::move(done.error()) };
      }
    }
    return std::move(value_);
  }

private:
  buffer_resource<> staging_ {};
  vk::raii::Semaphore copy_done_ { nullptr };
  std::inplace_vector<submission, 2> submissions_ {};
  Resource value_ {};
};

export struct buffer_upload_create_info
{
  device_context& device;
  command_pool& pool;
  std::optional<command_pool&> transfer_pool {};
  std::span<const std::byte> bytes {};
  vk::BufferUsageFlags gpu_usage {};
};

[[nodiscard]] inline auto
submit_buffer_upload(const buffer_upload_create_info& create_info,
  buffer_resource<>&& staging_buffer, buffer_resource<>&& device_local_buffer)
  -> std::expected<pending_upload<buffer_resource<>>, error_t>
{
  const vk::Buffer source_buffer = staging_buffer.buffer();
  const vk::Buffer destination_buffer = device_local_buffer.buffer();
  const vk::DeviceSize byte_size = device_local_buffer.size();

  const bool dual_queue = create_info.device.has_dedicated_transfer() &&
    create_info.transfer_pool.has_value();

  if (!dual_queue)
  {
    single_time_submit single_time {
      create_info.pool,
      create_info.device.device(),
      create_info.device.graphics_queue(),
    };
    return single_time.begin()
      .and_then(
        [ & ] -> std::expected<submission, error_t>
        {
          const vk::BufferCopy region {
            .srcOffset = 0UZ,
            .dstOffset = 0UZ,
            .size = byte_size,
          };
          single_time.command_buffer().copyBuffer(
            source_buffer, destination_buffer, region);
          return single_time.end_and_submit(upload::deferred);
        })
      .transform(
        [ & ](submission&& copy_submission) -> pending_upload<buffer_resource<>>
        {
          std::inplace_vector<submission, 2> submissions {};
          submissions.push_back(std::move(copy_submission));
          return pending_upload<buffer_resource<>> {
            std::move(staging_buffer),
            vk::raii::Semaphore { nullptr },
            std::move(submissions),
            std::move(device_local_buffer),
          };
        });
  }

  const ownership_transfer transfer {
    .src_queue_family = create_info.device.transfer_qf_index(),
    .dst_queue_family = create_info.device.graphics_qf_index(),
  };
  return UTILS_VK(create_info.device.device().createSemaphore({}),
    ^^vk::raii::Device::createSemaphore)
    .and_then(
      [ & ](vk::raii::Semaphore&& copy_done)
        -> std::expected<pending_upload<buffer_resource<>>, error_t>
      {
        single_time_submit transfer_submit {
          *create_info.transfer_pool,
          create_info.device.device(),
          create_info.device.transfer_queue(),
        };
        single_time_submit graphics_submit {
          create_info.pool,
          create_info.device.device(),
          create_info.device.graphics_queue(),
        };
        return transfer_submit.begin()
          .and_then(
            [ & ] -> std::expected<submission, error_t>
            {
              const vk::BufferCopy region {
                .srcOffset = 0UZ,
                .dstOffset = 0UZ,
                .size = byte_size,
              };
              transfer_submit.command_buffer().copyBuffer(
                source_buffer, destination_buffer, region);
              const buffer_barrier release = release_buffer_ownership(
                destination_buffer, transfer, vk::PipelineStageFlagBits2::eCopy,
                vk::AccessFlagBits2::eTransferWrite);
              record_barriers(
                transfer_submit.command_buffer(), std::span { &release, 1UZ });
              return transfer_submit.end_and_submit(
                upload::deferred, { .signal = *copy_done });
            })
          .and_then(
            [ & ](submission&& release_submitted)
              -> std::expected<pending_upload<buffer_resource<>>, error_t>
            {
              return graphics_submit.begin()
                .and_then(
                  [ & ] -> std::expected<submission, error_t>
                  {
                    const buffer_barrier acquire =
                      acquire_buffer_ownership(destination_buffer, transfer,
                        vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eMemoryRead);
                    record_barriers(graphics_submit.command_buffer(),
                      std::span { &acquire, 1UZ });
                    return graphics_submit.end_and_submit(
                      upload::deferred, { .wait = *copy_done });
                  })
                .transform(
                  [ &, release_submitted = std::move(release_submitted) ](
                    submission&& acquire_submitted) mutable
                    -> pending_upload<buffer_resource<>>
                  {
                    std::inplace_vector<submission, 2> submissions {};
                    submissions.push_back(std::move(release_submitted));
                    submissions.push_back(std::move(acquire_submitted));
                    return pending_upload<buffer_resource<>> {
                      std::move(staging_buffer),
                      std::move(copy_done),
                      std::move(submissions),
                      std::move(device_local_buffer),
                    };
                  });
            });
      });
}

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
        -> std::expected<pending_upload<buffer_resource<>>, error_t>
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
              -> std::expected<pending_upload<buffer_resource<>>, error_t>
            {
              return submit_buffer_upload(create_info,
                std::move(staging_buffer), std::move(device_local_buffer));
            });
      })
    .and_then([](pending_upload<buffer_resource<>>&& pending)
                -> std::expected<buffer_resource<>, error_t>
      { return std::move(pending).join(); });
}
}; // namespace vkpp
