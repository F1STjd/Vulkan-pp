module;

#include "error/vk_error_config.hpp"

export module vkpp.buffer.upload;

import std;
import vulkan;

export import vkpp.buffer;
import vkpp.memory;
import vkpp.memory.vma;
import vkpp.error;
import vkpp.device;
import vkpp.command;
import vkpp.barrier;

namespace vkpp
{
using namespace std::string_view_literals;

export template<typename Resource>
class pending_upload
{
public:
  pending_upload() = default;
  pending_upload(buffer_resource<>&& staging, vk::raii::Semaphore&& copy_done,
    submission&& primary, Resource&& value)
  : staging_ { std::move(staging) }, copy_done_ { std::move(copy_done) },
    submissions_ { std::move(primary), submission {} },
    submission_count_ { 1U }, value_ { std::move(value) }
  {}

  pending_upload(buffer_resource<>&& staging, vk::raii::Semaphore&& copy_done,
    submission&& primary, submission&& secondary, Resource&& value)
  : staging_ { std::move(staging) }, copy_done_ { std::move(copy_done) },
    submissions_ { std::move(primary), std::move(secondary) },
    submission_count_ { 2U }, value_ { std::move(value) }
  {}

  [[nodiscard]] auto
  join() && -> std::expected<Resource, error_t>
  {
    for (std::uint32_t index : std::views::iota(0U, submission_count_))
    {
      if (auto done = submissions_[ index ].wait(); !done)
      {
        return std::unexpected { std::move(done).error() };
      }
    }
    return std::move(value_);
  }

private:
  buffer_resource<> staging_ {};
  vk::raii::Semaphore copy_done_ { nullptr };
  std::array<submission, 2> submissions_ {};
  std::uint32_t submission_count_ { 0U };
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

[[nodiscard]] auto
submit_single_queue_buffer_copy(const buffer_upload_create_info& create_info,
  vk::Buffer source_buffer, vk::Buffer destination_buffer,
  vk::DeviceSize byte_size) -> std::expected<submission, error_t>
{
  single_time_submit single_time {
    create_info.pool,
    create_info.device.device(),
    create_info.device.graphics_queue(),
  };
  return single_time.begin().and_then(
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
    });
}

[[nodiscard]] auto
submit_transfer_copy_and_release(const buffer_upload_create_info& create_info,
  ownership_transfer transfer, vk::Semaphore signal_copy_done,
  vk::Buffer source_buffer, vk::Buffer destination_buffer,
  vk::DeviceSize byte_size) -> std::expected<submission, error_t>
{
  single_time_submit transfer_submit {
    *create_info.transfer_pool,
    create_info.device.device(),
    create_info.device.transfer_queue(),
  };
  return transfer_submit.begin().and_then(
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
        upload::deferred, { .signal = signal_copy_done });
    });
}

[[nodiscard]] auto
submit_graphics_acquire(const buffer_upload_create_info& create_info,
  ownership_transfer transfer, vk::Semaphore wait_copy_done,
  vk::Buffer destination_buffer) -> std::expected<submission, error_t>
{
  single_time_submit graphics_submit {
    create_info.pool,
    create_info.device.device(),
    create_info.device.graphics_queue(),
  };

  return graphics_submit.begin().and_then(
    [ & ] -> std::expected<submission, error_t>
    {
      const buffer_barrier acquire = acquire_buffer_ownership(
        destination_buffer, transfer, vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eMemoryRead);
      record_barriers(
        graphics_submit.command_buffer(), std::span { &acquire, 1UZ });
      return graphics_submit.end_and_submit(
        upload::deferred, { .wait = wait_copy_done });
    });
}

[[nodiscard]] auto
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
    return submit_single_queue_buffer_copy(
      create_info, source_buffer, destination_buffer, byte_size)
      .transform(
        [ & ](submission&& copy_submitted) -> pending_upload<buffer_resource<>>
        {
          return pending_upload<buffer_resource<>> {
            std::move(staging_buffer),
            vk::raii::Semaphore { nullptr },
            std::move(copy_submitted),
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
        return submit_transfer_copy_and_release(create_info, transfer,
          *copy_done, source_buffer, destination_buffer, byte_size)
          .and_then(
            [ & ](submission&& release_submitted)
              -> std::expected<pending_upload<buffer_resource<>>, error_t>
            {
              return submit_graphics_acquire(
                create_info, transfer, *copy_done, destination_buffer)
                .transform(
                  [ &, release_submitted = std::move(release_submitted) ](
                    submission&& acquire_submitted) mutable
                    -> pending_upload<buffer_resource<>>
                  {
                    return pending_upload<buffer_resource<>> {
                      std::move(staging_buffer),
                      std::move(copy_done),
                      std::move(release_submitted),
                      std::move(acquire_submitted),
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

} // namespace vkpp
