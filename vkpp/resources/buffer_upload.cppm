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
import vkpp.buffer.stage_pool;

namespace vkpp
{
using namespace std::string_view_literals;

export template<typename Resource>
class pending_upload
{
public:
  pending_upload() = default;

  pending_upload(staging_buffer&& staging, vk::raii::Semaphore&& copy_done,
    submission&& primary, Resource&& value)
  : staging_ { std::move(staging) }, copy_done_ { std::move(copy_done) },
    submissions_ { std::move(primary), submission {} },
    submission_count_ { 1U }, value_ { std::move(value) }
  {}

  pending_upload(staging_buffer&& staging, vk::raii::Semaphore&& copy_done,
    submission&& primary, submission&& secondary, Resource&& value)
  : staging_ { std::move(staging) }, copy_done_ { std::move(copy_done) },
    submissions_ { std::move(primary), std::move(secondary) },
    submission_count_ { 2U }, value_ { std::move(value) }
  {}

  pending_upload(stage_pool& pool, stage_allocation&& allocation,
    vk::raii::Semaphore&& copy_done, submission&& primary, Resource&& value)
  : pool_ { pool }, allocation_ { std::move(allocation) },
    copy_done_ { std::move(copy_done) },
    submissions_ { std::move(primary), submission {} },
    submission_count_ { 1U }, value_ { std::move(value) }
  {}

  pending_upload(stage_pool& pool, stage_allocation&& allocation,
    vk::raii::Semaphore&& copy_done, submission&& primary, submission&& secondary, Resource&& value)
  : pool_ { pool }, allocation_ { std::move(allocation) },
    copy_done_ { std::move(copy_done) },
    submissions_ { std::move(primary), std::move(secondary) },
    submission_count_ { 2U }, value_ { std::move(value) }
  {}

  [[nodiscard]] auto
  join() && -> std::expected<Resource, error_t>
  {
    for (std::uint32_t index : std::views::indices(submission_count_))
    {
      if (auto done = submissions_[ index ].wait(); !done)
      {
        return std::unexpected { std::move(done).error() };
      }
    }
    if(pool_.has_value())
    {
      pool_->free(allocation_);
      pool_ = std::nullopt;
    }
    return std::move(value_);
  }

private:
  staging_buffer staging_ {};
  std::optional<stage_pool&> pool_ {};
  stage_allocation allocation_ {};
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
  std::optional<stage_pool&> stage_pool {};
};

export template<buffer_kind Kind>
struct buffer_upload_create_info_for
{
  device_context& device;
  command_pool& pool;
  std::optional<command_pool&> transfer_pool {};
  std::span<const std::byte> bytes {};
  std::optional<stage_pool&> stage_pool {};
};

[[nodiscard]] auto
submit_single_queue_buffer_copy(const buffer_upload_create_info& create_info,
  vk::Buffer source_buffer, vk::Buffer destination_buffer,
  vk::DeviceSize byte_size, vk::DeviceSize src_offset)
  -> std::expected<submission, error_t>
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
        .srcOffset = src_offset,
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
  vk::DeviceSize byte_size, vk::DeviceSize src_offset)
  -> std::expected<submission, error_t>
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
        .srcOffset = src_offset,
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

template<buffer_kind Kind>
[[nodiscard]] auto
submit_buffer_upload(const buffer_upload_create_info_for<Kind>& kinded,
  staging_buffer&& staging, buffer_resource<Kind>&& device_local)
  -> std::expected<pending_upload<buffer_resource<Kind>>, error_t>
{
  const buffer_upload_create_info create_info {
    .device = kinded.device,
    .pool = kinded.pool,
    .transfer_pool = kinded.transfer_pool,
    .bytes = kinded.bytes,
    .gpu_usage = {},
    .stage_pool = kinded.stage_pool,
  };
  const vk::Buffer source_buffer = staging.buffer();
  const vk::Buffer destination_buffer = device_local.buffer();
  const vk::DeviceSize byte_size = device_local.size();
  const bool dual_queue = create_info.device.has_dedicated_transfer() &&
    create_info.transfer_pool.has_value();

  if (!dual_queue)
  {
    return submit_single_queue_buffer_copy(
      create_info, source_buffer, destination_buffer, byte_size, 0UZ)
      .transform(
        [ & ](
          submission&& copy_submitted) -> pending_upload<buffer_resource<Kind>>
        {
          return pending_upload<buffer_resource<Kind>> {
            std::move(staging),
            vk::raii::Semaphore { nullptr },
            std::move(copy_submitted),
            std::move(device_local),
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
        -> std::expected<pending_upload<buffer_resource<Kind>>, error_t>
      {
        return submit_transfer_copy_and_release(create_info, transfer,
          *copy_done, source_buffer, destination_buffer, byte_size, 0UZ)
          .and_then(
            [ & ](submission&& release_submitted)
              -> std::expected<pending_upload<buffer_resource<Kind>>, error_t>
            {
              return submit_graphics_acquire(
                create_info, transfer, *copy_done, destination_buffer)
                .transform(
                  [ &, release_submitted = std::move(release_submitted) ](
                    submission&& acquire_submitted) mutable
                    -> pending_upload<buffer_resource<Kind>>
                  {
                    return pending_upload<buffer_resource<Kind>> {
                      std::move(staging),
                      std::move(copy_done),
                      std::move(release_submitted),
                      std::move(acquire_submitted),
                      std::move(device_local),
                    };
                  });
            });
      });
}

template<buffer_kind Kind>
[[nodiscard]] auto
submit_buffer_upload(const buffer_upload_create_info_for<Kind>& kinded,
  stage_pool& pool, stage_allocation&& allocation,
  buffer_resource<Kind>&& device_local)
  -> std::expected<pending_upload<buffer_resource<Kind>>, error_t>
{
  const buffer_upload_create_info create_info {
    .device = kinded.device,
    .pool = kinded.pool,
    .transfer_pool = kinded.transfer_pool,
    .bytes = kinded.bytes,
    .gpu_usage = {},
    .stage_pool = kinded.stage_pool,
  };
  const vk::Buffer source_buffer = pool.buffer();
  const vk::DeviceSize src_offset = allocation.offset;
  const vk::Buffer destination_buffer = device_local.buffer();
  const vk::DeviceSize byte_size = device_local.size();
  const bool dual_queue = create_info.device.has_dedicated_transfer() &&
    create_info.transfer_pool.has_value();

  if (!dual_queue)
  {
    return submit_single_queue_buffer_copy(
      create_info, source_buffer, destination_buffer, byte_size, src_offset)
      .transform(
        [ & ](
          submission&& copy_submitted) -> pending_upload<buffer_resource<Kind>>
        {
          return pending_upload<buffer_resource<Kind>> {
            pool,
            std::move(allocation),
            vk::raii::Semaphore { nullptr },
            std::move(copy_submitted),
            std::move(device_local),
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
        -> std::expected<pending_upload<buffer_resource<Kind>>, error_t>
      {
        return submit_transfer_copy_and_release(create_info, transfer,
          *copy_done, source_buffer, destination_buffer, byte_size, src_offset)
          .and_then(
            [ & ](submission&& release_submitted)
              -> std::expected<pending_upload<buffer_resource<Kind>>, error_t>
            {
              return submit_graphics_acquire(
                create_info, transfer, *copy_done, destination_buffer)
                .transform(
                  [ &, release_submitted = std::move(release_submitted) ](
                    submission&& acquire_submitted) mutable
                    -> pending_upload<buffer_resource<Kind>>
                  {
                    return pending_upload<buffer_resource<Kind>> {
                      pool,
                      std::move(allocation),
                      std::move(copy_done),
                      std::move(release_submitted),
                      std::move(acquire_submitted),
                      std::move(device_local),
                    };
                  });
            });
      });
}

export template<buffer_kind Kind>
auto
upload_device_local_buffer(
  const buffer_upload_create_info_for<Kind>& create_info)
  -> std::expected<buffer_resource<Kind>, error_t>
{
  const vk::DeviceSize byte_size = create_info.bytes.size_bytes();

  if(create_info.stage_pool.has_value())
  {
    return create_info.stage_pool->allocate(byte_size, 4UZ)
      .and_then([&](stage_allocation&& allocation)
        -> std::expected<pending_upload<buffer_resource<Kind>>, error_t>
      {
        std::memcpy(allocation.mapped, create_info.bytes.data(), byte_size);
        return buffer_resource<Kind>::create(
          create_info.device.allocator(), byte_size)
          .and_then(
            [&, allocation = std::move(allocation)](
              buffer_resource<Kind>&& device_local) mutable
              -> std::expected<pending_upload<buffer_resource<Kind>>, error_t>
            {
              return submit_buffer_upload(create_info, *create_info.stage_pool,
                std::move(allocation), std::move(device_local));
            });
      })
      .and_then([](pending_upload<buffer_resource<Kind>>&& pending)
        -> std::expected<buffer_resource<Kind>, error_t>
      { return std::move(pending).join(); });
  }

  return staging_buffer::create(create_info.device.allocator(), byte_size)
    .and_then(
      [ & ](staging_buffer&& staging)
        -> std::expected<pending_upload<buffer_resource<Kind>>, error_t>
      {
        if (staging.mapped() == nullptr)
        {
          return std::unexpected {
            app_error {
              .kind = app_error_kind::mapping_failed,
              .detail = "Staging buffer map returned nullptr"sv,
            },
          };
        }

        std::memcpy(staging.mapped(), create_info.bytes.data(), byte_size);
        return buffer_resource<Kind>::create(
          create_info.device.allocator(), byte_size)
          .and_then(
            [ &, staging = std::move(staging) ](
              buffer_resource<Kind>&& device_local) mutable
              -> std::expected<pending_upload<buffer_resource<Kind>>, error_t>
            {
              return submit_buffer_upload(
                create_info, std::move(staging), std::move(device_local));
            });
      })
    .and_then([](pending_upload<buffer_resource<Kind>>&& pending)
                -> std::expected<buffer_resource<Kind>, error_t>
      { return std::move(pending).join(); });
}

} // namespace vkpp
