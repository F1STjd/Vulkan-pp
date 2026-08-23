module;

#include "error/vk_error_config.hpp"

export module vkpp.buffer.arena.upload;

import std;
import vulkan;

export import vkpp.buffer.arena;
import vkpp.memory;
import vkpp.memory.vma;
import vkpp.error;
import vkpp.device;
import vkpp.command;
import vkpp.barrier;

namespace vkpp
{
using namespace std::string_view_literals;

export struct arena_slice_upload
{
  std::span<const std::byte> bytes {};
  vk::DeviceSize dst_offset {};
};

export struct arena_upload_create_info
{
  device_context& device;
  command_pool& pool;
  std::optional<command_pool&> transfer_pool {};
  vk::Buffer arena {};
  std::span<const arena_slice_upload> slices {};
};

[[nodiscard]] auto
fill_staging_and_regions(const arena_upload_create_info& create_info,
  buffer_resource<>& staging) -> std::vector<vk::BufferCopy>
{
  std::vector<vk::BufferCopy> regions {};
  regions.reserve(create_info.slices.size());
  vk::DeviceSize src_offset { 0UZ };
  auto* mapped = static_cast<std::byte*>(staging.mapped());
  for (const auto& slice : create_info.slices)
  {
    std::memcpy(
      mapped + src_offset, slice.bytes.data(), slice.bytes.size_bytes());
    regions.push_back({
      .srcOffset = src_offset,
      .dstOffset = slice.dst_offset,
      .size = slice.bytes.size_bytes(),
    });
    src_offset += slice.bytes.size_bytes();
  }
  return regions;
}

[[nodiscard]] auto
submit_arena_copy_single_queue(const arena_upload_create_info& create_info,
  vk::Buffer staging, std::span<const vk::BufferCopy> regions)
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
      single_time.command_buffer().copyBuffer(
        staging, create_info.arena, regions);
      return single_time.end_and_submit(upload::deferred);
    });
}

[[nodiscard]] auto
submit_arena_copy_dual_queue(const arena_upload_create_info& create_info,
  vk::Buffer staging, std::span<const vk::BufferCopy> regions)
  -> std::expected<void, error_t>
{
  const ownership_transfer transfer {
    .src_queue_family = create_info.device.transfer_qf_index(),
    .dst_queue_family = create_info.device.graphics_qf_index(),
  };
  // I like the idea of single nesting monadic chain more. So transfer one
  // branch in nesting levels, the grafics comes to the top again, and starts
  // nesting again.
  // TODO (Konrad): all vkpp should follow this nesting rule - later it will be
  // better to isolate monadic chain parts for helper functions (if needed).
  // This is like one function one responsibility <=> first level of nesting in
  // the monadic chain one resposnsibility
  // Inside the plan each level of responsibility is at +1 level of nesting. If
  // nothing I wrote here is easy to understand then look at this diagram:
  //
  // return createSemaphore
  //   | do the transfer submit
  //     | if it needs nesting its ok (it has to be about transfer)
  //   | do the graphics submit
  //     | if it needs nesting its ok (it has to be about graphics)
  //
  //  Edit: It does not work, because of semaphore dependency. I would have to
  //  move it out of monadic chain to use in the second top level and_then.
  //  Think if there could be a way of splitting the responsibilities without
  //  createing global dependencies. Returning both submission and semaphore is
  //  not the idea I lean to
  return UTILS_VK(create_info.device.device().createSemaphore({}),
    ^^vk::raii::Device::createSemaphore)
    .and_then(
      [ & ](vk::raii::Semaphore&& copy_done) -> std::expected<void, error_t>
      {
        single_time_submit transfer_submit {
          *create_info.transfer_pool,
          create_info.device.device(),
          create_info.device.transfer_queue(),
        };
        return transfer_submit.begin()
          .and_then(
            [ & ] -> std::expected<submission, error_t>
            {
              transfer_submit.command_buffer().copyBuffer(
                staging, create_info.arena, regions);
              const buffer_barrier release = release_buffer_ownership(
                create_info.arena, transfer, vk::PipelineStageFlagBits2::eCopy,
                vk::AccessFlagBits2::eTransferWrite);
              record_barriers(
                transfer_submit.command_buffer(), std::span { &release, 1UZ });
              return transfer_submit.end_and_submit(
                upload::deferred, { .signal = *copy_done });
            })
          .and_then(
            [ & ](
              submission&& release_submitted) -> std::expected<void, error_t>
            {
              single_time_submit graphics_submit {
                create_info.pool,
                create_info.device.device(),
                create_info.device.graphics_queue(),
              };
              return graphics_submit.begin()
                .and_then(
                  [ & ] -> std::expected<submission, error_t>
                  {
                    const buffer_barrier acquire =
                      acquire_buffer_ownership(create_info.arena, transfer,
                        vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eMemoryRead);
                    record_barriers(graphics_submit.command_buffer(),
                      std::span { &acquire, 1UZ });
                    return graphics_submit.end_and_submit(
                      upload::deferred, { .wait = *copy_done });
                  })
                .and_then(
                  [ &release_submitted ](submission&& acquire_submitted)
                    -> std::expected<void, error_t>
                  {
                    return release_submitted.wait().and_then(
                      [ & ] -> std::expected<void, error_t>
                      { return acquire_submitted.wait(); });
                  });
            });
      });
}

export [[nodiscard]] auto
upload_arena_slices(const arena_upload_create_info& create_info)
  -> std::expected<void, error_t>
{
  vk::DeviceSize total { 0UZ };
  for (const auto& slice : create_info.slices)
  {
    total += slice.bytes.size_bytes();
  }

  return make_buffer_resource(create_info.device.allocator(), total,
    vk::BufferUsageFlagBits::eTransferSrc, memory_intent::staging)
    .and_then(
      [ & ](vkpp::buffer_resource<>&& staging) -> std::expected<void, error_t>
      {
        if (staging.mapped() == nullptr)
        {
          return std::unexpected {
            app_error { .kind = app_error_kind::mapping_failed,
              .detail = "Staging buffer map returned nullptr"sv },
          };
        }
        const std::vector<vk::BufferCopy> regions =
          fill_staging_and_regions(create_info, staging);
        const bool dual_queue = create_info.device.has_dedicated_transfer() &&
          create_info.transfer_pool.has_value();
        if (dual_queue)
        {
          return submit_arena_copy_dual_queue(
            create_info, staging.buffer(), regions);
        }
        return submit_arena_copy_single_queue(
          create_info, staging.buffer(), regions)
          .and_then([](submission&& done) -> std::expected<void, error_t>
            { return done.wait(); });
      });
}

} // namespace vkpp
