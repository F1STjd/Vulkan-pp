module;

#include "error/vk_error_config.hpp"

export module vkpp.buffer.download;

import std;
import vulkan;

export import vkpp.buffer;
import vkpp.memory;
import vkpp.memory.vma;
import vkpp.error;
import vkpp.device;
import vkpp.command;
import vkpp.barrier;
import vkpp.graph;

namespace vkpp
{
using namespace std::string_view_literals;

export class pending_download
{
public:
  pending_download() = default;

  pending_download(buffer_resource<>&& staging, submission&& copy_sobmitted)
  : staging_ { std::move(staging) },
    submissions_ { { std::move(copy_sobmitted), submission {} } },
    submission_count_ { 1U }
  {}

  [[nodiscard]] auto
  join() && -> std::expected<buffer_resource<>, error_t>
  {
    // TODO (Konrad): Check if there are more wrong usecases of
    // std::views::indeices in the codebase.
    // Top 3 is:
    // 1. dedicated algorithm
    // 1.5 sometimes range for loop is clearer than an algorithm
    // 2. std::views::indices if index is needed
    // 3. std::views::iota if we do not start from 0
    // ... everything else
    // 99. C-style for loop
    for (auto& submission : submissions_)
    {
      if (auto done = submission.wait(); !done)
      {
        return std::unexpected { std::move(done).error() };
      }
    }
    return std::move(staging_);
  }

private:
  buffer_resource<> staging_ {};
  std::array<submission, 2> submissions_ {};
  std::uint32_t submission_count_ { 0U };
};

export struct buffer_download_create_info
{
  device_context& device;
  command_pool& pool;
  std::optional<command_pool&> transfer_pool {};
  vk::Buffer source {};
  vk::DeviceSize offset {};
  vk::DeviceSize size {};
};

export auto
download_device_local_buffer(const buffer_download_create_info& create_info)
  -> std::expected<pending_download, error_t>
{
  const vk::DeviceSize byte_size = create_info.size;
  return make_buffer_resource<buffer_kind::readback>(
    create_info.device.allocator(), byte_size)
    .and_then(
      [ & ](
        buffer_resource<>&& staging) -> std::expected<pending_download, error_t>
      {
        if (staging.mapped() == nullptr)
        {
          return std::unexpected {
            app_error {
              .kind = app_error_kind::mapping_failed,
              .detail = "readback staging map returned nullptr"sv,
            },
          };
        }
        single_time_submit single_time {
          create_info.pool,
          create_info.device.device(),
          create_info.device.graphics_queue(),
        };
        return single_time.begin().and_then(
          [ &, staging = std::move(staging) ] mutable
            -> std::expected<pending_download, error_t>
          {
            const vk::BufferCopy region {
              .srcOffset = create_info.offset,
              .dstOffset = 0UZ,
              .size = byte_size,
            };
            single_time.command_buffer().copyBuffer(
              create_info.source, staging.buffer(), region);
            return single_time.end_and_submit(upload::deferred)
              .transform(
                [ &, staging = std::move(staging) ](
                  submission&& submitted) mutable -> pending_download
                {
                  return pending_download {
                    std::move(staging),
                    std::move(submitted),
                  };
                });
          });
      });
}

export auto
download_device_local_buffer_and_join(
  const buffer_download_create_info& create_info)
  -> std::expected<buffer_resource<>, error_t>
{
  return download_device_local_buffer(create_info)
    .and_then(
      [](
        pending_download&& pending) -> std::expected<buffer_resource<>, error_t>
      { return std::move(pending).join(); });
}

} // namespace vkpp
