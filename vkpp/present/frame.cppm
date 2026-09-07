module;

#include "error/vk_error_config.hpp"

export module vkpp.frame;

import std;
import vulkan;

import vkpp.buffer;
import vkpp.command;
import vkpp.device;
import vkpp.error;
import vkpp.memory;

namespace vkpp
{
using namespace std::string_view_literals;

export struct frame
{
  vk::raii::Semaphore present_complete { nullptr };
  vk::raii::CommandBuffer command_buffer { nullptr };
  uniform_buffer uniform_buffer {};
  vk::DescriptorSet descriptor_set {};
};

export struct frames_create_info
{
  device_context& device;
  command_pool& pool;
  vk::DeviceSize min_ubo_alignment { 1UZ };
};

export template<std::size_t N, typename UniformType>
[[nodiscard]] auto
create_frames(const frames_create_info& info)
  -> std::expected<std::array<frame, N>, error_t>
{
  std::array<frame, N> frames {};

  return info.pool.allocate_primary(info.device.device(), N)
    .transform(
      [ & ](std::vector<vk::raii::CommandBuffer>&& command_buffers) -> void
      {
        for (std::size_t index : std::views::indices(N))
        {
          frames[ index ].command_buffer = std::move(command_buffers[ index ]);
        }
      })
    .and_then(
      [ & ] -> std::expected<void, error_t>
      {
        for (std::size_t index : std::views::indices(N))
        {
          auto semaphore = UTILS_VK(info.device.device().createSemaphore({}),
            ^^vk::raii::Device::createSemaphore);
          if (!semaphore)
          {
            return std::unexpected { std::move(semaphore).error() };
          }
          frames[ index ].present_complete = std::move(*semaphore);
        }
        return {};
      })
    .and_then(
      [ & ] -> std::expected<void, error_t>
      {
        for (std::size_t index : std::views::indices(N))
        {
          auto ubo = uniform_buffer::create<UniformType>(
            info.device.allocator(), info.min_ubo_alignment);
          if (!ubo) { return std::unexpected { std::move(ubo).error() }; }
          if (ubo->mapped() == nullptr)
          {
            return std::unexpected {
              app_error {
                .kind = app_error_kind::mapping_failed,
                .detail = "UBO mapping returned nullptr"sv,
              },
            };
          }
          frames[ index ].uniform_buffer = std::move(*ubo);
        }
        return {};
      })
    .transform([ & ] { return std::move(frames); });
}

}; // namespace vkpp
