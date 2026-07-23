module;

#include "error/vk_error_config.hpp"

export module vkpp.frame;

import vkpp.buffer;
import vkpp.command;
import vkpp.device;
import vkpp.error;
import vkpp.memory;

import std;
import vulkan;

namespace vkpp
{
using namespace std::string_view_literals;

export struct frame
{
  vk::raii::Semaphore present_complete { nullptr };
  vk::raii::Fence in_flight { nullptr };
  vk::raii::CommandBuffer command_buffer { nullptr };
  mapped_buffer<> uniform_buffer {};
  vk::raii::DescriptorSet descriptor_set { nullptr };
};

export struct frames_create_info
{
  device_context* device {};
  command_pool* pool {};
  vk::DeviceSize ubo_size {};
};

export template<std::size_t N>
[[nodiscard]] auto
create_frames(const frames_create_info& info)
  -> std::expected<std::array<frame, N>, error_t>
{
  std::array<frame, N> frames {};

  return info.pool->allocate_primary(info.device->device(), N)
    .transform(
      // TODO: Konrad - passing by value silences clang-tidy, but idomatic
      // approach should be to pass by r-value ref (3 pointer move overhead)
      [ & ](std::vector<vk::raii::CommandBuffer> command_buffers) -> void
      {
        for (std::size_t index : std::views::iota(0UZ, N))
        {
          frames[ index ].command_buffer = std::move(command_buffers[ index ]);
        }
      })
    .and_then(
      [ & ] -> std::expected<void, error_t>
      {
        for (std::size_t index : std::views::iota(0UZ, N))
        {
          auto semaphore = UTILS_VK(info.device->device().createSemaphore({}),
            ^^vk::raii::Device::createSemaphore);
          if (!semaphore)
          {
            return std::unexpected { std::move(semaphore).error() };
          }
          frames[ index ].present_complete = std::move(*semaphore);

          auto fence = UTILS_VK( //
            info.device->device().createFence(
              { .flags = vk::FenceCreateFlagBits::eSignaled }),
            ^^vk::raii::Device::createFence);
          if (!fence) { return std::unexpected { std::move(fence).error() }; }
          frames[ index ].in_flight = std::move(*fence);
        }
        return {};
      })
    .and_then(
      [ & ] -> std::expected<void, error_t>
      {
        for (std::size_t index : std::views::iota(0UZ, N))
        {
          auto ubo = make_buffer_resource(info.device->allocator(),
            info.ubo_size, vk::BufferUsageFlagBits::eUniformBuffer,
            memory_intent::cpu_to_gpu);
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
          frames[ index ].uniform_buffer = mapped_buffer<> { std::move(*ubo) };
        }
        return {};
      })
    // TODO: Konrad - Find the difference between returning by value or r-value
    // ref. R-value ref is disabling (N)RVO
    .transform([ & ] { return std::move(frames); });
}

[[nodiscard]] auto
create_frame_slot(device_context& device,
  vk::raii::CommandBuffer&& command_buffer, vk::DeviceSize ubo_size)
  -> std::expected<frame, error_t>
{
  frame output {
    .command_buffer = std::move(command_buffer),
  };
  return UTILS_VK(
    device.device().createSemaphore({}), ^^vk::raii::Device::createSemaphore)
    .transform([ & ](vk::raii::Semaphore&& semaphore) -> void
      { output.present_complete = std::move(semaphore); })
    .and_then(
      [ & ] -> std::expected<vk::raii::Fence, error_t>
      {
        return UTILS_VK( //
          device.device().createFence(
            { .flags = vk::FenceCreateFlagBits::eSignaled }),
          ^^vk::raii::Device::createFence);
      })
    .transform([ & ](vk::raii::Fence&& fence) -> void
      { output.in_flight = std::move(fence); })
    .and_then(
      [ & ] -> std::expected<mapped_buffer<>, error_t>
      {
        return make_buffer_resource(device.allocator(), ubo_size,
          vk::BufferUsageFlagBits::eUniformBuffer, memory_intent::cpu_to_gpu)
          .and_then(
            [](buffer_resource<>&& buffer)
              -> std::expected<mapped_buffer<>, error_t>
            {
              if (buffer.mapped() == nullptr)
              {
                return std::unexpected {
                  app_error {
                    .kind = app_error_kind::mapping_failed,
                    .detail = "UBO mapping returned nullptr"sv,
                  },
                };
              }
              return mapped_buffer<> { std::move(buffer) };
            });
      })
    .transform(
      [ & ](mapped_buffer<>&& ubo) -> frame
      {
        output.uniform_buffer = std::move(ubo);
        return std::move(output);
      });
}

}; // namespace vkpp
