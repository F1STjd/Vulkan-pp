module;

#include "error/vk_error_config.hpp"

export module vkpp.command;

import vkpp.error;

import std;
import vulkan;

namespace vkpp
{

export class command_pool
{
public:
  [[nodiscard]] static auto
  create(const vk::raii::Device& device, std::uint32_t queue_family_index)
    -> std::expected<command_pool, error_t>
  {
    const vk::CommandPoolCreateInfo command_pool_info {
      .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      .queueFamilyIndex = queue_family_index,
    };
    return UTILS_VK(device.createCommandPool(command_pool_info),
      ^^vk::raii::Device::createCommandPool)
      .transform(
        [](vk::raii::CommandPool&& pool)
        {
          command_pool output {};
          output.pool_ = std::move(pool);
          return output;
        });
  }

  [[nodiscard]] auto
  allocate_primary(const vk::raii::Device& device, std::uint32_t count)
    -> std::expected<std::vector<vk::raii::CommandBuffer>, error_t>
  {
    const vk::CommandBufferAllocateInfo allocate_info {
      .commandPool = *pool_,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = count,
    };
    return UTILS_VK(device.allocateCommandBuffers(allocate_info),
      ^^vk::raii::Device::allocateCommandBuffers);
  }

  [[nodiscard]] auto
  handle(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.pool_); }

private:
  vk::raii::CommandPool pool_ { nullptr };
};

export class single_time_submit
{
public:
  single_time_submit(command_pool& pool, const vk::raii::Device& device,
    const vk::raii::Queue& queue)
  : pool_ { &pool }, device_ { &device }, queue_ { &queue }
  {}

  single_time_submit(const single_time_submit&) = delete;

  auto
  operator=(const single_time_submit&) -> single_time_submit& = delete;

  single_time_submit(single_time_submit&&) noexcept = delete;

  auto
  operator=(single_time_submit&&) noexcept -> single_time_submit& = delete;

  [[nodiscard]] auto
  begin() -> std::expected<vk::raii::CommandBuffer*, error_t>
  {
    const vk::CommandBufferAllocateInfo allocate_info {
      .commandPool = *pool_->handle(),
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = 1U,
    };
    return UTILS_VK(device_->allocateCommandBuffers(allocate_info),
      ^^vk::raii::Device::allocateCommandBuffers)
      .and_then(
        [ this ](std::vector<vk::raii::CommandBuffer> buffers)
          -> std::expected<vk::raii::CommandBuffer*, error_t>
        {
          command_buffer_ = std::move(buffers.front());
          return UTILS_VK(
            command_buffer_.begin({
              .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
            }),
            ^^vk::raii::CommandBuffer::begin)
            .transform([ this ] { return &command_buffer_; });
        });
  }

  [[nodiscard]] auto
  end_and_submit() -> std::expected<void, error_t>
  {
    return UTILS_VK(command_buffer_.end(), ^^vk::raii::CommandBuffer::end)
      .and_then(
        [ this ] -> std::expected<void, error_t>
        {
          const vk::CommandBuffer handle = *command_buffer_;
          const vk::SubmitInfo submit_info {
            .commandBufferCount = 1U,
            .pCommandBuffers = &handle,
          };
          return UTILS_VK(
            queue_->submit(submit_info, nullptr), ^^vk::raii::Queue::submit);
        })
      .and_then([ this ] -> std::expected<void, error_t>
        { return UTILS_VK(queue_->waitIdle(), ^^vk::raii::Queue::waitIdle); });
  }

private:
  command_pool* pool_ {};
  const vk::raii::Device* device_ {};
  const vk::raii::Queue* queue_ {};
  vk::raii::CommandBuffer command_buffer_ { nullptr };
};

}; // namespace vkpp
