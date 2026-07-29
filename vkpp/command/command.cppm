module;

#include "error/vk_error_config.hpp"

export module vkpp.command;

import std;
import vulkan;

import vkpp.error;

namespace vkpp
{

export namespace upload
{

struct wait_idle_t
{};

inline constexpr wait_idle_t wait_idle {};

struct deferred_t
{};

inline constexpr deferred_t deferred {};

} // namespace upload

export class submission
{
public:
  submission() = default;
  submission(const vk::raii::Device& device, vk::raii::Fence&& fence,
    vk::raii::CommandBuffer&& command_buffer)
  : device_ { device }, fence_ { std::move(fence) },
    command_buffer_ { std::move(command_buffer) }
  {}

  submission(const submission&) = delete;

  auto
  operator=(const submission&) -> submission& = delete;

  submission(submission&&) noexcept = default;

  auto
  operator=(submission&&) noexcept -> submission& = default;

  ~submission()
  {
    if (device_.has_value() && fence_ != nullptr) { (void)wait(); }
  }

  [[nodiscard]] auto
  wait(std::uint64_t timeout_ns = std::numeric_limits<std::uint64_t>::max())
    -> std::expected<void, error_t>
  {
    if (const auto result =
          device_->waitForFences(*fence_, vk::True, timeout_ns);
      result != vk::Result::eSuccess)
    {
      return std::unexpected {
        vk_error {
          .function = "waitForFences",
          .type = "vk::raii::Device",
          .result = result,
        },
      };
    }
    return {};
  }

  [[nodiscard]] auto
  is_done() -> std::expected<bool, error_t>
  {
    const auto result = fence_.getStatus();
    if (result == vk::Result::eSuccess) { return true; }
    if (result == vk::Result::eNotReady) { return false; }
    return std::unexpected {
      vk_error {
        .function = "getStatus",
        .type = "vk::raii::Fence",
        .result = result,
      },
    };
  }

private:
  std::optional<const vk::raii::Device&> device_ {};
  vk::raii::Fence fence_ { nullptr };
  vk::raii::CommandBuffer command_buffer_ { nullptr };
};

export class command_pool
{
public:
  command_pool() = default;
  explicit command_pool(vk::raii::CommandPool&& pool)
  : pool_ { std::move(pool) }
  {}

  [[nodiscard]] static auto
  create(const vk::raii::Device& device, std::uint32_t queue_family_index,
    vk::CommandPoolCreateFlags flags = {})
    -> std::expected<command_pool, error_t>
  {
    const vk::CommandPoolCreateInfo command_pool_info {
      .flags = flags,
      .queueFamilyIndex = queue_family_index,
    };
    return UTILS_VK(device.createCommandPool(command_pool_info),
      ^^vk::raii::Device::createCommandPool)
      .transform([](vk::raii::CommandPool&& pool) -> command_pool
        { return command_pool { std::move(pool) }; });
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
  : pool_ { pool }, device_ { device }, queue_ { queue }
  {}

  single_time_submit(const single_time_submit&) = delete;

  auto
  operator=(const single_time_submit&) -> single_time_submit& = delete;

  single_time_submit(single_time_submit&&) noexcept = delete;

  auto
  operator=(single_time_submit&&) noexcept -> single_time_submit& = delete;

  [[nodiscard]] auto
  begin() -> std::expected<void, error_t>
  {
    const vk::CommandBufferAllocateInfo allocate_info {
      .commandPool = *pool_.handle(),
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = 1U,
    };
    return UTILS_VK(device_.allocateCommandBuffers(allocate_info),
      ^^vk::raii::Device::allocateCommandBuffers)
      .and_then(
        [ this ](std::vector<vk::raii::CommandBuffer> buffers)
          -> std::expected<void, error_t>
        {
          command_buffer_ = std::move(buffers.front());
          return UTILS_VK(
            command_buffer_.begin({
              .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
            }),
            ^^vk::raii::CommandBuffer::begin);
        });
  }

  [[nodiscard]] auto
  end_and_submit(upload::wait_idle_t) -> std::expected<void, error_t>
  {
    return UTILS_VK(command_buffer_.end(), ^^vk::raii::CommandBuffer::end)
      .and_then(
        [ this ] -> std::expected<void, error_t>
        {
          const vk::CommandBufferSubmitInfo command_buffer_info {
            .commandBuffer = *command_buffer_,
          };
          const vk::SubmitInfo2 submit_info {
            .commandBufferInfoCount = 1U,
            .pCommandBufferInfos = &command_buffer_info,
          };
          return UTILS_VK(
            queue_.submit2(submit_info, nullptr), ^^vk::raii::Queue::submit2);
        })
      .and_then([ this ] -> std::expected<void, error_t>
        { return UTILS_VK(queue_.waitIdle(), ^^vk::raii::Queue::waitIdle); });
  }

  [[nodiscard]] auto
  end_and_submit(upload::deferred_t) -> std::expected<submission, error_t>
  {
    return UTILS_VK(command_buffer_.end(), ^^vk::raii::CommandBuffer::end)
      .and_then(
        [ this ] -> std::expected<vk::raii::Fence, error_t>
        {
          return UTILS_VK(
            device_.createFence({}), ^^vk::raii::Device::createFence);
        })
      .and_then(
        [ this ](vk::raii::Fence&& fence) -> std::expected<submission, error_t>
        {
          const vk::CommandBufferSubmitInfo command_buffer_info {
            .commandBuffer = *command_buffer_,
          };
          const vk::SubmitInfo2 submit_info {
            .commandBufferInfoCount = 1U,
            .pCommandBufferInfos = &command_buffer_info,
          };
          return UTILS_VK(
            queue_.submit2(submit_info, nullptr), ^^vk::raii::Queue::submit2)
            .transform(
              [ this, fence = std::move(fence) ] mutable -> submission
              {
                return submission { device_, std::move(fence),
                  std::move(command_buffer_) };
              });
        });
  }

  [[nodiscard]] auto
  command_buffer(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.command_buffer_); }

private:
  command_pool& pool_;
  const vk::raii::Device& device_;
  const vk::raii::Queue& queue_;
  vk::raii::CommandBuffer command_buffer_ { nullptr };
};

}; // namespace vkpp
