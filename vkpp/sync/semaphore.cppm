module;

#include "error/vk_error_config.hpp"

export module vkpp.semaphore;

import std;
import vulkan;

import vkpp.error;

namespace vkpp
{

export [[nodiscard]] auto
make_timeline_semaphore(
  const vk::raii::Device& device, std::uint64_t initial_value = 0ULL)
  -> std::expected<vk::raii::Semaphore, error_t>
{
  const vk::StructureChain chain {
    vk::SemaphoreCreateInfo {},
    vk::SemaphoreTypeCreateInfo {
      .semaphoreType = vk::SemaphoreType::eTimeline,
      .initialValue = initial_value,
    },
  };
  return UTILS_VK(device.createSemaphore(chain.get<vk::SemaphoreCreateInfo>()),
    ^^vk::raii::Device::createSemaphore);
}

} // namespace vkpp
