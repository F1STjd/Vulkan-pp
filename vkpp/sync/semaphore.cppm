export module vkpp.semaphore;

import std;
import vulkan;

import vkpp.error;
import vkpp.diagnostics;

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
  return map_vk_error(
    device.createSemaphore(chain.get<vk::SemaphoreCreateInfo>()), std::nullopt);
}

} // namespace vkpp
