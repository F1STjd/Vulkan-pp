export module vkpp.memory;

import std;
import vulkan;
import vkpp.error;

namespace vkpp
{

export enum class memory_intent {
  // GPU reads/writes, CPU never touches: attachments, device-local vertex/index
  gpu_only,
  // CPU writes, GPU reads: staging buffers, uniform buffers
  cpu_to_gpu,
  // GPU writes, CPU reads back: screenshots, occlusion query results
  gpu_to_cpu,
};

export template<typename Policy>
concept device_allocator =
  requires(Policy policy, const vk::ImageCreateInfo& image_info,
    const vk::BufferCreateInfo& buffer_info, memory_intent intent,
    const typename Policy::image_handle& image,
    const typename Policy::buffer_handle& buffer) {
    {
      policy.create_image(image_info, intent)
    } -> std::same_as<std::expected<typename Policy::image_handle, error_t>>;
    {
      policy.create_buffer(buffer_info, intent)
    } -> std::same_as<std::expected<typename Policy::buffer_handle, error_t>>;

    { image.get() } -> std::same_as<vk::Image>;
    { buffer.get() } -> std::same_as<vk::Buffer>;
    { buffer.mapped() } -> std::same_as<void*>;
  };

}; // namespace vkpp
