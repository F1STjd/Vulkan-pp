export module vkpp.descriptor.cis;

import std;
import vulkan;
import vkpp.error;
import vkpp.sampler;
import vkpp.texture;
import vkpp.descriptor.indexing;

namespace vkpp
{

export [[nodiscard]] auto
register_combined_image_sampler(classic_bindless_table& table,
  const vk::raii::Device& device, vk::Sampler sampler, const texture<>& image)
  -> std::expected<std::uint32_t, error_t>
{
  return table.register_combined_image_sampler(device, sampler, *image.view());
}

export template<class Layout, class Tag>
[[nodiscard]] auto
register_combined_image_sampler(heap_bindless_table& table,
  const vk::raii::Device& device, const vk::raii::PhysicalDevice& physical,
  const sampler_create_info& sampler_info, const texture<>& image)
  -> std::expected<std::uint32_t, error_t>
{
  return table.template register_combined_image_sampler<Layout, Tag>(
    device, to_vk(physical, sampler_info), image.view_create_info());
}

export template<class Layout, class Tag>
[[nodiscard]] auto
write_cis(descriptor_heap_arena& arena, const vk::raii::Device& device,
  const vk::raii::PhysicalDevice& physical,
  const sampler_create_info& sampler_info, const texture<>& image,
  vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal)
  -> std::expected<void, error_t>
{
  return arena.template write_cis<Layout, Tag>(
    device, to_vk(physical, sampler_info), image.view_create_info(), layout);
}

} // namespace vkpp
