export module vkpp.descriptor.cis;

import std;
import vulkan;
import vkpp.error;
import vkpp.sampler;
import vkpp.texture;
import vkpp.descriptor.indexing;
import vkpp.descriptor;
import vkpp.descriptor.layout_decl;

namespace vkpp
{

export struct descriptor_write_context
{
  const vk::raii::Device&         device;
  const vk::raii::PhysicalDevice& physical;
  sampler_cache&                  samplers;
};

export
template<declared_resource_layout Layout, class Tag>
[[nodiscard]]
auto
register_combined_image_sampler(
  classic_bindless_table& table,
  const vk::raii::Device& device,
  vk::Sampler             sampler,
  const texture<>&        image) -> std::expected<std::uint32_t, error_t>
{
  static_assert(
    Layout::template declaration_of<Tag>::kind ==
      resource_declaration_kind::cis_table,
    "register_combined_image_sampler: Tag must name a cis_table_resource");
  return table.register_combined_image_sampler(device, sampler, *image.view());
}

export
template<class Layout, class Tag>
[[nodiscard]]
auto
register_combined_image_sampler(
  heap_bindless_table&            table,
  const vk::raii::Device&         device,
  const vk::raii::PhysicalDevice& physical,
  const sampler_create_info&      sampler_info,
  const texture<>& image) -> std::expected<std::uint32_t, error_t>
{
  return table.template register_combined_image_sampler<Layout, Tag>(
    device, to_vk(physical, sampler_info), image.view_create_info());
}

export
template<class Layout, class Tag>
[[nodiscard]]
auto
write_cis(
  descriptor_heap_arena&          arena,
  const vk::raii::Device&         device,
  const vk::raii::PhysicalDevice& physical,
  const sampler_create_info&      sampler_info,
  const texture<>&                image,
  vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal)
  -> std::expected<void, error_t>
{
  return arena.template write_cis<Layout, Tag>(
    device, to_vk(physical, sampler_info), image.view_create_info(), layout);
}

export
template<class Tag, declared_resource_layout Layout, resource_scope Scope>
[[nodiscard]]
auto
write_descriptor(
  const descriptor_scope_arena<Layout, Scope>& arena,
  const descriptor_write_context&              context,
  const sampler_create_info&                   sampler_info,
  const texture<>&                             image,
  std::uint32_t set_index = 0U) -> std::expected<void, error_t>
{
  return context.samplers.get_or_create(sampler_info)
    .transform([ & ](vk::Sampler sampler) -> void {
      write_descriptor<Tag>(
        arena,
        context.device,
        sampler,
        *image.view(),
        vk::ImageLayout::eShaderReadOnlyOptimal,
        set_index);
    });
}

export
template<declared_resource_layout Layout, class Tag>
[[nodiscard]]
auto
register_combined_image_sampler(
  classic_bindless_table&         table,
  const descriptor_write_context& context,
  const sampler_create_info&      sampler_info,
  const texture<>& image) -> std::expected<std::uint32_t, error_t>
{
  return context.samplers.get_or_create(sampler_info)
    .and_then(
      [ & ](vk::Sampler sampler) -> std::expected<std::uint32_t, error_t> {
        return register_combined_image_sampler<Layout, Tag>(
          table, context.device, sampler, image);
      });
}

export
template<class Tag, class Layout>
[[nodiscard]]
auto
write_descriptor(
  descriptor_heap_arena&          arena,
  const descriptor_write_context& context,
  const sampler_create_info&      sampler_info,
  const texture<>&                image) -> std::expected<void, error_t>
{
  return arena.template write_cis<Layout, Tag>(
    context.device,
    to_vk(context.physical, sampler_info),
    image.view_create_info());
}

export
template<class Layout, class Tag>
[[nodiscard]]
auto
register_combined_image_sampler(
  classic_bindless_table&         table,
  const descriptor_write_context& context,
  const sampler_create_info&      sampler_info,
  const texture<>& image) -> std::expected<std::uint32_t, error_t>
{
  return register_combined_image_sampler<Layout, Tag>(
    table, context.device, context.physical, sampler_info, image);
}

} // namespace vkpp
