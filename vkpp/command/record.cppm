export module vkpp.command.record;

import std;
import vulkan;

import vkpp.capabilities;
import vkpp.descriptor.indexing;

namespace vkpp
{

export inline void
bind_compute(vk::raii::CommandBuffer& command_buffer, vk::Pipeline pipeline,
  vk::PipelineLayout layout, std::span<const vk::DescriptorSet> sets,
  std::uint32_t first_set = 0U)
{
  command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
  if (!sets.empty())
  {
    command_buffer.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute, layout, first_set, sets, {});
  }
}

export inline void
dispatch(vk::raii::CommandBuffer& command_buffer, std::uint32_t group_count_x,
  std::uint32_t group_count_y = 1U, std::uint32_t group_count_z = 1U)
{ command_buffer.dispatch(group_count_x, group_count_y, group_count_z); }

export inline void
bind_graphics(vk::raii::CommandBuffer& command_buffer, vk::Pipeline pipeline,
  vk::PipelineLayout layout, std::span<const vk::DescriptorSet> sets,
  std::uint32_t first_set = 0U,
  std::span<const std::uint32_t> dynamic_offsets = {})
{
  command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
  if (!sets.empty())
  {
    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout,
      first_set, sets, dynamic_offsets);
  }
}

export template<typename T>
void
push_constants(vk::raii::CommandBuffer& command_buffer,
  vk::PipelineLayout layout, vk::ShaderStageFlags stages, const T& value)
{
  static_assert(std::is_trivially_copyable_v<T>);
  command_buffer.pushConstants(
    layout, stages, 0U, sizeof(T), static_cast<const void*>(&value));
}

export template<typename T>
void
push_compute_constants(vk::raii::CommandBuffer& command_buffer,
  vk::PipelineLayout layout, const T& value)
{
  push_constants(
    command_buffer, layout, vk::ShaderStageFlagBits::eCompute, value);
}

export inline void
bind_shaders(vk::raii::CommandBuffer& command_buffer,
  std::span<const vk::ShaderStageFlagBits> stages,
  std::span<const vk::ShaderEXT> shaders)
{ command_buffer.bindShadersEXT(stages, shaders); }

export template<descriptor_table_backend Backend>
  requires(Backend == descriptor_table_backend::classic)
void
bind_graphics_descriptors(vk::raii::CommandBuffer& command_buffer,
  vk::Pipeline pipeline, vk::PipelineLayout layout,
  std::span<const vk::DescriptorSet> sets,
  std::span<const std::uint32_t> dynamic_offsets = {})
{ bind_graphics(command_buffer, pipeline, layout, sets, 0U, dynamic_offsets); }

export template<descriptor_table_backend Backend>
  requires(Backend == descriptor_table_backend::heap)
void
bind_graphics_descriptors(vk::raii::CommandBuffer& command_buffer,
  vk::Pipeline pipeline, [[maybe_unused]] vk::PipelineLayout layout,
  [[maybe_unused]] std::span<const vk::DescriptorSet> sets,
  [[maybe_unused]] std::span<const std::uint32_t> dynamic_offsets,
  const descriptor_heap_arena& arena)
{
  command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
  arena.bind(command_buffer);
}

export template<descriptor_table_backend Backend>
  requires(Backend == descriptor_table_backend::classic)
void
bind_compute_descriptors(vk::raii::CommandBuffer& command_buffer,
  vk::Pipeline pipeline, vk::PipelineLayout layout,
  std::span<const vk::DescriptorSet> sets)
{ bind_compute(command_buffer, pipeline, layout, sets); }

export template<descriptor_table_backend Backend>
  requires(Backend == descriptor_table_backend::heap)
void
bind_compute_descriptors(vk::raii::CommandBuffer& command_buffer,
  vk::Pipeline pipeline, [[maybe_unused]] vk::PipelineLayout layout,
  [[maybe_unused]] std::span<const vk::DescriptorSet> sets,
  const descriptor_heap_arena& arena)
{
  command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
  arena.bind(command_buffer);
}

export template<typename T>
void
push_data(
  vk::raii::CommandBuffer& command_buffer, std::uint32_t offset, const T& value)
{
  static_assert(std::is_trivially_copyable_v<T>);
  const vk::PushDataInfoEXT info {
    .offset = offset,
    .data =
      vk::HostAddressRangeConstEXT {
        .address = &value,
        .size = sizeof(T),
      },
  };
  command_buffer.pushDataEXT(info);
}

} // namespace vkpp
