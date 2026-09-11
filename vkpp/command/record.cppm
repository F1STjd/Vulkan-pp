module;

export module vkpp.command.record;

import std;
import vulkan;

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

} // namespace vkpp
