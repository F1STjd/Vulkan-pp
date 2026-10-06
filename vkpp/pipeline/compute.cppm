export module vkpp.pipeline.compute;

import std;
import vulkan;

import vkpp.error;
import vkpp.diagnostics;
import vkpp.capabilities;
import vkpp.descriptor.indexing;

namespace vkpp
{
using namespace std::string_view_literals;

export struct compute_pipeline_runtime_args
{
  std::span<const vk::DescriptorSetLayout> set_layouts {};
  std::uint32_t push_constant_size { 0U };
};

export struct compute_shader
{
  std::span<const char> spirv {};
  const char* entry { "compute_main" };
};

export class compute_pipeline
{
public:
  compute_pipeline() = default;

  compute_pipeline(
    vk::raii::PipelineLayout&& layout, vk::raii::Pipeline&& pipeline)
  : layout_ { std::move(layout) }, pipeline_ { std::move(pipeline) }
  {}

  [[nodiscard]] auto
  pipeline(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.pipeline_); }

  [[nodiscard]] auto
  layout(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.layout_); }

private:
  vk::raii::PipelineLayout layout_ { nullptr };
  vk::raii::Pipeline pipeline_ { nullptr };
};

export template<
  descriptor_table_backend Backend = descriptor_table_backend::classic>
  requires(Backend == descriptor_table_backend::classic)
[[nodiscard]] auto
make_compute_pipeline(const vk::raii::Device& device,
  const compute_pipeline_runtime_args& runtime_args,
  const compute_shader& shader,
  const vk::raii::PipelineCache& cache = { nullptr },
  std::optional<diagnostic_buffer&> diagnostics = {})
  -> std::expected<compute_pipeline, error_t>
{
  if (runtime_args.set_layouts.empty() || shader.spirv.empty())
  {
    return std::unexpected {
      make_app_error(app_error_code::missing_required_argument),
    };
  }

  const vk::ShaderModuleCreateInfo module_info {
    .codeSize = shader.spirv.size_bytes(),
    .pCode = std::start_lifetime_as<std::uint32_t>(shader.spirv.data()),
  };

  return map_vk_error(device.createShaderModule(module_info), diagnostics)
    .and_then(
      [ & ](vk::raii::ShaderModule&& module)
        -> std::expected<compute_pipeline, error_t>
      {
        const vk::PushConstantRange push_range {
          .stageFlags = vk::ShaderStageFlagBits::eCompute,
          .offset = 0U,
          .size = runtime_args.push_constant_size,
        };
        const vk::PipelineLayoutCreateInfo layout_info {
          .setLayoutCount =
            static_cast<std::uint32_t>(runtime_args.set_layouts.size()),
          .pSetLayouts = runtime_args.set_layouts.data(),
          .pushConstantRangeCount =
            runtime_args.push_constant_size > 0U ? 1U : 0U,
          .pPushConstantRanges = &push_range,
        };

        return map_vk_error(
          device.createPipelineLayout(layout_info), diagnostics)
          .and_then(
            [ &, module = std::move(module) ](vk::raii::PipelineLayout&& layout)
              -> std::expected<compute_pipeline, error_t>
            {
              const vk::ComputePipelineCreateInfo compute_pipeline_info {
                .stage = {
                  .stage = vk::ShaderStageFlagBits::eCompute,
                  .module = *module,
                  .pName = shader.entry,
                },
                .layout = *layout,
              };

              return map_vk_error(
                device.createComputePipeline(cache, compute_pipeline_info),
                diagnostics)
                .transform(
                  [ &layout ](
                    vk::raii::Pipeline&& pipeline) mutable -> compute_pipeline
                  {
                    return compute_pipeline {
                      std::move(layout),
                      std::move(pipeline),
                    };
                  });
            });
      });
}

export template<descriptor_table_backend Backend>
  requires(Backend == descriptor_table_backend::heap)
[[nodiscard]] auto
make_compute_pipeline(const vk::raii::Device& device,
  [[maybe_unused]] const compute_pipeline_runtime_args& runtime_args,
  const compute_shader& shader, const descriptor_heap_arena& arena,
  const vk::raii::PipelineCache& cache = { nullptr },
  std::optional<diagnostic_buffer&> diagnostics = {})
  -> std::expected<compute_pipeline, error_t>
{
  if (shader.spirv.empty())
  {
    return std::unexpected {
      make_app_error(app_error_code::missing_required_argument),
    };
  }

  const vk::ShaderModuleCreateInfo module_info {
    .codeSize = shader.spirv.size_bytes(),
    .pCode = std::start_lifetime_as<std::uint32_t>(shader.spirv.data()),
  };

  return map_vk_error(device.createShaderModule(module_info), diagnostics)
    .and_then(
      [ & ](vk::raii::ShaderModule&& module)
        -> std::expected<compute_pipeline, error_t>
      {
        auto mapping = arena.shader_and_mapping_info();

        vk::ComputePipelineCreateInfo compute_pipeline_info {
          .stage = {
            .pNext = &mapping,
            .stage = vk::ShaderStageFlagBits::eCompute,
            .module = *module,
            .pName = shader.entry,
          },
          .layout = {},
        };

        vk::PipelineCreateFlags2CreateInfoKHR flags2 {
          .flags = vk::PipelineCreateFlagBits2::eDescriptorHeapEXT,
        };
        flags2.pNext = compute_pipeline_info.pNext;
        compute_pipeline_info.pNext = &flags2;
        compute_pipeline_info.flags = {};

        return map_vk_error(
          device.createComputePipeline(cache, compute_pipeline_info),
          diagnostics)
          .transform(
            [](vk::raii::Pipeline&& pipeline) mutable -> compute_pipeline
            {
              return compute_pipeline {
                { nullptr },
                std::move(pipeline),
              };
            });
      });
}

} // namespace vkpp
