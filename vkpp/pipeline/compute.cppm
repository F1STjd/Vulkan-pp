module;

#include "error/vk_error_config.hpp"

export module vkpp.pipeline.compute;

import std;
import vulkan;

import vkpp.error;

namespace vkpp
{
using namespace std::string_view_literals;

export struct compute_pipeline_runtime_args
{
  const vk::raii::DescriptorSetLayout& set_layout;
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

export [[nodiscard]] auto
make_compute_pipeline(const vk::raii::Device& device,
  const compute_pipeline_runtime_args& runtime_args,
  const compute_shader& shader) -> std::expected<compute_pipeline, error_t>
{
  if (runtime_args.set_layout == nullptr || shader.spirv.empty())
  {
    return std::unexpected {
      app_error {
        .kind = app_error_kind::invalid_argument,
        .detail = "compute_pipeline_runtime_args missing set_layout/spirv"sv,
      },
    };
  }

  const vk::ShaderModuleCreateInfo module_info {
    .codeSize = shader.spirv.size_bytes(),
    .pCode = std::start_lifetime_as<std::uint32_t>(shader.spirv.data()),
  };

  return UTILS_VK(device.createShaderModule(module_info),
    ^^vk::raii::Device::createShaderModule)
    .and_then(
      [ & ](vk::raii::ShaderModule&& module)
        -> std::expected<compute_pipeline, error_t>
      {
        const std::array set_layouts { *runtime_args.set_layout };
        const vk::PushConstantRange push_range {
          .stageFlags = vk::ShaderStageFlagBits::eCompute,
          .offset = 0U,
          .size = runtime_args.push_constant_size,
        };
        const vk::PipelineLayoutCreateInfo layout_info {
          .setLayoutCount = 1U,
          .pSetLayouts = set_layouts.data(),
          .pushConstantRangeCount =
            runtime_args.push_constant_size > 0U ? 1U : 0U,
          .pPushConstantRanges = &push_range,
        };

        return UTILS_VK(device.createPipelineLayout(layout_info),
          ^^vk::raii::Device::createPipelineLayout)
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

              return UTILS_VK(
                device.createComputePipeline(nullptr, compute_pipeline_info),
                ^^vk::raii::Device::createComputePipeline)
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

} // namespace vkpp
