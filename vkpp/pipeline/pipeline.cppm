module;

#include "error/vk_error_config.hpp"

export module vkpp.pipeline;

import std;
import vulkan;

import vkpp.error;

namespace vkpp
{

using namespace std::string_view_literals;

export struct graphics_pipeline_spec
{
  vk::PrimitiveTopology topology { vk::PrimitiveTopology::eTriangleList };
  vk::PolygonMode polygon_mode { vk::PolygonMode::eFill };
  vk::CullModeFlagBits cull_mode { vk::CullModeFlagBits::eBack };
  vk::FrontFace front_face { vk::FrontFace::eCounterClockwise };
  vk::CompareOp depth_compare { vk::CompareOp::eLess };
  bool depth_test { true };
  bool depth_write { true };
  bool blend_enable { false };
  bool sample_shading { false };
  float min_sample_shading { 1.0F };
};

export consteval auto
validate(const graphics_pipeline_spec& spec) -> bool
{
  if (spec.min_sample_shading < 0.0F || spec.min_sample_shading > 1.0F)
  {
    return false;
  }
  return true;
}

export struct graphics_pipeline_runtime_args
{
  std::span<const vk::Format> color_formats {};
  vk::Format depth_format { vk::Format::eUndefined };
  vk::SampleCountFlagBits samples { vk::SampleCountFlagBits::e1 };
  std::span<const vk::DescriptorSetLayout> set_layouts;
  std::span<const vk::VertexInputBindingDescription> vertex_bindings {};
  std::span<const vk::VertexInputAttributeDescription> vertex_attributes {};
  std::uint32_t push_constant_size { 0U };
  vk::ShaderStageFlags push_constant_stages {
    vk::ShaderStageFlagBits::eVertex
  };
};

export class graphics_pipeline
{
public:
  graphics_pipeline() = default;
  graphics_pipeline(
    vk::raii::PipelineLayout&& layout, vk::raii::Pipeline&& pipeline)
  : layout_ { std::move(layout) }, pipeline_ { std::move(pipeline) } {};

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

export struct graphics_pipeline_shaders
{
  std::span<const char> spirv {};
  const char* vertex_entry { "vertex_main" };
  const char* fragment_entry { "fragment_main" };
};

export template<graphics_pipeline_spec Spec = graphics_pipeline_spec {}>
  requires(validate(Spec))
auto make_graphics_pipeline(const vk::raii::Device& device,
  const graphics_pipeline_runtime_args& runtime_args,
  const graphics_pipeline_shaders& shaders)
  -> std::expected<graphics_pipeline, error_t>
{
  if (runtime_args.set_layouts.empty() || runtime_args.color_formats.empty())
  {
    return std::unexpected {
      app_error {
        .kind = app_error_kind::invalid_argument,
        .detail =
          "graphics_pipeline_runtime_args missing set_layouts/color_formats"sv,
      },
    };
  }

  const vk::ShaderModuleCreateInfo module_info {
    .codeSize = shaders.spirv.size_bytes(),
    .pCode = std::start_lifetime_as<std::uint32_t>(shaders.spirv.data()),
  };

  return UTILS_VK(device.createShaderModule(module_info),
    ^^vk::raii::Device::createShaderModule)
    .and_then(
      [ & ](vk::raii::ShaderModule&& module)
        -> std::expected<graphics_pipeline, error_t>
      {
        const vk::PushConstantRange push_range {
          .stageFlags = runtime_args.push_constant_stages,
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

        return UTILS_VK(device.createPipelineLayout(layout_info),
          ^^vk::raii::Device::createPipelineLayout)
          .and_then(
            [ &, module = std::move(module) ](vk::raii::PipelineLayout&& layout)
              -> std::expected<graphics_pipeline, error_t>
            {
              // Removed static constexpr from everything, because GCC's ICE was
              // triggered
              const std::array stages {
                vk::PipelineShaderStageCreateInfo {
                  .stage = vk::ShaderStageFlagBits::eVertex,
                  .module = *module,
                  .pName = shaders.vertex_entry,
                },
                vk::PipelineShaderStageCreateInfo {
                  .stage = vk::ShaderStageFlagBits::eFragment,
                  .module = *module,
                  .pName = shaders.fragment_entry,
                },
              };

              const vk::PipelineVertexInputStateCreateInfo vertex_input {
                .vertexBindingDescriptionCount = static_cast<std::uint32_t>(
                  runtime_args.vertex_bindings.size()),
                .pVertexBindingDescriptions =
                  runtime_args.vertex_bindings.data(),
                .vertexAttributeDescriptionCount = static_cast<std::uint32_t>(
                  runtime_args.vertex_attributes.size()),
                .pVertexAttributeDescriptions =
                  runtime_args.vertex_attributes.data(),
              };
              const vk::PipelineInputAssemblyStateCreateInfo input_assembly {
                .topology = Spec.topology
              };
              const vk::PipelineViewportStateCreateInfo viewport_state {
                .viewportCount = 1U,
                .scissorCount = 1U,
              };
              const vk::PipelineRasterizationStateCreateInfo raster {
                .depthClampEnable = vk::False,
                .rasterizerDiscardEnable = vk::False,
                .polygonMode = Spec.polygon_mode,
                .cullMode = Spec.cull_mode,
                .frontFace = Spec.front_face,
                .depthBiasEnable = vk::False,
                .lineWidth = 1.0F,
              };
              const vk::PipelineMultisampleStateCreateInfo multisample {
                .rasterizationSamples = runtime_args.samples,
                .sampleShadingEnable = vk::Bool32 { Spec.sample_shading },
                .minSampleShading = Spec.min_sample_shading,
              };
              const vk::PipelineDepthStencilStateCreateInfo depth {
                .depthTestEnable = vk::Bool32 { Spec.depth_test },
                .depthWriteEnable = vk::Bool32 { Spec.depth_write },
                .depthCompareOp = Spec.depth_compare,
                .depthBoundsTestEnable = vk::False,
                .stencilTestEnable = vk::False,
              };
              const vk::PipelineColorBlendAttachmentState blend_attachment {
                .blendEnable = vk::Bool32 { Spec.blend_enable },
                .colorWriteMask = vk::ColorComponentFlagBits::eR |
                  vk::ColorComponentFlagBits::eG |
                  vk::ColorComponentFlagBits::eB |
                  vk::ColorComponentFlagBits::eA,
              };
              const vk::PipelineColorBlendStateCreateInfo blend {
                .logicOpEnable = vk::False,
                .attachmentCount = 1U,
                .pAttachments = &blend_attachment,
              };
              const std::array dynamic_states {
                vk::DynamicState::eViewport,
                vk::DynamicState::eScissor,
                vk::DynamicState::eCullMode,
                vk::DynamicState::eFrontFace,
              };
              const vk::PipelineDynamicStateCreateInfo dynamic {
                .dynamicStateCount =
                  static_cast<std::uint32_t>(dynamic_states.size()),
                .pDynamicStates = dynamic_states.data(),
              };

              vk::StructureChain chain {
                vk::GraphicsPipelineCreateInfo {
                  .stageCount = static_cast<std::uint32_t>(stages.size()),
                  .pStages = stages.data(),
                  .pVertexInputState = &vertex_input,
                  .pInputAssemblyState = &input_assembly,
                  .pViewportState = &viewport_state,
                  .pRasterizationState = &raster,
                  .pMultisampleState = &multisample,
                  .pDepthStencilState = &depth,
                  .pColorBlendState = &blend,
                  .pDynamicState = &dynamic,
                  .layout = *layout,
                  .renderPass = nullptr,
                },
                vk::PipelineRenderingCreateInfo {
                  .colorAttachmentCount = static_cast<std::uint32_t>(
                    runtime_args.color_formats.size()),
                  .pColorAttachmentFormats = runtime_args.color_formats.data(),
                  .depthAttachmentFormat = runtime_args.depth_format,
                },
              };

              return UTILS_VK(device.createGraphicsPipeline(nullptr,
                                chain.get<vk::GraphicsPipelineCreateInfo>()),
                ^^vk::raii::Device::createGraphicsPipeline)
                .transform(
                  [ &, layout = std::move(layout) ](
                    vk::raii::Pipeline&& pipeline) mutable
                  {
                    return graphics_pipeline {
                      std::move(layout),
                      std::move(pipeline),
                    };
                  });
            });
      });
}

} // namespace vkpp
