module;

#include "error/vk_error_config.hpp"

export module vkpp.pipeline.gpl;

import std;
import vulkan;

import vkpp.error;
import vkpp.pipeline;

namespace vkpp
{
using namespace std::string_view_literals;

export enum class graphics_pipeline_library_kind : std::uint8_t {
  vertex_input,
  pre_rasterization,
  fragment,
  fragment_output,
};

export class graphics_pipeline_library
{
public:
  graphics_pipeline_library() = default;

  graphics_pipeline_library(vk::raii::Pipeline&& pipeline)
  : pipeline_ { std::move(pipeline) }
  {}

  [[nodiscard]] auto
  pipeline(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.pipeline_); }

private:
  vk::raii::Pipeline pipeline_ { nullptr };
};

export struct graphics_pipeline_library_runtime_args
{
  // vertex_input
  std::span<const vk::VertexInputBindingDescription> vertex_bindings {};
  std::span<const vk::VertexInputAttributeDescription> vertex_attributes {};
  // pre_rasterization + fragment (shader)
  std::span<const char> spirv {};
  const char* vertex_entry { "vertex_main" };
  const char* fragment_entry { "fragment_main" };
  std::span<const vk::DescriptorSetLayout> set_layouts {};
  std::uint32_t push_constant_size { 0U };
  vk::ShaderStageFlags push_constant_stages {
    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
  };
  vk::PipelineLayout layout {};
  // fragment_output / fragment (rendering + MSAA)
  std::span<const vk::Format> color_formats {};
  vk::Format depth_format { vk::Format::eUndefined };
  vk::SampleCountFlagBits samples { vk::SampleCountFlagBits::e1 };
};

[[nodiscard]] consteval auto
library_flags_for(graphics_pipeline_library_kind kind)
  -> vk::GraphicsPipelineLibraryFlagsEXT
{
  using enum graphics_pipeline_library_kind;
  using enum vk::GraphicsPipelineLibraryFlagBitsEXT;
  switch (kind)
  {
  case vertex_input     : return eVertexInputInterface;
  case pre_rasterization: return ePreRasterizationShaders;
  case fragment         : return eFragmentShader;
  case fragment_output  : return eFragmentOutputInterface;
  }
}

export template<graphics_pipeline_library_kind Kind,
  graphics_pipeline_spec Spec = {}>
  requires(validate(Spec))
auto
make_graphics_pipeline_library(const vk::raii::Device& device,
  const graphics_pipeline_library_runtime_args& runtime_args,
  const vk::raii::PipelineCache& cache = { nullptr })
  -> std::expected<graphics_pipeline_library, error_t>
{
  using enum graphics_pipeline_library_kind;

  if constexpr (Kind == vertex_input)
  {
    const vk::PipelineVertexInputStateCreateInfo vertex_input {
      .vertexBindingDescriptionCount =
        static_cast<std::uint32_t>(runtime_args.vertex_bindings.size()),
      .pVertexBindingDescriptions = runtime_args.vertex_bindings.data(),
      .vertexAttributeDescriptionCount =
        static_cast<std::uint32_t>(runtime_args.vertex_attributes.size()),
      .pVertexAttributeDescriptions = runtime_args.vertex_attributes.data(),
    };
    const vk::PipelineInputAssemblyStateCreateInfo input_assembly {
      .topology = Spec.topology,
    };
    vk::StructureChain chain {
      vk::GraphicsPipelineCreateInfo {
        .flags = vk::PipelineCreateFlagBits::eLibraryKHR,
        .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &input_assembly,
      },
      vk::GraphicsPipelineLibraryCreateInfoEXT {
        .flags = library_flags_for(Kind),
      },
    };
    return UTILS_VK(device.createGraphicsPipeline(
                      cache, chain.get<vk::GraphicsPipelineCreateInfo>()),
      ^^vk::raii::Device::createGraphicsPipeline)
      .transform([](vk::raii::Pipeline&& pipeline) -> graphics_pipeline_library
        { return graphics_pipeline_library { std::move(pipeline) }; });
  }
  else if constexpr (Kind == pre_rasterization)
  {
    if (runtime_args.layout == vk::PipelineLayout {})
    {
      return std::unexpected {
        app_error {
          .kind = app_error_kind::invalid_argument,
          .detail = "pre_rasterization library requires layout"sv,
        },
      };
    }
    if (runtime_args.spirv.empty())
    {
      return std::unexpected {
        app_error {
          .kind = app_error_kind::invalid_argument,
          .detail = "pre_rasterization library spirv empty"sv,
        },
      };
    }
    const vk::ShaderModuleCreateInfo module_info {
      .codeSize = runtime_args.spirv.size_bytes(),
      .pCode = std::start_lifetime_as<std::uint32_t>(runtime_args.spirv.data()),
    };
    return UTILS_VK(device.createShaderModule(module_info),
      ^^vk::raii::Device::createShaderModule)
      .and_then(
        [ & ](vk::raii::ShaderModule&& module)
          -> std::expected<graphics_pipeline_library, error_t>
        {
          const vk::PipelineShaderStageCreateInfo stage {
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = *module,
            .pName = runtime_args.vertex_entry,
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
          const std::array dynamic_states {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor,
            vk::DynamicState::eCullMode,
            vk::DynamicState::eFrontFace,
          };
          vk::PipelineDynamicStateCreateInfo dynamic {
            .dynamicStateCount =
              static_cast<std::uint32_t>(dynamic_states.size()),
            .pDynamicStates = dynamic_states.data(),
          };
          vk::StructureChain chain {
            vk::GraphicsPipelineCreateInfo {
              .flags = vk::PipelineCreateFlagBits::eLibraryKHR,
              .stageCount = 1U,
              .pStages = &stage,
              .pViewportState = &viewport_state,
              .pRasterizationState = &raster,
              .pDynamicState = &dynamic,
              .layout = runtime_args.layout,
            },
            vk::GraphicsPipelineLibraryCreateInfoEXT {
              .flags = library_flags_for(Kind),
            },
          };
          return UTILS_VK(device.createGraphicsPipeline(
                            cache, chain.get<vk::GraphicsPipelineCreateInfo>()),
            ^^vk::raii::Device::createGraphicsPipeline)
            .transform(
              [ module = std::move(module) ](
                vk::raii::Pipeline&& pipeline) mutable
                -> graphics_pipeline_library
              {
                (void)module;
                return graphics_pipeline_library { std::move(pipeline) };
              });
        });
  }
  else if constexpr (Kind == fragment)
  {
    if (runtime_args.layout == vk::PipelineLayout {})
    {
      return std::unexpected {
        app_error {
          .kind = app_error_kind::invalid_argument,
          .detail = "fragment library requires layout"sv,
        },
      };
    }
    if (runtime_args.spirv.empty())
    {
      return std::unexpected {
        app_error {
          .kind = app_error_kind::invalid_argument,
          .detail = "fragment library spirv empty"sv,
        },
      };
    }
    const vk::ShaderModuleCreateInfo module_info {
      .codeSize = runtime_args.spirv.size_bytes(),
      .pCode = std::start_lifetime_as<std::uint32_t>(runtime_args.spirv.data()),
    };
    return UTILS_VK(device.createShaderModule(module_info),
      ^^vk::raii::Device::createShaderModule)
      .and_then(
        [ & ](vk::raii::ShaderModule&& module)
          -> std::expected<graphics_pipeline_library, error_t>
        {
          const vk::PipelineShaderStageCreateInfo stage {
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = *module,
            .pName = runtime_args.fragment_entry,
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
          vk::StructureChain chain {
            vk::GraphicsPipelineCreateInfo {
              .flags = vk::PipelineCreateFlagBits::eLibraryKHR,
              .stageCount = 1U,
              .pStages = &stage,
              .pMultisampleState = &multisample,
              .pDepthStencilState = &depth,
              .layout = runtime_args.layout,
            },
            vk::GraphicsPipelineLibraryCreateInfoEXT {
              .flags = library_flags_for(Kind),
            },
          };
          return UTILS_VK(device.createGraphicsPipeline(
                            cache, chain.get<vk::GraphicsPipelineCreateInfo>()),
            ^^vk::raii::Device::createGraphicsPipeline)
            .transform(
              [ module = std::move(module) ](
                vk::raii::Pipeline&& pipeline) mutable
                -> graphics_pipeline_library
              {
                (void)module;
                return graphics_pipeline_library { std::move(pipeline) };
              });
        });
  }
  else if constexpr (Kind == fragment_output)
  {
    if (runtime_args.color_formats.empty())
    {
      return std::unexpected {
        app_error {
          .kind = app_error_kind::invalid_argument,
          .detail = "fragment_output library missing color_formats"sv,
        },
      };
    }
    const vk::PipelineMultisampleStateCreateInfo multisample {
      .rasterizationSamples = runtime_args.samples,
      .sampleShadingEnable = vk::Bool32 { Spec.sample_shading },
      .minSampleShading = Spec.min_sample_shading,
    };
    const vk::PipelineColorBlendAttachmentState blend_attachment {
      .blendEnable = vk::Bool32 { Spec.blend_enable },
      .srcColorBlendFactor = Spec.src_color_blend_factor,
      .dstColorBlendFactor = Spec.dst_color_blend_factor,
      .colorBlendOp = Spec.color_blend_op,
      .srcAlphaBlendFactor = Spec.src_alpha_blend_factor,
      .dstAlphaBlendFactor = Spec.dst_alpha_blend_factor,
      .alphaBlendOp = Spec.alpha_blend_op,
      .colorWriteMask = vk::ColorComponentFlagBits::eR |
        vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB |
        vk::ColorComponentFlagBits::eA,
    };
    const vk::PipelineColorBlendStateCreateInfo blend {
      .logicOpEnable = vk::False,
      .attachmentCount = 1U,
      .pAttachments = &blend_attachment,
    };
    vk::StructureChain chain {
      vk::GraphicsPipelineCreateInfo {
        .flags = vk::PipelineCreateFlagBits::eLibraryKHR,
        .pMultisampleState = &multisample,
        .pColorBlendState = &blend,
      },
      vk::GraphicsPipelineLibraryCreateInfoEXT {
        .flags = library_flags_for(Kind),
      },
      vk::PipelineRenderingCreateInfo {
        .colorAttachmentCount =
          static_cast<std::uint32_t>(runtime_args.color_formats.size()),
        .pColorAttachmentFormats = runtime_args.color_formats.data(),
        .depthAttachmentFormat = runtime_args.depth_format,
      },
    };
    return UTILS_VK(device.createGraphicsPipeline(
                      cache, chain.get<vk::GraphicsPipelineCreateInfo>()),
      ^^vk::raii::Device::createGraphicsPipeline)
      .transform(
        [](vk::raii::Pipeline&& pipeline) mutable -> graphics_pipeline_library
        { return graphics_pipeline_library { std::move(pipeline) }; });
  }
  else
  {
    static_assert(Kind == vertex_input || Kind == pre_rasterization ||
        Kind == fragment || Kind == fragment_output,
      "unhandled graphics_pipeline_library_kind");
    return std::unexpected {
      app_error {
        .kind = app_error_kind::invalid_argument,
        .detail = "unhandled graphics_pipeline_library_kind"sv,
      },
    };
  }
}

export auto
link_graphics_pipeline(const vk::raii::Device& device,
  std::span<const vk::Pipeline> libraries, bool optimize,
  vk::PipelineLayout layout, const vk::raii::PipelineCache& cache = { nullptr })
  -> std::expected<vk::raii::Pipeline, error_t>
{
  if (libraries.size() != 4UZ)
  {
    return std::unexpected {
      app_error {
        .kind = app_error_kind::invalid_argument,
        .detail = "link_graphics_pipeline expects 4 libraries"sv,
      },
    };
  }
  const vk::PipelineLibraryCreateInfoKHR library_info {
    .libraryCount = 4U,
    .pLibraries = libraries.data(),
  };
  vk::PipelineCreateFlags flags {};
  if (optimize)
  {
    flags |= vk::PipelineCreateFlagBits::eLinkTimeOptimizationEXT;
  }
  const vk::GraphicsPipelineCreateInfo create_info {
    .pNext = &library_info,
    .flags = flags,
    .layout = layout,
  };

  return UTILS_VK(device.createGraphicsPipeline(cache, create_info),
    ^^vk::raii::Device::createGraphicsPipeline)
    .transform([](vk::raii::Pipeline&& pipeline) -> vk::raii::Pipeline
      { return std::move(pipeline); });
}

} // namespace vkpp
