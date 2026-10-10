export module vkpp.pipeline.gpl;

import std;
import vulkan;

import vkpp.error;
import vkpp.diagnostics;
import vkpp.pipeline;
import vkpp.pipeline.persistence;
import vkpp.pipeline.binary;
import vkpp.capabilities;
import vkpp.descriptor.indexing;
import vkpp.descriptor.spirv_binding_check;

namespace vkpp
{

using namespace std::string_view_literals;

export enum class graphics_pipeline_library_kind : std::uint8_t
{
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
  : pipeline_ {std::move(pipeline)}
  {}

  [[nodiscard]]
  auto
  pipeline(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.pipeline_); }

private:
  vk::raii::Pipeline pipeline_ {nullptr};
};

export struct graphics_pipeline_library_runtime_args
{
  // vertex_input
  std::span<const vk::VertexInputBindingDescription>   vertex_bindings {};
  std::span<const vk::VertexInputAttributeDescription> vertex_attributes {};
  // pre_rasterization + fragment (shader)
  std::span<const char>                                spirv {};
  const char*                              vertex_entry {"vertex_main"};
  const char*                              fragment_entry {"fragment_main"};
  std::span<const vk::DescriptorSetLayout> set_layouts {};
  std::uint32_t                            push_constant_size {0U};
  vk::ShaderStageFlags                     push_constant_stages {
    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
  };
  vk::PipelineLayout          layout {};
  // fragment_output / fragment (rendering + MSAA)
  std::span<const vk::Format> color_formats {};
  vk::Format                  depth_format {vk::Format::eUndefined};
  vk::SampleCountFlagBits     samples {vk::SampleCountFlagBits::e1};
};

[[nodiscard]]
consteval
auto
library_flags_for(graphics_pipeline_library_kind kind)
  -> vk::GraphicsPipelineLibraryFlagsEXT
{
  using enum graphics_pipeline_library_kind;
  using enum vk::GraphicsPipelineLibraryFlagBitsEXT;
  switch (kind)
  {
  case vertex_input      : return eVertexInputInterface;
  case pre_rasterization : return ePreRasterizationShaders;
  case fragment          : return eFragmentShader;
  case fragment_output   : return eFragmentOutputInterface;
  }
}

template<descriptor_table_backend Backend>
[[nodiscard]]
auto
create_graphics_pipeline_library_application_miss(
  const vk::raii::Device&           device,
  vk::GraphicsPipelineCreateInfo&   graphics_create_info,
  const vk::PipelineBinaryKeyKHR&   pipeline_key,
  pipeline_persistence&             persistence,
  std::optional<diagnostic_buffer&> diagnostics)
  -> std::expected<graphics_pipeline_library, error_t>
{
  vk::PipelineCreateFlags2KHR bits =
    vk::PipelineCreateFlagBits2::eLibraryKHR |
    vk::PipelineCreateFlagBits2::eCaptureDataKHR;
  if constexpr (Backend == descriptor_table_backend::heap)
  {
    bits |= vk::PipelineCreateFlagBits2::eDescriptorHeapEXT;
  }
  vk::PipelineCreateFlags2CreateInfoKHR flags2 {
    .pNext = graphics_create_info.pNext,
    .flags = bits,
  };
  graphics_create_info.pNext = &flags2;
  graphics_create_info.flags = {};
  if constexpr (Backend == descriptor_table_backend::heap)
  {
    graphics_create_info.layout = vk::PipelineLayout {};
  }

  return map_vk_error(
    device.createGraphicsPipeline(
      persistence.cache_for_create(), graphics_create_info),
    diagnostics)
    .and_then(
      [ & ](vk::raii::Pipeline&& pipeline)
        -> std::expected<graphics_pipeline_library, error_t> {
        return pipeline_binaries::capture(device, *pipeline)
          .and_then(
            [ & ](pipeline_binaries&& binaries)
              -> std::expected<graphics_pipeline_library, error_t> {
              return binaries.extract_blobs(device).and_then(
                [ & ](std::vector<pipeline_binary_blob>&& blobs)
                  -> std::expected<graphics_pipeline_library, error_t> {
                  return persistence.store()
                    ->upsert_capture(pipeline_key, blobs, diagnostics)
                    .transform([ & ] -> graphics_pipeline_library {
                      persistence.mark_store_dirty();
                      if (diagnostics)
                      {
                        diagnostics->report(
                          diagnostic_severity::info,
                          std::nullopt,
                          std::source_location::current(),
                          "pipeline_persistence: captured {} binaries",
                          blobs.size());
                      }
                      return graphics_pipeline_library {
                        std::move(pipeline),
                      };
                    });
                });
            });
      });
}

template<descriptor_table_backend Backend>
[[nodiscard]]
auto
create_graphics_pipeline_library_internal(
  const vk::raii::Device&         device,
  vk::GraphicsPipelineCreateInfo& graphics_create_info,
  [[maybe_unused]] std::optional<const descriptor_heap_arena&> arena       = {},
  std::optional<diagnostic_buffer&>                            diagnostics = {})
  -> std::expected<graphics_pipeline_library, error_t>
{
  vk::PipelineCreateFlags2CreateInfoKHR flags2 {};
  if constexpr (Backend == descriptor_table_backend::heap)
  {
    flags2.flags =
      vk::PipelineCreateFlagBits2::eDescriptorHeapEXT |
      vk::PipelineCreateFlagBits2::eLibraryKHR;
    flags2.pNext                = graphics_create_info.pNext;
    graphics_create_info.pNext  = &flags2;
    graphics_create_info.flags  = {};
    graphics_create_info.layout = vk::PipelineLayout {};
  }
  auto internal =
    try_internal_pipeline_binaries(device, graphics_create_info, diagnostics);
  if (!internal) { return std::unexpected {internal.error()}; }
  std::optional<pipeline_binaries> held {};
  vk::PipelineBinaryInfoKHR        binary_local {};
  if (*internal)
  {
    held                       = std::move(**internal);
    binary_local               = held->info();
    binary_local.pNext         = graphics_create_info.pNext;
    graphics_create_info.pNext = &binary_local;
  }
  const vk::raii::PipelineCache null_cache {nullptr};
  return map_vk_error(
    device.createGraphicsPipeline(null_cache, graphics_create_info), diagnostics)
    .transform(
      [ _ = std::move(held) ](
        vk::raii::Pipeline&& pipeline) mutable -> graphics_pipeline_library {
        return graphics_pipeline_library {std::move(pipeline)};
      });
}

template<descriptor_table_backend Backend>
[[nodiscard]]
auto
create_graphics_pipeline_library_with_persistence(
  const vk::raii::Device&         device,
  vk::GraphicsPipelineCreateInfo& graphics_create_info,
  pipeline_persistence&           persistence,
  [[maybe_unused]] std::optional<const descriptor_heap_arena&> arena       = {},
  std::optional<diagnostic_buffer&>                            diagnostics = {})
  -> std::expected<graphics_pipeline_library, error_t>
{
  using enum pipeline_persistence_mode;
  switch (persistence.mode())
  {
  case pipeline_cache :
  {
    vk::PipelineCreateFlags2CreateInfoKHR flags2 {};
    if constexpr (Backend == descriptor_table_backend::heap)
    {
      flags2.flags =
        vk::PipelineCreateFlagBits2::eDescriptorHeapEXT |
        vk::PipelineCreateFlagBits2::eLibraryKHR;
      flags2.pNext                = graphics_create_info.pNext;
      graphics_create_info.pNext  = &flags2;
      graphics_create_info.flags  = {};
      graphics_create_info.layout = vk::PipelineLayout {};
    }
    return map_vk_error(
      device.createGraphicsPipeline(
        persistence.cache_for_create(), graphics_create_info),
      diagnostics)
      .transform([](vk::raii::Pipeline&& pipeline) -> graphics_pipeline_library {
        return graphics_pipeline_library {std::move(pipeline)};
      });
  }
  case application_binary :
  {
    auto key =
      query_pipeline_binary_key(device, graphics_create_info, diagnostics);
    if (!key) { return std::unexpected {key.error()}; }
    auto hit = make_pipeline_binaries_from_store(
      device, *persistence.store(), *key, diagnostics);
    if (!hit) { return std::unexpected {hit.error()}; }
    if (*hit)
    {
      if (diagnostics)
      {
        diagnostics->report(
          diagnostic_severity::info,
          std::nullopt,
          std::source_location::current(),
          "pipeline_persistence: application-binary hit");
      }
      vk::PipelineCreateFlags2CreateInfoKHR flags2 {};
      if constexpr (Backend == descriptor_table_backend::heap)
      {
        flags2.flags =
          vk::PipelineCreateFlagBits2::eDescriptorHeapEXT |
          vk::PipelineCreateFlagBits2::eLibraryKHR;
        flags2.pNext                = graphics_create_info.pNext;
        graphics_create_info.pNext  = &flags2;
        graphics_create_info.flags  = {};
        graphics_create_info.layout = vk::PipelineLayout {};
      }
      pipeline_binaries         binaries     = std::move(**hit);
      vk::PipelineBinaryInfoKHR binary_local = binaries.info();
      binary_local.pNext                     = graphics_create_info.pNext;
      graphics_create_info.pNext             = &binary_local;
      return map_vk_error(
        device.createGraphicsPipeline(
          persistence.cache_for_create(), graphics_create_info),
        diagnostics)
        .transform(
          [ _ = std::move(binaries) ](
            vk::raii::Pipeline&& pipeline) -> graphics_pipeline_library {
            return graphics_pipeline_library {std::move(pipeline)};
          });
    }
    if (diagnostics)
    {
      diagnostics->report(
        diagnostic_severity::info,
        std::nullopt,
        std::source_location::current(),
        "pipeline_persistence: application binary miss");
    }
    return create_graphics_pipeline_library_application_miss<Backend>(
      device, graphics_create_info, *key, persistence, diagnostics);
  }
  case internal_binary :
    return create_graphics_pipeline_library_internal<Backend>(
      device, graphics_create_info, arena, diagnostics);
  }
  std::unreachable();
}

export template<
  graphics_pipeline_library_kind Kind,
  graphics_pipeline_spec         Spec    = {},
  descriptor_table_backend       Backend = descriptor_table_backend::classic,
  class Layout                           = void,
  class DiagnosticsPolicy                = diagnostics_off
>
  requires (validate(Spec) && Backend == descriptor_table_backend::classic)
auto make_graphics_pipeline_library(
  const vk::raii::Device&                       device,
  const graphics_pipeline_library_runtime_args& runtime_args,
  pipeline_persistence&                         persistence,
  std::optional<diagnostic_buffer&>             diagnostics = {})
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
    vk::GraphicsPipelineLibraryCreateInfoEXT library_info {
      .flags = library_flags_for(Kind),
    };
    vk::GraphicsPipelineCreateInfo graphics_create_info {
      .pNext               = &library_info,
      .flags               = vk::PipelineCreateFlagBits::eLibraryKHR,
      .pVertexInputState   = &vertex_input,
      .pInputAssemblyState = &input_assembly,
    };
    return create_graphics_pipeline_library_with_persistence<Backend>(
      device, graphics_create_info, persistence, {}, diagnostics);
  }
  else if constexpr (Kind == pre_rasterization)
  {
    if (runtime_args.layout == vk::PipelineLayout {})
    {
      return std::unexpected {
        make_app_error(app_error_code::missing_required_argument),
      };
    }
    if (runtime_args.spirv.empty())
    {
      return std::unexpected {
        make_app_error(app_error_code::missing_required_argument),
      };
    }
    if constexpr (
      !std::is_void_v<Layout> && DiagnosticsPolicy::check_spirv_bindings)
    {
      const auto* words =
        std::start_lifetime_as<std::uint32_t>(runtime_args.spirv.data());
      const std::span<const std::uint32_t> spirv_u32 {
        words,
        runtime_args.spirv.size_bytes() / sizeof(std::uint32_t),
      };
      if (
        auto check = validate_spirv_bindings(
          spirv_u32,
          Layout {},
          vk::ShaderStageFlagBits::eVertex |
            vk::ShaderStageFlagBits::eFragment |
            vk::ShaderStageFlagBits::eGeometry |
            vk::ShaderStageFlagBits::eTessellationControl |
            vk::ShaderStageFlagBits::eTessellationEvaluation);
        !check)
      {
        return std::unexpected {check.error()};
      }
    }
    const vk::ShaderModuleCreateInfo module_info {
      .codeSize = runtime_args.spirv.size_bytes(),
      .pCode = std::start_lifetime_as<std::uint32_t>(runtime_args.spirv.data()),
    };
    return map_vk_error(device.createShaderModule(module_info), diagnostics)
      .and_then(
        [ & ](vk::raii::ShaderModule&& module)
          -> std::expected<graphics_pipeline_library, error_t> {
          const vk::PipelineShaderStageCreateInfo stage {
            .stage  = vk::ShaderStageFlagBits::eVertex,
            .module = *module,
            .pName  = runtime_args.vertex_entry,
          };
          const vk::PipelineViewportStateCreateInfo viewport_state {
            .viewportCount = 1U,
            .scissorCount  = 1U,
          };
          const vk::PipelineRasterizationStateCreateInfo raster {
            .depthClampEnable        = vk::False,
            .rasterizerDiscardEnable = vk::False,
            .polygonMode             = Spec.polygon_mode,
            .cullMode                = Spec.cull_mode,
            .frontFace               = Spec.front_face,
            .depthBiasEnable         = vk::False,
            .lineWidth               = 1.0F,
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
          vk::GraphicsPipelineLibraryCreateInfoEXT library_info {
            .flags = library_flags_for(Kind),
          };
          vk::GraphicsPipelineCreateInfo graphics_create_info {
            .pNext               = &library_info,
            .flags               = vk::PipelineCreateFlagBits::eLibraryKHR,
            .stageCount          = 1U,
            .pStages             = &stage,
            .pViewportState      = &viewport_state,
            .pRasterizationState = &raster,
            .pDynamicState       = &dynamic,
            .layout              = runtime_args.layout,
          };
          return create_graphics_pipeline_library_with_persistence<Backend>(
            device, graphics_create_info, persistence, {}, diagnostics);
        });
  }
  else if constexpr (Kind == fragment)
  {
    if (runtime_args.layout == vk::PipelineLayout {})
    {
      return std::unexpected {
        make_app_error(app_error_code::missing_required_argument),
      };
    }
    if (runtime_args.spirv.empty())
    {
      return std::unexpected {
        make_app_error(app_error_code::missing_required_argument),
      };
    }
    if constexpr (
      !std::is_void_v<Layout> && DiagnosticsPolicy::check_spirv_bindings)
    {
      const auto* words =
        std::start_lifetime_as<std::uint32_t>(runtime_args.spirv.data());
      const std::span<const std::uint32_t> spirv_u32 {
        words,
        runtime_args.spirv.size_bytes() / sizeof(std::uint32_t),
      };
      if (
        auto check = validate_spirv_bindings(
          spirv_u32,
          Layout {},
          vk::ShaderStageFlagBits::eVertex |
            vk::ShaderStageFlagBits::eFragment |
            vk::ShaderStageFlagBits::eGeometry |
            vk::ShaderStageFlagBits::eTessellationControl |
            vk::ShaderStageFlagBits::eTessellationEvaluation);
        !check)
      {
        return std::unexpected {check.error()};
      }
    }
    const vk::ShaderModuleCreateInfo module_info {
      .codeSize = runtime_args.spirv.size_bytes(),
      .pCode = std::start_lifetime_as<std::uint32_t>(runtime_args.spirv.data()),
    };
    return map_vk_error(device.createShaderModule(module_info), diagnostics)
      .and_then(
        [ & ](vk::raii::ShaderModule&& module)
          -> std::expected<graphics_pipeline_library, error_t> {
          const vk::PipelineShaderStageCreateInfo stage {
            .stage  = vk::ShaderStageFlagBits::eFragment,
            .module = *module,
            .pName  = runtime_args.fragment_entry,
          };
          const vk::PipelineMultisampleStateCreateInfo multisample {
            .rasterizationSamples = runtime_args.samples,
            .sampleShadingEnable  = vk::Bool32 {Spec.sample_shading},
            .minSampleShading     = Spec.min_sample_shading,
          };
          const vk::PipelineDepthStencilStateCreateInfo depth {
            .depthTestEnable       = vk::Bool32 {Spec.depth_test},
            .depthWriteEnable      = vk::Bool32 {Spec.depth_write},
            .depthCompareOp        = Spec.depth_compare,
            .depthBoundsTestEnable = vk::False,
            .stencilTestEnable     = vk::False,
          };
          vk::GraphicsPipelineLibraryCreateInfoEXT library_info {
            .flags = library_flags_for(Kind),
          };
          vk::GraphicsPipelineCreateInfo graphics_create_info {
            .pNext              = &library_info,
            .flags              = vk::PipelineCreateFlagBits::eLibraryKHR,
            .stageCount         = 1U,
            .pStages            = &stage,
            .pMultisampleState  = &multisample,
            .pDepthStencilState = &depth,
            .layout             = runtime_args.layout,
          };
          return create_graphics_pipeline_library_with_persistence<Backend>(
            device, graphics_create_info, persistence, {}, diagnostics);
        });
  }
  else if constexpr (Kind == fragment_output)
  {
    if (runtime_args.color_formats.empty())
    {
      return std::unexpected {
        make_app_error(app_error_code::missing_required_argument),
      };
    }
    const vk::PipelineMultisampleStateCreateInfo multisample {
      .rasterizationSamples = runtime_args.samples,
      .sampleShadingEnable  = vk::Bool32 {Spec.sample_shading},
      .minSampleShading     = Spec.min_sample_shading,
    };
    const vk::PipelineColorBlendAttachmentState blend_attachment {
      .blendEnable         = vk::Bool32 {Spec.blend_enable},
      .srcColorBlendFactor = Spec.src_color_blend_factor,
      .dstColorBlendFactor = Spec.dst_color_blend_factor,
      .colorBlendOp        = Spec.color_blend_op,
      .srcAlphaBlendFactor = Spec.src_alpha_blend_factor,
      .dstAlphaBlendFactor = Spec.dst_alpha_blend_factor,
      .alphaBlendOp        = Spec.alpha_blend_op,
      .colorWriteMask =
        vk::ColorComponentFlagBits::eR |
        vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB |
        vk::ColorComponentFlagBits::eA,
    };
    const vk::PipelineColorBlendStateCreateInfo blend {
      .logicOpEnable   = vk::False,
      .attachmentCount = 1U,
      .pAttachments    = &blend_attachment,
    };
    vk::PipelineRenderingCreateInfo rendering_info {
      .colorAttachmentCount =
        static_cast<std::uint32_t>(runtime_args.color_formats.size()),
      .pColorAttachmentFormats = runtime_args.color_formats.data(),
      .depthAttachmentFormat   = runtime_args.depth_format,
    };
    vk::GraphicsPipelineLibraryCreateInfoEXT library_info {
      .pNext = &rendering_info,
      .flags = library_flags_for(Kind),
    };
    vk::GraphicsPipelineCreateInfo graphics_create_info {
      .pNext             = &library_info,
      .flags             = vk::PipelineCreateFlagBits::eLibraryKHR,
      .pMultisampleState = &multisample,
      .pColorBlendState  = &blend,
    };
    return create_graphics_pipeline_library_with_persistence<Backend>(
      device, graphics_create_info, persistence, {}, diagnostics);
  }
  else
  {
    static_assert(
      Kind == vertex_input ||
        Kind == pre_rasterization ||
        Kind == fragment ||
        Kind == fragment_output,
      "unhandled graphics_pipeline_library_kind");
    return std::unexpected {make_app_error(app_error_code::invalid_state)};
  }
}

export template<
  graphics_pipeline_library_kind Kind,
  graphics_pipeline_spec         Spec = {},
  descriptor_table_backend       Backend,
  class Layout            = void,
  class DiagnosticsPolicy = diagnostics_off
>
  requires (validate(Spec) && Backend == descriptor_table_backend::heap)
auto make_graphics_pipeline_library(
  const vk::raii::Device&                       device,
  const graphics_pipeline_library_runtime_args& runtime_args,
  pipeline_persistence&                         persistence,
  const descriptor_heap_arena&                  arena,
  std::optional<diagnostic_buffer&>             diagnostics = {})
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
    vk::GraphicsPipelineLibraryCreateInfoEXT library_info {
      .flags = library_flags_for(Kind),
    };
    vk::GraphicsPipelineCreateInfo graphics_create_info {
      .pNext               = &library_info,
      .flags               = vk::PipelineCreateFlagBits::eLibraryKHR,
      .pVertexInputState   = &vertex_input,
      .pInputAssemblyState = &input_assembly,
    };
    return create_graphics_pipeline_library_with_persistence<Backend>(
      device, graphics_create_info, persistence, arena, diagnostics);
  }
  else if constexpr (Kind == pre_rasterization)
  {
    if (runtime_args.spirv.empty())
    {
      return std::unexpected {
        make_app_error(app_error_code::missing_required_argument),
      };
    }
    if constexpr (
      !std::is_void_v<Layout> && DiagnosticsPolicy::check_spirv_bindings)
    {
      const auto* words =
        std::start_lifetime_as<std::uint32_t>(runtime_args.spirv.data());
      const std::span<const std::uint32_t> spirv_u32 {
        words,
        runtime_args.spirv.size_bytes() / sizeof(std::uint32_t),
      };
      if (
        auto check = validate_spirv_bindings(
          spirv_u32,
          Layout {},
          vk::ShaderStageFlagBits::eVertex |
            vk::ShaderStageFlagBits::eFragment |
            vk::ShaderStageFlagBits::eGeometry |
            vk::ShaderStageFlagBits::eTessellationControl |
            vk::ShaderStageFlagBits::eTessellationEvaluation);
        !check)
      {
        return std::unexpected {check.error()};
      }
    }
    const vk::ShaderModuleCreateInfo module_info {
      .codeSize = runtime_args.spirv.size_bytes(),
      .pCode = std::start_lifetime_as<std::uint32_t>(runtime_args.spirv.data()),
    };
    return map_vk_error(device.createShaderModule(module_info), diagnostics)
      .and_then(
        [ & ](vk::raii::ShaderModule&& module)
          -> std::expected<graphics_pipeline_library, error_t> {
          auto mapping = arena.shader_and_mapping_info();
          const vk::PipelineShaderStageCreateInfo stage {
            .pNext  = &mapping,
            .stage  = vk::ShaderStageFlagBits::eVertex,
            .module = *module,
            .pName  = runtime_args.vertex_entry,
          };
          const vk::PipelineViewportStateCreateInfo viewport_state {
            .viewportCount = 1U,
            .scissorCount  = 1U,
          };
          const vk::PipelineRasterizationStateCreateInfo raster {
            .depthClampEnable        = vk::False,
            .rasterizerDiscardEnable = vk::False,
            .polygonMode             = Spec.polygon_mode,
            .cullMode                = Spec.cull_mode,
            .frontFace               = Spec.front_face,
            .depthBiasEnable         = vk::False,
            .lineWidth               = 1.0F,
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
          vk::GraphicsPipelineLibraryCreateInfoEXT library_info {
            .flags = library_flags_for(Kind),
          };
          vk::GraphicsPipelineCreateInfo graphics_create_info {
            .pNext               = &library_info,
            .flags               = vk::PipelineCreateFlagBits::eLibraryKHR,
            .stageCount          = 1U,
            .pStages             = &stage,
            .pViewportState      = &viewport_state,
            .pRasterizationState = &raster,
            .pDynamicState       = &dynamic,
            .layout              = {},
          };
          return create_graphics_pipeline_library_with_persistence<Backend>(
            device, graphics_create_info, persistence, arena, diagnostics);
        });
  }
  else if constexpr (Kind == fragment)
  {
    if (runtime_args.spirv.empty())
    {
      return std::unexpected {
        make_app_error(app_error_code::missing_required_argument),
      };
    }
    if constexpr (
      !std::is_void_v<Layout> && DiagnosticsPolicy::check_spirv_bindings)
    {
      const auto* words =
        std::start_lifetime_as<std::uint32_t>(runtime_args.spirv.data());
      const std::span<const std::uint32_t> spirv_u32 {
        words,
        runtime_args.spirv.size_bytes() / sizeof(std::uint32_t),
      };
      if (
        auto check = validate_spirv_bindings(
          spirv_u32,
          Layout {},
          vk::ShaderStageFlagBits::eVertex |
            vk::ShaderStageFlagBits::eFragment |
            vk::ShaderStageFlagBits::eGeometry |
            vk::ShaderStageFlagBits::eTessellationControl |
            vk::ShaderStageFlagBits::eTessellationEvaluation);
        !check)
      {
        return std::unexpected {check.error()};
      }
    }
    const vk::ShaderModuleCreateInfo module_info {
      .codeSize = runtime_args.spirv.size_bytes(),
      .pCode = std::start_lifetime_as<std::uint32_t>(runtime_args.spirv.data()),
    };
    return map_vk_error(device.createShaderModule(module_info), diagnostics)
      .and_then(
        [ & ](vk::raii::ShaderModule&& module)
          -> std::expected<graphics_pipeline_library, error_t> {
          auto mapping = arena.shader_and_mapping_info();
          const vk::PipelineShaderStageCreateInfo stage {
            .pNext  = &mapping,
            .stage  = vk::ShaderStageFlagBits::eFragment,
            .module = *module,
            .pName  = runtime_args.fragment_entry,
          };
          const vk::PipelineMultisampleStateCreateInfo multisample {
            .rasterizationSamples = runtime_args.samples,
            .sampleShadingEnable  = vk::Bool32 {Spec.sample_shading},
            .minSampleShading     = Spec.min_sample_shading,
          };
          const vk::PipelineDepthStencilStateCreateInfo depth {
            .depthTestEnable       = vk::Bool32 {Spec.depth_test},
            .depthWriteEnable      = vk::Bool32 {Spec.depth_write},
            .depthCompareOp        = Spec.depth_compare,
            .depthBoundsTestEnable = vk::False,
            .stencilTestEnable     = vk::False,
          };
          vk::GraphicsPipelineLibraryCreateInfoEXT library_info {
            .flags = library_flags_for(Kind),
          };
          vk::GraphicsPipelineCreateInfo graphics_create_info {
            .pNext              = &library_info,
            .flags              = vk::PipelineCreateFlagBits::eLibraryKHR,
            .stageCount         = 1U,
            .pStages            = &stage,
            .pMultisampleState  = &multisample,
            .pDepthStencilState = &depth,
            .layout             = {},
          };
          return create_graphics_pipeline_library_with_persistence<Backend>(
            device, graphics_create_info, persistence, arena, diagnostics);
        });
  }
  else if constexpr (Kind == fragment_output)
  {
    if (runtime_args.color_formats.empty())
    {
      return std::unexpected {
        make_app_error(app_error_code::missing_required_argument),
      };
    }
    const vk::PipelineMultisampleStateCreateInfo multisample {
      .rasterizationSamples = runtime_args.samples,
      .sampleShadingEnable  = vk::Bool32 {Spec.sample_shading},
      .minSampleShading     = Spec.min_sample_shading,
    };
    const vk::PipelineColorBlendAttachmentState blend_attachment {
      .blendEnable         = vk::Bool32 {Spec.blend_enable},
      .srcColorBlendFactor = Spec.src_color_blend_factor,
      .dstColorBlendFactor = Spec.dst_color_blend_factor,
      .colorBlendOp        = Spec.color_blend_op,
      .srcAlphaBlendFactor = Spec.src_alpha_blend_factor,
      .dstAlphaBlendFactor = Spec.dst_alpha_blend_factor,
      .alphaBlendOp        = Spec.alpha_blend_op,
      .colorWriteMask =
        vk::ColorComponentFlagBits::eR |
        vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB |
        vk::ColorComponentFlagBits::eA,
    };
    const vk::PipelineColorBlendStateCreateInfo blend {
      .logicOpEnable   = vk::False,
      .attachmentCount = 1U,
      .pAttachments    = &blend_attachment,
    };
    vk::PipelineRenderingCreateInfo rendering_info {
      .colorAttachmentCount =
        static_cast<std::uint32_t>(runtime_args.color_formats.size()),
      .pColorAttachmentFormats = runtime_args.color_formats.data(),
      .depthAttachmentFormat   = runtime_args.depth_format,
    };
    vk::GraphicsPipelineLibraryCreateInfoEXT library_info {
      .pNext = &rendering_info,
      .flags = library_flags_for(Kind),
    };
    vk::GraphicsPipelineCreateInfo graphics_create_info {
      .pNext             = &library_info,
      .flags             = vk::PipelineCreateFlagBits::eLibraryKHR,
      .pMultisampleState = &multisample,
      .pColorBlendState  = &blend,
    };
    return create_graphics_pipeline_library_with_persistence<Backend>(
      device, graphics_create_info, persistence, arena, diagnostics);
  }
  else
  {
    static_assert(
      Kind == vertex_input ||
        Kind == pre_rasterization ||
        Kind == fragment ||
        Kind == fragment_output,
      "unhandled graphics_pipeline_library_kind");
    return std::unexpected {make_app_error(app_error_code::invalid_state)};
  }
}

export
template<descriptor_table_backend Backend = descriptor_table_backend::classic>
auto
link_graphics_pipeline(
  const vk::raii::Device&                         device,
  std::span<const vk::Pipeline>                   libraries,
  bool                                            optimize,
  vk::PipelineLayout                              layout,
  const vk::raii::PipelineCache&                  cache       = {nullptr},
  const std::optional<vk::PipelineBinaryInfoKHR>& binary_info = {})
  -> std::expected<vk::raii::Pipeline, error_t>
{
  if (libraries.size() != 4UZ)
  {
    return std::unexpected {make_app_error(app_error_code::invalid_state)};
  }
  const vk::PipelineLibraryCreateInfoKHR library_info {
    .libraryCount = 4U,
    .pLibraries   = libraries.data(),
  };

  const void*             p_next = &library_info;
  vk::PipelineCreateFlags flags {};
  if constexpr (Backend == descriptor_table_backend::classic)
  {
    if (optimize)
    {
      flags |= vk::PipelineCreateFlagBits::eLinkTimeOptimizationEXT;
    }
  }

  vk::PipelineBinaryInfoKHR binary_local {};
  const bool                replaying_binaries =
    binary_info.has_value() && binary_info->binaryCount != 0U;
  if (binary_info.has_value())
  {
    binary_local       = *binary_info;
    binary_local.pNext = p_next;
    p_next             = &binary_local;
  }

  vk::GraphicsPipelineCreateInfo create_info {
    .pNext  = p_next,
    .flags  = flags,
    .layout = layout,
  };

  vk::PipelineCreateFlags2CreateInfoKHR flags2 {};
  if constexpr (Backend == descriptor_table_backend::heap)
  {
    flags2.flags = vk::PipelineCreateFlagBits2::eDescriptorHeapEXT;
    if (optimize)
    {
      flags |= vk::PipelineCreateFlagBits::eLinkTimeOptimizationEXT;
    }
    flags2.pNext       = create_info.pNext;
    create_info.pNext  = &flags2;
    create_info.flags  = {};
    create_info.layout = vk::PipelineLayout {};
  }

  const vk::raii::PipelineCache null_cache {nullptr};
  const auto& cache_for_create = replaying_binaries ? null_cache : cache;

  return map_vk_error(
    device.createGraphicsPipeline(cache_for_create, create_info), std::nullopt)
    .transform([](vk::raii::Pipeline&& pipeline) -> vk::raii::Pipeline {
      return std::move(pipeline);
    });
}

} // namespace vkpp
