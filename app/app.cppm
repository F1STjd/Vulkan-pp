module;

#include "contracts_config.hpp"
#include "error/vk_error_config.hpp"

#include <SFML/Window.hpp>
#include <SFML/Window/VideoMode.hpp>
#include <vulkan/vk_platform.h>

#include <stb_image.h>

export module f1st.app;

import std;
import vulkan;
import glm;
import f1st.uniform_buffer;
import vkpp.io;
import vkpp.error;
import vkpp.vertex;
import vkpp.memory;
import vkpp.memory.vma;
import vkpp.image;
import vkpp.buffer;
import vkpp.instance;
import vkpp.device;
import vkpp.swapchain;
import vkpp.command;
import vkpp.frame;

namespace f1st
{
using namespace std::string_view_literals;

constexpr std::uint32_t window_width { 800 };
constexpr std::uint32_t window_height { 600 };

// During the development i want validation layers (for corectness) in the
// release build
#ifdef NDEBUG
constexpr std::array validation_layers {
  "VK_LAYER_KHRONOS_validation",
  // This one is not checked in the code :(, but should be
  "VK_LAYER_LUNARG_monitor",
};
constexpr bool enable_validation_layers { true };
#else
constexpr std::array<const char*, 0> validation_layers {};
constexpr bool enable_validation_layers { false };
#endif

[[nodiscard]] auto
required_instance_extensions() -> std::vector<const char*>
{
  const auto& sfml = sf::Vulkan::getGraphicsRequiredInstanceExtensions();
  std::vector<const char*> extensions { std::from_range, sfml };
  if constexpr (enable_validation_layers)
  {
    extensions.push_back(vk::EXTDebugUtilsExtensionName);
  }
  return extensions;
}

constexpr std::array required_device_extensions {
  vk::KHRSwapchainExtensionName,
  vk::EXTExtendedDynamicStateExtensionName,
};

constexpr std::uint32_t max_frames_in_flight { 2U };
static_assert(max_frames_in_flight > 0,
  "variable % max_frames_in_flight is used later, so being 0 is UB");

export class app
{
public:
  void
  run()
  {
    const auto result = init_vulkan().and_then(
      [ this ]() -> std::expected<void, vkpp::error_t> { return main_loop(); });

    (void)device_.device().waitIdle();
    if (!result) { std::println(stderr, "{}", vkpp::message(result.error())); }
    swap_chain_.release();
  }

private:
  auto
  init_vulkan() -> std::expected<void, vkpp::error_t>
  {
    return create_instance_context()
      .and_then(std::bind_front(&app::create_surface, this))
      .and_then(std::bind_front(&app::create_device_context, this))
      .and_then(std::bind_front(&app::create_swap_chain, this))
      .and_then(std::bind_front(&app::create_command_pool, this))
      .and_then(std::bind_front(&app::create_descriptor_set_layout, this))
      .and_then(std::bind_front(&app::create_graphics_pipeline, this))
      .and_then(std::bind_front(&app::create_texture_image, this))
      .and_then(std::bind_front(&app::create_texture_image_view, this))
      .and_then(std::bind_front(&app::create_texture_sampler, this))
      .and_then([ this ] { return vkpp::load_model_obj(vertices_, indices_); })
      .and_then(std::bind_front(&app::create_vertex_buffer, this))
      .and_then(std::bind_front(&app::create_index_buffer, this))
      .and_then(std::bind_front(&app::create_uniform_buffers, this))
      .and_then(std::bind_front(&app::create_descriptor_pool, this))
      .and_then(std::bind_front(&app::create_descriptor_sets, this))
      .and_then(std::bind_front(&app::create_command_buffers, this))
      .and_then(std::bind_front(&app::create_sync_objects, this));
  }

  auto
  main_loop() -> std::expected<void, vkpp::error_t>
  {
    const auto on_close = [ this ](const sf::Event::Closed&) -> void
    { window_.close(); };

    const auto on_resize = [ this ](const sf::Event::Resized&) -> void
    { resized_ = true; };

    while (window_.isOpen())
    {
      window_.handleEvents(on_close, on_resize);
      if (!window_.isOpen()) { break; }
      if (auto result = draw_frame(); !result) { return result; }
    }
    return UTILS_VK(device_.device().waitIdle(), ^^vk::raii::Device::waitIdle);
  }

private:
  auto
  create_instance_context() -> std::expected<void, vkpp::error_t>
  {
    static constexpr vk::ApplicationInfo app_info {
      .pApplicationName = "f1st",
      .applicationVersion = vk::makeVersion(1, 0, 0),
      .pEngineName = "No Engine",
      .engineVersion = vk::makeVersion(1, 0, 0),
      .apiVersion = vk::ApiVersion14,
    };
    const auto extensions = required_instance_extensions();
    return vkpp::instance_context::create(
      {
        .app_info = app_info,
        .extensions = extensions,
        .layers = validation_layers,
        .enable_validation = enable_validation_layers,
      })
      .transform([ this ](vkpp::instance_context&& context)
        { instance_ = std::move(context); });
  }

  auto
  create_surface() -> std::expected<void, vkpp::error_t>
  {
    VkSurfaceKHR _surface {};
    if (!window_.createVulkanSurface(*instance_.instance(), _surface))
    {
      return std::unexpected {
        vkpp::app_error {
          .kind = vkpp::app_error_kind::surface_creation,
          .detail = "Faild to create window surface"sv,
        },
      };
    }
    instance_.adopt_surface(
      vk::raii::SurfaceKHR(instance_.instance(), _surface));
    return {};
  }

  auto
  create_device_context() -> std::expected<void, vkpp::error_t>
  {
    return vkpp::device_context::create(instance_,
      {
        .extensions = required_device_extensions,
        .min_api_version = vk::ApiVersion13,
        .features = {
          .sampler_anisotropy = true,
          .sample_rate_shading = true,
          .dynamic_rendering = true,
          .synchronization2 = true,
          .extended_dynamic_state = true,
        },
        .require_present = true,
      })
      .transform([ this ](vkpp::device_context&& device)
        { device_ = std::move(device); });
  }

  auto
  create_swap_chain() -> std::expected<void, vkpp::error_t>
  {
    return vkpp::swapchain::create(device_, instance_.surface(),
      framebuffer_extent_request(),
      [ this ](const vk::SurfaceCapabilitiesKHR& capabilities,
        vk::Extent2D framebuffer)
      { return choose_swap_extent(capabilities, framebuffer); })
      .transform([ this ](vkpp::swapchain&& swap_chain)
        { swap_chain_ = std::move(swap_chain); });
  }

  auto
  framebuffer_extent_request() const -> vkpp::extent_request
  {
    const auto [ width, height ] = window_.getSize();
    return {
      .framebuffer_size = { width, height },
    };
  }

  auto
  choose_swap_extent(const vk::SurfaceCapabilitiesKHR& capabilities,
    vk::Extent2D framebuffer) -> vk::Extent2D
  {
    if (capabilities.currentExtent.width !=
      std::numeric_limits<std::uint32_t>::max())
    {
      return capabilities.currentExtent;
    }
    return {
      .width = std::clamp(framebuffer.width, capabilities.minImageExtent.width,
        capabilities.maxImageExtent.width),
      .height = std::clamp(framebuffer.height,
        capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
    };
  }

  auto
  create_image_view(const vk::Image& image, vk::Format format,
    vk::ImageAspectFlags aspect_flags, std::uint32_t mip_levels)
    -> std::expected<vk::raii::ImageView, vkpp::error_t>
  {
    vk::ImageViewCreateInfo image_view_info {
        .image = image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = {
          .aspectMask = aspect_flags,
          .baseMipLevel = 0U,
          .levelCount = mip_levels,
          .baseArrayLayer = 0U,
          .layerCount = 1U,
        },
      };

    return UTILS_VK(device_.device().createImageView(image_view_info),
      ^^vk::raii::Device::createImageView);
  }

  auto
  create_descriptor_set_layout() -> std::expected<void, vkpp::error_t>
  {
    std::array bindings {
      vk::DescriptorSetLayoutBinding {
        .binding = 0U,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 1U,
        .stageFlags = vk::ShaderStageFlagBits::eVertex,
      },
      vk::DescriptorSetLayoutBinding {
        .binding = 1U,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1U,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
      },
    };
    vk::DescriptorSetLayoutCreateInfo ubo_layout_create_info {
      .bindingCount = static_cast<std::uint32_t>(bindings.size()),
      .pBindings = bindings.data(),
    };

    return UTILS_VK(
      device_.device().createDescriptorSetLayout(ubo_layout_create_info),
      ^^vk::raii::Device::createDescriptorSetLayout)
      .transform([ this ](vk::raii::DescriptorSetLayout&& layout) -> void
        { descriptor_set_layout_ = std::move(layout); });
  }

  auto
  create_graphics_pipeline() -> std::expected<void, vkpp::error_t>
  {
    static const vk::PipelineLayoutCreateInfo pipeline_layout_create_info {
      .setLayoutCount = 1,
      .pSetLayouts = &*descriptor_set_layout_,
      .pushConstantRangeCount = 0,
    };

    return UTILS_VK(
      device_.device().createPipelineLayout(pipeline_layout_create_info),
      ^^vk::raii::Device::createPipelineLayout)
      .and_then(
        [ this ](vk::raii::PipelineLayout&& layout)
          -> std::expected<std::vector<char>, vkpp::error_t>
        {
          pipeline_layout_ = std::move(layout);
          return vkpp::load_shader_file(SHADER_DIRECTORY "slang.spv");
        })
      .and_then([ this ](std::span<const char> code)
        { return create_shader_module(code); })
      .and_then(
        [ this ](const vk::raii::ShaderModule& shader_module)
          -> std::expected<vk::raii::Pipeline, vkpp::error_t>
        {
          const vk::PipelineShaderStageCreateInfo
            vertex_shader_stage_create_info {
              .stage = vk::ShaderStageFlagBits::eVertex,
              .module = shader_module,
              .pName = "vertex_main",
              .pSpecializationInfo = nullptr,
            };
          const vk::PipelineShaderStageCreateInfo
            fragment_shader_stage_create_info {
              .stage = vk::ShaderStageFlagBits::eFragment,
              .module = shader_module,
              .pName = "fragment_main",
              .pSpecializationInfo = nullptr,
            };
          const std::array shader_stages {
            vertex_shader_stage_create_info,
            fragment_shader_stage_create_info,
          };

          static constexpr auto binding_description =
            vkpp::vertex::get_binding_description();
          static constexpr auto attribute_descriptions =
            vkpp::vertex::get_attribute_descriptions();
          static constexpr vk::PipelineVertexInputStateCreateInfo
            vertex_input_create_info {
              .vertexBindingDescriptionCount = 1U,
              .pVertexBindingDescriptions = &binding_description,
              .vertexAttributeDescriptionCount =
                static_cast<std::uint32_t>(attribute_descriptions.size()),
              .pVertexAttributeDescriptions = attribute_descriptions.data(),
            };

          static constexpr vk::PipelineInputAssemblyStateCreateInfo
            input_assembly_create_info {
              .topology = vk::PrimitiveTopology::eTriangleList,
            };
          static constexpr vk::PipelineViewportStateCreateInfo
            viewport_state_create_info {
              .viewportCount = 1U,
              .scissorCount = 1U,
            };
          static constexpr vk::PipelineRasterizationStateCreateInfo
            rasterizer_create_info {
              .depthClampEnable = vk::False,
              .rasterizerDiscardEnable = vk::False,
              .polygonMode = vk::PolygonMode::eFill,
              .cullMode = vk::CullModeFlagBits::eBack,
              .frontFace = vk::FrontFace::eCounterClockwise,
              .depthBiasEnable = vk::False,
              .lineWidth = 1.0F,
            };
          const vk::PipelineMultisampleStateCreateInfo
            multisampling_create_info {
              .rasterizationSamples = device_.msaa_samples(),
              .sampleShadingEnable = vk::True,
              .minSampleShading = 0.2F,
            };

          static constexpr vk::PipelineDepthStencilStateCreateInfo
            depth_stencil_create_info {
              .depthTestEnable = vk::True,
              .depthWriteEnable = vk::True,
              .depthCompareOp = vk::CompareOp::eLess,
              .depthBoundsTestEnable = vk::False,
              .stencilTestEnable = vk::False,
            };

          static constexpr vk::PipelineColorBlendAttachmentState
            color_blend_attachment {
              .blendEnable = vk::False,
              .colorWriteMask = vk::ColorComponentFlagBits::eR |
                vk::ColorComponentFlagBits::eG |
                vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
            };
          static constexpr vk::PipelineColorBlendStateCreateInfo
            color_blend_create_info {
              .logicOpEnable = vk::False,
              .logicOp = vk::LogicOp::eCopy,
              .attachmentCount = 1U,
              .pAttachments = &color_blend_attachment,
            };

          static constexpr std::array dynamic_states {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor,
          };
          static constexpr vk::PipelineDynamicStateCreateInfo dynamic_state {
            .dynamicStateCount =
              static_cast<std::uint32_t>(dynamic_states.size()),
            .pDynamicStates = dynamic_states.data(),
          };

          const std::array color_attachment_formats { swap_chain_.format() };
          vk::StructureChain pipeline_create_info_chain {
            vk::GraphicsPipelineCreateInfo {
              .stageCount = 2U,
              .pStages = shader_stages.data(),
              .pVertexInputState = &vertex_input_create_info,
              .pInputAssemblyState = &input_assembly_create_info,
              .pViewportState = &viewport_state_create_info,
              .pRasterizationState = &rasterizer_create_info,
              .pMultisampleState = &multisampling_create_info,
              .pDepthStencilState = &depth_stencil_create_info,
              .pColorBlendState = &color_blend_create_info,
              .pDynamicState = &dynamic_state,
              .layout = pipeline_layout_,
              .renderPass = nullptr,
            },
            vk::PipelineRenderingCreateInfo {
              .colorAttachmentCount =
                static_cast<std::uint32_t>(color_attachment_formats.size()),
              .pColorAttachmentFormats = color_attachment_formats.data(),
              .depthAttachmentFormat = swap_chain_.depth().format(),
            },
          };
          return UTILS_VK(
            device_.device().createGraphicsPipeline(nullptr,
              pipeline_create_info_chain.get<vk::GraphicsPipelineCreateInfo>()),
            ^^vk::raii::Device::createGraphicsPipeline);
        })
      .transform([ this ](vk::raii::Pipeline&& pipeline) -> void
        { graphics_pipeline_ = std::move(pipeline); });
  }

  [[nodiscard]] auto
  create_shader_module(std::span<const char> code)
    -> std::expected<vk::raii::ShaderModule, vkpp::error_t>
  {
    vk::ShaderModuleCreateInfo shader_module_create_info {
      .codeSize = code.size_bytes(),
      .pCode = std::start_lifetime_as<std::uint32_t>(code.data()),
    };

    return UTILS_VK(
      device_.device().createShaderModule(shader_module_create_info),
      ^^vk::raii::Device::createShaderModule);
  }

  auto
  create_command_pool() -> std::expected<void, vkpp::error_t>
  {
    vk::CommandPoolCreateInfo command_pool_create_info {
      .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      .queueFamilyIndex = device_.graphics_qf_index(),
    };

    return UTILS_VK(
      device_.device().createCommandPool(command_pool_create_info),
      ^^vk::raii::Device::createCommandPool)
      .transform([ this ](vk::raii::CommandPool&& command_pool) -> void
        { command_pool_ = std::move(command_pool); });
  }

  using image_memory_pair = std::pair<vk::raii::Image, vk::raii::DeviceMemory>;

  // TODO: Konrad - Buffer and image creation is almost same, so create some
  // abstraction, for creating allocated objects
  auto
  create_image(std::uint32_t width, std::uint32_t height,
    std::uint32_t mip_levels, vk::SampleCountFlagBits samples,
    vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage,
    vk::MemoryPropertyFlags properties)
    -> std::expected<image_memory_pair, vkpp::error_t>
  {
    const vk::ImageCreateInfo image_create_info {
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {
          .width = width,
          .height = height,
          .depth = 1U,
        },
        .mipLevels = mip_levels,
        .arrayLayers = 1,
        .samples = samples,
        .tiling = tiling,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
      };

    vk::raii::Image temp_image { nullptr };
    vk::raii::DeviceMemory temp_memory { nullptr };
    vk::DeviceSize memory_requirements_size; // NOLINT

    return UTILS_VK(device_.device().createImage(image_create_info),
      ^^vk::raii::Device::createImage)
      .and_then(
        [ &, this ](vk::raii::Image&& image)
        {
          temp_image = std::move(image);
          const auto memory_requirements = temp_image.getMemoryRequirements();
          memory_requirements_size = memory_requirements.size;
          return find_memory_type(
            memory_requirements.memoryTypeBits, properties);
        })
      .and_then(
        [ &, this ](auto memory_type)
        {
          vk::MemoryAllocateInfo memory_allocate_info {
            .allocationSize = memory_requirements_size,
            .memoryTypeIndex = memory_type,
          };

          return UTILS_VK(device_.device().allocateMemory(memory_allocate_info),
            ^^vk::raii::Device::allocateMemory);
        })
      .and_then(
        [ & ](vk::raii::DeviceMemory&& memory)
        {
          temp_memory = std::move(memory);
          return UTILS_VK(temp_image.bindMemory(*temp_memory, 0ULL),
            ^^vk::raii::Image::bindMemory);
        })
      .transform(
        [ & ]() -> image_memory_pair
        {
          return std::pair { std::move(temp_image), std::move(temp_memory) };
        });
  }

  auto
  find_supported_format(std::span<const vk::Format> candidates,
    vk::ImageTiling tiling, vk::FormatFeatureFlags features)
    -> std::expected<vk::Format, vkpp::error_t>
  {
    for (const auto format : candidates)
    {
      const auto properties =
        device_.physical_device().getFormatProperties(format);

      if (((tiling == vk::ImageTiling::eLinear) &&
            ((properties.linearTilingFeatures & features) == features)) ||
        ((tiling == vk::ImageTiling::eOptimal) &&
          ((properties.optimalTilingFeatures & features) == features)))
      {
        return format;
      }
    }

    return std::unexpected {
      vkpp::app_error {
        .kind = vkpp::app_error_kind::no_supported_format,
        .detail = "Failed to find supported format"sv,
      },
    };
  }

  auto
  create_texture_image() -> std::expected<void, vkpp::error_t>
  {
    std::int32_t texture_width;  // NOLINT
    std::int32_t texture_height; // NOLINT

    std::uint8_t* image_p {};
    std::size_t image_size {};

    vk::raii::Buffer staging_buffer { nullptr };
    vk::raii::DeviceMemory staging_buffer_memory { nullptr };

    auto upload = [ & ]
    {
      vkpp::single_time_submit sts { command_pool_, device_.device(),
        device_.graphics_queue() };
      return sts.begin()
        .and_then(
          [ &, this ](vk::raii::CommandBuffer* command_buffer)
            -> std::expected<void, vkpp::error_t>
          {
            transition_image_layout(*command_buffer, texture_image_,
              vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
              mip_levels_);
            copy_buffer_to_image(*command_buffer, staging_buffer,
              texture_image_, static_cast<std::uint32_t>(texture_width),
              static_cast<std::uint32_t>(texture_height));
            return generate_mipmaps(*command_buffer, texture_image_,
              vk::Format::eR8G8B8A8Srgb, texture_width, texture_height,
              mip_levels_);
          })
        .and_then([ & ] -> std::expected<void, vkpp::error_t>
          { return sts.end_and_submit(); });
    };

    return vkpp::load_texture_file(
      vkpp::texture_path, texture_width, texture_height, mip_levels_)
      .and_then(
        [ &, this ](const auto& image)
        {
          image_p = image.data();
          image_size = image.size();
          return create_buffer(image.size(),
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible |
              vk::MemoryPropertyFlagBits::eHostCoherent);
        })
      .and_then(
        [ & ](buffer_memory_pair&& pair) -> std::expected<void*, vkpp::error_t>
        {
          std::tie(staging_buffer, staging_buffer_memory) = std::move(pair);
          return UTILS_VK(staging_buffer_memory.mapMemory(0ULL, image_size),
            ^^vk::raii::DeviceMemory::mapMemory);
        })
      .and_then(
        [ &, this ](void* data_staging) -> std::expected<void, vkpp::error_t>
        {
          std::memcpy(data_staging, image_p, image_size);
          staging_buffer_memory.unmapMemory();
          stbi_image_free(image_p);
          image_p = nullptr;
          return create_image(static_cast<std::uint32_t>(texture_width),
            static_cast<std::uint32_t>(texture_height), mip_levels_,
            vk::SampleCountFlagBits::e1, vk::Format::eR8G8B8A8Srgb,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlags {
              vk::ImageUsageFlagBits::eTransferSrc |
                vk::ImageUsageFlagBits::eTransferDst |
                vk::ImageUsageFlagBits::eSampled,
            },
            vk::MemoryPropertyFlagBits::eDeviceLocal)
            .transform(
              [ this ](image_memory_pair&& pair) -> void
              {
                std::tie(texture_image_, texture_image_memory_) =
                  std::move(pair);
              });
        })
      .and_then(upload)
      .or_else(
        [ & ](vkpp::error_t error) -> std::expected<void, vkpp::error_t>
        {
          if (image_p != nullptr) { stbi_image_free(image_p); }
          return std::unexpected { error };
        });
  }

  auto
  create_texture_image_view() -> std::expected<void, vkpp::error_t>
  {
    return create_image_view(*texture_image_, vk::Format::eR8G8B8A8Srgb,
      vk::ImageAspectFlagBits::eColor, mip_levels_)
      .transform([ this ](vk::raii::ImageView&& view) -> void
        { texture_image_view_ = std::move(view); });
  }

  auto
  create_texture_sampler() -> std::expected<void, vkpp::error_t>
  {
    const auto properties = device_.physical_device().getProperties();
    vk::SamplerCreateInfo sampler_create_info {
      .magFilter = vk::Filter::eLinear,
      .minFilter = vk::Filter::eLinear,
      .mipmapMode = vk::SamplerMipmapMode::eLinear,
      .addressModeU = vk::SamplerAddressMode::eRepeat,
      .addressModeV = vk::SamplerAddressMode::eRepeat,
      .addressModeW = vk::SamplerAddressMode::eRepeat,
      .mipLodBias = 0.0F,
      .anisotropyEnable = vk::True,
      .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
      .compareEnable = vk::False,
      .compareOp = vk::CompareOp::eAlways,
      .minLod = 0.0F,
      .maxLod = vk::LodClampNone,
      .borderColor = vk::BorderColor::eIntOpaqueBlack,
      .unnormalizedCoordinates = vk::False,
    };

    return UTILS_VK(device_.device().createSampler(sampler_create_info),
      ^^vk::raii::Device::createSampler)
      .transform([ this ](vk::raii::Sampler&& sampler) -> void
        { texture_sampler_ = std::move(sampler); });
  }

  using buffer_memory_pair =
    std::pair<vk::raii::Buffer, vk::raii::DeviceMemory>;

  auto
  create_buffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
    vk::MemoryPropertyFlags properties)
    -> std::expected<buffer_memory_pair, vkpp::error_t>
  {
    vk::BufferCreateInfo buffer_create_info {
      .size = size,
      .usage = usage,
      .sharingMode = vk::SharingMode::eExclusive,
    };

    vk::raii::Buffer temp_buffer { nullptr };
    vk::raii::DeviceMemory temp_memory { nullptr };
    vk::DeviceSize memory_requirements_size; // NOLINT

    return UTILS_VK(device_.device().createBuffer(buffer_create_info),
      ^^vk::raii::Device::createBuffer)
      .and_then(
        [ &, this, properties ](vk::raii::Buffer&& buffer)
        {
          temp_buffer = std::move(buffer);
          const auto memory_requirements = temp_buffer.getMemoryRequirements();
          memory_requirements_size = memory_requirements.size;
          return find_memory_type(
            memory_requirements.memoryTypeBits, properties);
        })
      .and_then(
        [ &, this ](std::uint32_t memory_type)
        {
          vk::MemoryAllocateInfo memory_allocate_info {
            .allocationSize = memory_requirements_size,
            .memoryTypeIndex = memory_type,
          };

          return UTILS_VK(device_.device().allocateMemory(memory_allocate_info),
            ^^vk::raii::Device::allocateMemory);
        })
      .and_then(
        [ & ](vk::raii::DeviceMemory&& memory)
        {
          temp_memory = std::move(memory);
          return UTILS_VK(temp_buffer.bindMemory(*temp_memory, 0ULL),
            ^^vk::raii::Buffer::bindMemory);
        })
      .transform(
        [ & ]
        {
          return std::pair { std::move(temp_buffer), std::move(temp_memory) };
        });
  }

  // TODO: Create new command pool for copying (for short-lived buffers), with
  // vk::CommandPoolCreateFlagBits::eTransient
  auto
  copy_buffer(vk::raii::Buffer& source, vk::raii::Buffer& destination,
    vk::DeviceSize size) -> std::expected<void, vkpp::error_t>
  {
    vkpp::single_time_submit sts {
      command_pool_,
      device_.device(),
      device_.graphics_queue(),
    };

    return sts.begin().and_then(
      [ &, this ](vk::raii::CommandBuffer* command_buffer)
        -> std::expected<void, vkpp::error_t>
      {
        command_buffer->copyBuffer(*source, *destination,
          vk::BufferCopy {
            .srcOffset = 0UZ,
            .dstOffset = 0UZ,
            .size = size,
          });
        return sts.end_and_submit();
      });
  }

  auto
  create_vertex_buffer() -> std::expected<void, vkpp::error_t>
  {
    auto buffer_size = std::span { vertices_ }.size_bytes();
    vk::raii::Buffer staging_buffer { nullptr };
    vk::raii::DeviceMemory staging_buffer_memory { nullptr };

    return create_buffer(buffer_size, vk::BufferUsageFlagBits::eTransferSrc,
      vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent)
      .and_then(
        [ &, buffer_size ](
          buffer_memory_pair&& pair) -> std::expected<void*, vkpp::error_t>
        {
          std::tie(staging_buffer, staging_buffer_memory) = std::move(pair);
          return UTILS_VK(staging_buffer_memory.mapMemory(0ULL, buffer_size),
            ^^vk::raii::DeviceMemory::mapMemory);
        })
      .and_then(
        [ &, this, buffer_size ](void* data_staging)
          -> std::expected<buffer_memory_pair, vkpp::error_t>
        {
          std::memcpy(data_staging, vertices_.data(), buffer_size);
          staging_buffer_memory.unmapMemory();
          return create_buffer(buffer_size,
            vk::BufferUsageFlagBits::eVertexBuffer |
              vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
        })
      .and_then(
        [ &, this, buffer_size ](
          buffer_memory_pair&& pair) -> std::expected<void, vkpp::error_t>
        {
          std::tie(vertex_buffer_, vertex_buffer_memory_) = std::move(pair);
          return copy_buffer(staging_buffer, vertex_buffer_, buffer_size);
        });
  }

  // TODO: only data and buffers change - maybe some abstraction:
  // create_buffer(...)
  auto
  create_index_buffer() -> std::expected<void, vkpp::error_t>
  {
    auto buffer_size = std::span { indices_ }.size_bytes();
    vk::raii::Buffer staging_buffer { nullptr };
    vk::raii::DeviceMemory staging_buffer_memory { nullptr };

    return create_buffer(buffer_size, vk::BufferUsageFlagBits::eTransferSrc,
      vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent)
      .and_then(
        [ &, buffer_size ](
          buffer_memory_pair&& pair) -> std::expected<void*, vkpp::error_t>
        {
          std::tie(staging_buffer, staging_buffer_memory) = std::move(pair);
          return UTILS_VK(staging_buffer_memory.mapMemory(0ULL, buffer_size),
            ^^vk::raii::DeviceMemory::mapMemory);
        })
      .and_then(
        [ &, this, buffer_size ](void* data_staging)
          -> std::expected<buffer_memory_pair, vkpp::error_t>
        {
          std::memcpy(data_staging, indices_.data(), buffer_size);
          staging_buffer_memory.unmapMemory();
          return create_buffer(buffer_size,
            vk::BufferUsageFlagBits::eIndexBuffer |
              vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
        })
      .and_then(
        [ &, this, buffer_size ](
          buffer_memory_pair&& pair) -> std::expected<void, vkpp::error_t>
        {
          std::tie(index_buffer_, index_buffer_memory_) = std::move(pair);
          return copy_buffer(staging_buffer, index_buffer_, buffer_size);
        });
  }

  auto
  create_uniform_buffers() -> std::expected<void, vkpp::error_t>
  {
    uniform_buffers_.reserve(max_frames_in_flight);
    uniform_buffers_memory_.reserve(max_frames_in_flight);
    uniform_buffers_mapped_.reserve(max_frames_in_flight);
    for (auto _ : std::views::iota(0U, max_frames_in_flight))
    {
      static constexpr vk::DeviceSize buffer_size =
        sizeof(uniform_buffer_object);
      vk::raii::Buffer buffer { nullptr };
      vk::raii::DeviceMemory buffer_memory { nullptr };

      const auto result =
        create_buffer(buffer_size, vk::BufferUsageFlagBits::eUniformBuffer,
          vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent)
          .and_then(
            [ & ](
              buffer_memory_pair&& pair) -> std::expected<void*, vkpp::error_t>
            {
              uniform_buffers_.emplace_back(std::move(pair.first));
              uniform_buffers_memory_.emplace_back(std::move(pair).second);
              return UTILS_VK(
                uniform_buffers_memory_.back().mapMemory(0ULL, buffer_size),
                ^^vk::raii::DeviceMemory::mapMemory);
            })
          .transform([ this ](void* mapped_memory) -> void
            { uniform_buffers_mapped_.emplace_back(mapped_memory); });
      if (!result) { return result; }
    }
    return {};
  }

  auto
  update_uniform_buffer(std::uint32_t current_image)
  {
    static auto start_time = std::chrono::high_resolution_clock::now();
    auto current_time = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration<float>(current_time - start_time).count();
    static constexpr auto degrees { glm::radians(90.0F) };
    static constexpr auto camera_position { glm::vec3 { 2.0F, 2.0F, 2.0F } };
    static constexpr auto target { glm::vec3 { 0.0F, 0.0F, 0.0F } };
    static constexpr auto up { glm::vec3 { 0.0F, 0.0F, 1.0F } };
    static constexpr auto fov_vertical { glm::radians(45.0F) };
    static const auto aspect_ratio { //
      static_cast<float>(swap_chain_.extent().width) /
      static_cast<float>(swap_chain_.extent().height)
    };
    static constexpr auto near_plane { 0.1F };
    static constexpr auto far_plane { 10.0F };

    uniform_buffer_object ubo {
      .model = glm::gtc::rotate(
        glm::mat4 { 1.0F }, time * degrees, glm::vec3 { 0.0F, 0.0F, 1.0F }),
      .view = glm::gtc::lookAt(camera_position, target, up),
      .projection = glm::gtc::perspective(
        fov_vertical, aspect_ratio, near_plane, far_plane),
    };
    ubo.projection[ 1 ][ 1 ] *= -1;

    std::memcpy(uniform_buffers_mapped_[ current_image ], &ubo, sizeof(ubo));
  }

  auto
  create_descriptor_pool() -> std::expected<void, vkpp::error_t>
  {
    std::array pool_sizes {
      vk::DescriptorPoolSize {
        .type = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = max_frames_in_flight,
      },
      vk::DescriptorPoolSize {
        .type = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = max_frames_in_flight,
      },
    };
    vk::DescriptorPoolCreateInfo descriptor_pool_create_info {
      .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      .maxSets = max_frames_in_flight,
      .poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size()),
      .pPoolSizes = pool_sizes.data(),
    };

    return UTILS_VK(
      device_.device().createDescriptorPool(descriptor_pool_create_info),
      ^^vk::raii::Device::createDescriptorPool)
      .transform([ this ](vk::raii::DescriptorPool&& pool) -> void
        { descriptor_pool_ = std::move(pool); });
  }

  auto
  create_descriptor_sets() -> std::expected<void, vkpp::error_t>
  {
    std::vector layouts(max_frames_in_flight, *descriptor_set_layout_);
    vk::DescriptorSetAllocateInfo descriptor_set_allocate_info {
      .descriptorPool = *descriptor_pool_,
      .descriptorSetCount = max_frames_in_flight,
      .pSetLayouts = layouts.data(),
    };

    return UTILS_VK(
      device_.device().allocateDescriptorSets(descriptor_set_allocate_info),
      ^^vk::raii::Device::allocateDescriptorSets)
      .transform(
        [ this ](std::vector<vk::raii::DescriptorSet>&& sets) -> void
        {
          descriptor_sets_ = std::move(sets);

          for (auto frame_index : std::views::iota(0UZ, max_frames_in_flight))
          {
            vk::DescriptorBufferInfo buffer_info {
              .buffer = uniform_buffers_[ frame_index ],
              .offset = 0U,
              .range = sizeof(uniform_buffer_object),
            };
            vk::DescriptorImageInfo image_info {
              .sampler = texture_sampler_,
              .imageView = texture_image_view_,
              .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            };
            std::array descriptor_set_writes {
              vk::WriteDescriptorSet {
                .dstSet = descriptor_sets_[ frame_index ],
                .dstBinding = 0U,
                .dstArrayElement = 0U,
                .descriptorCount = 1U,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo = &buffer_info,
              },
              vk::WriteDescriptorSet {
                .dstSet = descriptor_sets_[ frame_index ],
                .dstBinding = 1U,
                .dstArrayElement = 0U,
                .descriptorCount = 1U,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &image_info,
              },
            };

            device_.device().updateDescriptorSets(descriptor_set_writes, {});
          }
        });
  }

  auto
  find_memory_type(
    std::uint32_t type_filter, vk::MemoryPropertyFlags properties)
    -> std::expected<std::uint32_t, vkpp::error_t>
  {
    const auto available_properties =
      device_.physical_device().getMemoryProperties();
    const auto memory_types =
      std::views::iota(0U, available_properties.memoryTypeCount);
    auto memory_type_it = std::ranges::find_if(memory_types,
      [ type_filter, properties, &available_properties ](
        std::uint32_t memory_type) -> bool
      {
        return (type_filter & (1U << memory_type)) &&
          (available_properties.memoryTypes[ memory_type ].propertyFlags &
            properties) == properties;
      });
    if (memory_type_it == memory_types.end())
    {
      return std::unexpected {
        vkpp::app_error {
          .kind = vkpp::app_error_kind::no_memory_type,
          .detail = "Failed to find suitable memory type"sv,
        },
      };
    }

    return *memory_type_it;
  }

  auto
  create_command_buffers() -> std::expected<void, vkpp::error_t>
  {
    vk::CommandBufferAllocateInfo command_buffer_acclocate_info {
      .commandPool = *command_pool_.handle(),
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = max_frames_in_flight,
    };

    return UTILS_VK(
      device_.device().allocateCommandBuffers(command_buffer_acclocate_info),
      ^^vk::raii::Device::allocateCommandBuffers)
      .transform(
        [ this ](std::vector<vk::raii::CommandBuffer>&& command_buffers) -> void
        { command_buffers_ = std::move(command_buffers); });
  }

  auto
  record_command_buffer(std::uint32_t image_index)
    -> std::expected<void, vkpp::error_t>
  {
    const auto& command_buffer = command_buffers_[ frame_index_ ];
    return UTILS_VK(command_buffer.begin({}), ^^vk::raii::CommandBuffer::begin)
      .transform(
        [ this, image_index, &command_buffer ]() -> void
        {
          transition_image_layout(swap_chain_.images()[ image_index ],
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal, {},
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::ImageAspectFlagBits::eColor);

          transition_image_layout(swap_chain_.color().image(),
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal, {},
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::ImageAspectFlagBits::eColor);

          transition_image_layout(swap_chain_.depth().image(),
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eDepthAttachmentOptimal,
            vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            vk::PipelineStageFlagBits2::eEarlyFragmentTests |
              vk::PipelineStageFlagBits2::eLateFragmentTests,
            vk::PipelineStageFlagBits2::eEarlyFragmentTests |
              vk::PipelineStageFlagBits2::eLateFragmentTests,
            vk::ImageAspectFlagBits::eDepth);

          vk::ClearValue clear_color { vk::ClearColorValue {
            0.0F,
            0.0F,
            0.0F,
            1.0F,
          } };
          vk::RenderingAttachmentInfo color_attachment_info {
            .imageView = *swap_chain_.color().view(),
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .resolveMode = vk::ResolveModeFlagBits::eAverage,
            .resolveImageView = swap_chain_.image_view(image_index),
            .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = clear_color,
          };

          vk::ClearValue clear_depth { vk::ClearDepthStencilValue {
            .depth = 1.0F,
            .stencil = 0,
          } };
          vk::RenderingAttachmentInfo depth_attachment_info {
            .imageView = *swap_chain_.depth().view(),
            .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eDontCare,
            .clearValue = clear_depth,
          };

          vk::RenderingInfo rendering_info {
            .renderArea =
              vk::Rect2D {
                .offset = { .x = 0, .y = 0 },
                .extent = swap_chain_.extent(),
              },
            .layerCount = 1U,
            .colorAttachmentCount = 1U,
            .pColorAttachments = &color_attachment_info,
            .pDepthAttachment = &depth_attachment_info,
          };
          command_buffer.beginRendering(rendering_info);
          command_buffer.bindPipeline(
            vk::PipelineBindPoint::eGraphics, *graphics_pipeline_);
          command_buffer.setViewport(0U,
            vk::Viewport {
              .x = 0.0F,
              .y = 0.0F,
              .width = static_cast<float>(swap_chain_.extent().width),
              .height = static_cast<float>(swap_chain_.extent().height),
              .minDepth = 0.0F,
              .maxDepth = 1.0F,
            });
          command_buffer.setScissor(0U,
            vk::Rect2D {
              .offset = vk::Offset2D { .x = 0, .y = 0 },
              .extent = swap_chain_.extent(),
            });
          command_buffer.bindVertexBuffers(0U, *vertex_buffer_, { 0UZ });
          command_buffer.bindIndexBuffer(*index_buffer_, 0UZ,
            vk::IndexTypeValue<decltype(indices_)::value_type>::value);
          command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
            pipeline_layout_, 0U, *descriptor_sets_[ frame_index_ ], nullptr);
          command_buffer.drawIndexed(
            static_cast<std::uint32_t>(indices_.size()), 1U, 0U, 0U, 0U);
          command_buffer.endRendering();

          transition_image_layout(swap_chain_.images()[ image_index ],
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::ePresentSrcKHR,
            vk::AccessFlagBits2::eColorAttachmentWrite, {},
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eBottomOfPipe,
            vk::ImageAspectFlagBits::eColor);
        })
      .and_then(
        [ &command_buffer ]() -> std::expected<void, vkpp::error_t>
        {
          return UTILS_VK(command_buffer.end(), ^^vk::raii::CommandBuffer::end);
        });
  }

  void
  transition_image_layout(vk::Image image, vk::ImageLayout old_layout,
    vk::ImageLayout new_layout, vk::AccessFlags2 source_access_mask,
    vk::AccessFlags2 destination_access_mask,
    vk::PipelineStageFlags2 source_stage_mask,
    vk::PipelineStageFlags2 destination_stage_mask,
    vk::ImageAspectFlags image_aspect_flags)
  {
    vk::ImageMemoryBarrier2 memory_barrier {
        .srcStageMask = source_stage_mask,
        .srcAccessMask = source_access_mask,
        .dstStageMask = destination_stage_mask,
        .dstAccessMask = destination_access_mask,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = image,
        .subresourceRange = {
          .aspectMask = image_aspect_flags,
          .baseMipLevel = 0U,
          .levelCount = 1U,
          .baseArrayLayer = 0U,
          .layerCount = 1U,
        },
      };
    vk::DependencyInfo dependency_info {
      .dependencyFlags = {},
      .imageMemoryBarrierCount = 1U,
      .pImageMemoryBarriers = &memory_barrier,
    };
    command_buffers_[ frame_index_ ].pipelineBarrier2(dependency_info);
  }

  void
  transition_image_layout(vk::raii::CommandBuffer& command_buffer,
    const vk::raii::Image& image, vk::ImageLayout old_layout,
    vk::ImageLayout new_layout, std::uint32_t mip_levels)
  {
    vk::ImageMemoryBarrier barrier {
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = image,
        .subresourceRange = { .aspectMask = vk::ImageAspectFlagBits::eColor,
          .levelCount = mip_levels,
          .layerCount = 1U, },
      };

    vk::PipelineStageFlags source_stage;
    vk::PipelineStageFlags destination_stage;

    if (old_layout == vk::ImageLayout::eUndefined &&
      new_layout == vk::ImageLayout::eTransferDstOptimal)
    {
      barrier.srcAccessMask = {};
      barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
      source_stage = vk::PipelineStageFlagBits::eTopOfPipe;
      destination_stage = vk::PipelineStageFlagBits::eTransfer;
    }
    else if (old_layout == vk::ImageLayout::eTransferDstOptimal &&
      new_layout == vk::ImageLayout::eShaderReadOnlyOptimal)
    {
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
      barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
      source_stage = vk::PipelineStageFlagBits::eTransfer;
      destination_stage = vk::PipelineStageFlagBits::eFragmentShader;
    }

    command_buffer.pipelineBarrier(
      source_stage, destination_stage, {}, {}, nullptr, barrier);
  }

  auto
  generate_mipmaps(vk::raii::CommandBuffer& command_buffer,
    vk::raii::Image& image, vk::Format format, std::int32_t texture_width,
    std::int32_t texture_height, std::uint32_t mip_levels)
    -> std::expected<void, vkpp::error_t>
  {
    auto format_properties =
      device_.physical_device().getFormatProperties(format);
    if (!(format_properties.optimalTilingFeatures &
          vk::FormatFeatureFlagBits::eSampledImageFilterLinear))
    {

      return std::unexpected {
        vkpp::app_error {
          .kind = vkpp::app_error_kind::no_supported_format,
          .detail = "Texture image format does not support linear blitting"sv,
        },
      };
    }

    vk::ImageMemoryBarrier barrier {
      .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
      .dstAccessMask = vk::AccessFlagBits::eTransferRead,
      .oldLayout = vk::ImageLayout::eTransferDstOptimal,
      .newLayout = vk::ImageLayout::eTransferSrcOptimal,
      .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
      .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
      .image = *image,
      .subresourceRange {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .levelCount = 1U,
        .baseArrayLayer = 0U,
        .layerCount = 1U,
      },
    };

    auto mip_width = texture_width;
    auto mip_height = texture_height;

    for (auto mip_level : std::views::iota(1U, mip_levels_))
    {
      barrier.subresourceRange.baseMipLevel = mip_level - 1U;
      barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
      barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
      barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

      command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, barrier);

      std::array src_offsets {
        vk::Offset3D { .x = 0, .y = 0, .z = 0 },
        vk::Offset3D { .x = mip_width, .y = mip_height, .z = 1 },
      };
      std::array dst_offsets {
        vk::Offset3D { .x = 0, .y = 0, .z = 0 },
        vk::Offset3D {
          .x = mip_width > 1 ? mip_width / 2 : 1,
          .y = mip_height > 1 ? mip_height / 2 : 1,
          .z = 1,
        },
      };
      vk::ImageBlit blit {
                .srcSubresource = {
                  .aspectMask=vk::ImageAspectFlagBits::eColor,
                  .mipLevel=mip_level - 1U,
                  .baseArrayLayer=0,
                  .layerCount=1,
                },
                .srcOffsets = src_offsets,
                .dstSubresource = {
                  .aspectMask=vk::ImageAspectFlagBits::eColor,
                  .mipLevel=mip_level,
                  .baseArrayLayer=0,
                  .layerCount=1,
                },
                .dstOffsets = dst_offsets,
              };

      command_buffer.blitImage(image, vk::ImageLayout::eTransferSrcOptimal,
        image, vk::ImageLayout::eTransferDstOptimal, { blit },
        vk::Filter::eLinear);

      barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
      barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
      barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

      command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);

      if (mip_width > 1) { mip_width /= 2; }
      if (mip_height > 1) { mip_height /= 2; }
    }

    barrier.subresourceRange.baseMipLevel = mip_levels - 1U;
    barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
      vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);
    return {};
  }

  void
  copy_buffer_to_image(vk::raii::CommandBuffer& command_buffer,
    const vk::raii::Buffer& buffer, vk::raii::Image& image, std::uint32_t width,
    std::uint32_t height)
  {
    vk::BufferImageCopy region {
        .bufferOffset = 0UZ,
        .bufferRowLength = 0U,
        .bufferImageHeight = 0U,
        .imageSubresource = {
          .aspectMask = vk::ImageAspectFlagBits::eColor,
          .mipLevel = 0U,
          .baseArrayLayer = 0U,
          .layerCount = 1U,
        },
        .imageOffset = {
          .x = 0,
          .y = 0,
          .z = 0,
        },
        .imageExtent = {
          .width=width,
          .height=height,
          .depth=1U,
        },
      };

    command_buffer.copyBufferToImage(
      buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
  }

  auto
  suspend_rendering() -> std::expected<void, vkpp::error_t>
  {
    if (frame_rendering_state_ == frame_rendering_state::suspended &&
      swap_chain_.empty())
    {
      return {};
    }

    frame_rendering_state_ = frame_rendering_state::suspended;
    resized_ = false;

    if (swap_chain_.empty()) { return {}; }

    return UTILS_VK(device_.device().waitIdle(), ^^vk::raii::Device::waitIdle)
      .transform([ this ] { swap_chain_.release(); });
  }

  auto
  recreate_swap_chain() -> std::expected<void, vkpp::error_t>
  {
    return swap_chain_.recreate(device_, instance_.surface(),
      framebuffer_extent_request(),
      [ this ](const vk::SurfaceCapabilitiesKHR& capabilities,
        vk::Extent2D framebuffer)
      { return choose_swap_extent(capabilities, framebuffer); });
  }

  auto
  query_presentability() -> std::expected<vkpp::presentability, vkpp::error_t>
  {
    return vkpp::swapchain::query_presentability(device_, instance_.surface(),
      framebuffer_extent_request(),
      [ this ](const vk::SurfaceCapabilitiesKHR& capabilities,
        vk::Extent2D framebuffer)
      { return choose_swap_extent(capabilities, framebuffer); });
  }

  auto
  recreate_or_suspend() -> std::expected<void, vkpp::error_t>
  {
    return query_presentability().and_then(
      [ this ](vkpp::presentability presentability)
        -> std::expected<void, vkpp::error_t>
      {
        if (!presentability.presentable) { return suspend_rendering(); }
        return recreate_swap_chain();
      });
  }

  auto
  draw_frame_resume() -> std::expected<void, vkpp::error_t>
  {
    return recreate_swap_chain().and_then(
      [ this ] -> std::expected<void, vkpp::error_t>
      {
        frame_rendering_state_ = frame_rendering_state::active;
        return draw_frame_active();
      });
  }

  auto
  draw_frame_active() -> std::expected<void, vkpp::error_t>
  {
    if (auto result =
          device_.device().waitForFences(*in_flight_fences_[ frame_index_ ],
            vk::True, std::numeric_limits<std::uint64_t>::max());
      result != vk::Result::eSuccess)
    {
      return std::unexpected {
        vkpp::vk_error {
          .function = "waitForFences",
          .type = "vk::raii::Device",
          .result = result,
        },
      };
    }
    auto [ result, image_index ] = swap_chain_.swap_chain().acquireNextImage(
      std::numeric_limits<std::uint64_t>::max(),
      *present_complete_semaphores_[ frame_index_ ], nullptr);
    if (result == vk::Result::eErrorOutOfDateKHR)
    {
      return recreate_or_suspend();
    }
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
    {
      return std::unexpected {
        vkpp::vk_error {
          .function = "acquireNextImage",
          .type = "vk::raii::SwapchainKHR",
          .result = result,
        },
      };
    }

    update_uniform_buffer(frame_index_);
    return UTILS_VK(
      device_.device().resetFences(*in_flight_fences_[ frame_index_ ]),
      ^^vk::raii::Device::resetFences)
      .and_then(
        [ this ] -> std::expected<void, vkpp::error_t>
        {
          return UTILS_VK(command_buffers_[ frame_index_ ].reset(),
            ^^vk::raii::CommandBuffer::reset);
        })
      .and_then([ this, &image_index ] -> std::expected<void, vkpp::error_t>
        { return record_command_buffer(image_index); })
      .and_then(
        [ this, image_index ] -> std::expected<void, vkpp::error_t>
        {
          vk::PipelineStageFlags wait_destination_stage_mask {
            vk::PipelineStageFlagBits::eColorAttachmentOutput
          };
          const vk::SubmitInfo submit_info {
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*present_complete_semaphores_[ frame_index_ ],
            .pWaitDstStageMask = &wait_destination_stage_mask,
            .commandBufferCount = 1,
            .pCommandBuffers = &*command_buffers_[ frame_index_ ],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &*swap_chain_.render_finished(image_index),
          };
          return UTILS_VK(device_.graphics_queue().submit(
                            submit_info, *in_flight_fences_[ frame_index_ ]),
            ^^vk::raii::Queue::submit);
        })
      .and_then(
        [ this, &image_index, &result ] -> std::expected<void, vkpp::error_t>
        {
          const vk::PresentInfoKHR present_info {
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*swap_chain_.render_finished(image_index),
            .swapchainCount = 1,
            .pSwapchains = &*swap_chain_.swap_chain(),
            .pImageIndices = &image_index,
            .pResults = nullptr,
          };

          result = device_.graphics_queue().presentKHR(present_info);

          if (result == vk::Result::eSuccess)
          {
            ++frame_index_ %= max_frames_in_flight;
            return {};
          }
          if (result == vk::Result::eSuboptimalKHR ||
            result == vk::Result::eErrorOutOfDateKHR || resized_)
          {
            resized_ = false;
            return recreate_or_suspend();
          }
          return std::unexpected {
            vkpp::vk_error {
              .function = "presentKHR",
              .type = "vk::raii::Queue",
              .result = result,
            },
          };
        });
  }

  auto
  draw_frame() -> std::expected<void, vkpp::error_t>
  {
    return query_presentability().and_then(
      [ this ](vkpp::presentability presentability)
        -> std::expected<void, vkpp::error_t>
      {
        if (!presentability.presentable) { return suspend_rendering(); }
        if (frame_rendering_state_ == frame_rendering_state::suspended)
        {
          return draw_frame_resume();
        }
        return draw_frame_active();
      });
  }

  auto
  create_sync_objects() -> std::expected<void, vkpp::error_t>
  {
    for (auto _ : std::views::iota(0U, max_frames_in_flight))
    {
      if (auto error = //
        UTILS_VK(device_.device().createSemaphore({}),
          ^^vk::raii::Device::createSemaphore)
          .transform([ & ](vk::raii::Semaphore&& semaphore) -> void
            { present_complete_semaphores_.push_back(std::move(semaphore)); });
        !error)
      {
        return error;
      }

      if (auto error = UTILS_VK( //
            device_.device().createFence({
              .flags = vk::FenceCreateFlagBits::eSignaled,
            }),
            ^^vk::raii::Device::createFence)
            .transform([ & ](vk::raii::Fence&& fence) -> void
              { in_flight_fences_.push_back(std::move(fence)); });
        !error)
      {
        return error;
      }
    }
    return {};
  }

private:
  enum class frame_rendering_state : std::uint8_t
  {
    active,
    suspended,
  };

private:
  sf::WindowBase window_ {
    sf::VideoMode { { window_width, window_height } },
    "Window_title",
  };
  vkpp::instance_context instance_ {};
  vkpp::device_context device_ {};
  vkpp::swapchain swap_chain_ {};

  vk::raii::PipelineLayout pipeline_layout_ { nullptr };
  vk::raii::Pipeline graphics_pipeline_ { nullptr };

  vkpp::command_pool command_pool_ {};
  std::vector<vk::raii::CommandBuffer> command_buffers_;
  std::vector<vk::raii::Semaphore> present_complete_semaphores_;
  std::vector<vk::raii::Fence> in_flight_fences_;
  std::uint32_t frame_index_ {};

  std::uint32_t mip_levels_ {};

  vk::raii::Image texture_image_ { nullptr };
  vk::raii::DeviceMemory texture_image_memory_ { nullptr };
  vk::raii::ImageView texture_image_view_ { nullptr };
  vk::raii::Sampler texture_sampler_ { nullptr };

  // TODO: https://developer.nvidia.com/vulkan-memory-management suggests to use
  // one vk::raii::Buffer to have more buffers inside, and use offsets
  std::vector<vkpp::vertex> vertices_;
  std::vector<std::uint32_t> indices_;
  vk::raii::Buffer vertex_buffer_ { nullptr };
  vk::raii::DeviceMemory vertex_buffer_memory_ { nullptr };
  vk::raii::Buffer index_buffer_ { nullptr };
  vk::raii::DeviceMemory index_buffer_memory_ { nullptr };
  std::vector<vk::raii::Buffer> uniform_buffers_;
  std::vector<vk::raii::DeviceMemory> uniform_buffers_memory_;
  std::vector<void*> uniform_buffers_mapped_;
  vk::raii::DescriptorSetLayout descriptor_set_layout_ { nullptr };
  vk::raii::DescriptorPool descriptor_pool_ { nullptr };
  std::vector<vk::raii::DescriptorSet> descriptor_sets_;

  bool resized_ { false };
  frame_rendering_state frame_rendering_state_ {
    frame_rendering_state::active
  };
};
} // namespace f1st
