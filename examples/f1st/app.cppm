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
import vkpp.texture;
import vkpp.barrier;
import vkpp.pipeline;
import vkpp.descriptor;

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

constexpr std::size_t max_frames_in_flight { 2UZ };
static_assert(max_frames_in_flight > 0,
  "variable % max_frames_in_flight is used later, so being 0 is UB");

// global?
static constexpr std::array k_set0_bindings {
  vkpp::descriptor_set_layout_binding {
    .binding = 0U,
    .type = vk::DescriptorType::eUniformBuffer,
    .count = 1U,
    .stages = vk::ShaderStageFlagBits::eVertex,
  },
  vkpp::descriptor_set_layout_binding {
    .binding = 1U,
    .type = vk::DescriptorType::eCombinedImageSampler,
    .count = 1U,
    .stages = vk::ShaderStageFlagBits::eFragment,
  },
};

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
      .and_then(std::bind_front(&app::create_frames, this))
      .and_then(std::bind_front(&app::create_descriptor_set_layout, this))
      .and_then(std::bind_front(&app::create_graphics_pipeline, this))
      .and_then(std::bind_front(&app::create_texture_image, this))
      .and_then([ this ] { return vkpp::load_model_obj(vertices_, indices_); })
      .and_then(std::bind_front(&app::create_vertex_buffer, this))
      .and_then(std::bind_front(&app::create_index_buffer, this))
      .and_then(std::bind_front(&app::create_descriptor_pool, this))
      .and_then(std::bind_front(&app::create_descriptor_sets, this));
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
    return vkpp::make_descriptor_set_layout(device_.device(), k_set0_bindings)
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
    return vkpp::command_pool::create(
      device_.device(), device_.graphics_qf_index())
      .transform([ this ](vkpp::command_pool&& pool)
        { command_pool_ = std::move(pool); });
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
    return vkpp::load_host_image_rgba8(vkpp::texture_path)
      .and_then(
        [ &, this ](const vkpp::host_image& host_texture)
        {
          return vkpp::make_texture({
            .device = &device_,
            .pool = &command_pool_,
            .pixels = host_texture.pixels,
            .extent = host_texture.extent,
            .format = host_texture.format,
            .mip_levels = host_texture.mip_levels_present,
            .mip_policy = host_texture.suggested_mip_policy,
            .sampler = {},
          });
        })
      .transform([ this ](vkpp::texture<>&& texture) -> void
        { texture_ = std::move(texture); });
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
  create_vertex_buffer() -> std::expected<void, vkpp::error_t>
  {
    return vkpp::upload_device_local_buffer(
      {
        .device = &device_,
        .pool = &command_pool_,
        .bytes = std::as_bytes(std::span { vertices_ }),
        .gpu_usage = vk::BufferUsageFlagBits::eVertexBuffer,
      })
      .transform([ this ](vkpp::buffer_resource<>&& buffer) -> void
        { vertex_buffer_ = std::move(buffer); });
  }

  auto
  create_index_buffer() -> std::expected<void, vkpp::error_t>
  {
    return vkpp::upload_device_local_buffer(
      {
        .device = &device_,
        .pool = &command_pool_,
        .bytes = std::as_bytes(std::span { indices_ }),
        .gpu_usage = vk::BufferUsageFlagBits::eIndexBuffer,
      })
      .transform([ this ](vkpp::buffer_resource<>&& buffer) -> void
        { index_buffer_ = std::move(buffer); });
  }

  auto
  create_frames() -> std::expected<void, vkpp::error_t>
  {
    return vkpp::create_frames<max_frames_in_flight>(
      {
        .device = &device_,
        .pool = &command_pool_,
        .ubo_size = sizeof(uniform_buffer_object),
      })
      .transform(
        [ this ](std::array<vkpp::frame, max_frames_in_flight>&& frames) -> void
        { frames_ = std::move(frames); });
  }

  void
  update_uniform_buffer(std::uint32_t current_frame)
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
    frames_[ current_frame ].uniform_buffer.write(&ubo, sizeof(ubo));
  }

  auto
  create_descriptor_pool() -> std::expected<void, vkpp::error_t>
  {
    auto sizes = vkpp::pool_sizes_for(k_set0_bindings, max_frames_in_flight);
    return vkpp::descriptor_pool::create(
      device_.device(), max_frames_in_flight, sizes)
      .transform([ this ](vkpp::descriptor_pool&& pool) -> void
        { descriptor_pool_ = std::move(pool); });
  }

  auto
  create_descriptor_sets() -> std::expected<void, vkpp::error_t>
  {
    return descriptor_pool_
      .allocate(device_.device(), descriptor_set_layout_, max_frames_in_flight)
      .transform(
        [ this ](std::vector<vk::raii::DescriptorSet>&& sets) -> void
        {
          for (auto i : std::views::iota(0UZ, max_frames_in_flight))
          {
            // Ideal: frames_[i].descriptor_set = sets[i].release(); //
            // non-owning
            frames_[ i ].descriptor_set = std::move(sets[ i ]);
            vkpp::write_ubo_and_combined_image(device_.device(),
              *frames_[ i ].descriptor_set,
              frames_[ i ].uniform_buffer.buffer(),
              sizeof(uniform_buffer_object), *texture_.sampler(),
              *texture_.view());
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
  record_command_buffer(std::uint32_t image_index)
    -> std::expected<void, vkpp::error_t>
  {
    const auto& command_buffer = frames_[ frame_index_ ].command_buffer;
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
          command_buffer.bindVertexBuffers(
            0U, vertex_buffer_.buffer(), { 0UZ });
          command_buffer.bindIndexBuffer(index_buffer_.buffer(), 0UZ,
            vk::IndexTypeValue<decltype(indices_)::value_type>::value);
          command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
            pipeline_layout_, 0U, *frames_[ frame_index_ ].descriptor_set,
            nullptr);
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
    frames_[ frame_index_ ].command_buffer.pipelineBarrier2(dependency_info);
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

    for (auto mip_level : std::views::iota(1U, texture_.mip_levels()))
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
    auto& frame = frames_[ frame_index_ ];

    if (auto result = device_.device().waitForFences(*frame.in_flight, vk::True,
          std::numeric_limits<std::uint64_t>::max());
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
      std::numeric_limits<std::uint64_t>::max(), *frame.present_complete,
      nullptr);
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

    return UTILS_VK(device_.device().resetFences(*frame.in_flight),
      ^^vk::raii::Device::resetFences)
      .and_then(
        [ & ] -> std::expected<void, vkpp::error_t>
        {
          return UTILS_VK(
            frame.command_buffer.reset(), ^^vk::raii::CommandBuffer::reset);
        })
      .and_then([ this, image_index ] -> std::expected<void, vkpp::error_t>
        { return record_command_buffer(image_index); })
      .and_then(
        [ this, image_index, &frame ] -> std::expected<void, vkpp::error_t>
        {
          vk::PipelineStageFlags wait_destination_stage_mask {
            vk::PipelineStageFlagBits::eColorAttachmentOutput
          };
          const vk::Semaphore wait_semaphore = *frame.present_complete;
          const vk::CommandBuffer command_buffer = *frame.command_buffer;
          const vk::Semaphore signal_semaphore =
            *swap_chain_.render_finished(image_index);

          const vk::SubmitInfo submit_info {
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &wait_semaphore,
            .pWaitDstStageMask = &wait_destination_stage_mask,
            .commandBufferCount = 1,
            .pCommandBuffers = &command_buffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &signal_semaphore,
          };
          return UTILS_VK(
            device_.graphics_queue().submit(submit_info, *frame.in_flight),
            ^^vk::raii::Queue::submit);
        })
      .and_then(
        [ this, &image_index, &result ] -> std::expected<void, vkpp::error_t>
        {
          const vk::Semaphore wait_semaphore =
            *swap_chain_.render_finished(image_index);
          const vk::SwapchainKHR swapchain = *swap_chain_.swap_chain();

          const vk::PresentInfoKHR present_info {
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &wait_semaphore,
            .swapchainCount = 1,
            .pSwapchains = &swapchain,
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

  vk::raii::DescriptorSetLayout descriptor_set_layout_ { nullptr };
  vkpp::descriptor_pool descriptor_pool_ {};

  vkpp::command_pool command_pool_ {};
  std::array<vkpp::frame, max_frames_in_flight> frames_ {};
  std::uint32_t frame_index_ {};

  vkpp::texture<> texture_ {};

  // TODO: https://developer.nvidia.com/vulkan-memory-management suggests to use
  // one vk::raii::Buffer to have more buffers inside, and use offsets
  std::vector<vkpp::vertex> vertices_;
  std::vector<std::uint32_t> indices_;
  vkpp::buffer_resource<> vertex_buffer_ {};
  vkpp::buffer_resource<> index_buffer_ {};

  bool resized_ { false };
  frame_rendering_state frame_rendering_state_ {
    frame_rendering_state::active
  };
};
} // namespace f1st
