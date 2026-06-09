module;

#include "contracts_config.hpp"
#include "vk_error_config.hpp"

#include <SFML/Window.hpp>
#include <SFML/Window/VideoMode.hpp>
#include <vulkan/vk_platform.h>

export module lbn;

import std;
import vulkan;
import load;
import utils;

namespace lbn
{
constexpr std::uint32_t window_width { 800 };
constexpr std::uint32_t window_height { 600 };

#ifdef NDEBUG
constexpr std::array validation_layers { "VK_LAYER_KHRONOS_validation" };
constexpr bool enable_validation_layers { false };
#else
constexpr std::array<const char*, 0> validation_layers {};
constexpr bool enable_validation_layers { true };
#endif

constexpr std::array required_device_extensions {
  vk::KHRSwapchainExtensionName
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
      [ this ]() -> std::expected<void, std::string> { return main_loop(); });

    if (!result) { std::println(stderr, "{}", result.error()); }
  }

private:
  auto
  init_vulkan() -> std::expected<void, std::string>
  {
    return create_instance() //
      .and_then([ this ] -> std::expected<void, std::string>
        { return setup_debug_messenger(); })
      .and_then([ this ] -> std::expected<void, std::string> //
        { return create_surface(); })
      .and_then([ this ] -> std::expected<void, std::string>
        { return pick_physical_device(); })
      .and_then([ this ] -> std::expected<void, std::string>
        { return create_logical_device(); })
      .and_then([ this ] -> std::expected<void, std::string>
        { return create_swap_chain(); })
      .and_then([ this ] -> std::expected<void, std::string>
        { return create_image_views(); })
      .and_then([ this ] -> std::expected<void, std::string>
        { return create_graphics_pipeline(); })
      .and_then([ this ] -> std::expected<void, std::string>
        { return create_command_pool(); })
      .and_then([ this ] -> std::expected<void, std::string>
        { return create_command_buffers(); })
      .and_then([ this ] -> std::expected<void, std::string>
        { return create_sync_objects(); });
  }

  auto
  main_loop() -> std::expected<void, std::string>
  {
    while (window_.isOpen())
    {
      while (const auto event = window_.pollEvent())
      {
        if (event->is<sf::Event::Closed>()) { window_.close(); }
      }
      if (auto result = draw_frame(); !result)
      {
        return std::expected<void, std::string> {
          std::unexpect,
          std::move(result).error(),
        };
      }
    }
    return UTILS_VK(device_.waitIdle(), ^^vk::raii::Device::waitIdle);
  }

private:
  auto
  create_instance() -> std::expected<void, std::string>
  {
    static constexpr vk::ApplicationInfo app_info {
      .pApplicationName = "App name",
      .applicationVersion = vk::makeVersion(1, 0, 0),
      .pEngineName = "No Engine",
      .engineVersion = vk::makeVersion(1, 0, 0),
      .apiVersion = vk::ApiVersion14,
    };

    const auto instance_extensions = get_required_instance_extensions();

    const vk::InstanceCreateInfo create_info {
      .pApplicationInfo = &app_info,
      .enabledLayerCount = static_cast<std::uint32_t>(validation_layers.size()),
      .ppEnabledLayerNames = validation_layers.data(),
      .enabledExtensionCount =
        static_cast<std::uint32_t>(instance_extensions.size()),
      .ppEnabledExtensionNames = instance_extensions.data(),
    };

    return check_validation_layer_support(validation_layers)
      .and_then(
        [ this, &instance_extensions ]() -> std::expected<void, std::string>
        { return check_window_vulkan_extensions_support(instance_extensions); })
      .and_then(
        [ & ]() -> std::expected<vk::raii::Instance, std::string>
        {
          return UTILS_VK(context_.createInstance(create_info),
            ^^vk::raii::Context::createInstance);
        })
      .transform([ & ](vk::raii::Instance&& instance) -> void
        { instance_ = std::move(instance); });
  }

  auto
  check_window_vulkan_extensions_support(
    std::span<const char* const> instance_extensions)
    -> std::expected<void, std::string>
  {
    return UTILS_VK(context_.enumerateInstanceExtensionProperties(),
      ^^vk::raii::Context::enumerateInstanceExtensionProperties)
      .and_then(
        [ &instance_extensions ](
          std::span<const vk::ExtensionProperties> vulkan_extensions)
          -> std::expected<void, std::string>
        {
          auto unsupported_extension_it =
            std::ranges::find_if(instance_extensions,
              [ &vulkan_extensions ](const auto& instance_extension)
              {
                return std::ranges::none_of(vulkan_extensions,
                  [ instance_extension ](const auto& vulkan_extension)
                  {
                    return std::strcmp(vulkan_extension.extensionName,
                             instance_extension) == 0;
                  });
              });

          if (unsupported_extension_it != instance_extensions.end())
          {
            return std::expected<void, std::string> {
              std::unexpect,
              std::format(
                "Missing window extension: {}", *unsupported_extension_it),
            };
          }
          return {};
        });
  }

  auto
  check_validation_layer_support(std::span<const char* const> required_layers)
    -> std::expected<void, std::string>
  {
    // this pipeline triggers: -Wfree-nonheap-object - report sometime
    return UTILS_VK(context_.enumerateInstanceLayerProperties(),
      ^^vk::raii::Context::enumerateInstanceLayerProperties)
      .and_then(
        [ &required_layers ](
          std::span<const vk::LayerProperties> layer_extensions)
          -> std::expected<void, std::string>
        {
          auto unsupported_layer_it = std::ranges::find_if(required_layers,
            [ &layer_extensions ](const auto& required_layer)
            {
              return std::ranges::none_of(layer_extensions,
                [ &required_layer ](const auto& layer_extension)
                {
                  return std::strcmp(
                           layer_extension.layerName, required_layer) == 0;
                });
            });
          if (unsupported_layer_it != required_layers.end())
          {
            return std::expected<void, std::string> {
              std::unexpect,
              std::format(
                "Required layer nor supported: {}", *unsupported_layer_it),
            };
          }
          return {};
        });
  }

  auto
  get_required_instance_extensions() -> std::vector<const char*>
  {
    const auto& ext_temp = sf::Vulkan::getGraphicsRequiredInstanceExtensions();
    std::vector<const char*> instance_extensions(
      ext_temp.begin(), ext_temp.end());
    if constexpr (enable_validation_layers)
    {
      instance_extensions.push_back(vk::EXTDebugUtilsExtensionName);
    }
    return instance_extensions;
  }

  static VKAPI_ATTR auto VKAPI_CALL
  debug_callback(
    [[maybe_unused]] vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT type,
    const vk::DebugUtilsMessengerCallbackDataEXT* callback_data,
    [[maybe_unused]] void* user_data) -> vk::Bool32
  {
    std::println(std::cerr, "Validation layer:\nType: {}\nMessage: {}",
      vk::to_string(type), callback_data->pMessage);

    return vk::False;
  }

  auto
  setup_debug_messenger() -> std::expected<void, std::string>
  {
    if constexpr (!enable_validation_layers) { return {}; }

    static constexpr vk::DebugUtilsMessageSeverityFlagsEXT
      message_severity_flags {
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
      };

    static constexpr vk::DebugUtilsMessageTypeFlagsEXT message_type_flags {
      vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
      vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
      vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
    };

    static constexpr vk::DebugUtilsMessengerCreateInfoEXT create_info {
      .messageSeverity = message_severity_flags,
      .messageType = message_type_flags,
      .pfnUserCallback = &debug_callback
    };

    return UTILS_VK(instance_.createDebugUtilsMessengerEXT(create_info),
      ^^vk::raii::Instance::createDebugUtilsMessengerEXT)
      .transform(
        [ this ](vk::raii::DebugUtilsMessengerEXT&& debug_messenger) -> void
        { debug_messenger_ = std::move(debug_messenger); });
  }

  auto
  create_surface() -> std::expected<void, std::string>
  {
    VkSurfaceKHR _surface {};
    if (!window_.createVulkanSurface(*instance_, _surface))
    {
      return std::expected<void, std::string> {
        std::unexpect,
        "Faild to create window surface",
      };
    }
    surface_ = vk::raii::SurfaceKHR(instance_, _surface);
    return {};
  }

  auto
  pick_physical_device() -> std::expected<void, std::string>
  {
    return UTILS_VK(instance_.enumeratePhysicalDevices(),
      ^^vk::raii::Instance::enumeratePhysicalDevices)
      .and_then(
        [ this ](std::span<const vk::raii::PhysicalDevice> physical_devices)
          -> std::expected<void, std::string>
        {
          const auto suitable_device_it = std::ranges::find_if(physical_devices,
            [ this ](const vk::raii::PhysicalDevice& device)
            { return is_device_suitable(device); });
          if (suitable_device_it == physical_devices.end())
          {
            return std::expected<void, std::string> {
              std::unexpect,
              "No suitable GPU found",
            };
          }
          physical_device_ = *suitable_device_it;
          return {};
        });
  }

  auto
  is_device_suitable(const vk::raii::PhysicalDevice& physical_device) -> bool
  {
    bool supports_vulkan_13 =
      physical_device.getProperties().apiVersion >= vk::ApiVersion13;

    const auto queue_family_properties =
      physical_device.getQueueFamilyProperties();
    bool supports_graphics = std::ranges::any_of(queue_family_properties,
      [](const auto& qf_property)
      {
        return (static_cast<bool>(
          qf_property.queueFlags & vk::QueueFlagBits::eGraphics));
      });

    auto supports_required_device_extensions =
      physical_device.enumerateDeviceExtensionProperties()
        .transform(
          [ & ](std::span<const vk::ExtensionProperties>
              available_device_extensions)
          {
            return std::ranges::all_of(required_device_extensions,
              [ &available_device_extensions ](
                const auto& required_device_extension)
              {
                return std::ranges::any_of(available_device_extensions,
                  [ required_device_extension ](
                    const auto& available_device_extension)
                  {
                    return strcmp(available_device_extension.extensionName,
                             required_device_extension) == 0;
                  });
              });
          })
        .value_or(false);

    auto features = physical_device.getFeatures2<vk::PhysicalDeviceFeatures2,
      vk::PhysicalDeviceVulkan13Features,
      vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
    bool supports_required_features =
      features.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering ==
        vk::True &&
      features.get<vk::PhysicalDeviceVulkan13Features>().synchronization2 ==
        vk::True &&
      features.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>()
          .extendedDynamicState == vk::True;

    return supports_vulkan_13 && supports_graphics &&
      supports_required_device_extensions && supports_required_features;
  }

  auto
  create_logical_device() -> std::expected<void, std::string>
  {
    const auto qf_properties = physical_device_.getQueueFamilyProperties();
    const auto qf_count = static_cast<std::uint32_t>(qf_properties.size());
    const auto qf_indices = std::views::iota(0U, qf_count);
    const auto graphics_qf_index_it = std::ranges::find_if(qf_indices,
      [ this, &qf_properties ](std::uint32_t i)
      {
        return (qf_properties[ i ].queueFlags & vk::QueueFlagBits::eGraphics) &&
          physical_device_.getSurfaceSupportKHR(i, surface_);
      });
    if (graphics_qf_index_it == std::ranges::end(qf_indices))
    {
      return std::expected<void, std::string> {
        std::unexpect,
        "No queue family with graphics and surface present support found",
      };
    }
    graphics_qf_index_ = *graphics_qf_index_it;

    static constexpr float graphics_queue_priority { 0.5F };
    vk::DeviceQueueCreateInfo device_queue_create_info {
      .queueFamilyIndex = graphics_qf_index_,
      .queueCount = 1,
      .pQueuePriorities = &graphics_queue_priority,
    };

    vk::StructureChain feature_chain {
      vk::PhysicalDeviceFeatures2 {},
      vk::PhysicalDeviceVulkan13Features {
        .synchronization2 = vk::True,
        .dynamicRendering = vk::True,
      },
      vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT {
        .extendedDynamicState = vk::True,
      },
    };

    vk::DeviceCreateInfo device_create_info {
      .pNext = &feature_chain.get<vk::PhysicalDeviceFeatures2>(),
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &device_queue_create_info,
      .enabledExtensionCount =
        static_cast<std::uint32_t>(required_device_extensions.size()),
      .ppEnabledExtensionNames = required_device_extensions.data(),
    };

    return UTILS_VK(physical_device_.createDevice(device_create_info),
      ^^vk::raii::PhysicalDevice::createDevice)
      .transform(
        [ this ](vk::raii::Device&& device) -> void
        {
          device_ = std::move(device);
          graphics_queue_ = device_.getQueue(graphics_qf_index_, 0);
        });
  }

  auto
  choose_swap_surface_format(std::span<const vk::SurfaceFormatKHR> formats)
    -> vk::SurfaceFormatKHR PRE(formats.size() > 0)
  {
    const auto format_it = std::ranges::find_if(formats,
      [](const auto& format) -> bool
      {
        return format.format == vk::Format::eB8G8R8A8Srgb &&
          format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
      });
    return format_it != formats.end() ? *format_it : formats[ 0 ];
  }

  auto
  choose_swap_present_mode(std::span<const vk::PresentModeKHR> present_modes)
    -> vk::PresentModeKHR
  {
    // We look for mailbox present mode (tripple buffering), but there are
    // available some new (maybe better) modes, that were not presented in the
    // tutorial => The revision/study of them needs to be done
    return std::ranges::any_of(present_modes,
             [](vk::PresentModeKHR present_mode) -> bool
             { return present_mode == vk::PresentModeKHR::eMailbox; })
      ? vk::PresentModeKHR::eMailbox
      : vk::PresentModeKHR::eFifo;
  }

  auto
  choose_swap_extent(const vk::SurfaceCapabilitiesKHR& capabilities)
    -> vk::Extent2D
  {
    if (capabilities.currentExtent.width !=
      std::numeric_limits<std::uint32_t>::max())
    {
      return capabilities.currentExtent;
    }
    auto [ width, height ] = window_.getSize();

    return {
      .width = std::clamp(width, capabilities.minImageExtent.width,
        capabilities.maxImageExtent.width),
      .height = std::clamp(height, capabilities.minImageExtent.height,
        capabilities.maxImageExtent.height),
    };
  }

  auto
  create_swap_chain() -> std::expected<void, std::string>
  {
    auto surface_capabilities =
      UTILS_VK(physical_device_.getSurfaceCapabilitiesKHR(*surface_),
        ^^vk::raii::PhysicalDevice::getSurfaceCapabilitiesKHR);
    if (!surface_capabilities)
    {
      return std::expected<void, std::string> {
        std::unexpect,
        std::move(surface_capabilities).error(),
      };
    }
    swap_chain_extent_ = choose_swap_extent(*surface_capabilities);
    const auto min_image_count =
      choose_swap_min_image_count(*surface_capabilities);

    auto available_formats =
      UTILS_VK(physical_device_.getSurfaceFormatsKHR(*surface_),
        ^^vk::raii::PhysicalDevice::getSurfaceFormatsKHR);
    if (!available_formats)
    {
      return std::expected<void, std::string> {
        std::unexpect,
        std::move(available_formats).error(),
      };
    }
    swap_chain_surface_format_ = choose_swap_surface_format(*available_formats);

    auto available_present_modes =
      UTILS_VK(physical_device_.getSurfacePresentModesKHR(*surface_),
        ^^vk::raii::PhysicalDevice::getSurfacePresentModesKHR);
    if (!available_present_modes)
    {
      return std::expected<void, std::string> {
        std::unexpect,
        std::move(available_present_modes).error(),
      };
    }
    const auto present_mode =
      choose_swap_present_mode(*available_present_modes);

    vk::SwapchainCreateInfoKHR swap_chain_create_info {
      .surface = *surface_,
      .minImageCount = min_image_count,
      .imageFormat = swap_chain_surface_format_.format,
      .imageColorSpace = swap_chain_surface_format_.colorSpace,
      .imageExtent = swap_chain_extent_,
      .imageArrayLayers = 1U,
      .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
      .imageSharingMode = vk::SharingMode::eExclusive,
      .preTransform = surface_capabilities->currentTransform,
      .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
      .presentMode = present_mode,
      .clipped = vk::True,
      .oldSwapchain = nullptr,
    };

    return UTILS_VK(device_.createSwapchainKHR(swap_chain_create_info),
      ^^vk::raii::Device::createSwapchainKHR)
      .and_then(
        [ this ](vk::raii::SwapchainKHR&& swap_chain)
        {
          swap_chain_ = std::move(swap_chain);
          return UTILS_VK(
            swap_chain_.getImages(), ^^vk::raii::SwapchainKHR::getImages);
        })
      .transform([ this ](std::vector<vk::Image>&& images) -> void
        { swap_chain_images_ = std::move(images); });
  }

  auto
  choose_swap_min_image_count(
    const vk::SurfaceCapabilitiesKHR& surface_capabilities) -> std::uint32_t
  {
    auto min_image_count = std::max(3U, surface_capabilities.minImageCount);
    if ((surface_capabilities.maxImageCount > 0U) &&
      (surface_capabilities.maxImageCount < min_image_count))
    {
      min_image_count = surface_capabilities.maxImageCount;
    }
    return min_image_count;
  }

  auto
  create_image_views() -> std::expected<void,
    std::string> /* PRE(swap_chain_image_views_.empty()) */
  {
    vk::ImageViewCreateInfo image_view_create_info {
    .viewType = vk::ImageViewType::e2D,
    .format = swap_chain_surface_format_.format,
    .subresourceRange = {
      .aspectMask = vk::ImageAspectFlagBits::eColor,
      .baseMipLevel = 0,
      .levelCount = 1,
      .baseArrayLayer = 0,
      .layerCount = 1,
    },
  };
    swap_chain_image_views_.clear();
    swap_chain_image_views_.reserve(swap_chain_images_.size());
    for (const auto& image : swap_chain_images_)
    {
      image_view_create_info.image = image;
      auto image_view =
        UTILS_VK(device_.createImageView(image_view_create_info),
          ^^vk::raii::Device::createImageView);
      if (!image_view)
      {
        return std::expected<void, std::string> {
          std::unexpect,
          std::move(image_view).error(),
        };
      }
      swap_chain_image_views_.push_back(std::move(*image_view));
    }
    return {};
  }

  auto
  create_graphics_pipeline() -> std::expected<void, std::string>
  {
    return load::read_shader_file(SHADER_DIRECTORY "slang.spv")
      .and_then([ this ](std::span<const char> code)
        { return create_shader_module(code); })
      .and_then(
        [ this ](const vk::raii::ShaderModule& shader_module)
          -> std::expected<vk::raii::Pipeline, std::string>
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

          constexpr vk::PipelineVertexInputStateCreateInfo
            vertex_shader_input_create_info {};
          constexpr vk::PipelineInputAssemblyStateCreateInfo
            input_assembly_create_info {
              .topology = vk::PrimitiveTopology::eTriangleList,
            };
          constexpr vk::PipelineViewportStateCreateInfo
            viewport_state_create_info {
              .viewportCount = 1U,
              .scissorCount = 1U,
            };
          constexpr vk::PipelineRasterizationStateCreateInfo
            rasterizer_create_info {
              .depthClampEnable = vk::False,
              .rasterizerDiscardEnable = vk::False,
              .polygonMode = vk::PolygonMode::eFill,
              .cullMode = vk::CullModeFlagBits::eBack,
              .frontFace = vk::FrontFace::eClockwise,
              .depthBiasEnable = vk::False,
              .lineWidth = 1.0F,
            };
          constexpr vk::PipelineMultisampleStateCreateInfo
            multisampling_create_info {
              .rasterizationSamples = vk::SampleCountFlagBits::e1,
              .sampleShadingEnable = vk::False,
            };

          static constexpr vk::PipelineColorBlendAttachmentState
            color_blend_attachment {
              .blendEnable = vk::False,
              .colorWriteMask = vk::ColorComponentFlagBits::eR |
                vk::ColorComponentFlagBits::eG |
                vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
            };
          constexpr vk::PipelineColorBlendStateCreateInfo
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
          constexpr vk::PipelineDynamicStateCreateInfo dynamic_state {
            .dynamicStateCount =
              static_cast<std::uint32_t>(dynamic_states.size()),
            .pDynamicStates = dynamic_states.data(),
          };

          vk::PipelineLayoutCreateInfo pipeline_layout_create_info {
            .setLayoutCount = 0,
            .pushConstantRangeCount = 0,
          };

          auto maybe_layout =
            UTILS_VK(device_.createPipelineLayout(pipeline_layout_create_info),
              ^^vk::raii::Device::createPipelineLayout);
          if (!maybe_layout)
          {
            return std::expected<vk::raii::Pipeline, std::string> {
              std::unexpect,
              std::move(maybe_layout).error(),
            };
          }
          pipeline_layout_ = std::move(*maybe_layout);

          vk::StructureChain pipeline_create_info_chain {
            vk::GraphicsPipelineCreateInfo {
              .stageCount = 2,
              .pStages = shader_stages.data(),
              .pVertexInputState = &vertex_shader_input_create_info,
              .pInputAssemblyState = &input_assembly_create_info,
              .pViewportState = &viewport_state_create_info,
              .pRasterizationState = &rasterizer_create_info,
              .pMultisampleState = &multisampling_create_info,
              .pColorBlendState = &color_blend_create_info,
              .pDynamicState = &dynamic_state,
              .layout = pipeline_layout_,
              .renderPass = nullptr,
            },
            vk::PipelineRenderingCreateInfo {
              .colorAttachmentCount = 1,
              .pColorAttachmentFormats = &swap_chain_surface_format_.format,
            },
          };
          return UTILS_VK(
            device_.createGraphicsPipeline(nullptr,
              pipeline_create_info_chain.get<vk::GraphicsPipelineCreateInfo>()),
            ^^vk::raii::Device::createGraphicsPipeline);
        })
      .transform([ this ](vk::raii::Pipeline&& pipeline) -> void
        { graphics_pipeline_ = std::move(pipeline); });
  }

  [[nodiscard]] auto
  create_shader_module(std::span<const char> code)
    -> std::expected<vk::raii::ShaderModule, std::string>
  {
    vk::ShaderModuleCreateInfo shader_module_create_info {
      .codeSize = code.size_bytes(),
      .pCode = std::start_lifetime_as<std::uint32_t>(code.data()),
    };

    return UTILS_VK(device_.createShaderModule(shader_module_create_info),
      ^^vk::raii::Device::createShaderModule);
  }

  auto
  create_command_pool() -> std::expected<void, std::string>
  {
    vk::CommandPoolCreateInfo command_pool_create_info {
      .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      .queueFamilyIndex = graphics_qf_index_,
    };

    return UTILS_VK(device_.createCommandPool(command_pool_create_info),
      ^^vk::raii::Device::createCommandPool)
      .transform([ this ](vk::raii::CommandPool&& command_pool) -> void
        { command_pool_ = std::move(command_pool); });
  }

  auto
  create_command_buffers() -> std::expected<void, std::string>
  {
    vk::CommandBufferAllocateInfo command_buffer_acclocate_info {
      .commandPool = command_pool_,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = max_frames_in_flight,
    };

    return UTILS_VK(
      device_.allocateCommandBuffers(command_buffer_acclocate_info),
      ^^vk::raii::Device::allocateCommandBuffers)
      .transform(
        [ this ](std::vector<vk::raii::CommandBuffer>&& command_buffers) -> void
        { command_buffers_ = std::move(command_buffers); });
  }

  auto
  record_command_buffer(std::uint32_t image_index)
    -> std::expected<void, std::string>
  {
    const auto& command_buffer = command_buffers_[ frame_index_ ];
    return UTILS_VK(command_buffer.begin({}), ^^vk::raii::CommandBuffer::begin)
      .transform(
        [ this, image_index, &command_buffer ]() -> void
        {
          transition_image_layout(image_index, vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal, {},
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput);

          vk::ClearValue clear_color { vk::ClearColorValue {
            0.0F, 0.0F, 0.0F, 1.0F } };
          vk::RenderingAttachmentInfo rendering_attachment_info {
            .imageView = swap_chain_image_views_[ image_index ],
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = clear_color,
          };
          vk::RenderingInfo rendering_info {
            .renderArea =
              vk::Rect2D {
                .offset = { .x = 0, .y = 0 },
                .extent = swap_chain_extent_,
              },
            .layerCount = 1U,
            .colorAttachmentCount = 1U,
            .pColorAttachments = &rendering_attachment_info,
          };
          command_buffer.beginRendering(rendering_info);
          command_buffer.bindPipeline(
            vk::PipelineBindPoint::eGraphics, *graphics_pipeline_);
          command_buffer.setViewport(0,
            vk::Viewport {
              .x = 0.0F,
              .y = 0.0F,
              .width = static_cast<float>(swap_chain_extent_.width),
              .height = static_cast<float>(swap_chain_extent_.height),
              .minDepth = 0.0F,
              .maxDepth = 1.0F,
            });
          command_buffer.setScissor(0,
            vk::Rect2D {
              .offset = vk::Offset2D { .x = 0, .y = 0 },
              .extent = swap_chain_extent_,
            });
          command_buffer.draw(3U, 1U, 0U, 0U);
          command_buffer.endRendering();

          transition_image_layout(image_index,
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::ePresentSrcKHR,
            vk::AccessFlagBits2::eColorAttachmentWrite, {},
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eBottomOfPipe);
        })
      .and_then(
        [ &command_buffer ]() -> std::expected<void, std::string>
        {
          return UTILS_VK(command_buffer.end(), ^^vk::raii::CommandBuffer::end);
        });
  }

  void
  transition_image_layout(std::uint32_t image_index, vk::ImageLayout old_layout,
    vk::ImageLayout new_layout, vk::AccessFlags2 source_access_mask,
    vk::AccessFlags2 destination_access_mask,
    vk::PipelineStageFlags2 source_stage_mask,
    vk::PipelineStageFlags2 destination_stage_mask)
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
      .image = swap_chain_images_[ image_index ],
      .subresourceRange = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
      },
    };
    vk::DependencyInfo dependency_info {
      .dependencyFlags = {},
      .imageMemoryBarrierCount = 1U,
      .pImageMemoryBarriers = &memory_barrier,
    };
    command_buffers_[ frame_index_ ].pipelineBarrier2(dependency_info);
  }

  auto
  draw_frame() -> std::expected<void, std::string>
  {
    if (auto result = device_.waitForFences(*in_flight_fences_[ frame_index_ ],
          vk::True, std::numeric_limits<std::uint64_t>::max());
      result != vk::Result::eSuccess)
    {
      return std::expected<void, std::string> { std::unexpect,
        std::format("Failed to wait for fence: {}", vk::to_string(result)) };
    }
    std::uint32_t image_index {};
    return UTILS_VK(device_.resetFences(*in_flight_fences_[ frame_index_ ]),
      ^^vk::raii::Device::resetFences)
      .and_then(
        [ this, &image_index ]() -> std::expected<void, std::string>
        {
          auto [ result, idx_temp ] = swap_chain_.acquireNextImage(
            std::numeric_limits<std::uint64_t>::max(),
            *present_complete_semaphores_[ frame_index_ ], nullptr);
          if (result != vk::Result::eSuccess)
          {
            return std::expected<void, std::string> {
              std::unexpect,
              std::format(
                "Failed to acquire next image from the swap chain: {}",
                vk::to_string(result)),
            };
          }
          image_index = idx_temp;
          return UTILS_VK(command_buffers_[ frame_index_ ].reset(),
            ^^vk::raii::CommandBuffer::reset);
        })
      .and_then([ this, &image_index ]() -> std::expected<void, std::string>
        { return record_command_buffer(image_index); })
      .and_then(
        [ this, image_index ]() -> std::expected<void, std::string>
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
            .pSignalSemaphores = &*render_finished_semaphores_[ image_index ],
          };
          return UTILS_VK(graphics_queue_.submit(
                            submit_info, *in_flight_fences_[ frame_index_ ]),
            ^^vk::raii::Queue::submit);
        })
      .and_then(
        [ this, &image_index ]() -> std::expected<void, std::string>
        {
          const vk::PresentInfoKHR present_info {
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*render_finished_semaphores_[ image_index ],
            .swapchainCount = 1,
            .pSwapchains = &*swap_chain_,
            .pImageIndices = &image_index,
            .pResults = nullptr,
          };

          auto result = graphics_queue_.presentKHR(present_info);
          switch (result)
          {
          case vk::Result::eSuccess:
          {
            ++frame_index_ %= max_frames_in_flight;
            return {};
          }
          default:
          {
            return std::expected<void, std::string> {
              std::unexpect,
              std::format(
                "vk::Qeueue::presentKHR() returned: {}", vk::to_string(result)),
            };
          }
          }
        });
  }

  auto
  create_sync_objects() -> std::expected<void, std::string>
  {
    for (auto _ : std::views::iota(0UZ, swap_chain_images_.size()))
    {
      auto maybe_render_semaphore = UTILS_VK(
        device_.createSemaphore({}), ^^vk::raii::Device::createSemaphore);
      if (!maybe_render_semaphore)
      {
        return std::expected<void, std::string> {
          std::unexpect,
          std::move(maybe_render_semaphore).error(),
        };
      }
      render_finished_semaphores_.push_back(std::move(*maybe_render_semaphore));
    }

    for (auto _ : std::views::iota(0U, max_frames_in_flight))
    {
      auto maybe_present_semaphore = UTILS_VK(
        device_.createSemaphore({}), ^^vk::raii::Device::createSemaphore);
      if (!maybe_present_semaphore)
      {
        return std::expected<void, std::string> {
          std::unexpect,
          std::move(maybe_present_semaphore).error(),
        };
      }
      present_complete_semaphores_.push_back(
        std::move(*maybe_present_semaphore));

      auto maybe_draw_fence = UTILS_VK(
        device_.createFence({ .flags = vk::FenceCreateFlagBits::eSignaled }),
        ^^vk::raii::Device::createFence);
      if (!maybe_draw_fence)
      {
        return std::expected<void, std::string> {
          std::unexpect,
          std::move(maybe_draw_fence).error(),
        };
      }
      in_flight_fences_.push_back(std::move(*maybe_draw_fence));
    }
    return {};
  }

private:
  sf::WindowBase window_ {
    sf::VideoMode { { window_width, window_height } },
    "Window_title",
  };
  vk::raii::Context context_;
  vk::raii::Instance instance_ { nullptr };
  vk::raii::DebugUtilsMessengerEXT debug_messenger_ { nullptr };
  vk::raii::SurfaceKHR surface_ { nullptr };
  vk::raii::PhysicalDevice physical_device_ { nullptr };
  vk::raii::Device device_ { nullptr };
  vk::raii::Queue graphics_queue_ { nullptr };
  vk::raii::SwapchainKHR swap_chain_ { nullptr };
  std::vector<vk::Image> swap_chain_images_;
  vk::SurfaceFormatKHR swap_chain_surface_format_;
  vk::Extent2D swap_chain_extent_;
  std::vector<vk::raii::ImageView> swap_chain_image_views_;
  vk::raii::PipelineLayout pipeline_layout_ { nullptr };
  vk::raii::Pipeline graphics_pipeline_ { nullptr };
  vk::raii::CommandPool command_pool_ { nullptr };
  std::vector<vk::raii::CommandBuffer> command_buffers_;
  std::vector<vk::raii::Semaphore> present_complete_semaphores_;
  std::vector<vk::raii::Semaphore> render_finished_semaphores_;
  std::vector<vk::raii::Fence> in_flight_fences_;
  std::uint32_t graphics_qf_index_ { ~0U };
  std::uint32_t frame_index_ { 0U };
};
} // namespace lbn
