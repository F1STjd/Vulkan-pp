module;

#include "contracts_config.hpp"

#include <SFML/Window.hpp>
#include <SFML/Window/VideoMode.hpp>
#include <vulkan/vk_platform.h>

export module lbn;

import std;
import vulkan;
import load;

namespace
{
template<typename T>
auto
map_vk_error(std::expected<T, vk::Result>&& result)
  -> std::expected<T, std::string>
{
  return std::move(result).transform_error(
    [](vk::Result error) -> std::string { return vk::to_string(error); });
}
} // namespace

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

export class app
{
public:
  void
  run()
  {
    const auto result =
      init_vulkan().transform([ this ]() -> void { main_loop(); });

    if (!result) { std::println(stderr, "{}", result.error()); }
  }

private:
  void
  init_window();

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
        { return create_command_pool(); });
  }

  void
  main_loop()
  {
    while (window_.isOpen())
    {
      while (const auto event = window_.pollEvent())
      {
        if (event->is<sf::Event::Closed>()) { window_.close(); }
      }
    }
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
      .and_then([ & ]() -> std::expected<vk::raii::Instance, std::string>
        { return map_vk_error(context_.createInstance(create_info)); })
      .transform([ & ](vk::raii::Instance&& instance) -> void
        { instance_ = std::move(instance); });
  }

  auto
  check_window_vulkan_extensions_support(
    std::span<const char* const> instance_extensions)
    -> std::expected<void, std::string>
  {
    return map_vk_error(context_.enumerateInstanceExtensionProperties())
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
    return map_vk_error(context_.enumerateInstanceLayerProperties())
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

    return map_vk_error(instance_.createDebugUtilsMessengerEXT(create_info))
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
    return map_vk_error(instance_.enumeratePhysicalDevices())
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

    return map_vk_error(physical_device_.createDevice(device_create_info))
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
      map_vk_error(physical_device_.getSurfaceCapabilitiesKHR(*surface_));
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
      map_vk_error(physical_device_.getSurfaceFormatsKHR(*surface_));
    if (!available_formats)
    {
      return std::expected<void, std::string> {
        std::unexpect,
        std::move(available_formats).error(),
      };
    }
    swap_chain_surface_format_ = choose_swap_surface_format(*available_formats);

    auto available_present_modes =
      map_vk_error(physical_device_.getSurfacePresentModesKHR(*surface_));
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

    return map_vk_error(device_.createSwapchainKHR(swap_chain_create_info))
      .and_then(
        [ this ](vk::raii::SwapchainKHR&& swap_chain)
        {
          swap_chain_ = std::move(swap_chain);
          return map_vk_error(swap_chain_.getImages());
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
        map_vk_error(device_.createImageView(image_view_create_info));
      if (!image_view)
      {
        return std::expected<void, std::string> {
          std::unexpect,
          std::move(image_view).error(),
        };
      }
      swap_chain_image_views_.emplace_back(std::move(*image_view));
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

          auto maybe_layout = map_vk_error(
            device_.createPipelineLayout(pipeline_layout_create_info));
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
          return map_vk_error(device_.createGraphicsPipeline(nullptr,
            pipeline_create_info_chain.get<vk::GraphicsPipelineCreateInfo>()));
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

    return map_vk_error(device_.createShaderModule(shader_module_create_info));
  }

  auto
  create_command_pool() -> std::expected<void, std::string>
  {
    vk::CommandPoolCreateInfo command_pool_create_info {
      .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      .queueFamilyIndex = graphics_qf_index_,
    };

    return map_vk_error(device_.createCommandPool(command_pool_create_info))
      .transform([ this ](vk::raii::CommandPool&& command_pool) -> void
        { command_pool_ = std::move(command_pool); });
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
  vk::raii::CommandBuffer command_buffer_ { nullptr };

  std::uint32_t graphics_qf_index_ { ~0U };
};
} // namespace lbn
