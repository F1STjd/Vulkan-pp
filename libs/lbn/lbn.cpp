module;

#include "contracts_config.hpp"

#include <SFML/Window.hpp>
#include <SFML/Window/Vulkan.hpp>
#include <vulkan/vk_platform.h>

module lbn;

import std;
import vulkan;

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

void
app::run()
{
  const auto result = init_vulkan().and_then(
    [ this ]() -> std::expected<void, std::string>
    {
      main_loop();
      return {};
    });

  if (!result) { std::println(stderr, "{}", result.error()); }
}

auto
app::init_vulkan() -> std::expected<void, std::string>
{
  return create_instance() //
    .and_then([ this ] -> std::expected<void, std::string>
      { return setup_debug_messenger(); })
    .and_then([ this ] -> std::expected<void, std::string> //
      { return create_surface(); })
    .and_then([ this ] -> std::expected<void, std::string>
      { return pick_physical_device(); });
}

void
app::main_loop()
{
  while (window_.isOpen())
  {
    while (const auto event = window_.pollEvent())
    {
      if (event->is<sf::Event::Closed>()) { window_.close(); }
    }
  }
}

auto
app::create_instance() -> std::expected<void, std::string>
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
    .and_then(
      [ & ](vk::raii::Instance&& instance) -> std::expected<void, std::string>
      {
        instance_ = std::move(instance);
        return {};
      });
}

auto
app::check_window_vulkan_extensions_support(
  std::span<const char* const> instance_extensions)
  -> std::expected<void, std::string>
{
  return context_.enumerateInstanceExtensionProperties()
    .transform_error(
      [](vk::Result result) -> std::string { return vk::to_string(result); })
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
app::check_validation_layer_support(
  std::span<const char* const> required_layers)
  -> std::expected<void, std::string>
{
  return context_.enumerateInstanceLayerProperties()
    .transform_error(
      [](vk::Result result) -> std::string { return vk::to_string(result); })
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
                return std::strcmp(layer_extension.layerName, required_layer) ==
                  0;
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
app::get_required_instance_extensions() -> std::vector<const char*>
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

VKAPI_ATTR auto VKAPI_CALL
app::debug_callback(
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
app::setup_debug_messenger() -> std::expected<void, std::string>
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
    .and_then(
      [ this ](vk::raii::DebugUtilsMessengerEXT debug_messenger)
        -> std::expected<void, std::string>
      {
        debug_messenger_ = std::move(debug_messenger);
        return {};
      });
}

auto
app::create_surface() -> std::expected<void, std::string>
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
app::pick_physical_device() -> std::expected<void, std::string>
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
app::is_device_suitable(const vk::raii::PhysicalDevice& physical_device) -> bool
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
        [ & ](
          std::span<const vk::ExtensionProperties> available_device_extensions)
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
app::create_logical_device() -> std::expected<void, std::string>
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
  const auto graphics_qf_index = *graphics_qf_index_it;

  static constexpr float graphics_queue_priority { 0.5F };
  vk::DeviceQueueCreateInfo device_queue_create_info {
    .queueFamilyIndex = graphics_qf_index,
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
    .and_then(
      [ this, graphics_qf_index ](
        vk::raii::Device&& device) -> std::expected<void, std::string>
      {
        device_ = std::move(device);
        graphics_queue_ = device_.getQueue(graphics_qf_index, 0);
        return {};
      });
}
} // namespace lbn
