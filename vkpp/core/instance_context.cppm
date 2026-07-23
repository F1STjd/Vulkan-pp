module;
#include <vulkan/vk_platform.h>

#include "error/vk_error_config.hpp"

export module vkpp.instance;

import vulkan;
import std;
import vkpp.error;

namespace vkpp
{
using namespace std::string_view_literals;

export struct instance_create_info
{
  vk::ApplicationInfo app_info {};
  std::span<const char* const> extensions {};
  std::span<const char* const> layers {};
  bool enable_validation { false };
};

export class instance_context
{
public:
  instance_context() = default;

  [[nodiscard]] static auto
  create(const instance_create_info& info)
    -> std::expected<instance_context, error_t>
  {
    instance_context output {};

    const std::vector<const char*> extensions {
      std::from_range,
      info.extensions,
    };

    auto check_layers = [ & ]() -> std::expected<void, error_t>
    {
      if (info.layers.empty()) { return {}; }
      return UTILS_VK(output.context_.enumerateInstanceLayerProperties(),
        ^^vk::raii::Context::enumerateInstanceLayerProperties)
        .and_then(
          [ & ](std::span<const vk::LayerProperties> available)
            -> std::expected<void, error_t>
          {
            const auto missing_it = std::ranges::find_if(info.layers,
              [ & ](const char* required)
              {
                return std::ranges::none_of(available,
                  [ & ](const vk::LayerProperties& property)
                  { return std::strcmp(property.layerName, required) == 0; });
              });
            if (missing_it != info.layers.end())
            {
              return std::unexpected {
                app_error {
                  .kind = app_error_kind::missing_validation_layer,
                  .detail = std::format(
                    "Required validation layer not supported: {}", *missing_it),
                },
              };
            }
            return {};
          });
    };

    auto check_extensions = [ & ]() -> std::expected<void, error_t>
    {
      return UTILS_VK(output.context_.enumerateInstanceExtensionProperties(),
        ^^vk::raii::Context::enumerateInstanceExtensionProperties)
        .and_then(
          [ & ](std::span<const vk::ExtensionProperties> available)
            -> std::expected<void, error_t>
          {
            const auto missing = std::ranges::find_if(extensions,
              [ & ](const char* required)
              {
                return std::ranges::none_of(available,
                  [ & ](const vk::ExtensionProperties& property)
                  {
                    return std::strcmp(property.extensionName, required) == 0;
                  });
              });
            if (missing != extensions.end())
            {
              return std::unexpected {
                app_error {
                  .kind = app_error_kind::missing_instance_extension,
                  .detail =
                    std::format("Missing instance extension: {}", *missing),
                },
              };
            }
            return {};
          });
    };

    return check_layers()
      .and_then([ & ] { return check_extensions(); })
      .and_then(
        [ & ]
        {
          const vk::InstanceCreateInfo create_info {
            .pApplicationInfo = &info.app_info,
            .enabledLayerCount = static_cast<std::uint32_t>(info.layers.size()),
            .ppEnabledLayerNames = info.layers.data(),
            .enabledExtensionCount =
              static_cast<std::uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data(),
          };
          return UTILS_VK(output.context_.createInstance(create_info),
            ^^vk::raii::Context::createInstance);
        })
      .and_then(
        [ & ](vk::raii::Instance&& instance)
          -> std::expected<instance_context, error_t>
        {
          output.instance_ = std::move(instance);
          if (!info.enable_validation) { return std::move(output); }
          return setup_debug_messenger(output).transform(
            [ & ] { return std::move(output); });
        });
  }

  void
  adopt_surface(vk::raii::SurfaceKHR&& surface)
  { surface_ = std::move(surface); }

  [[nodiscard]]
  auto
  instance() const -> const vk::raii::Instance&
  { return instance_; }

  [[nodiscard]]
  auto
  surface() const -> const vk::raii::SurfaceKHR&
  { return surface_; }

private:
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

  static auto
  setup_debug_messenger(instance_context& context)
    -> std::expected<void, vkpp::error_t>
  {
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

    return UTILS_VK(context.instance_.createDebugUtilsMessengerEXT(create_info),
      ^^vk::raii::Instance::createDebugUtilsMessengerEXT)
      .transform(
        [ & ](vk::raii::DebugUtilsMessengerEXT&& debug_messenger) -> void
        { context.debug_messenger_ = std::move(debug_messenger); });
  }

  vk::raii::Context context_;
  vk::raii::Instance instance_ { nullptr };
  vk::raii::DebugUtilsMessengerEXT debug_messenger_ { nullptr };
  vk::raii::SurfaceKHR surface_ { nullptr };
};
}; // namespace vkpp
