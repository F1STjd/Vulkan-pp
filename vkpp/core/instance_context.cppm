module;

#include <vulkan/vk_platform.h>

export module vkpp.instance;

import vulkan;
import std;

import vkpp.error;
import vkpp.diagnostics;

namespace vkpp
{
using namespace std::string_view_literals;

export struct instance_create_info
{
  vk::ApplicationInfo app_info {};
  std::span<const char* const> extensions {};
  std::span<const char* const> layers {};
  bool enable_validation { false };
  std::optional<vkpp::diagnostic_buffer&> diagnostics {};
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
      return map_vk_error(
        output.context_.enumerateInstanceLayerProperties(), info.diagnostics)
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
                make_app_error(app_error_code::missing_validation_layer),
              };
            }
            return {};
          });
    };

    auto check_extensions = [ & ]() -> std::expected<void, error_t>
    {
      return map_vk_error(
        output.context_.enumerateInstanceExtensionProperties(),
        info.diagnostics)
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
                make_app_error(app_error_code::missing_instance_extension),
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
          return map_vk_error(
            output.context_.createInstance(create_info), info.diagnostics);
        })
      .and_then(
        [ & ](vk::raii::Instance&& instance)
          -> std::expected<instance_context, error_t>
        {
          output.instance_ = std::move(instance);
          if (!info.enable_validation || !info.diagnostics.has_value() ||
              info.diagnostics->threshold() == diagnostic_severity::off)
          {
            return std::move(output);
          }
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
  debug_callback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT type,
    const vk::DebugUtilsMessengerCallbackDataEXT* callback_data,
    void* user_data) -> vk::Bool32
  {
    auto* buffer = static_cast<diagnostic_buffer*>(user_data);
    if (buffer == nullptr || callback_data == nullptr) { return vk::False; }

    const diagnostic_severity level = [ & ] -> diagnostic_severity
    {
      if (severity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eError)
      {
        return diagnostic_severity::error;
      }
      if (severity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
      {
        return diagnostic_severity::warning;
      }
      if (severity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo)
      {
        return diagnostic_severity::info;
      }
      return diagnostic_severity::verbose;
    }();

    buffer->report(level, std::nullopt, std::source_location::current(),
      "validation type={} id={} name={} msg={}", vk::to_string(type),
      callback_data->messageIdNumber,
      callback_data->pMessageIdName != nullptr ? callback_data->pMessageIdName
                                               : "",
      callback_data->pMessage != nullptr ? callback_data->pMessage : "");
    return vk::False;
  }

  static auto
  setup_debug_messenger(instance_context& context)
    -> std::expected<void, vkpp::error_t>
  {
    const auto threshold = context.diagnostics_->threshold();
    vk::DebugUtilsMessageSeverityFlagsEXT message_severity_flags {};
    switch (threshold)
    {
    case diagnostic_severity::error:
      message_severity_flags = vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
      break;
    case diagnostic_severity::warning:
      message_severity_flags =
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
      break;
    case diagnostic_severity::info:
      message_severity_flags = vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
      break;
    case diagnostic_severity::verbose:
      message_severity_flags =
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
      break;
    case diagnostic_severity::off: return {};
    }

    const vk::DebugUtilsMessageTypeFlagsEXT message_type_flags {
      vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
      vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
      vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
    };

    const vk::DebugUtilsMessengerCreateInfoEXT create_info {
      .messageSeverity = message_severity_flags,
      .messageType = message_type_flags,
      .pfnUserCallback = &debug_callback,
      .pUserData = static_cast<void*>(&(*context.diagnostics_)),
    };

    return map_vk_error(
      context.instance_.createDebugUtilsMessengerEXT(create_info),
      context.diagnostics_)
      .transform(
        [ & ](vk::raii::DebugUtilsMessengerEXT&& debug_messenger) -> void
        { context.debug_messenger_ = std::move(debug_messenger); });
  }

  vk::raii::Context context_;
  vk::raii::Instance instance_ { nullptr };
  vk::raii::DebugUtilsMessengerEXT debug_messenger_ { nullptr };
  vk::raii::SurfaceKHR surface_ { nullptr };
  std::optional<vkpp::diagnostic_buffer&> diagnostics_ {};
};
}; // namespace vkpp
