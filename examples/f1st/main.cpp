#include <SFML/Window.hpp>

import std;
import vulkan;

import f1st.app;

import vkpp.error;
import vkpp.diagnostics;
import vkpp.instance;
import vkpp.device;
import vkpp.capabilities;

auto
main() -> int
{
  std::array<vkpp::diagnostic_record, 256> diagnostic_storage {};
  vkpp::diagnostic_buffer                  diagnostic_buffer {
    diagnostic_storage,
    vkpp::diagnostic_severity::warning,
  };

  sf::WindowBase window {
    sf::VideoMode {{f1st::window_width, f1st::window_height}},
    "Window title",
  };

  static constexpr vk::ApplicationInfo app_info {
    .pApplicationName   = "f1st",
    .applicationVersion = vk::makeVersion(1, 0, 0),
    .pEngineName        = "No Engine",
    .engineVersion      = vk::makeVersion(1, 0, 0),
    .apiVersion         = vk::ApiVersion14,
  };
  const auto extensions = f1st::required_instance_extensions();
  auto       instance   = vkpp::instance_context::create({
    .app_info          = app_info,
    .extensions        = extensions,
    .layers            = f1st::validation_layers,
    .enable_validation = f1st::enable_validation_layers,
    .diagnostics       = diagnostic_buffer,
  });
  if (!instance) { return EXIT_FAILURE; }

  VkSurfaceKHR raw_surface {};
  if (!window.createVulkanSurface(*instance->instance(), raw_surface))
  {
    return EXIT_FAILURE;
  }
  instance->adopt_surface(
    vk::raii::SurfaceKHR(instance->instance(), raw_surface));

  auto device = vkpp::device_context::create(
    *instance,
    {
      .profiles                   = f1st::device_profiles,
      .extra_extensions           = f1st::extra_device_extensions,
      .min_api_version            = vk::ApiVersion14,
      .require_present            = true,
      .request_dedicated_transfer = true,
      .rank                       = {.prefer_discrete = true},
    },
    diagnostic_buffer);
  if (!device) { return EXIT_FAILURE; }

  const auto result = vkpp::dispatch_descriptor_backend(
    device->selected_capabilities(),
    [ & ]<vkpp::descriptor_table_backend Backend>()
      -> std::expected<void, vkpp::error_t> {
      f1st::app_session<Backend> session;
      return session
        .init(std::move(window), std::move(*instance), std::move(*device))
        .and_then([ & ] { return session.run(); });
    });
  return result ? EXIT_SUCCESS : EXIT_FAILURE;
}
