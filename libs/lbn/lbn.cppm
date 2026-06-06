module;

#include "contracts_config.hpp"

#include <SFML/Window.hpp>
#include <SFML/Window/VideoMode.hpp>
#include <vulkan/vk_platform.h>

export module lbn;

import std;
import vulkan;

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
  run();

private:
  void
  init_window();

  auto
  init_vulkan() -> std::expected<void, std::string>;

  void
  main_loop();

private:
  auto
  create_instance() -> std::expected<void, std::string>;

  auto check_window_vulkan_extensions_support(std::span<const char* const>)
    -> std::expected<void, std::string>;

  auto check_validation_layer_support(std::span<const char* const>)
    -> std::expected<void, std::string>;

  auto
  get_required_instance_extensions() -> std::vector<const char*>;

  static VKAPI_ATTR auto VKAPI_CALL
  debug_callback(vk::DebugUtilsMessageSeverityFlagBitsEXT,
    vk::DebugUtilsMessageTypeFlagsEXT,
    const vk::DebugUtilsMessengerCallbackDataEXT*, void*) -> vk::Bool32;

  auto
  setup_debug_messenger() -> std::expected<void, std::string>;

  auto
  create_surface() -> std::expected<void, std::string>;

  auto
  pick_physical_device() -> std::expected<void, std::string>;

  auto
  is_device_suitable(const vk::raii::PhysicalDevice&) -> bool;

  auto
  create_logical_device() -> std::expected<void, std::string>;

  auto
  choose_swap_surface_format(std::span<const vk::SurfaceFormatKHR> formats)
    -> vk::SurfaceFormatKHR PRE(formats.size() > 0);

  auto choose_swap_present_mode(std::span<const vk::PresentModeKHR>)
    -> vk::PresentModeKHR;

  auto
  choose_swap_extent(const vk::SurfaceCapabilitiesKHR&) -> vk::Extent2D;

  auto
  create_swap_chain() -> std::expected<void, std::string>;

  auto
  choose_swap_min_image_count(const vk::SurfaceCapabilitiesKHR&)
    -> std::uint32_t;

  auto
  create_image_views() -> std::expected<void,
    std::string> /* PRE(swap_chain_image_views_.empty()) */;
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
};
} // namespace lbn
