module;

#include "error/vk_error_config.hpp"

export module vkpp.swapchain;

import vkpp.device;
import vkpp.image;
import vkpp.error;

import std;
import vulkan;

namespace vkpp
{
using namespace std::string_view_literals;

export struct extent_request
{
  vk::Extent2D framebuffer_size {};
};

export struct presentability
{
  bool presentable {};
  vk::Extent2D extent {};
};

export class swapchain
{
public:
  [[nodiscard]] static auto
  create(device_context& device, const vk::raii::SurfaceKHR& surface,
    extent_request window,
    std::invocable<const vk::SurfaceCapabilitiesKHR&, vk::Extent2D> auto&&
      choose_extent) -> std::expected<swapchain, error_t>
  {
    swapchain output {};
    surface_build_info build {};

    return UTILS_VK(
      device.physical_device().getSurfaceCapabilitiesKHR(*surface),
      ^^vk::raii::PhysicalDevice::getSurfaceCapabilitiesKHR)
      .and_then(
        [ & ](const vk::SurfaceCapabilitiesKHR& caps)
          -> std::expected<void, error_t>
        {
          build.capabilities = caps;
          build.extent =
            choose_extent(build.capabilities, window.framebuffer_size);
          if (build.extent.width == 0U || build.extent.height == 0U)
          {
            return std::unexpected {
              error_t {
                app_error {
                  .kind = app_error_kind::surface_not_presentable,
                  .detail =
                    "Surface not presentable: chosen swapchain extent is 0x0"sv,
                },
              },
            };
          }
          build.min_image_count =
            choose_swap_min_image_count(build.capabilities);
          build.pre_transform = build.capabilities.currentTransform;
          return {};
        })
      .and_then(
        [ & ] -> std::expected<void, error_t>
        {
          return UTILS_VK(
            device.physical_device().getSurfaceFormatsKHR(*surface),
            ^^vk::raii::PhysicalDevice::getSurfaceFormatsKHR)
            .transform(
              [ & ](std::span<const vk::SurfaceFormatKHR> available_formats)
              {
                build.surface_format =
                  choose_swap_surface_format(available_formats);
              });
        })
      .and_then(
        [ & ] -> std::expected<void, error_t>
        {
          return UTILS_VK(
            device.physical_device().getSurfacePresentModesKHR(*surface),
            ^^vk::raii::PhysicalDevice::getSurfacePresentModesKHR)
            .transform(
              [ & ](std::span<const vk::PresentModeKHR> present_modes)
              {
                build.present_mode = choose_swap_present_mode(present_modes);
              });
        })
      .and_then(
        [ & ] -> std::expected<void, error_t>
        {
          output.surface_format_ = build.surface_format;
          output.extent_ = build.extent;

          vk::SwapchainCreateInfoKHR swap_chain_create_info {
            .surface = *surface,
            .minImageCount = build.min_image_count,
            .imageFormat = output.format(),
            .imageColorSpace = output.surface_format_.colorSpace,
            .imageExtent = output.extent(),
            .imageArrayLayers = 1U,
            .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
            .imageSharingMode = vk::SharingMode::eExclusive,
            .preTransform = build.pre_transform,
            .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode = build.present_mode,
            .clipped = vk::True,
            .oldSwapchain = nullptr,
          };

          return UTILS_VK(
            device.device().createSwapchainKHR(swap_chain_create_info),
            ^^vk::raii::Device::createSwapchainKHR)
            .transform([ & ](vk::raii::SwapchainKHR&& swap_chain)
              { output.swap_chain_ = std::move(swap_chain); });
        })
      .and_then(
        [ & ] -> std::expected<void, error_t>
        {
          return UTILS_VK(
            output.swap_chain_.getImages(), ^^vk::raii::SwapchainKHR::getImages)
            .transform([ & ](std::vector<vk::Image>&& images)
              { output.images_ = std::move(images); });
        })
      .and_then(
        [ & ] -> std::expected<void, error_t>
        {
          output.image_views_.clear();
          output.image_views_.reserve(output.images_.size());
          for (vk::Image image : output.images_)
          {
            auto view = make_image_view<image_kind::resolve>(
              device.device(), image, output.format());
            if (!view) { return std::unexpected { std::move(view).error() }; }
            output.image_views_.push_back(std::move(*view));
          }
          return {};
        })
      .and_then(
        [ & ] -> std::expected<void, error_t>
        {
          return make_image_resource<image_kind::color>(device.allocator(),
            device.device(),
            image_runtime_args {
              .extent = output.extent(),
              .format = output.format(),
              .samples = device.msaa_samples(),
            })
            .transform([ & ](image_resource<>&& color)
              { output.color_ = std::move(color); });
        })
      .and_then(
        [ & ] -> std::expected<void, error_t>
        {
          return find_depth_format(device.physical_device())
            .and_then(
              [ & ](vk::Format format)
              {
                return make_image_resource<image_kind::depth>(
                  device.allocator(), device.device(),
                  image_runtime_args {
                    .extent = output.extent(),
                    .format = format,
                    .samples = device.msaa_samples(),
                  });
              })
            .transform([ & ](image_resource<>&& depth)
              { output.depth_ = std::move(depth); });
        })
      .and_then(
        [ & ] -> std::expected<void, error_t>
        {
          output.render_finished_semaphores_.clear();
          output.render_finished_semaphores_.reserve(output.images_.size());
          for (auto _ : std::views::iota(0UZ, output.images_.size()))
          {
            if (auto error = UTILS_VK(device.device().createSemaphore({}),
                  ^^vk::raii::Device::createSemaphore)
                  .transform(
                    [ & ](vk::raii::Semaphore&& semaphore)
                    {
                      output.render_finished_semaphores_.push_back(
                        std::move(semaphore));
                    });
              !error)
            {
              return error;
            }
          }
          return {};
        })
      .transform([ & ] -> swapchain { return std::move(output); });
  }

  // It is possible to create a new swap chain while drawing commands on an
  // image from the old swap chain are still in-flight. You need to pass the
  // previous swap chain to the oldSwapchain field in the
  // vk::SwapchainCreateInfoKHR struct and destroy the old swap chain as soon as
  // you’ve finished using it.
  [[nodiscard]] auto
  recreate(device_context& device, const vk::raii::SurfaceKHR& surface,
    extent_request window,
    std::invocable<const vk::SurfaceCapabilitiesKHR&, vk::Extent2D> auto&&
      choose_extent) -> std::expected<void, error_t>
  {
    return UTILS_VK(device.device().waitIdle(), ^^vk::raii::Device::waitIdle)
      .and_then(
        [ & ]() -> std::expected<void, error_t>
        {
          release();
          return create(device, surface, window, choose_extent)
            .transform(
              [ & ](swapchain&& swapchain) { *this = std::move(swapchain); });
        });
  }

  [[nodiscard]] static auto
  query_presentability(const device_context& device,
    const vk::raii::SurfaceKHR& surface, extent_request window,
    std::invocable<const vk::SurfaceCapabilitiesKHR&, vk::Extent2D> auto&&
      choose_extent) -> std::expected<presentability, error_t>
  {
    return UTILS_VK(
      device.physical_device().getSurfaceCapabilitiesKHR(*surface),
      ^^vk::raii::PhysicalDevice::getSurfaceCapabilitiesKHR)
      .transform(
        [ & ](const vk::SurfaceCapabilitiesKHR& capabilities) -> presentability
        {
          const vk::Extent2D extent =
            choose_extent(capabilities, window.framebuffer_size);
          return presentability {
            .presentable = extent.width > 0U && extent.height > 0U,
            .extent = extent,
          };
        });
  }

  void
  release()
  {
    image_views_.clear();
    images_.clear();
    swap_chain_ = nullptr;
    color_ = {};
    depth_ = {};
    render_finished_semaphores_.clear();
    extent_ = vk::Extent2D {};
    surface_format_ = vk::SurfaceFormatKHR {};
  }

  [[nodiscard]] auto
  empty() const -> bool
  { return !*swap_chain_; }

  [[nodiscard]] auto
  swap_chain() const -> const vk::raii::SwapchainKHR&
  { return swap_chain_; }

  [[nodiscard]] auto
  extent() const -> vk::Extent2D
  { return extent_; }

  [[nodiscard]] auto
  format() const -> vk::Format
  { return surface_format_.format; }

  [[nodiscard]] auto
  images() const -> std::span<const vk::Image>
  { return images_; }

  [[nodiscard]] auto
  image_view(std::size_t index) const -> const vk::raii::ImageView&
  { return image_views_[ index ]; }

  [[nodiscard]] auto
  color() -> image_resource<>&
  { return color_; }

  [[nodiscard]] auto
  depth() -> image_resource<>&
  { return depth_; }

  [[nodiscard]] auto
  render_finished(std::size_t image_index) const -> const vk::raii::Semaphore&
  { return render_finished_semaphores_[ image_index ]; }

private:
  struct surface_build_info
  {
    vk::SurfaceCapabilitiesKHR capabilities;
    vk::Extent2D extent;
    vk::SurfaceFormatKHR surface_format;
    vk::PresentModeKHR present_mode;
    std::uint32_t min_image_count;
    vk::SurfaceTransformFlagBitsKHR pre_transform;
  };

  static auto
  choose_swap_surface_format(std::span<const vk::SurfaceFormatKHR> formats)
    -> vk::SurfaceFormatKHR
  {
    const auto format_it = std::ranges::find_if(formats,
      [](const auto& format) -> bool
      {
        return format.format == vk::Format::eB8G8R8A8Srgb &&
          format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
      });
    return format_it != formats.end() ? *format_it : formats[ 0 ];
  }

  [[gnu::pure]]
  static auto
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

  [[gnu::pure]]
  static auto
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

  static auto
  find_depth_format(const vk::raii::PhysicalDevice& physical_device)
    -> std::expected<vk::Format, vkpp::error_t>
  {
    static constexpr std::array candidates {
      vk::Format::eD32Sfloat,
      vk::Format::eD32SfloatS8Uint,
      vk::Format::eD24UnormS8Uint,
    };

    return find_supported_format(physical_device, candidates,
      vk::ImageTiling::eOptimal,
      vk::FormatFeatureFlagBits::eDepthStencilAttachment);
  }

  static auto
  find_supported_format(const vk::raii::PhysicalDevice& physical_device,
    std::span<const vk::Format> candidates, vk::ImageTiling tiling,
    vk::FormatFeatureFlags features) -> std::expected<vk::Format, vkpp::error_t>
  {
    for (const auto format : candidates)
    {
      const auto properties = physical_device.getFormatProperties(format);

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

  vk::raii::SwapchainKHR swap_chain_ { nullptr };
  std::vector<vk::Image> images_;
  std::vector<vk::raii::ImageView> image_views_;
  vk::SurfaceFormatKHR surface_format_ {};
  vk::Extent2D extent_ {};
  image_resource<> color_ {};
  image_resource<> depth_ {};
  std::vector<vk::raii::Semaphore> render_finished_semaphores_;
};

}; // namespace vkpp
