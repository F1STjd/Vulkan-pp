export module vkpp.image;

import std;
import vulkan;

import vkpp.memory;
import vkpp.memory.vma;
import vkpp.error;
import vkpp.diagnostics;

namespace vkpp
{

export template<device_allocator Alloc = vma_policy>
class image_resource
{
public:
  image_resource() = default;
  image_resource(
    Alloc::image_handle&&          handle,
    vk::raii::ImageView&&          view,
    vk::Extent2D                   extent,
    vk::Format                     format,
    const vk::ImageViewCreateInfo& view_info)
  : handle_ {std::move(handle)},
    view_ {std::move(view)},
    extent_ {extent},
    format_ {format},
    view_info_ {view_info}
  {}

  [[nodiscard]]
  auto
  image() const -> vk::Image
  { return handle_.get(); }

  [[nodiscard]]
  auto
  view() const -> const vk::raii::ImageView&
  { return view_; }

  [[nodiscard]]
  auto
  view_create_info() const noexcept -> const vk::ImageViewCreateInfo&
  { return view_info_; }

  [[nodiscard]]
  auto
  extent() const -> vk::Extent2D
  { return extent_; }

  [[nodiscard]]
  auto
  format() const -> vk::Format
  { return format_; }

private:
  Alloc::image_handle     handle_ {};
  vk::raii::ImageView     view_ {nullptr};
  vk::Extent2D            extent_ {};
  vk::Format              format_ {};
  vk::ImageViewCreateInfo view_info_ {};
};

export struct image_type_spec
{
  vk::ImageType        image_type {vk::ImageType::e2D};
  vk::ImageViewType    view_type {vk::ImageViewType::e2D};
  std::uint32_t        array_layers {1U};
  vk::ImageTiling      tiling {vk::ImageTiling::eOptimal};
  memory_intent        intent {memory_intent::gpu_only};
  vk::ImageUsageFlags  usage {};
  vk::ImageAspectFlags aspect {};
  vk::ImageCreateFlags flags {};
};

export enum class image_kind : std::uint8_t
{
  color,
  depth,
  resolve,
  color_sampled,
  color_transfer,
  sampled_texture,
  shadow_map,
  sampled_cube,
};

export struct image_runtime_args
{
  vk::Extent2D            extent {};
  vk::Format              format {};
  vk::SampleCountFlagBits samples {vk::SampleCountFlagBits::e1};
  std::uint32_t           mip_levels {1U};
  std::uint32_t           array_layers {1U};
};

export template<image_kind Kind>
struct image_traits;

template<>
struct image_traits<image_kind::color>
{
  static constexpr image_type_spec spec {
    .usage =
      vk::ImageUsageFlagBits::eTransientAttachment |
      vk::ImageUsageFlagBits::eColorAttachment,
    .aspect = vk::ImageAspectFlagBits::eColor,
  };
};

template<>
struct image_traits<image_kind::depth>
{
  static constexpr image_type_spec spec {
    .usage  = vk::ImageUsageFlagBits::eDepthStencilAttachment,
    .aspect = vk::ImageAspectFlagBits::eDepth,
  };
};

template<>
struct image_traits<image_kind::resolve>
{
  static constexpr image_type_spec spec {
    .usage  = vk::ImageUsageFlagBits::eColorAttachment,
    .aspect = vk::ImageAspectFlagBits::eColor,
  };
};

template<>
struct image_traits<image_kind::color_sampled>
{
  static constexpr image_type_spec spec {
    .usage =
      vk::ImageUsageFlagBits::eColorAttachment |
      vk::ImageUsageFlagBits::eSampled,
    .aspect = vk::ImageAspectFlagBits::eColor,
  };
};

template<>
struct image_traits<image_kind::color_transfer>
{
  static constexpr image_type_spec spec {
    .usage =
      vk::ImageUsageFlagBits::eColorAttachment |
      vk::ImageUsageFlagBits::eTransferSrc,
    .aspect = vk::ImageAspectFlagBits::eColor,
  };
};

template<>
struct image_traits<image_kind::sampled_texture>
{
  static constexpr image_type_spec spec {
    .intent = memory_intent::gpu_only,
    .usage =
      vk::ImageUsageFlagBits::eTransferDst |
      vk::ImageUsageFlagBits::eTransferSrc |
      vk::ImageUsageFlagBits::eSampled,
    .aspect = vk::ImageAspectFlagBits::eColor,
  };
};

template<>
struct image_traits<image_kind::sampled_cube>
{
  static constexpr image_type_spec spec {
    .view_type    = vk::ImageViewType::eCube,
    .array_layers = 6U,
    .intent       = memory_intent::gpu_only,
    .usage =
      vk::ImageUsageFlagBits::eTransferDst |
      vk::ImageUsageFlagBits::eTransferSrc |
      vk::ImageUsageFlagBits::eSampled,
    .aspect = vk::ImageAspectFlagBits::eColor,
    .flags  = vk::ImageCreateFlagBits::eCubeCompatible,
  };
};

export consteval
auto
validate(const image_type_spec& spec) -> bool
{
  // depth
  if (
    (spec.usage & vk::ImageUsageFlagBits::eDepthStencilAttachment) &&
    !(spec.aspect & vk::ImageAspectFlagBits::eDepth))
  {
    return false;
  }
  // cube
  if (
    (spec.flags & vk::ImageCreateFlagBits::eCubeCompatible) &&
    (spec.view_type != vk::ImageViewType::eCube || spec.array_layers % 6U != 0U))
  {
    return false;
  }
  // color
  if (
    (spec.usage & vk::ImageUsageFlagBits::eTransientAttachment) &&
    spec.tiling != vk::ImageTiling::eOptimal)
  {
    return false;
  }
  // color sampled
  if (
    (spec.usage & vk::ImageUsageFlagBits::eSampled) &&
    (spec.usage & vk::ImageUsageFlagBits::eTransientAttachment))
  {
    return false;
  }
  return true;
}

static_assert(validate(image_traits<image_kind::sampled_texture>::spec));
static_assert(validate(image_traits<image_kind::color_sampled>::spec));
static_assert(validate(image_traits<image_kind::color_transfer>::spec));
static_assert(validate(image_traits<image_kind::sampled_cube>::spec));

export
template<image_kind Kind>
  requires (validate(image_traits<Kind>::spec))
constexpr auto
make_image_view_create_info(
  vk::Image     image,
  vk::Format    format,
  std::uint32_t mip_levels,
  std::uint32_t array_layers) -> vk::ImageViewCreateInfo
{
  constexpr image_type_spec spec = image_traits<Kind>::spec;
  return {
    .image            = image,
    .viewType         = spec.view_type,
    .format           = format,
    .subresourceRange = {
      .aspectMask     = spec.aspect,
      .baseMipLevel   = 0U,
      .levelCount     = mip_levels,
      .baseArrayLayer = 0U,
      .layerCount     = array_layers,
    },
  };
}

export
template<image_kind Kind>
  requires (validate(image_traits<Kind>::spec))
auto
make_image_view(
  const vk::raii::Device& device,
  vk::Image               image,
  vk::Format              format,
  std::uint32_t           mip_levels,
  std::uint32_t array_layers) -> std::expected<vk::raii::ImageView, error_t>
{
  return map_vk_error(
    device.createImageView(
      make_image_view_create_info<Kind>(
        image, format, mip_levels, array_layers)),
    std::nullopt);
}

export
template<image_kind Kind, device_allocator Alloc = vma_policy>
  requires (validate(image_traits<Kind>::spec))
auto
make_image_resource(
  Alloc&                    allocator,
  const vk::raii::Device&   device,
  const image_runtime_args& args)
  -> std::expected<image_resource<Alloc>, error_t>
{
  constexpr image_type_spec spec = image_traits<Kind>::spec;
  const vk::ImageCreateInfo image_info {
    .flags     = spec.flags,
    .imageType = spec.image_type,
    .format    = args.format,
    .extent =
      {
        .width  = args.extent.width,
        .height = args.extent.height,
        .depth  = 1U,
      },
    .mipLevels   = args.mip_levels,
    .arrayLayers = args.array_layers,
    .samples     = args.samples,
    .tiling      = spec.tiling,
    .usage       = spec.usage,
    .sharingMode = vk::SharingMode::eExclusive,
  };

  return allocator.create_image(image_info, spec.intent)
    .and_then(
      [ & ](Alloc::image_handle&& handle)
        -> std::expected<image_resource<Alloc>, error_t> {
        const auto view_info = make_image_view_create_info<Kind>(
          handle.get(), args.format, args.mip_levels, args.array_layers);
        return map_vk_error(device.createImageView(view_info), std::nullopt)
          .transform([ & ](vk::raii::ImageView&& view) {
            return image_resource<Alloc> {
              std::move(handle),
              std::move(view),
              args.extent,
              args.format,
              view_info,
            };
          });
      });
}

} // namespace vkpp
