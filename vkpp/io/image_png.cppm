module;

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

export module vkpp.io.image.png;

import std;
import vulkan;

import vkpp.io.types;
import vkpp.io.image;
import vkpp.io.channels;
import vkpp.error;

namespace vkpp
{
using namespace std::string_view_literals;

[[nodiscard]] auto
host_image_from_stb(stbi_uc* pixels, std::int32_t width, std::int32_t height)
  -> std::expected<host_image, error_t>
{
  if (pixels == nullptr)
  {
    return std::unexpected {
      app_error {
        .kind = app_error_kind::image_decode,
        .detail = "stb_image failed to decode"sv,
      },
    };
  }

  const std::size_t image_size {
    channels::four * static_cast<std::size_t>(width) *
      static_cast<std::size_t>(height),
  };
  const auto mip_levels = static_cast<std::uint32_t>(
    std::floor(std::log2(std::max(width, height))) + 1);
  constexpr auto to_byte = [](stbi_uc byte) static -> std::byte
  { return std::byte { byte }; };
  auto pixels_span = std::span { pixels, image_size };

  host_image image {
    .pixels = {
      std::from_range,
      pixels_span | std::views::transform(to_byte),
    },
    .extent = {
      .width = static_cast<std::uint32_t>(width),
      .height = static_cast<std::uint32_t>(height),
    },
    .mip_levels_present = mip_levels,
  };

  stbi_image_free(pixels);
  return image;
}

export [[nodiscard]] auto
load_host_image_stb_from_memory(std::span<const std::byte> bytes)
  -> std::expected<host_image, error_t>
{
  std::int32_t width { 0 };
  std::int32_t height { 0 };
  std::int32_t file_channels { 0 };
  auto* pixels =
    stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(bytes.data()),
      static_cast<std::int32_t>(bytes.size()), &width, &height, &file_channels,
      STBI_rgb_alpha);
  return host_image_from_stb(pixels, width, height);
}

template<>
[[nodiscard]] auto
load_host_image<image_file_type::png>(const std::filesystem::path& path,
  const png_load_runtime_args&) -> std::expected<host_image, error_t>
{
  std::int32_t width { 0 };
  std::int32_t height { 0 };
  std::int32_t file_channels { 0 };
  auto* pixels = stbi_load(
    path.string().c_str(), &width, &height, &file_channels, STBI_rgb_alpha);
  return host_image_from_stb(pixels, width, height);
}

} // namespace vkpp
