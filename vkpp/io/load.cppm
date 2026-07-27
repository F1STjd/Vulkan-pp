module;

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <tiny_obj_loader.h>

export module vkpp.io;

import std;
import vulkan;

import vkpp.io.channels;
import vkpp.vertex;
import vkpp.error;
import vkpp.texture;

namespace vkpp
{

using namespace std::string_view_literals;

export constexpr const char* model_path { MODEL_DIRECTORY "viking_room.obj" };
export constexpr const char* texture_path { TEXTURE_DIRECTORY
  "viking_room.png" };

export [[nodiscard]] constexpr auto
load_shader_file(const std::filesystem::path& filename)
  -> std::expected<std::vector<char>, vkpp::error_t>
{
  std::ifstream input_file { filename, std::ios::ate | std::ios::binary };
  if (!input_file.is_open())
  {
    return std::unexpected {
      vkpp::app_error {
        .kind = vkpp::app_error_kind::file_open,
        .detail = "Failed to open shader file"sv,
      },
    };
  }

  std::vector<char> buffer(input_file.tellg());
  input_file.seekg(0, std::ios::beg);
  input_file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
  return { buffer };
}

export struct host_image
{
  std::vector<std::byte> pixels {};
  vk::Extent2D extent {};
  vk::Format format { vk::Format::eR8G8B8A8Srgb };
  std::uint32_t mip_levels_present { 1U };
  texture_mip_policy suggested_mip_policy {
    texture_mip_policy::generate_gpu_blit
  };
};

// This will be made generic in future
export auto
load_host_image_rgba8(const std::filesystem::path& path)
  -> std::expected<host_image, error_t>
{
  std::int32_t texture_channels; // NOLINT
  std::int32_t width;            // NOLINT
  std::int32_t height;           // NOLINT
  auto* pixels = stbi_load(
    path.string().c_str(), &width, &height, &texture_channels, STBI_rgb_alpha);
  if (pixels == nullptr)
  {
    return std::unexpected {
      vkpp::app_error {
        .kind = vkpp::app_error_kind::file_open,
        .detail = "Failed to load texture file"sv,
      },
    };
  }

  const vk::DeviceSize image_size { channels::four * width * height };
  const auto mip_levels = static_cast<std::uint32_t>(
    std::floor(std::log2(std::max(width, height))) + 1U);
  auto to_byte = [](stbi_uc byte) static { return std::byte { byte }; };
  auto pixels_span = std::span { pixels, image_size };

  auto texture = host_image {
    .pixels = {
      std::from_range,
      pixels_span | std::views::transform(to_byte),
    },
    .extent = {
      .width = static_cast<std::uint32_t>(width),
      .height = static_cast <std::uint32_t>(height),
    },
    .mip_levels_present = mip_levels,
  };

  stbi_image_free(pixels);
  return texture;
}

template<std::size_t Components>
using obj_attribute_view = std::mdspan<const float,
  std::extents<std::size_t, std::dynamic_extent, Components>>;

export [[nodiscard]] constexpr auto
load_model_obj(std::vector<vkpp::vertex>& vertices,
  std::vector<std::uint32_t>& indices) -> std::expected<void, vkpp::error_t>
{
  tinyobj::attrib_t attributes;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string warnings;
  std::string errors;

  if (!tinyobj::LoadObj(
        &attributes, &shapes, &materials, &warnings, &errors, model_path))
  {
    return std::unexpected {
      vkpp::app_error {
        .kind = vkpp::app_error_kind::model_parse,
        .detail = std::format("warnings: {}\nerrors: {}", warnings, errors),
      },
    };
  }

  obj_attribute_view<3> positions {
    attributes.vertices.data(),
    attributes.vertices.size() / 3UZ,
  };
  obj_attribute_view<2> texture_coordinates {
    attributes.texcoords.data(),
    attributes.texcoords.size() / 2UZ,
  };

  std::unordered_map<vkpp::vertex, std::uint32_t> unique_vertices {};

  for (const auto& shape : shapes)
  {
    for (const auto& index : shape.mesh.indices)
    {
      const auto xyz =
        std::submdspan(positions, index.vertex_index, std::full_extent);
      const auto uv = std::submdspan(
        texture_coordinates, index.texcoord_index, std::full_extent);

      vkpp::vertex vertex {};
      vertex.position = {
        xyz[ 0UZ ],
        xyz[ 1UZ ],
        xyz[ 2UZ ],
      };
      vertex.texture_coordinates = {
        uv[ 0UZ ],
        1.0F - uv[ 1UZ ],
      };
      vertex.color = { 1.0F, 1.0F, 1.0F };

      auto [ it, inserted ] = unique_vertices.insert(
        { vertex, static_cast<std::uint32_t>(vertices.size()) });
      if (inserted) { vertices.push_back(vertex); }
      indices.push_back(it->second);
    }
  }

  return {};
}
} // namespace vkpp
