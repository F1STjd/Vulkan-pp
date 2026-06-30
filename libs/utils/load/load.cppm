module;

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <tiny_obj_loader.h>

export module load;

import std;
import vulkan;
import load.channels;
import vertex;

namespace load
{
export constexpr const char* model_path { MODEL_DIRECTORY "viking_room.obj" };
export constexpr const char* texture_path { TEXTURE_DIRECTORY
  "viking_room.png" };

export [[nodiscard]] constexpr auto
shader_file(const std::filesystem::path& filename)
  -> std::expected<std::vector<char>, std::string>
{
  std::ifstream input_file { filename, std::ios::ate | std::ios::binary };
  if (!input_file.is_open())
  {
    return std::expected<std::vector<char>, std::string> {
      std::unexpect,
      "Failed to open shader file",
    };
  }

  std::vector<char> buffer(input_file.tellg());
  input_file.seekg(0, std::ios::beg);
  input_file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
  return { buffer };
}

export [[nodiscard]] constexpr auto
texture_file(const std::filesystem::path& filename, std::int32_t& texture_width,
  std::int32_t& texture_height, std::uint32_t& mip_levels)
  -> std::expected<std::span<stbi_uc>, std::string>
{
  std::int32_t texture_channels; // NOLINT
  auto* pixels = stbi_load(filename.string().c_str(), &texture_width,
    &texture_height, &texture_channels, STBI_rgb_alpha);
  const vk::DeviceSize image_size { channels::four * texture_width *
    texture_height };

  if (pixels == nullptr)
  {
    return std::expected<std::span<stbi_uc>, std::string> {
      std::unexpect,
      std::format("Failed to load texture file: {}", filename),
    };
  }
  mip_levels = static_cast<std::uint32_t>(
    std::floor(std::log2(std::max(texture_width, texture_height))) + 1U);
  return std::span { pixels, image_size };
}

template<std::size_t Components>
using obj_attribute_view = std::mdspan<const float,
  std::extents<std::size_t, std::dynamic_extent, Components>>;

export [[nodiscard]] constexpr auto
model_obj(std::vector<lbn::vertex>& vertices,
  std::vector<std::uint32_t>& indices) -> std::expected<void, std::string>
{
  tinyobj::attrib_t attributes;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string warnings;
  std::string errors;

  if (!tinyobj::LoadObj(
        &attributes, &shapes, &materials, &warnings, &errors, model_path))
  {
    return std::expected<void, std::string> {
      std::unexpect,
      std::format("warnings: {}\nerrors: {}", warnings, errors),
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

  std::unordered_map<lbn::vertex, std::uint32_t> unique_vertices {};

  for (const auto& shape : shapes)
  {
    for (const auto& index : shape.mesh.indices)
    {
      const auto xyz =
        std::submdspan(positions, index.vertex_index, std::full_extent);
      const auto uv = std::submdspan(
        texture_coordinates, index.texcoord_index, std::full_extent);

      lbn::vertex vertex {};
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

} // namespace load
