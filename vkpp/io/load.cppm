module;

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <tiny_obj_loader.h>

#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>

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

export constexpr const char* model_path { MODEL_DIRECTORY "911.glb" };
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

export enum class file_type : std::uint8_t {
  gltf,
  obj,
  ktx2,
  png,
};

export template<file_type Type>
struct file_type_traits;

template<>
struct file_type_traits<file_type::gltf>
{
  static constexpr bool loads_mesh { true };
  static constexpr bool loads_image { false };
};

template<>
struct file_type_traits<file_type::obj>
{
  static constexpr bool loads_mesh { true };
  static constexpr bool loads_image { false };
};

template<>
struct file_type_traits<file_type::ktx2>
{
  static constexpr bool loads_mesh { false };
  static constexpr bool loads_image { true };
  /* using image_result = host_image_mip_chain; */
};

template<>
struct file_type_traits<file_type::png>
{
  static constexpr bool loads_mesh { false };
  static constexpr bool loads_image { true };
  using image_result = host_image;
};

template<std::size_t Components>
using attribute_mdspan = std::mdspan<const float,
  std::extents<std::size_t, std::dynamic_extent, Components>>;

export struct mesh_streams_cpu
{
  std::vector<float> positions {};
  std::vector<float> colors {};
  std::vector<float> texcoords {};
  std::vector<std::byte> indices {};
  vk::IndexType index_type { vk::IndexType::eUint16 };
  std::uint32_t vertex_count { 0U };
  std::uint32_t index_count { 0U };

  [[nodiscard]] auto
  positions_view() const -> attribute_mdspan<3>
  { return attribute_mdspan<3> { positions.data(), vertex_count }; }

  [[nodiscard]] auto
  colors_view() const -> attribute_mdspan<3>
  { return attribute_mdspan<3> { colors.data(), vertex_count }; }

  [[nodiscard]] auto
  texcoords_view() const -> attribute_mdspan<2>
  { return attribute_mdspan<2> { texcoords.data(), vertex_count }; }
};

export struct mesh_cpu
{
  std::vector<mesh_streams_cpu> primitives {};
};

export struct mesh_interleaved_cpu
{
  std::vector<std::byte> vertices {};
  std::vector<std::byte> indices {};
  vk::IndexType index_type { vk::IndexType::eUint16 };
  std::uint32_t vertex_count { 0U };
  std::uint32_t index_count { 0U };
};

export [[nodiscard]] auto
pack_interleaved_vertices(const mesh_streams_cpu& streams)
  -> mesh_interleaved_cpu
{
  mesh_interleaved_cpu packed {
    .indices = streams.indices,
    .index_type = streams.index_type,
    .vertex_count = streams.vertex_count,
    .index_count = streams.index_count,
  };
  packed.vertices.resize(streams.vertex_count * sizeof(vertex));
  auto* out = std::start_lifetime_as_array<vertex>(
    packed.vertices.data(), streams.vertex_count);

  const attribute_mdspan<3> positions = streams.positions_view();
  const attribute_mdspan<3> colors = streams.colors_view();
  const attribute_mdspan<2> texcoords = streams.texcoords_view();

  for (auto index : std::views::iota(0U, streams.vertex_count))
  {
    const auto xyz = std::submdspan(positions, index, std::full_extent);
    const auto rgb = std::submdspan(colors, index, std::full_extent);
    const auto uv = std::submdspan(texcoords, index, std::full_extent);
    out[ index ] = vertex {
      .position = { xyz[ 0 ], xyz[ 1 ], xyz[ 2 ] },
      .color = { rgb[ 0 ], rgb[ 1 ], rgb[ 2 ] },
      .texture_coordinates = { uv[ 0 ], uv[ 1 ] },
    };
  }

  return packed;
}

[[nodiscard]] auto
map_fastgltf_error(fastgltf::Error error) -> error_t
{
  return app_error {
    .kind = app_error_kind::model_parse,
    .detail = std::format("fastgltf: {}", fastgltf::getErrorMessage(error)),
  };
}

export template<file_type Type>
  requires(file_type_traits<Type>::loads_mesh)
[[nodiscard]] auto
load_mesh_cpu(const std::filesystem::path& path)
  -> std::expected<mesh_cpu, error_t>;

template<>
[[nodiscard]] auto
load_mesh_cpu<file_type::gltf>(const std::filesystem::path& path)
  -> std::expected<mesh_cpu, error_t>
{
  auto data = fastgltf::GltfDataBuffer::FromPath(path);
  if (data.error() != fastgltf::Error::None)
  {
    return std::unexpected { map_fastgltf_error(data.error()) };
  }

  fastgltf::Parser parser {};
  auto asset = parser.loadGltf(
    data.get(), path.parent_path(), fastgltf::Options::LoadExternalBuffers);
  if (asset.error() != fastgltf::Error::None)
  {
    return std::unexpected { map_fastgltf_error(asset.error()) };
  }

  const fastgltf::Asset& gltf = asset.get();
  mesh_cpu mesh {};

  for (const fastgltf::Mesh& gltf_mesh : gltf.meshes)
  {
    for (const fastgltf::Primitive& primitive : gltf_mesh.primitives)
    {
      const auto* position_it = primitive.findAttribute("POSITION");
      if (position_it == primitive.attributes.end())
      {
        return std::unexpected {
          app_error {
            .kind = app_error_kind::model_parse,
            .detail = "glTF primitive missing POSITION"sv,
          },
        };
      }

      const fastgltf::Accessor& position_accessor =
        gltf.accessors[ position_it->accessorIndex ];
      mesh_streams_cpu streams {
        .vertex_count = static_cast<std::uint32_t>(position_accessor.count),
      };
      streams.positions.resize(streams.vertex_count * 3U);
      streams.colors.resize(streams.vertex_count * 3U, 1.0F);
      streams.texcoords.resize(streams.vertex_count * 2U, 0.0F);

      fastgltf::copyFromAccessor<float>(
        gltf, position_accessor, streams.positions.data());

      if (const auto* color_it = primitive.findAttribute("COLOR_0");
        color_it != primitive.attributes.end())
      {
        fastgltf::copyFromAccessor<float>(gltf,
          gltf.accessors[ color_it->accessorIndex ], streams.colors.data());
      }

      if (const auto* uv_it = primitive.findAttribute("TEXCOORD_0");
        uv_it != primitive.attributes.end())
      {
        fastgltf::iterateAccessorWithIndex<glm::vec2>(gltf,
          gltf.accessors[ uv_it->accessorIndex ],
          [ & ](glm::vec2 uv, std::size_t index)
          {
            streams.texcoords[ index * 2U ] = uv.x;
            streams.texcoords[ index * 2U + 1U ] = 1.0F - uv.y;
          });
      }

      if (!primitive.indicesAccessor.has_value())
      {
        return std::unexpected {
          app_error {
            .kind = app_error_kind::model_parse,
            .detail = "glTF primitive missing indices (unsupported for now)"sv,
          },
        };
      }

      const fastgltf::Accessor& index_accessor =
        gltf.accessors[ *primitive.indicesAccessor ];
      streams.index_count = static_cast<std::uint32_t>(index_accessor.count);

      if (index_accessor.componentType ==
        fastgltf::ComponentType::UnsignedShort)
      {
        streams.index_type = vk::IndexType::eUint16;
        std::vector<std::uint16_t> indices(streams.index_count);
        fastgltf::copyFromAccessor<std::uint16_t>(
          gltf, index_accessor, indices.data());
        streams.indices.resize(indices.size() * sizeof(std::uint16_t));
        std::memcpy(
          streams.indices.data(), indices.data(), streams.indices.size());
      }
      else if (index_accessor.componentType ==
        fastgltf::ComponentType::UnsignedInt)
      {
        streams.index_type = vk::IndexType::eUint32;
        std::vector<std::uint32_t> indices(streams.index_count);
        fastgltf::copyFromAccessor<std::uint32_t>(
          gltf, index_accessor, indices.data());
        streams.indices.resize(indices.size() * sizeof(std::uint32_t));
        std::memcpy(
          streams.indices.data(), indices.data(), streams.indices.size());
      }
      else
      {
        return std::unexpected {
          app_error {
            .kind = app_error_kind::model_parse,
            .detail = "unsupported glTF index component type"sv,
          },
        };
      }

      mesh.primitives.push_back(std::move(streams));
    }
  }

  if (mesh.primitives.empty())
  {
    return std::unexpected {
      app_error {
        .kind = app_error_kind::model_parse,
        .detail = "glTF contained no mesh primitives"sv,
      },
    };
  }
  return mesh;
}

// refactor obj, so it does what ^^^ is doing
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

export template<file_type Type>
  requires(file_type_traits<Type>::loads_image)
[[nodiscard]] auto
load_host_image(const std::filesystem::path& path)
  -> std::expected<host_image, error_t>;

template<>
[[nodiscard]] auto
load_host_image<file_type::png>(const std::filesystem::path& path)
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

// deprecated, only for compilation, will be removed
export auto
load_host_image_rgba8(const std::filesystem::path& path)
  -> std::expected<host_image, error_t>
{ return load_host_image<file_type::png>(path); }

} // namespace vkpp
