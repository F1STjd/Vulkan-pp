module;

#include <tiny_obj_loader.h>

export module vkpp.io.mesh.obj;

import std;
import vulkan;

import vkpp.io.types;
import vkpp.io.mesh;
import vkpp.error;

namespace vkpp
{
using namespace std::string_view_literals;

template<std::size_t Components>
using obj_attribute_view = std::mdspan<const float,
  std::extents<std::size_t, std::dynamic_extent, Components>>;

template<>
[[nodiscard]] auto
load_mesh_cpu<mesh_file_type::obj>(const std::filesystem::path& path)
  -> std::expected<mesh_cpu, error_t>
{
  tinyobj::attrib_t attributes;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string warnings;
  std::string errors;

  if (!tinyobj::LoadObj(&attributes, &shapes, &materials, &warnings, &errors,
        path.string().c_str()))
  {
    return std::unexpected {
      vkpp::app_error {
        .kind = vkpp::app_error_kind::model_parse,
        .detail = std::format("warnings: {}\nerrors: {}", warnings, errors),
      },
    };
  }

  mesh_streams_cpu streams { .index_type = vk::IndexType::eUint32 };
  std::vector<std::uint32_t> indices {};
  std::unordered_map<std::uint64_t, std::uint32_t> unique {};

  for (const auto& shape : shapes)
  {
    for (const auto& index : shape.mesh.indices)
    {
      const std::uint64_t key =
        (static_cast<std::uint64_t>(
           static_cast<std::uint32_t>(index.vertex_index))
          << 32U) |
        static_cast<std::uint32_t>(index.texcoord_index);

      auto [ it, inserted ] = unique.insert(
        { key, static_cast<std::uint32_t>(streams.vertex_count) });

      if (inserted)
      {
        const auto position_base =
          static_cast<std::size_t>(index.vertex_index) * 3UZ;
        streams.positions.push_back(attributes.vertices[ position_base ]);
        streams.positions.push_back(attributes.vertices[ position_base + 1UZ ]);
        streams.positions.push_back(attributes.vertices[ position_base + 2UZ ]);
        streams.colors.insert(streams.colors.end(), { 1.0F, 1.0F, 1.0F });
        if (index.texcoord_index >= 0)
        {
          const auto uv_base =
            static_cast<std::size_t>(index.texcoord_index) * 2UZ;
          streams.texcoords.push_back(attributes.texcoords[ uv_base ]);
          streams.texcoords.push_back(
            1.0F - attributes.texcoords[ uv_base + 1UZ ]);
        }
        else
        {
          streams.texcoords.insert(streams.texcoords.end(), { 0.0F, 0.0F });
        }
        ++streams.vertex_count;
      }
      indices.push_back(it->second);
    }
  }

  streams.index_count = static_cast<std::uint32_t>(indices.size());
  streams.indices.resize(indices.size() * sizeof(std::uint32_t));
  std::memcpy(streams.indices.data(), indices.data(), streams.indices.size());

  mesh_cpu mesh {};
  mesh.primitives.push_back(std::move(streams));
  return mesh;
}
} // namespace vkpp
