module;

#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>

export module vkpp.io.mesh.gltf;

import std;
import vulkan;

import vkpp.io.types;
import vkpp.io.mesh;
import vkpp.io.image.png;
import vkpp.io.image.ktx2;
import vkpp.error;
import vkpp.vertex;

namespace vkpp
{
using namespace std::string_view_literals;

[[nodiscard]] auto
map_fastgltf_error(fastgltf::Error error) -> error_t
{
  return app_error {
    .kind = app_error_kind::model_parse,
    .detail = std::format("fastgltf: {}", fastgltf::getErrorMessage(error)),
  };
}

[[nodiscard]] auto
make_gltf_parser(const gltf::load_runtime_args& runtime_args)
  -> fastgltf::Parser
{
  fastgltf::Extensions extensions { fastgltf::Extensions::None };
  if (runtime_args.enable_texture_basisu)
  {
    extensions |= fastgltf::Extensions::KHR_texture_basisu;
  }
  return fastgltf::Parser { extensions };
}

[[nodiscard]] auto
gltf_options(const gltf::load_runtime_args& runtime_args) -> fastgltf::Options
{
  fastgltf::Options options = fastgltf::Options::None;
  if (runtime_args.load_external_buffers)
  {
    options |= fastgltf::Options::LoadExternalBuffers;
  }
  if (runtime_args.load_external_images &&
    runtime_args.content != gltf::content_policy::geometry_only)
  {
    options |= fastgltf::Options::LoadExternalImages;
  }
  return options;
}

[[nodiscard]] auto
category_for(gltf::content_policy content) -> fastgltf::Category
{
  using fastgltf::Category;
  switch (content)
  {
  case gltf::content_policy::geometry_only:
    return Category::Buffers | Category::BufferViews | Category::Accessors |
      Category::Meshes;
  case gltf::content_policy::geometry_and_materials:
    return Category::OnlyRenderable;
  case gltf::content_policy::geometry_and_host_images:
    return Category::OnlyRenderable;
  }
  return Category::Meshes;
}

[[nodiscard]] auto
extract_mesh_cpu(const fastgltf::Asset& gltf)
  -> std::expected<mesh_cpu, error_t>
{
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

      fastgltf::copyComponentsFromAccessor<float>(
        gltf, position_accessor, streams.positions.data());
      if (const auto* normal_it = primitive.findAttribute("NORMAL");
        normal_it != primitive.attributes.end())
      {
        streams.normals.resize(streams.vertex_count * 3U);
        fastgltf::copyComponentsFromAccessor<float>(gltf,
          gltf.accessors[ normal_it->accessorIndex ], streams.normals.data());
      }

      if (const auto* color_it = primitive.findAttribute("COLOR_0");
        color_it != primitive.attributes.end())
      {
        const fastgltf::Accessor& color_accessor =
          gltf.accessors[ color_it->accessorIndex ];
        if (color_accessor.type == fastgltf::AccessorType::Vec3)
        {
          fastgltf::copyComponentsFromAccessor<float>(
            gltf, color_accessor, streams.colors.data());
        }
        else
        {
          fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec4>(gltf,
            color_accessor,
            [ & ](fastgltf::math::fvec4 rgba, std::size_t index) -> void
            {
              streams.colors[ index * 3U ] = rgba[ 0 ];
              streams.colors[ index * 3U + 1U ] = rgba[ 1 ];
              streams.colors[ index * 3U + 2U ] = rgba[ 2 ];
            });
        }
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

      if (primitive.materialIndex.has_value())
      {
        streams.material_index =
          static_cast<std::uint32_t>(*primitive.materialIndex);
      }

      mesh.primitives.push_back(std::move(streams));
    }
  }
  return mesh;
}

[[nodiscard]] constexpr auto
map_gltf_sampler(const fastgltf::Sampler& sampler) -> gltf::sampler_cpu
{
  constexpr auto address = [](fastgltf::Wrap wrap) -> vk::SamplerAddressMode
  {
    switch (wrap)
    {
    case fastgltf::Wrap::ClampToEdge:
      return vk::SamplerAddressMode::eClampToEdge;
    case fastgltf::Wrap::MirroredRepeat:
      return vk::SamplerAddressMode::eMirroredRepeat;
    case fastgltf::Wrap::Repeat:
      return vk::SamplerAddressMode::eRepeat;
    }
    return vk::SamplerAddressMode::eRepeat;
  };

  gltf::sampler_cpu mapped {
    .address_u = address(sampler.wrapS),
    .address_v = address(sampler.wrapT),
  };

  if (sampler.magFilter == fastgltf::Filter::Nearest)
  {
    mapped.mag_filter = vk::Filter::eNearest;
  }

  switch (sampler.minFilter.value_or(fastgltf::Filter::LinearMipMapLinear))
  {
  case fastgltf::Filter::Nearest:
  case fastgltf::Filter::NearestMipMapNearest:
    mapped.min_filter = vk::Filter::eNearest;
    mapped.mipmap_mode = vk::SamplerMipmapMode::eNearest;
    break;
  case fastgltf::Filter::Linear:
  case fastgltf::Filter::LinearMipMapNearest:
    mapped.min_filter = vk::Filter::eLinear;
    mapped.mipmap_mode = vk::SamplerMipmapMode::eNearest;
    break;
  case fastgltf::Filter::NearestMipMapLinear:
    mapped.min_filter = vk::Filter::eNearest;
    mapped.mipmap_mode = vk::SamplerMipmapMode::eLinear;
    break;
  default:
    break; // std::unreachable(); ?
  }
  return mapped;
}

[[nodiscard]] auto
build_draw_list(const fastgltf::Asset& gltf) -> std::vector<gltf::draw_item_cpu>
{
  std::vector<gltf::draw_item_cpu> draws {};
  if (gltf.scenes.empty()) { return draws; }

  std::vector<std::uint32_t> first_primitive(gltf.meshes.size(), 0U);
  std::uint32_t running { 0U };
  // TODO (Konrad): change all views::iota that were used for indices to this
  for (auto mesh_index : std::views::indices(gltf.meshes.size()))
  {
    first_primitive[ mesh_index ] = running;
    running +=
      static_cast<std::uint32_t>(gltf.meshes[ mesh_index ].primitives.size());
  }

  const auto scene_index = gltf.defaultScene.value_or(0UZ);
  fastgltf::iterateSceneNodes(gltf, scene_index, fastgltf::math::fmat4x4 {},
    [ & ](
      const fastgltf::Node& node, const fastgltf::math::fmat4x4& world) -> void
    {
      if (!node.meshIndex.has_value()) { return; }
      const auto& mesh = gltf.meshes[ *node.meshIndex ];
      for (auto offset : std::views::indices(mesh.primitives.size()))
      {
        gltf::draw_item_cpu item {
          .primitive_index = first_primitive[ *node.meshIndex ] +
            static_cast<std::uint32_t>(offset),
        };
        for (auto column : std::views::indices(4UZ))
        {
          for (auto row : std::views::indices(4UZ))
          {
            item.world_transform[ column * 4UZ + row ] = world[ column ][ row ];
          }
        }
        draws.push_back(item);
      }
    });
  return draws;
}

[[nodiscard]] auto
read_file_bytes(const std::filesystem::path& path, std::size_t byte_offset)
  -> std::expected<std::vector<std::byte>, error_t>
{
  std::ifstream input { path, std::ios::ate | std::ios::binary };
  if (!input.is_open())
  {
    return std::unexpected {
      app_error {
        .kind = app_error_kind::file_open,
        .detail = "Failed to open glTF-referenced image file"sv,
      },
    };
  }

  const auto total = static_cast<std::size_t>(input.tellg());
  std::vector<std::byte> bytes(total - byte_offset);
  input.seekg(static_cast<std::streamoff>(byte_offset), std::ios::beg);
  input.read(reinterpret_cast<char*>(bytes.data()),
    static_cast<std::streamsize>(bytes.size()));
  return bytes;
}

// maybe in some utility directory/module - I like this better than if constexpr
// over types
template<typename... Ts>
struct match : Ts...
{
  using Ts::operator()...;
};

template<typename... Ts>
match(Ts...) -> match<Ts...>;

[[nodiscard]] auto
extract_image_source(const fastgltf::Asset& gltf, const fastgltf::Image& image,
  const std::filesystem::path& directory)
  -> std::expected<gltf::image_source_cpu, error_t>
{
  gltf::image_source_cpu source {};
  const auto visit_bytes =
    [ & ](std::span<const std::byte> bytes,
      fastgltf::MimeType mime) -> std::expected<gltf::image_source_cpu, error_t>
  {
    switch (mime)
    {
    case fastgltf::MimeType::PNG:
    {
      source.kind = gltf::image_kind::encoded_png;
      break;
    }
    case fastgltf::MimeType::JPEG:
    {
      source.kind = gltf::image_kind::encoded_jpeg;
      break;
    }
    case fastgltf::MimeType::KTX2:
    {
      source.kind = gltf::image_kind::encoded_ktx2;
      break;
    }
    default:
    {
      source.kind = gltf::image_kind::encoded_other;
      break;
    }
    }
    source.encoded_bytes.assign_range(bytes);
    return source;
  };

  using return_value = std::expected<gltf::image_source_cpu, error_t>;
  return std::visit(
    match {
      [ & ](const fastgltf::sources::BufferView& data) -> return_value
      {
        const std::span<const std::byte> bytes =
          fastgltf::DefaultBufferDataAdapter {}(gltf, data.bufferViewIndex);
        return visit_bytes(bytes, data.mimeType);
      },
      [ & ](const fastgltf::sources::Array& data) -> return_value
      {
        return visit_bytes(
          std::span { data.bytes.data(), data.bytes.size() }, data.mimeType);
      },
      [ & ](const fastgltf::sources::ByteView& data) -> return_value
      { return visit_bytes(data.bytes, data.mimeType); },
      [ & ](const fastgltf::sources::URI& file) -> return_value
      {
        if (!file.uri.isLocalPath())
        {
          return std::unexpected {
            app_error {
              .kind = app_error_kind::model_parse,
              .detail = "non-local glTF image URI"sv,
            },
          };
        }
        const std::filesystem::path full_path = directory / file.uri.fspath();
        return read_file_bytes(full_path, file.fileByteOffset)
          .and_then(
            [ & ](std::vector<std::byte>&& bytes) -> return_value
            {
              fastgltf::MimeType mime = file.mimeType;
              if (mime == fastgltf::MimeType::None)
              {
                const auto extension = full_path.extension();
                if (extension == ".png") { mime = fastgltf::MimeType::PNG; }
                else if (extension == ".jpg" || extension == ".jpeg")
                {
                  mime = fastgltf::MimeType::JPEG;
                }
                else if (extension == ".ktx2")
                {
                  mime = fastgltf::MimeType::KTX2;
                }
              }
              auto result = visit_bytes(std::span { bytes }, mime);
              if (result) { result->debug_uri = full_path; }
              return result;
            });
      },
      [ & ](const auto& data) -> return_value
      {
        return std::unexpected {
          app_error {
            .kind = app_error_kind::model_parse,
            .detail = "unsupported glTF image DataSource"sv,
          },
        };
      },
    },
    image.data);
}

export [[nodiscard]] auto
realize_gltf_host_images(std::span<const gltf::image_source_cpu> sources,
  const gltf::load_runtime_args& runtime_args)
  -> std::expected<std::vector<gltf::host_image_cpu>, error_t>
{
  std::vector<gltf::host_image_cpu> out {};
  out.resize(sources.size());
  for (auto index : std::views::indices(sources.size()))
  {
    const gltf::image_source_cpu& source = sources[ index ];
    switch (source.kind)
    {
    case gltf::image_kind::encoded_png:
    case gltf::image_kind::encoded_jpeg:
    {
      auto decoded = load_host_image_stb_from_memory(source.encoded_bytes);
      if (!decoded) { return std::unexpected { std::move(decoded).error() }; }
      out[ index ].decoded = std::move(*decoded);
      break;
    }
    case gltf::image_kind::encoded_ktx2:
    {
      auto chain = load_host_image_ktx2_from_memory(
        source.encoded_bytes, runtime_args.ktx2);
      if (!chain) { return std::unexpected { std::move(chain).error() }; }
      out[ index ].mip_chain = std::move(*chain);
      break;
    }
    case gltf::image_kind::encoded_other:
      return std::unexpected {
        app_error {
          .kind = app_error_kind::image_decode,
          .detail = "unsupported glTF image mime for host realize"sv,
        },
      };
    }
  }
  return out;
}

export [[nodiscard]] auto
load_gltf_asset_cpu(const std::filesystem::path& path,
  const gltf::load_runtime_args& runtime_args = {})
  -> std::expected<gltf::asset_cpu, error_t>
{
  auto data = fastgltf::GltfDataBuffer::FromPath(path);
  if (data.error() != fastgltf::Error::None)
  {
    return std::unexpected { map_fastgltf_error(data.error()) };
  }

  fastgltf::Parser parser = make_gltf_parser(runtime_args);
  auto asset = parser.loadGltf(data.get(), path.parent_path(),
    gltf_options(runtime_args), category_for(runtime_args.content));
  if (asset.error() != fastgltf::Error::None)
  {
    return std::unexpected { map_fastgltf_error(asset.error()) };
  }

  const fastgltf::Asset& gltf = asset.get();
  gltf::asset_cpu out {};
  return extract_mesh_cpu(gltf).and_then(
    [ & ](mesh_cpu&& meshes) -> std::expected<gltf::asset_cpu, error_t>
    {
      out.meshes = std::move(meshes);
      out.draw_list = build_draw_list(gltf);
      if (runtime_args.content == gltf::content_policy::geometry_only)
      {
        return out;
      }

      out.materials.reserve(gltf.materials.size());
      for (const auto& material : gltf.materials)
      {
        gltf::material_cpu mapped {
          .base_color_factor = {
            material.pbrData.baseColorFactor[ 0 ],
            material.pbrData.baseColorFactor[ 1 ],
            material.pbrData.baseColorFactor[ 2 ],
            material.pbrData.baseColorFactor[ 3 ],
          },
          .metallic_factor = material.pbrData.metallicFactor,
          .roughness_factor = material.pbrData.roughnessFactor,
        };
        if (material.pbrData.baseColorTexture.has_value())
        {
          const auto texture_index =
            material.pbrData.baseColorTexture->textureIndex;
          const auto& texture = gltf.textures[ texture_index ];
          mapped.base_color_texture = {
            .image_index = texture.imageIndex.has_value()
              ? std::optional { static_cast<std::uint32_t>(
                  *texture.imageIndex) }
              : std::nullopt,
            .basisu_image_index = texture.basisuImageIndex.has_value()
              ? std::optional { static_cast<std::uint32_t>(
                  *texture.basisuImageIndex) }
              : std::nullopt,
            .sampler_index = texture.samplerIndex.has_value()
              ? std::optional { static_cast<std::uint32_t>(
                  *texture.samplerIndex) }
              : std::nullopt,
          };
        }
        out.materials.push_back(std::move(mapped));
        out.samplers.reserve(gltf.samplers.size());
        for (const fastgltf::Sampler& sampler : gltf.samplers)
        {
          out.samplers.push_back(map_gltf_sampler(sampler));
        }
      }

      for (const auto& texture : gltf.textures)
      {
        out.textures.push_back({
          .image_index = texture.imageIndex.has_value()
            ? std::optional { static_cast<std::uint32_t>(*texture.imageIndex) }
            : std::nullopt,
          .basisu_image_index = texture.basisuImageIndex.has_value()
            ? std::optional { static_cast<std::uint32_t>(
                *texture.basisuImageIndex) }
            : std::nullopt,
          .sampler_index = texture.samplerIndex.has_value()
            ? std::optional { static_cast<std::uint32_t>(
                *texture.samplerIndex) }
            : std::nullopt,
        });
      }
      if (runtime_args.content == gltf::content_policy::geometry_and_materials)
      {
        return out;
      }
      out.image_sources.reserve(gltf.images.size());
      for (const auto& image : gltf.images)
      {
        auto source = extract_image_source(gltf, image, path.parent_path());
        if (!source) { return std::unexpected { std::move(source).error() }; }
        out.image_sources.push_back(std::move(*source));
      }
      return realize_gltf_host_images(out.image_sources, runtime_args)
        .transform(
          [ & ](std::vector<gltf::host_image_cpu>&& images) -> gltf::asset_cpu
          {
            out.host_images = std::move(images);
            return std::move(out);
          });
    });
}

template<>
[[nodiscard]] auto
load_mesh_cpu<mesh_file_type::gltf>(const std::filesystem::path& path)
  -> std::expected<mesh_cpu, error_t>
{
  return load_gltf_asset_cpu(
    path, { .content = gltf::content_policy::geometry_only })
    .transform([](gltf::asset_cpu&& asset) { return std::move(asset.meshes); });
}

} // namespace vkpp
