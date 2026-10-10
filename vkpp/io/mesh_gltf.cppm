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

export [[nodiscard]]
auto
make_fastgltf_error(fastgltf::Error error) -> error_t
{
  return error_t {
    .domain = error_domain::fastgltf,
    .code   = static_cast<std::int32_t>(error),
  };
}

[[nodiscard]]
auto
make_gltf_parser(const gltf::load_runtime_args& runtime_args) -> fastgltf::Parser
{
  fastgltf::Extensions extensions {fastgltf::Extensions::None};
  if (runtime_args.enable_texture_basisu)
  {
    extensions |= fastgltf::Extensions::KHR_texture_basisu;
  }
  extensions |= fastgltf::Extensions::KHR_materials_clearcoat;
  extensions |= fastgltf::Extensions::KHR_materials_transmission;
  return fastgltf::Parser {extensions};
}

[[nodiscard]]
auto
gltf_options(const gltf::load_runtime_args& runtime_args) -> fastgltf::Options
{
  fastgltf::Options options = fastgltf::Options::None;
  if (runtime_args.load_external_buffers)
  {
    options |= fastgltf::Options::LoadExternalBuffers;
  }
  if (
    runtime_args.load_external_images &&
    runtime_args.content != gltf::content_policy::geometry_only)
  {
    options |= fastgltf::Options::LoadExternalImages;
  }
  return options;
}

[[nodiscard]]
auto
category_for(gltf::content_policy content) -> fastgltf::Category
{
  using fastgltf::Category;
  switch (content)
  {
  case gltf::content_policy::geometry_only :
    return Category::Buffers |
           Category::BufferViews |
           Category::Accessors |
           Category::Meshes |
           Category::Nodes |
           Category::Scenes;
  case gltf::content_policy::geometry_and_materials :
    return Category::OnlyRenderable;
  case gltf::content_policy::geometry_and_host_images :
    return Category::OnlyRenderable;
  }
  return Category::Meshes;
}

[[nodiscard]]
auto
extract_mesh_cpu(const fastgltf::Asset& gltf) -> std::expected<mesh_cpu, error_t>
{
  mesh_cpu mesh {};
  for (const fastgltf::Mesh& gltf_mesh : gltf.meshes)
  {
    for (const fastgltf::Primitive& primitive : gltf_mesh.primitives)
    {
      const auto* position_it = primitive.findAttribute("POSITION");
      if (position_it == primitive.attributes.end())
      {
        return std::unexpected {make_app_error(app_error_code::model_parse)};
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
      if (
        const auto* normal_it = primitive.findAttribute("NORMAL");
        normal_it != primitive.attributes.end())
      {
        streams.normals.resize(streams.vertex_count * 3U);
        fastgltf::copyComponentsFromAccessor<float>(
          gltf,
          gltf.accessors[ normal_it->accessorIndex ],
          streams.normals.data());
      }

      if (
        const auto* color_it = primitive.findAttribute("COLOR_0");
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
          fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec4>(
            gltf,
            color_accessor,
            [ & ](fastgltf::math::fvec4 rgba, std::size_t index) -> void {
              streams.colors[ index * 3U ]      = rgba[ 0 ];
              streams.colors[ index * 3U + 1U ] = rgba[ 1 ];
              streams.colors[ index * 3U + 2U ] = rgba[ 2 ];
            });
        }
      }

      if (
        const auto* uv_it = primitive.findAttribute("TEXCOORD_0");
        uv_it != primitive.attributes.end())
      {
        fastgltf::iterateAccessorWithIndex<glm::vec2>(
          gltf,
          gltf.accessors[ uv_it->accessorIndex ],
          [ & ](glm::vec2 uv, std::size_t index) {
            streams.texcoords[ index * 2U ]      = uv.x;
            streams.texcoords[ index * 2U + 1U ] = uv.y;
          });
      }

      if (!primitive.indicesAccessor.has_value())
      {
        return std::unexpected {
          make_app_error(app_error_code::unsupported_model_data),
        };
      }

      const fastgltf::Accessor& index_accessor =
        gltf.accessors[ *primitive.indicesAccessor ];
      streams.index_count = static_cast<std::uint32_t>(index_accessor.count);

      if (index_accessor.componentType == fastgltf::ComponentType::UnsignedShort)
      {
        streams.index_type = vk::IndexType::eUint16;
        std::vector<std::uint16_t> indices(streams.index_count);
        fastgltf::copyFromAccessor<std::uint16_t>(
          gltf, index_accessor, indices.data());
        streams.indices.resize(indices.size() * sizeof(std::uint16_t));
        std::memcpy(
          streams.indices.data(), indices.data(), streams.indices.size());
      }
      else if (
        index_accessor.componentType == fastgltf::ComponentType::UnsignedInt)
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
          make_app_error(app_error_code::unsupported_model_data),
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

[[nodiscard]]
constexpr
auto
map_gltf_sampler(const fastgltf::Sampler& sampler) -> gltf::sampler_cpu
{
  constexpr auto address = [](fastgltf::Wrap wrap) -> vk::SamplerAddressMode {
    switch (wrap)
    {
    case fastgltf::Wrap::ClampToEdge :
      return vk::SamplerAddressMode::eClampToEdge;
    case fastgltf::Wrap::MirroredRepeat :
      return vk::SamplerAddressMode::eMirroredRepeat;
    case fastgltf::Wrap::Repeat : return vk::SamplerAddressMode::eRepeat;
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
  case fastgltf::Filter::Nearest :
    mapped.min_filter  = vk::Filter::eNearest;
    mapped.mipmap_mode = vk::SamplerMipmapMode::eNearest;
    mapped.max_lod     = 0.0F;
    break;
  case fastgltf::Filter::Linear :
    mapped.min_filter  = vk::Filter::eLinear;
    mapped.mipmap_mode = vk::SamplerMipmapMode::eNearest;
    mapped.max_lod     = 0.0F;
    break;
  case fastgltf::Filter::NearestMipMapNearest :
    mapped.min_filter  = vk::Filter::eNearest;
    mapped.mipmap_mode = vk::SamplerMipmapMode::eNearest;
    break;
  case fastgltf::Filter::LinearMipMapNearest :
    mapped.min_filter  = vk::Filter::eLinear;
    mapped.mipmap_mode = vk::SamplerMipmapMode::eNearest;
    break;
  case fastgltf::Filter::NearestMipMapLinear :
    mapped.min_filter  = vk::Filter::eNearest;
    mapped.mipmap_mode = vk::SamplerMipmapMode::eLinear;
    break;
  case fastgltf::Filter::LinearMipMapLinear :
    mapped.min_filter  = vk::Filter::eLinear;
    mapped.mipmap_mode = vk::SamplerMipmapMode::eLinear;
    break;
  }
  return mapped;
}

[[nodiscard]]
constexpr
auto
map_texture_ref(const fastgltf::Texture& texture) -> gltf::texture_ref_cpu
{
  return {
    .image_index =
      texture.imageIndex.has_value()
        ? std::optional {static_cast<std::uint32_t>(*texture.imageIndex)}
        : std::nullopt,
    .basisu_image_index =
      texture.basisuImageIndex.has_value()
        ? std::optional {static_cast<std::uint32_t>(*texture.basisuImageIndex)}
        : std::nullopt,
    .sampler_index =
      texture.samplerIndex.has_value()
        ? std::optional {static_cast<std::uint32_t>(*texture.samplerIndex)}
        : std::nullopt,
  };
}

[[nodiscard]]
constexpr
auto
map_texture_use(
  const std::optional<fastgltf::TextureInfo>& info,
  image_color_space                           role,
  std::size_t                                 texture_count)
  -> std::expected<std::optional<gltf::texture_use_cpu>, error_t>
{
  if (!info.has_value()) { return std::nullopt; }
  if (info->textureIndex >= texture_count)
  {
    return std::unexpected {make_app_error(app_error_code::out_of_range)};
  }
  return gltf::texture_use_cpu {
    .texture_index  = static_cast<std::uint32_t>(info->textureIndex),
    .texcoord_index = static_cast<std::uint32_t>(info->texCoordIndex),
    .color_space    = role,
  };
}

[[nodiscard]]
constexpr
auto
map_normal_texture_use(
  const std::optional<fastgltf::NormalTextureInfo>& info,
  image_color_space                                 role,
  std::size_t                                       texture_count)
  -> std::expected<std::optional<gltf::texture_use_cpu>, error_t>
{
  if (!info.has_value()) { return std::nullopt; }
  if (info->textureIndex >= texture_count)
  {
    return std::unexpected {make_app_error(app_error_code::out_of_range)};
  }
  return gltf::texture_use_cpu {
    .texture_index  = static_cast<std::uint32_t>(info->textureIndex),
    .texcoord_index = static_cast<std::uint32_t>(info->texCoordIndex),
    .color_space    = role,
  };
}

[[nodiscard]]
constexpr
auto
map_occlusion_texture_use(
  const std::optional<fastgltf::OcclusionTextureInfo>& info,
  image_color_space                                    role,
  std::size_t                                          texture_count)
  -> std::expected<std::optional<gltf::texture_use_cpu>, error_t>
{
  if (!info.has_value()) { return std::nullopt; }
  if (info->textureIndex >= texture_count)
  {
    return std::unexpected {make_app_error(app_error_code::out_of_range)};
  }
  return gltf::texture_use_cpu {
    .texture_index  = static_cast<std::uint32_t>(info->textureIndex),
    .texcoord_index = static_cast<std::uint32_t>(info->texCoordIndex),
    .color_space    = role,
  };
}

[[nodiscard]]
constexpr
auto
map_alpha_mode(fastgltf::AlphaMode mode) -> gltf::alpha_mode
{
  switch (mode)
  {
  case fastgltf::AlphaMode::Opaque : return gltf::alpha_mode::opaque;
  case fastgltf::AlphaMode::Mask   : return gltf::alpha_mode::mask;
  case fastgltf::AlphaMode::Blend  : return gltf::alpha_mode::blend;
  }
  return gltf::alpha_mode::opaque;
}

[[nodiscard]]
auto
build_draw_list(const fastgltf::Asset& gltf) -> std::vector<gltf::draw_item_cpu>
{
  std::vector<gltf::draw_item_cpu> draws {};
  if (gltf.scenes.empty()) { return draws; }

  std::vector<std::uint32_t> first_primitive(gltf.meshes.size(), 0U);
  std::uint32_t              running {0U};
  for (auto mesh_index : std::views::indices(gltf.meshes.size()))
  {
    first_primitive[ mesh_index ] = running;
    running +=
      static_cast<std::uint32_t>(gltf.meshes[ mesh_index ].primitives.size());
  }

  const auto scene_index = gltf.defaultScene.value_or(0UZ);
  fastgltf::iterateSceneNodes(
    gltf,
    scene_index,
    fastgltf::math::fmat4x4 {1.0F},
    [ & ](const fastgltf::Node& node, const fastgltf::math::fmat4x4& world)
      -> void {
      if (!node.meshIndex.has_value()) { return; }
      const auto& mesh = gltf.meshes[ *node.meshIndex ];
      for (auto offset : std::views::indices(mesh.primitives.size()))
      {
        gltf::draw_item_cpu item {
          .primitive_index =
            first_primitive[ *node.meshIndex ] +
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

[[nodiscard]]
auto
read_file_bytes(const std::filesystem::path& path, std::size_t byte_offset)
  -> std::expected<std::vector<std::byte>, error_t>
{
  std::ifstream input {path, std::ios::ate | std::ios::binary};
  if (!input.is_open())
  {
    return std::unexpected {make_app_error(app_error_code::file_open)};
  }

  const auto             total = static_cast<std::size_t>(input.tellg());
  std::vector<std::byte> bytes(total - byte_offset);
  input.seekg(static_cast<std::streamoff>(byte_offset), std::ios::beg);
  input.read(
    reinterpret_cast<char*>(bytes.data()),
    static_cast<std::streamsize>(bytes.size()));
  return bytes;
}

// maybe in some utility directory/module - I like this better than if constexpr
// over types
template<typename... Ts>
struct match : Ts...
{
  using Ts::operator ()...;
};

template<typename... Ts>
match(Ts...) -> match<Ts...>;

[[nodiscard]]
auto
extract_image_source(
  const fastgltf::Asset&       gltf,
  const fastgltf::Image&       image,
  const std::filesystem::path& directory)
  -> std::expected<gltf::image_source_cpu, error_t>
{
  gltf::image_source_cpu source {};
  const auto             visit_bytes =
    [ & ](std::span<const std::byte> bytes, fastgltf::MimeType mime)
    -> std::expected<gltf::image_source_cpu, error_t> {
    switch (mime)
    {
    case fastgltf::MimeType::PNG :
    {
      source.kind = gltf::image_kind::encoded_png;
      break;
    }
    case fastgltf::MimeType::JPEG :
    {
      source.kind = gltf::image_kind::encoded_jpeg;
      break;
    }
    case fastgltf::MimeType::KTX2 :
    {
      source.kind = gltf::image_kind::encoded_ktx2;
      break;
    }
    default :
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
      [ & ](const fastgltf::sources::BufferView& data) -> return_value {
        const std::span<const std::byte> bytes =
          fastgltf::DefaultBufferDataAdapter {}(gltf, data.bufferViewIndex);
        return visit_bytes(bytes, data.mimeType);
      },
      [ & ](const fastgltf::sources::Array& data) -> return_value {
        return visit_bytes(
          std::span {data.bytes.data(), data.bytes.size()}, data.mimeType);
      },
      [ & ](const fastgltf::sources::ByteView& data) -> return_value {
        return visit_bytes(data.bytes, data.mimeType);
      },
      [ & ](const fastgltf::sources::URI& file) -> return_value {
        if (!file.uri.isLocalPath())
        {
          return std::unexpected {
            make_app_error(app_error_code::model_parse),
          };
        }
        const std::filesystem::path full_path = directory / file.uri.fspath();
        return read_file_bytes(full_path, file.fileByteOffset)
          .and_then([ & ](std::vector<std::byte>&& bytes) -> return_value {
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
            auto result = visit_bytes(std::span {bytes}, mime);
            if (result) { result->debug_uri = full_path; }
            return result;
          });
      },
      [ & ](const auto& data) -> return_value {
        return std::unexpected {
          make_app_error(app_error_code::unsupported_model_data),
        };
      },
    },
    image.data);
}

export [[nodiscard]]
auto
realize_gltf_host_images(
  std::span<const gltf::image_source_cpu>           sources,
  std::span<const gltf::host_image_realization_key> keys,
  const gltf::load_runtime_args&                    runtime_args)
  -> std::expected<std::vector<gltf::realized_host_image_cpu>, error_t>
{
  for (auto index : std::views::indices(keys.size()))
  {
    for (auto later : std::views::iota(index + 1UZ, keys.size()))
    {
      if (keys[ index ].source_index != keys[ later ].source_index)
      {
        continue;
      }
      if (keys[ index ].color_space != keys[ later ].color_space) { continue; }
      if (
        sources[ keys[ index ].source_index ].kind ==
        gltf::image_kind::encoded_ktx2)
      {
        return std::unexpected {
          make_app_error(app_error_code::unsupported_model_data),
        };
      }
    }
  }

  std::vector<gltf::realized_host_image_cpu> out {};
  out.reserve(keys.size());
  for (const auto& key : keys)
  {
    if (key.source_index >= sources.size())
    {
      return std::unexpected {make_app_error(app_error_code::out_of_range)};
    }
    const auto&                   source = sources[ key.source_index ];
    gltf::realized_host_image_cpu realized {
      .source_index = key.source_index,
      .color_space  = key.color_space,
    };
    switch (source.kind)
    {
    case gltf::image_kind::encoded_png :
    case gltf::image_kind::encoded_jpeg :
    {
      auto decoded = load_host_image_stb_from_memory(source.encoded_bytes);
      if (!decoded) { return std::unexpected {std::move(decoded).error()}; }
      if (key.color_space == image_color_space::linear)
      {
        decoded->format = to_linear_format(decoded->format);
      }
      realized.image.decoded = std::move(*decoded);
      break;
    }
    case gltf::image_kind::encoded_ktx2 :
    {
      auto chain = load_host_image_ktx2_from_memory(
        source.encoded_bytes, runtime_args.ktx2, key.color_space);
      if (!chain) { return std::unexpected {std::move(chain).error()}; }
      realized.image.mip_chain = std::move(*chain);
      break;
    }
    case gltf::image_kind::encoded_other :
      return std::unexpected {
        make_app_error(app_error_code::unsupported_image_format),
      };
    }
    out.push_back(std::move(realized));
  }
  return out;
}

[[nodiscard]]
auto
selected_image_index(const gltf::texture_ref_cpu& texture)
  -> std::optional<std::uint32_t>
{
  return texture.basisu_image_index.has_value()
           ? texture.basisu_image_index
           : texture.image_index;
}

[[nodiscard]]
auto
collect_gltf_host_image_realization_keys(
  std::span<const gltf::material_cpu>    materials,
  std::span<const gltf::texture_ref_cpu> textures)
  -> std::expected<std::vector<gltf::host_image_realization_key>, error_t>
{
  std::vector<gltf::host_image_realization_key> keys {};
  const auto consider = [ & ](const std::optional<gltf::texture_use_cpu>& use)
    -> std::expected<void, error_t> {
    if (!use.has_value()) { return {}; }
    if (use->texture_index >= textures.size())
    {
      return std::unexpected {make_app_error(app_error_code::out_of_range)};
    }
    const auto source = selected_image_index(textures[ use->texture_index ]);
    if (!source.has_value()) { return {}; }
    keys.push_back({
      .source_index = *source,
      .color_space  = use->color_space,
    });
    return {};
  };

  for (const auto& material : materials)
  {
    if (auto _ = consider(material.base_color_texture); !_)
    {
      return std::unexpected {_.error()};
    }
    if (auto _ = consider(material.metallic_roughness_texture); !_)
    {
      return std::unexpected {_.error()};
    }
    if (auto _ = consider(material.normal_texture); !_)
    {
      return std::unexpected {_.error()};
    }
    if (auto _ = consider(material.occlusion_texture); !_)
    {
      return std::unexpected {_.error()};
    }
    if (auto _ = consider(material.emissive_texture); !_)
    {
      return std::unexpected {_.error()};
    }
    if (auto _ = consider(material.transmission.texture); !_)
    {
      return std::unexpected {_.error()};
    }
    if (auto _ = consider(material.clearcoat.texture); !_)
    {
      return std::unexpected {_.error()};
    }
    if (auto _ = consider(material.clearcoat.roughness_texture); !_)
    {
      return std::unexpected {_.error()};
    }
    if (auto _ = consider(material.clearcoat.normal_texture); !_)
    {
      return std::unexpected {_.error()};
    }
  }

  std::ranges::sort(keys, {}, [](const gltf::host_image_realization_key& key) {
    return std::pair {
      key.source_index, static_cast<std::uint8_t>(key.color_space)
    };
  });
  const auto unique_end = std::ranges::unique(keys).begin();
  keys.erase(unique_end, keys.end());
  return keys;
}

export [[nodiscard]]
auto
load_gltf_asset_cpu(
  const std::filesystem::path&   path,
  const gltf::load_runtime_args& runtime_args = {})
  -> std::expected<gltf::asset_cpu, error_t>
{
  auto data = fastgltf::GltfDataBuffer::FromPath(path);
  if (data.error() != fastgltf::Error::None)
  {
    return std::unexpected {make_fastgltf_error(data.error())};
  }

  fastgltf::Parser parser = make_gltf_parser(runtime_args);
  auto             asset  = parser.loadGltf(
    data.get(),
    path.parent_path(),
    gltf_options(runtime_args),
    category_for(runtime_args.content));
  if (asset.error() != fastgltf::Error::None)
  {
    return std::unexpected {make_fastgltf_error(asset.error())};
  }

  const fastgltf::Asset& gltf = asset.get();
  gltf::asset_cpu        out {};
  return extract_mesh_cpu(gltf).and_then(
    [ & ](mesh_cpu&& meshes) -> std::expected<gltf::asset_cpu, error_t> {
      out.meshes    = std::move(meshes);
      out.draw_list = build_draw_list(gltf);
      if (runtime_args.content == gltf::content_policy::geometry_only)
      {
        return out;
      }

      out.materials.reserve(gltf.materials.size());
      for (const auto& material : gltf.materials)
      {
        const auto texture_count = gltf.textures.size();

        auto base_color_texture = map_texture_use(
          material.pbrData.baseColorTexture,
          image_color_space::srgb,
          texture_count);
        if (!base_color_texture)
        {
          return std::unexpected {base_color_texture.error()};
        }

        auto metallic_roughness_texture = map_texture_use(
          material.pbrData.metallicRoughnessTexture,
          image_color_space::linear,
          texture_count);
        if (!metallic_roughness_texture)
        {
          return std::unexpected {metallic_roughness_texture.error()};
        }

        auto normal_texture = map_normal_texture_use(
          material.normalTexture, image_color_space::linear, texture_count);
        if (!normal_texture)
        {
          return std::unexpected {normal_texture.error()};
        }

        auto occlusion_texture = map_occlusion_texture_use(
          material.occlusionTexture, image_color_space::linear, texture_count);
        if (!occlusion_texture)
        {
          return std::unexpected {occlusion_texture.error()};
        }

        auto emissive_texture = map_texture_use(
          material.emissiveTexture, image_color_space::srgb, texture_count);
        if (!emissive_texture)
        {
          return std::unexpected {emissive_texture.error()};
        }

        gltf::material_cpu mapped {
          .base_color_factor =
            {
              material.pbrData.baseColorFactor[ 0 ],
              material.pbrData.baseColorFactor[ 1 ],
              material.pbrData.baseColorFactor[ 2 ],
              material.pbrData.baseColorFactor[ 3 ],
            },
          .metallic_factor            = material.pbrData.metallicFactor,
          .roughness_factor           = material.pbrData.roughnessFactor,
          .base_color_texture         = std::move(*base_color_texture),
          .metallic_roughness_texture = std::move(*metallic_roughness_texture),
          .normal_texture             = std::move(*normal_texture),
          .normal_scale =
            material.normalTexture.has_value() ? material.normalTexture->scale : 1.0F,
          .occlusion_texture = std::move(*occlusion_texture),
          .occlusion_strength =
            material.occlusionTexture.has_value()
              ? material.occlusionTexture->strength
              : 1.0F,
          .emissive_texture = std::move(*emissive_texture),
          .emissive_factor =
            {
              material.emissiveFactor[ 0 ],
              material.emissiveFactor[ 1 ],
              material.emissiveFactor[ 2 ],
            },
          .alpha_mode   = map_alpha_mode(material.alphaMode),
          .alpha_cutoff = material.alphaCutoff,
          .double_sided = material.doubleSided,
        };

        if (material.transmission)
        {
          auto transmission_texture = map_texture_use(
            material.transmission->transmissionTexture,
            image_color_space::linear,
            texture_count);
          if (!transmission_texture)
          {
            return std::unexpected {transmission_texture.error()};
          }
          mapped.transmission.factor =
            static_cast<float>(material.transmission->transmissionFactor);
          mapped.transmission.texture = std::move(*transmission_texture);
        }

        if (material.clearcoat)
        {
          auto clearcoat_texture = map_texture_use(
            material.clearcoat->clearcoatTexture,
            image_color_space::linear,
            texture_count);
          if (!clearcoat_texture)
          {
            return std::unexpected {clearcoat_texture.error()};
          }
          auto clearcoat_roughness_texture = map_texture_use(
            material.clearcoat->clearcoatRoughnessTexture,
            image_color_space::linear,
            texture_count);
          if (!clearcoat_roughness_texture)
          {
            return std::unexpected {clearcoat_roughness_texture.error()};
          }
          auto clearcoat_normal_texture = map_normal_texture_use(
            material.clearcoat->clearcoatNormalTexture,
            image_color_space::linear,
            texture_count);
          if (!clearcoat_normal_texture)
          {
            return std::unexpected {clearcoat_normal_texture.error()};
          }

          mapped.clearcoat.factor =
            static_cast<float>(material.clearcoat->clearcoatFactor);
          mapped.clearcoat.roughness_factor =
            static_cast<float>(material.clearcoat->clearcoatRoughnessFactor);
          mapped.clearcoat.normal_scale =
            material.clearcoat->clearcoatNormalTexture.has_value()
              ? material.clearcoat->clearcoatNormalTexture->scale
              : 1.0F;

          mapped.clearcoat.texture = std::move(*clearcoat_texture);
          mapped.clearcoat.roughness_texture =
            std::move(*clearcoat_roughness_texture);
          mapped.clearcoat.normal_texture = std::move(*clearcoat_normal_texture);
        }

        out.materials.push_back(std::move(mapped));
      }

      out.samplers.reserve(gltf.samplers.size());
      for (const fastgltf::Sampler& sampler : gltf.samplers)
      {
        out.samplers.push_back(map_gltf_sampler(sampler));
      }

      out.textures.reserve(gltf.textures.size());
      for (const auto& texture : gltf.textures)
      {
        out.textures.push_back(map_texture_ref(texture));
      }

      if (runtime_args.content == gltf::content_policy::geometry_and_materials)
      {
        return out;
      }
      out.image_sources.reserve(gltf.images.size());
      for (const auto& image : gltf.images)
      {
        auto source = extract_image_source(gltf, image, path.parent_path());
        if (!source) { return std::unexpected {std::move(source).error()}; }
        out.image_sources.push_back(std::move(*source));
      }
      return collect_gltf_host_image_realization_keys(
        out.materials, out.textures)
        .and_then(
          [ & ](std::vector<gltf::host_image_realization_key>&& keys)
            -> std::expected<gltf::asset_cpu, error_t> {
            return realize_gltf_host_images(
              out.image_sources, keys, runtime_args)
              .transform(
                [ & ](std::vector<gltf::realized_host_image_cpu>&& images)
                  -> gltf::asset_cpu {
                  out.host_images = std::move(images);
                  return std::move(out);
                });
          });
    });
}

template<>
[[nodiscard]]
auto
load_mesh_cpu<mesh_file_type::gltf>(const std::filesystem::path& path)
  -> std::expected<mesh_cpu, error_t>
{
  return load_gltf_asset_cpu(
    path, {.content = gltf::content_policy::geometry_only})
    .transform([](gltf::asset_cpu&& asset) { return std::move(asset.meshes); });
}

} // namespace vkpp
