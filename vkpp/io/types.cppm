module;
export module vkpp.io.types;

import std;
import vulkan;

import vkpp.texture;
import vkpp.vertex;

namespace vkpp
{

export struct png_load_runtime_args
{};

export enum class ktx2_transcode_target : std::uint8_t {
  bc7_rgba,
  etc2_rgba,
  astc_4x4_rgba,
  rgba32,
};

export [[nodiscard]] constexpr auto
select_ktx2_transcode_target(const vk::PhysicalDeviceFeatures& features)
  -> ktx2_transcode_target
{
  if (features.textureCompressionBC) { return ktx2_transcode_target::bc7_rgba; }
  if (features.textureCompressionETC2)
  {
    return ktx2_transcode_target::etc2_rgba;
  }
  if (features.textureCompressionASTC_LDR)
  {
    return ktx2_transcode_target::astc_4x4_rgba;
  }
  return ktx2_transcode_target::rgba32;
}

export struct ktx2_load_runtime_args
{
  std::optional<ktx2_transcode_target> transcode_target {};
};

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

export struct host_image_mip_chain
{
  std::vector<std::byte> texels {};
  std::vector<vk::DeviceSize> level_offsets {};
  vk::Extent2D base_extent {};
  vk::Format format {};
  std::uint32_t mip_levels_present { 1U };
  texture_mip_policy suggested_mip_policy {
    texture_mip_policy::upload_precomputed_chain
  };
};

template<std::size_t Components>
using attribute_mdspan = std::mdspan<const float,
  std::extents<std::size_t, std::dynamic_extent, Components>>;

export struct mesh_streams_cpu
{
  std::vector<float> positions {};
  std::vector<float> colors {};
  std::vector<float> texcoords {};
  std::vector<float> normals {};
  std::vector<std::byte> indices {};
  std::optional<std::uint32_t> material_index {};
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

  [[nodiscard]] auto
  normals_view() const -> attribute_mdspan<3>
  { return attribute_mdspan<3> { normals.data(), vertex_count }; }
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

  for (auto index : std::views::indices(streams.vertex_count))
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

// should we do the same for htx2/png/mesh/etc... to stay consistant? I believe
// yes, but don't know what about `namespace mesh`. If it has logical sense then
// yes
export namespace gltf
{

enum class content_policy : std::uint8_t
{
  geometry_only,
  geometry_and_materials,
  geometry_and_host_images,
};

struct load_runtime_args
{
  content_policy content { content_policy::geometry_only };
  bool load_external_buffers { true };
  bool load_external_images { true };
  bool enable_texture_basisu { true };
  bool prefer_basisu_image { true };
  ktx2_load_runtime_args ktx2 {};
};

struct texture_ref_cpu
{
  std::optional<std::uint32_t> image_index {};
  std::optional<std::uint32_t> basisu_image_index {};
  std::optional<std::uint32_t> sampler_index {};
};

enum class alpha_mode : std::uint8_t
{
  opaque,
  mask,
  blend,
};

struct material_cpu
{
  std::array<float, 4> base_color_factor { 1.0F, 1.0F, 1.0F, 1.0F };
  float metallic_factor { 1.0F };
  float roughness_factor { 1.0F };
  std::optional<texture_ref_cpu> base_color_texture {};
  std::optional<texture_ref_cpu> metallic_roughness_texture {};
  std::optional<texture_ref_cpu> normal_texture {};
  float normal_scale { 1.0F };
  std::optional<texture_ref_cpu> occlusion_texture {};
  float occlusion_strength { 1.0F };
  std::optional<texture_ref_cpu> emissive_texture {};
  std::array<float, 3> emissive_factor { 0.0F, 0.0F, 0.0F };
  alpha_mode alpha_mode { alpha_mode::opaque };
  float alpha_cutoff { 0.5F };
  bool double_sided { false };
};

struct sampler_cpu
{
  vk::Filter mag_filter { vk::Filter::eLinear };
  vk::Filter min_filter { vk::Filter::eLinear };
  vk::SamplerMipmapMode mipmap_mode { vk::SamplerMipmapMode::eLinear };
  vk::SamplerAddressMode address_u { vk::SamplerAddressMode::eRepeat };
  vk::SamplerAddressMode address_v { vk::SamplerAddressMode::eRepeat };
};

struct draw_item_cpu
{
  std::uint32_t primitive_index { 0U };
  std::array<float, 16> world_transform {};
};

enum class image_kind : std::uint8_t
{
  encoded_png,
  encoded_jpeg,
  encoded_ktx2,
  encoded_other,
};

struct image_source_cpu
{
  image_kind kind {};
  std::vector<std::byte> encoded_bytes {};
  std::filesystem::path debug_uri {};
};

struct host_image_cpu
{
  std::optional<host_image> decoded {};
  std::optional<host_image_mip_chain> mip_chain {};
};

struct asset_cpu
{
  mesh_cpu meshes {};
  std::vector<material_cpu> materials {};
  std::vector<texture_ref_cpu> textures {};
  std::vector<image_source_cpu> image_sources {};
  std::vector<host_image_cpu> host_images {};
  std::vector<sampler_cpu> samplers {};
  std::vector<draw_item_cpu> draw_list {};
};

} // namespace gltf
} // namespace vkpp
