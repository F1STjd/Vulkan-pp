module;

#include <ktx.h>

export module vkpp.io.image.ktx2;

import std;
import vulkan;

import vkpp.io.types;
import vkpp.io.image;
import vkpp.error;

namespace vkpp
{
using namespace std::string_view_literals;

[[nodiscard]] constexpr auto
to_ktx_transcode_format(ktx2_transcode_target target) -> ktx_transcode_fmt_e
{
  switch (target)
  {
  case ktx2_transcode_target::bc7_rgba:
    return KTX_TTF_BC7_RGBA;
  case ktx2_transcode_target::etc2_rgba:
    return KTX_TTF_ETC2_RGBA;
  case ktx2_transcode_target::astc_4x4_rgba:
    return KTX_TTF_ASTC_4x4_RGBA;
  case ktx2_transcode_target::rgba32:
    return KTX_TTF_RGBA32;
  }
  return KTX_TTF_RGBA32;
}

[[nodiscard]] auto
map_ktx_error(KTX_error_code code) -> error_t
{
  return app_error {
    .kind = app_error_kind::image_decode,
    .detail = std::format("libktx error: {}", static_cast<std::int32_t>(code)),
  };
}

struct ktx_texture2_deleter
{
  void
  operator()(ktxTexture2* value) const noexcept
  {
    if (value != nullptr) { ktxTexture_Destroy(ktxTexture(value)); }
  }
};

[[nodiscard]] auto
chain_from_ktx_texture(
  ktxTexture2* raw, const ktx2_load_runtime_args& runtime_args)
  -> std::expected<host_image_mip_chain, error_t>
{
  std::unique_ptr<ktxTexture2, ktx_texture2_deleter> held { raw };

  if (ktxTexture2_NeedsTranscoding(held.get()))
  {
    if (runtime_args.transcode_target.has_value())
    {
      return std::unexpected {
        app_error {
          .kind = app_error_kind::no_supported_format,
          .detail = "KTX2 needs transcoding but no target was provided"sv,
        },
      };
    }

    const auto transcode_result = ktxTexture2_TranscodeBasis(
      held.get(), to_ktx_transcode_format(*runtime_args.transcode_target), 0);
    if (transcode_result != KTX_SUCCESS)
    {
      return std::unexpected { map_ktx_error(transcode_result) };
    }
  }

  host_image_mip_chain chain {
    .base_extent = {
      .width = held->baseWidth,
      .height = held->baseHeight,
    },
    .format = static_cast<vk::Format>(held->vkFormat),
    .mip_levels_present = held->numLevels,
  };
  chain.texels.resize(held->dataSize);
  std::memcpy(chain.texels.data(), held->pData, held->dataSize);
  chain.level_offsets.resize(held->numLevels);
  for (auto level : std::views::indices(held->numLevels))
  {
    ktx_size_t offset { 0 };
    const auto offset_result =
      ktxTexture_GetImageOffset(ktxTexture(held.get()), level, 0, 0, &offset);
    if (offset_result != KTX_SUCCESS)
    {
      return std::unexpected { map_ktx_error(offset_result) };
    }
    chain.level_offsets[ level ] = static_cast<vk::DeviceSize>(offset);
  }

  return chain;
}

export [[nodiscard]] auto
load_host_image_ktx2_from_memory(
  std::span<const std::byte> bytes, const ktx2_load_runtime_args& runtime_args)
  -> std::expected<host_image_mip_chain, error_t>
{
  ktxTexture2* texture { nullptr };
  const auto create_result = ktxTexture2_CreateFromMemory(
    reinterpret_cast<const ktx_uint8_t*>(bytes.data()), bytes.size(),
    KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &texture);

  if (create_result != KTX_SUCCESS)
  {
    return std::unexpected { map_ktx_error(create_result) };
  }
  return chain_from_ktx_texture(texture, runtime_args);
}

template<>
[[nodiscard]] auto
load_host_image<image_file_type::ktx2>(
  const std::filesystem::path& path, const ktx2_load_runtime_args& runtime_args)
  -> std::expected<host_image_mip_chain, error_t>
{
  ktxTexture2* texture { nullptr };
  const auto create_result = ktxTexture2_CreateFromNamedFile(
    path.string().c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &texture);

  if (create_result != KTX_SUCCESS)
  {
    return std::unexpected { map_ktx_error(create_result) };
  }
  return chain_from_ktx_texture(texture, runtime_args);
}

} // namespace vkpp
