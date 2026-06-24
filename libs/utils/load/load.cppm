module;

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

export module load;

import std;
import vulkan;
import load.channels;

export namespace load
{
[[nodiscard]] constexpr auto
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

[[nodiscard]] constexpr auto
texture_file(const std::filesystem::path& filename, std::int32_t& texture_width,
  std::int32_t& texture_height)
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
  return std::span { pixels, image_size };
}

} // namespace load