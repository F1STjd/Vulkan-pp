module;

#include "error/vk_error_config.hpp"

export module vkpp.texture;

import std;
import vulkan;

import vkpp.error;
import vkpp.memory;
import vkpp.memory.vma;
import vkpp.image;
import vkpp.device;
import vkpp.command;
export import vkpp.sampler;

namespace vkpp
{
using namespace std::string_view_literals;

export enum class texture_mip_policy : std::uint8_t {
  single_level,             // no mipmaps
  generate_gpu_blit,        // generate mipmaps
  upload_precomputed_chain, // file type has mipmaps inside
};

export template<device_allocator Alloc = vma_policy>
class texture
{
public:
  texture() = default;
  texture(image_resource<Alloc>&& image, vk::raii::Sampler&& sampler,
    std::uint32_t mip_levels)
  : image_ { std::move(image) }, sampler_ { std::move(sampler) },
    mip_levels_ { mip_levels }
  {}

  [[nodiscard]] auto
  image(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.image_.image()); }

  [[nodiscard]] auto
  view(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.image_.view()); }

  [[nodiscard]] auto
  sampler(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.sampler_); }

  [[nodiscard]] auto
  extent(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.image_.extent()); }

  [[nodiscard]] auto
  format(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.image_.format()); }

  [[nodiscard]] auto
  mip_levels(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.mip_levels_); }

  [[nodiscard]] auto
  resource(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.image_); }

private:
  image_resource<Alloc> image_ {};
  vk::raii::Sampler sampler_ { nullptr };
  std::uint32_t mip_levels_ { 1U };
};

export struct texture_create_info
{
  device_context& device;
  command_pool& pool;
  std::optional<command_pool&> transfer_pool;
  std::span<const std::byte> pixels {};
  vk::Extent2D extent {};
  vk::Format format { vk::Format::eR8G8B8A8Srgb };
  std::uint32_t mip_levels { 1U };
  texture_mip_policy mip_policy { texture_mip_policy::generate_gpu_blit };
  sampler_create_info sampler {};
  std::span<const vk::DeviceSize> level_offsets {};
};

} // namespace vkpp
