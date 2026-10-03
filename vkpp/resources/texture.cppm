export module vkpp.texture;

import std;
import vulkan;

export import vkpp.sampler;
import vkpp.error;
import vkpp.memory;
import vkpp.memory.vma;
import vkpp.image;
import vkpp.device;
import vkpp.command;
import vkpp.buffer.stage_pool;

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

  texture(image_resource<Alloc>&& image, vk::raii::Sampler&& owned_sampler,
    std::uint32_t mip_levels, std::uint32_t array_layers = 1U)
  : image_ { std::move(image) }, owned_sampler_ { std::move(owned_sampler) },
    sampler_ { *owned_sampler }, mip_levels_ { mip_levels },
    array_layers_ { array_layers }
  {}

  texture(image_resource<Alloc>&& image, vk::Sampler borrowed_sampler,
    std::uint32_t mip_levels, std::uint32_t array_layers = 1U)
  : image_ { std::move(image) }, sampler_ { borrowed_sampler },
    mip_levels_ { mip_levels }, array_layers_ { array_layers }
  {}

  [[nodiscard]] auto
  image() const -> vk::Image
  { return image_.image(); }

  [[nodiscard]] auto
  view(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.image_.view()); }

  [[nodiscard]] auto
  sampler() const -> vk::Sampler
  { return sampler_; }

  [[nodiscard]] auto
  extent() const -> vk::Extent2D
  { return image_.extent(); }

  [[nodiscard]] auto
  format() const -> vk::Format
  { return image_.format(); }

  [[nodiscard]] auto
  mip_levels() const -> std::uint32_t
  { return mip_levels_; }

  [[nodiscard]]
  auto
  array_layers() const -> std::uint32_t
  { return array_layers_; }

  [[nodiscard]] auto
  view_create_info(vk::ImageViewType view_type, std::uint32_t layer_count) const -> vk::ImageViewCreateInfo
  {
    return vk::ImageViewCreateInfo {
      .image = image(),
      .viewType = view_type,
      .format = format(),
      .subresourceRange =
        {
          .aspectMask = vk::ImageAspectFlagBits::eColor,
          .baseMipLevel = 0U,
          .levelCount = mip_levels(),
          .baseArrayLayer = 0U,
          .layerCount = layer_count,
        },
    };
  }

  [[nodiscard]] auto
  view_create_info() const -> vk::ImageViewCreateInfo
  {
    const auto view_type = (array_layers_ == 6U)
      ? vk::ImageViewType::eCube
      : vk::ImageViewType::e2D;
    return view_create_info(view_type, array_layers_);
  }

  [[nodiscard]] auto
  resource(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.image_); }

private:
  image_resource<Alloc> image_ {};
  vk::raii::Sampler owned_sampler_ { nullptr };
  vk::Sampler sampler_ {};
  std::uint32_t mip_levels_ { 1U };
  std::uint32_t array_layers_ { 1U };
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
  // 3 bytes of padding
  sampler_create_info sampler {};
  std::optional<vk::Sampler> borrowed_sampler {};
  std::span<const vk::DeviceSize> level_offsets {};
  std::optional<stage_pool&> stage_pool {};
  std::uint32_t array_layers { 1U };
};

} // namespace vkpp
