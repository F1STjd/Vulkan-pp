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
import vkpp.buffer;
import vkpp.barrier;

namespace vkpp
{
using namespace std::string_view_literals;

// should sampler be in texture module or some new?
export struct sampler_create_info
{
  vk::Filter mag_filter { vk::Filter::eLinear };
  vk::Filter min_filter { vk::Filter::eLinear };
  vk::SamplerMipmapMode mipmap_mode { vk::SamplerMipmapMode::eLinear };
  vk::SamplerAddressMode address_mode_u { vk::SamplerAddressMode::eRepeat };
  vk::SamplerAddressMode address_mode_v { vk::SamplerAddressMode::eRepeat };
  vk::SamplerAddressMode address_mode_w { vk::SamplerAddressMode::eRepeat };
  bool anisotropy_enable { true };
  float min_lod { 0.0F };
  float max_lod { vk::LodClampNone };
};

export auto
make_sampler(const vk::raii::Device& device,
  const vk::raii::PhysicalDevice& physical,
  const sampler_create_info& create_info)
  -> std::expected<vk::raii::Sampler, error_t>
{
  const auto properties = physical.getProperties();
  const vk::SamplerCreateInfo sampler_info {
    .magFilter = create_info.mag_filter,
    .minFilter = create_info.min_filter,
    .mipmapMode = create_info.mipmap_mode,
    .addressModeU = create_info.address_mode_u,
    .addressModeV = create_info.address_mode_v,
    .addressModeW = create_info.address_mode_w,
    .mipLodBias = 0.0F,
    .anisotropyEnable = create_info.anisotropy_enable ? vk::True : vk::False,
    .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
    .compareEnable = vk::False,
    .compareOp = vk::CompareOp::eAlways,
    .minLod = create_info.min_lod,
    .maxLod = create_info.max_lod,
    .borderColor = vk::BorderColor::eIntOpaqueBlack,
    .unnormalizedCoordinates = vk::False,
  };
  return UTILS_VK(
    device.createSampler(sampler_info), ^^vk::raii::Device::createSampler);
}

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
  std::span<const std::byte> pixels {};
  vk::Extent2D extent {};
  vk::Format format { vk::Format::eR8G8B8A8Srgb };
  std::uint32_t mip_levels { 1U };
  texture_mip_policy mip_policy { texture_mip_policy::generate_gpu_blit };
  sampler_create_info sampler {};
};

export auto
make_texture(const texture_create_info& create_info)
  -> std::expected<texture<>, error_t>
{
  // can validation be moved to consteval?
  if (create_info.mip_policy == texture_mip_policy::single_level &&
    create_info.mip_levels != 1U)
  {
    return std::unexpected {
      app_error {
        .kind = app_error_kind::invalid_argument,
        .detail = "single_level policy requires mip_levels == 1"sv,
      },
    };
  }
  if (create_info.mip_policy == texture_mip_policy::generate_gpu_blit &&
    create_info.mip_levels < 2U)
  {
    return std::unexpected {
      app_error {
        .kind = app_error_kind::invalid_argument,
        .detail = "generate_gpu_blit policy requires mip_levels >= 2"sv,
      },
    };
  }
  if (create_info.mip_policy == texture_mip_policy::upload_precomputed_chain)
  {
    return std::unexpected {
      app_error {
        .kind = app_error_kind::invalid_argument,
        .detail = "uplad_precomputed_chain policy not working"sv,
      },
    };
  }

  const vk::DeviceSize byte_size = create_info.pixels.size_bytes();

  return make_buffer_resource(create_info.device.allocator(), byte_size,
    vk::BufferUsageFlagBits::eTransferSrc, memory_intent::staging)
    .and_then(
      [ & ](
        buffer_resource<>&& staging_buffer) -> std::expected<texture<>, error_t>
      {
        if (staging_buffer.mapped() == nullptr)
        {
          return std::unexpected {
            app_error {
              .kind = app_error_kind::mapping_failed,
              .detail = "Staging buffer map returned nullptr"sv,
            },
          };
        }
        std::memcpy(
          staging_buffer.mapped(), create_info.pixels.data(), byte_size);

        return make_image_resource<image_kind::sampled_texture>(
          create_info.device.allocator(), create_info.device.device(),
          image_runtime_args {
            .extent = create_info.extent,
            .format = create_info.format,
            .samples = vk::SampleCountFlagBits::e1,
            .mip_levels = create_info.mip_levels,
          })
          .and_then(
            [ &, staging_buffer = std::move(staging_buffer) ](
              image_resource<>&& image) mutable
              -> std::expected<texture<>, error_t>
            {
              single_time_submit single_time {
                create_info.pool,
                create_info.device.device(),
                create_info.device.graphics_queue(),
              };

              const vk::Image image_handle = image.image();
              return single_time.begin()
                .and_then(
                  [ & ](vk::raii::CommandBuffer* command_buffer)
                    -> std::expected<void, error_t>
                  {
                    if (create_info.mip_policy ==
                      texture_mip_policy::generate_gpu_blit)
                    {
                      return record_upload_sampled_texture(*command_buffer,
                        create_info.device.physical_device(),
                        staging_buffer.buffer(), image_handle,
                        create_info.format, create_info.extent,
                        create_info.mip_levels);
                    }
                    const image_barrier to_transfer_dst =
                      undefined_dst_to_transfer_dst(image_handle, 1U);
                    record_barriers(
                      *command_buffer, std::span { &to_transfer_dst, 1Uz });
                    record_copy_buffer_to_image(*command_buffer,
                      staging_buffer.buffer(), image_handle,
                      create_info.extent);
                    const image_barrier to_shader_read =
                      transfer_dst_to_shader_read(image_handle, 0U, 1U);
                    record_barriers(
                      *command_buffer, std::span { &to_shader_read, 1UZ });
                    return {};
                  })
                .and_then([ & ] -> std::expected<void, error_t>
                  { return single_time.end_and_submit(); })
                .and_then(
                  [ & ]() -> std::expected<texture<>, error_t>
                  {
                    return make_sampler(create_info.device.device(),
                      create_info.device.physical_device(), create_info.sampler)
                      .transform(
                        [ &, image = std::move(image) ](
                          vk::raii::Sampler&& sampler) mutable -> texture<>
                        {
                          return texture<> {
                            std::move(image),
                            std::move(sampler),
                            create_info.mip_levels,
                          };
                        });
                  });
              ;
            });
      });
}

} // namespace vkpp
