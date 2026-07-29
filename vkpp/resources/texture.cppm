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
import vkpp.sampler;

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
        .detail = "upload_precomputed_chain policy not implemented"sv,
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
                  [ & ] -> std::expected<void, error_t>
                  {
                    auto& command_buffer = single_time.command_buffer();
                    if (create_info.mip_policy ==
                      texture_mip_policy::generate_gpu_blit)
                    {
                      return record_upload_sampled_texture(command_buffer,
                        create_info.device.physical_device(),
                        staging_buffer.buffer(), image_handle,
                        create_info.format, create_info.extent,
                        create_info.mip_levels);
                    }
                    const image_barrier to_transfer_dst =
                      undefined_dst_to_transfer_dst(image_handle, 1U);
                    record_barriers(
                      command_buffer, std::span { &to_transfer_dst, 1Uz });
                    record_copy_buffer_to_image(command_buffer,
                      staging_buffer.buffer(), image_handle,
                      create_info.extent);
                    const image_barrier to_shader_read =
                      transfer_dst_to_shader_read(image_handle, 0U, 1U);
                    record_barriers(
                      command_buffer, std::span { &to_shader_read, 1UZ });
                    return {};
                  })
                .and_then([ & ] -> std::expected<void, error_t>
                  { return single_time.end_and_submit(upload::wait_idle); })
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
            });
      });
}

} // namespace vkpp
