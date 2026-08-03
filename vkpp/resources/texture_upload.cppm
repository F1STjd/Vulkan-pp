module;

#include "error/vk_error_config.hpp"

export module vkpp.texture.upload;

import std;
import vulkan;

export import vkpp.texture;
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

[[nodiscard]] inline auto
upload_texture_via_graphics_queue(const texture_create_info& create_info,
  vk::Buffer staging_buffer, vk::Image image_handle)
  -> std::expected<void, error_t>
{
  single_time_submit single_time {
    create_info.pool,
    create_info.device.device(),
    create_info.device.graphics_queue(),
  };

  return single_time.begin()
    .and_then(
      [ & ] -> std::expected<void, error_t>
      {
        auto& command_buffer = single_time.command_buffer();
        if (create_info.mip_policy == texture_mip_policy::generate_gpu_blit)
        {
          return record_upload_sampled_texture(command_buffer,
            create_info.device.physical_device(), staging_buffer, image_handle,
            create_info.format, create_info.extent, create_info.mip_levels);
        }
        const image_barrier to_transfer_dst =
          undefined_dst_to_transfer_dst(image_handle, 1U);
        record_barriers(command_buffer, std::span { &to_transfer_dst, 1UZ });
        record_copy_buffer_to_image(
          command_buffer, staging_buffer, image_handle, create_info.extent);
        const image_barrier to_shader_read =
          transfer_dst_to_shader_read(image_handle, 0U, 1U);
        record_barriers(command_buffer, std::span { &to_shader_read, 1UZ });
        return {};
      })
    .and_then([ & ] -> std::expected<submission, error_t>
      { return single_time.end_and_submit(upload::deferred); })
    .and_then([](submission&& done) -> std::expected<void, error_t>
      { return done.wait(); });
}

[[nodiscard]] inline auto
upload_texture_via_tranfer_queue(const texture_create_info& create_info,
  vk::Buffer staging_buffer, vk::Image image_handle)
  -> std::expected<void, error_t>
{
  const ownership_transfer transfer {
    .src_queue_family = create_info.device.transfer_qf_index(),
    .dst_queue_family = create_info.device.graphics_qf_index(),
  };
  return UTILS_VK(create_info.device.device().createSemaphore({}),
    ^^vk::raii::Device::createSemaphore)
    .and_then(
      [ & ](vk::raii::Semaphore&& copy_done) -> std::expected<void, error_t>
      {
        single_time_submit transfer_submit {
          *create_info.transfer_pool,
          create_info.device.device(),
          create_info.device.transfer_queue(),
        };

        return transfer_submit.begin()
          .and_then(
            [ & ] -> std::expected<submission, error_t>
            {
              auto& command_buffer = transfer_submit.command_buffer();
              const image_barrier to_transfer_dst =
                undefined_dst_to_transfer_dst(
                  image_handle, create_info.mip_levels);
              record_barriers(
                command_buffer, std::span { &to_transfer_dst, 1UZ });
              record_copy_buffer_to_image(command_buffer, staging_buffer,
                image_handle, create_info.extent);
              const image_barrier release = release_image_ownership(
                image_handle, transfer, vk::ImageLayout::eTransferDstOptimal,
                vk::ImageLayout::eShaderReadOnlyOptimal,
                vk::PipelineStageFlagBits2::eCopy,
                vk::AccessFlagBits2::eTransferWrite, create_info.mip_levels);
              record_barriers(command_buffer, std::span { &release, 1UZ });
              return transfer_submit.end_and_submit(
                upload::deferred, { .signal = *copy_done });
            })
          .and_then(
            [ & ](
              submission&& release_submitted) -> std::expected<void, error_t>
            {
              single_time_submit graphics_submit {
                create_info.pool,
                create_info.device.device(),
                create_info.device.graphics_queue(),
              };

              return graphics_submit.begin()
                .and_then(
                  [ & ] -> std::expected<submission, error_t>
                  {
                    const image_barrier acquire =
                      acquire_image_ownership(image_handle, transfer,
                        vk::ImageLayout::eTransferDstOptimal,
                        vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::PipelineStageFlagBits2::eFragmentShader,
                        vk::AccessFlagBits2::eShaderSampledRead,
                        create_info.mip_levels);
                    record_barriers(graphics_submit.command_buffer(),
                      std::span { &acquire, 1UZ });
                    return graphics_submit.end_and_submit(
                      upload::deferred, { .wait = *copy_done });
                  })
                .and_then([](submission&& acquire_submitted)
                            -> std::expected<void, error_t>
                  { return acquire_submitted.wait(); });
            });
      });
}

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
              const vk::Image image_handle = image.image();
              const bool dual_queue =
                create_info.device.has_dedicated_transfer() &&
                create_info.transfer_pool.has_value() &&
                create_info.mip_policy != texture_mip_policy::generate_gpu_blit;
              return (dual_queue
                  ? upload_texture_via_tranfer_queue(
                      create_info, staging_buffer.buffer(), image_handle)
                  : upload_texture_via_graphics_queue(
                      create_info, staging_buffer.buffer(), image_handle))
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
