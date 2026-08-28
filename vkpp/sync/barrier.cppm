module;

#include "error/vk_error_config.hpp"

export module vkpp.barrier;

import std;
import vulkan;

import vkpp.error;

namespace vkpp
{

using namespace std::string_view_literals;

export struct image_barrier
{
  vk::PipelineStageFlags2 src_stage {};
  vk::AccessFlags2 src_access {};
  vk::PipelineStageFlags2 dst_stage {};
  vk::AccessFlags2 dst_access {};
  vk::ImageLayout old_layout {};
  vk::ImageLayout new_layout {};
  vk::Image image {};
  vk::ImageSubresourceRange range {
    .aspectMask = vk::ImageAspectFlagBits ::eColor,
    .baseMipLevel = 0U,
    .levelCount = 1U,
    .baseArrayLayer = 0U,
    .layerCount = 1U,
  };
  std::uint32_t src_queue_family { vk::QueueFamilyIgnored };
  std::uint32_t dst_queue_family { vk::QueueFamilyIgnored };
};

export struct buffer_barrier
{
  vk::PipelineStageFlags2 src_stage {};
  vk::AccessFlags2 src_access {};
  vk::PipelineStageFlags2 dst_stage {};
  vk::AccessFlags2 dst_access {};
  vk::Buffer buffer {};
  vk::DeviceSize offset { 0UZ };
  vk::DeviceSize size { vk::WholeSize };
  std::uint32_t src_queue_family { vk::QueueFamilyIgnored };
  std::uint32_t dst_queue_family { vk::QueueFamilyIgnored };
};

export void
record_barriers(vk::raii::CommandBuffer& command_buffer,
  std::span<const image_barrier> barriers)
{
  std::vector<vk::ImageMemoryBarrier2> native_barriers;
  native_barriers.reserve(barriers.size());
  for (const image_barrier& barrier : barriers)
  {
    native_barriers.push_back({
      .srcStageMask = barrier.src_stage,
      .srcAccessMask = barrier.src_access,
      .dstStageMask = barrier.dst_stage,
      .dstAccessMask = barrier.dst_access,
      .oldLayout = barrier.old_layout,
      .newLayout = barrier.new_layout,
      .srcQueueFamilyIndex = barrier.src_queue_family,
      .dstQueueFamilyIndex = barrier.dst_queue_family,
      .image = barrier.image,
      .subresourceRange = barrier.range,
    });
  }
  const vk::DependencyInfo dependency_info {
    .imageMemoryBarrierCount =
      static_cast<std::uint32_t>(native_barriers.size()),
    .pImageMemoryBarriers = native_barriers.data(),
  };
  command_buffer.pipelineBarrier2(dependency_info);
}

export void
record_barriers(vk::raii::CommandBuffer& command_buffer,
  std::span<const buffer_barrier> barriers)
{
  std::vector<vk::BufferMemoryBarrier2> native_barriers;
  native_barriers.reserve(barriers.size());
  for (const buffer_barrier& barrier : barriers)
  {
    native_barriers.push_back({
      .srcStageMask = barrier.src_stage,
      .srcAccessMask = barrier.src_access,
      .dstStageMask = barrier.dst_stage,
      .dstAccessMask = barrier.dst_access,
      .srcQueueFamilyIndex = barrier.src_queue_family,
      .dstQueueFamilyIndex = barrier.dst_queue_family,
      .buffer = barrier.buffer,
      .offset = barrier.offset,
      .size = barrier.size,
    });
  }
  const vk::DependencyInfo dependency_info {
    .bufferMemoryBarrierCount =
      static_cast<std::uint32_t>(native_barriers.size()),
    .pBufferMemoryBarriers = native_barriers.data(),
  };
  command_buffer.pipelineBarrier2(dependency_info);
}

export [[nodiscard]] constexpr auto
undefined_dst_to_transfer_dst(vk::Image image, std::uint32_t mip_count,
  vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor)
  -> image_barrier
{
  return {
    .src_stage = vk::PipelineStageFlagBits2::eTopOfPipe,
    .src_access = {},
    .dst_stage = vk::PipelineStageFlagBits2::eTransfer,
    .dst_access = vk::AccessFlagBits2::eTransferWrite,
    .old_layout = vk::ImageLayout::eUndefined,
    .new_layout = vk::ImageLayout::eTransferDstOptimal,
    .image = image,
    .range = {
      .aspectMask = aspect,
      .levelCount = mip_count,
      .layerCount = 1U,
    },
  };
}

export [[nodiscard]] constexpr auto
transfer_dst_to_shader_read(vk::Image image, std::uint32_t base_mip,
  std::uint32_t mip_count = 1U,
  vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor)
  -> image_barrier
{
  return {
    .src_stage = vk::PipelineStageFlagBits2::eTransfer,
    .src_access = vk::AccessFlagBits2::eTransferWrite,
    .dst_stage = vk::PipelineStageFlagBits2::eFragmentShader,
    .dst_access = vk::AccessFlagBits2::eShaderSampledRead,
    .old_layout = vk::ImageLayout::eTransferDstOptimal,
    .new_layout = vk::ImageLayout::eShaderReadOnlyOptimal,
    .image = image,
    .range = {
      .aspectMask = aspect,
      .baseMipLevel = base_mip,
      .levelCount = mip_count,
      .layerCount = 1U,
    },
  };
}

export [[nodiscard]] constexpr auto
transfer_dst_to_transfer_src(vk::Image image, std::uint32_t mip,
  vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor)
  -> image_barrier
{
  return {
    .src_stage = vk::PipelineStageFlagBits2::eTransfer,
    .src_access = vk::AccessFlagBits2::eTransferWrite,
    .dst_stage = vk::PipelineStageFlagBits2::eTransfer,
    .dst_access = vk::AccessFlagBits2::eTransferRead,
    .old_layout = vk::ImageLayout::eTransferDstOptimal,
    .new_layout = vk::ImageLayout::eTransferSrcOptimal,
    .image = image,
    .range = {
      .aspectMask = aspect,
      .baseMipLevel = mip,
      .levelCount = 1U,
      .layerCount = 1U,
    },
  };
}

export [[nodiscard]] constexpr auto
transfer_src_to_shader_read(vk::Image image, std::uint32_t mip,
  vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor)
  -> image_barrier
{
  return {
    .src_stage = vk::PipelineStageFlagBits2::eTransfer,
    .src_access = vk::AccessFlagBits2::eTransferRead,
    .dst_stage = vk::PipelineStageFlagBits2::eFragmentShader,
    .dst_access = vk::AccessFlagBits2::eShaderSampledRead,
    .old_layout = vk::ImageLayout::eTransferSrcOptimal,
    .new_layout = vk::ImageLayout::eShaderReadOnlyOptimal,
    .image = image,
    .range = {
      .aspectMask = aspect,
      .baseMipLevel = mip,
      .levelCount = 1U,
      .layerCount = 1U,
    },
  };
}

export void
record_copy_buffer_to_image(vk::raii::CommandBuffer& command_buffer,
  vk::Buffer buffer, vk::Image image, vk::Extent2D extent)
{
  vk::BufferImageCopy region {
    .bufferOffset = 0UZ,
    .bufferRowLength = 0U,
    .bufferImageHeight = 0U,
    .imageSubresource = {
      .aspectMask = vk::ImageAspectFlagBits::eColor,
      .mipLevel = 0U,
      .baseArrayLayer = 0U,
      .layerCount = 1U,
    },
    .imageOffset = {
      .x = 0,
      .y = 0,
      .z = 0,
    },
    .imageExtent = {
      .width = extent.width,
      .height = extent.height,
      .depth = 1U,
    },
  };

  command_buffer.copyBufferToImage(
    buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
}

export auto
record_generate_mipmaps(vk::raii::CommandBuffer& command_buffer,
  const vk::raii::PhysicalDevice& physical, vk::Image image, vk::Format format,
  std::int32_t width, std::int32_t height, std::uint32_t mip_levels)
  -> std::expected<void, error_t>
{
  const auto features =
    physical.getFormatProperties(format).optimalTilingFeatures;

  if (!(features & vk::FormatFeatureFlagBits::eSampledImageFilterLinear))
  {
    return std::unexpected {
      app_error {
        .kind = app_error_kind::no_supported_format,
        .detail = "Texture format lacks linear blit filter"sv,
      },
    };
  }

  auto mip_width = width;
  auto mip_height = height;

  for (std::uint32_t mip_level : std::views::indices(mip_levels))
  {
    const image_barrier to_transfer_src =
      transfer_dst_to_transfer_src(image, mip_level - 1);
    record_barriers(command_buffer, std::span { &to_transfer_src, 1UZ });

    const std::array src_offsets {
      vk::Offset3D { .x = 0, .y = 0, .z = 0 },
      vk::Offset3D { .x = mip_width, .y = mip_height, .z = 1 },
    };
    const std::array dst_offsets {
      vk::Offset3D { .x = 0, .y = 0, .z = 0 },
      vk::Offset3D {
        .x = mip_width > 1 ? mip_width / 2 : 1,
        .y = mip_height > 1 ? mip_height / 2 : 1,
        .z = 1,
      },
    };
    const vk::ImageBlit blit {
      .srcSubresource = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .mipLevel = mip_level - 1U,
        .baseArrayLayer = 0U,
        .layerCount= 1U,
      },
      .srcOffsets = src_offsets,
      .dstSubresource = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .mipLevel = mip_level,
        .baseArrayLayer = 0U,
        .layerCount= 1U,
      },
      .dstOffsets = dst_offsets,
    };

    command_buffer.blitImage(image, vk::ImageLayout::eTransferSrcOptimal, image,
      vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eLinear);

    const image_barrier to_shader_read =
      transfer_src_to_shader_read(image, mip_level - 1U);
    record_barriers(command_buffer, std::span { &to_shader_read, 1UZ });

    if (mip_width > 1) { mip_width /= 2; }
    if (mip_height > 1) { mip_height /= 2; }
  }

  const image_barrier last_mip_to_shader_read =
    transfer_dst_to_shader_read(image, mip_levels - 1U);
  record_barriers(command_buffer, std::span { &last_mip_to_shader_read, 1UZ });
  return {};
}

export auto
record_upload_sampled_texture(vk::raii::CommandBuffer& command_buffer,
  const vk::raii::PhysicalDevice& physical, vk::Buffer staging_buffer,
  vk::Image image, vk::Format format, vk::Extent2D extent,
  std::uint32_t mip_levels) -> std::expected<void, error_t>
{
  const image_barrier to_transfer_dst =
    undefined_dst_to_transfer_dst(image, mip_levels);
  record_barriers(command_buffer, std::span { &to_transfer_dst, 1UZ });
  record_copy_buffer_to_image(command_buffer, staging_buffer, image, extent);

  if (mip_levels > 1U)
  {
    return record_generate_mipmaps(command_buffer, physical, image, format,
      static_cast<std::uint32_t>(extent.width),
      static_cast<std::uint32_t>(extent.height), mip_levels);
  }
  const image_barrier to_shader_read =
    transfer_dst_to_shader_read(image, 0U, mip_levels);
  record_barriers(command_buffer, std::span { &to_shader_read, 1UZ });
  return {};
}

export struct ownership_transfer
{
  std::uint32_t src_queue_family {};
  std::uint32_t dst_queue_family {};
};

export [[nodiscard]] constexpr auto
release_buffer_ownership(vk::Buffer buffer, ownership_transfer transfer,
  vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access)
  -> buffer_barrier
{
  return {
    .src_stage = src_stage,
    .src_access = src_access,
    .dst_stage = vk::PipelineStageFlagBits2::eNone,
    .dst_access = {},
    .buffer = buffer,
    .src_queue_family = transfer.src_queue_family,
    .dst_queue_family = transfer.dst_queue_family,
  };
}

export [[nodiscard]] constexpr auto
acquire_buffer_ownership(vk::Buffer buffer, ownership_transfer transfer,
  vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access)
  -> buffer_barrier
{
  return {
    .src_stage = vk::PipelineStageFlagBits2::eNone,
    .src_access = {},
    .dst_stage = dst_stage,
    .dst_access = dst_access,
    .buffer = buffer,
    .src_queue_family = transfer.src_queue_family,
    .dst_queue_family = transfer.dst_queue_family,
  };
}

export [[nodiscard]] constexpr auto
release_image_ownership(vk::Image image, ownership_transfer transfer,
  vk::ImageLayout old_layout, vk::ImageLayout new_layout,
  vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access,
  std::uint32_t mip_count = 1U,
  vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor)
  -> image_barrier
{
  return {
    .src_stage = src_stage,
    .src_access = src_access,
    .dst_stage = vk::PipelineStageFlagBits2::eNone,
    .dst_access = {},
    .old_layout = old_layout,
    .new_layout = new_layout,
    .image = image,
    .range = {
      .aspectMask = aspect,
      .levelCount = mip_count,
      .layerCount = 1U,
    },
    .src_queue_family = transfer.src_queue_family,
    .dst_queue_family = transfer.dst_queue_family,
  };
}

export [[nodiscard]] constexpr auto
acquire_image_ownership(vk::Image image, ownership_transfer transfer,
  vk::ImageLayout old_layout, vk::ImageLayout new_layout,
  vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access,
  std::uint32_t mip_count = 1U,
  vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor)
  -> image_barrier
{
  return {
    .src_stage = vk::PipelineStageFlagBits2::eNone,
    .src_access = {},
    .dst_stage = dst_stage,
    .dst_access = dst_access,
    .old_layout = old_layout,
    .new_layout = new_layout,
    .image = image,
    .range = {
      .aspectMask = aspect,
      .levelCount = mip_count,
      .layerCount = 1U,
    },
    .src_queue_family = transfer.src_queue_family,
    .dst_queue_family = transfer.dst_queue_family,
  };
}

} // namespace vkpp
