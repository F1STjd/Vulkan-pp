module;

#include "error/vk_error_config.hpp"

export module vkpp.graph;

import std;
import vulkan;

import vkpp.barrier;

namespace vkpp
{

export enum class image_use : std::uint8_t {
  none,
  color_attachment,
  color_blend,
  depth_attachment,
  present,
  sampled_fragment,
  transfer_src,
};

export struct image_use_fields
{
  vk::PipelineStageFlags2 stage {};
  vk::AccessFlags2 access {};
  vk::ImageLayout layout {};
};

export [[nodiscard]] constexpr auto
fields_for(image_use use) -> image_use_fields
{
  switch (use)
  {
  case image_use::none:
    return {
      .stage = {},
      .access = {},
      .layout = vk::ImageLayout::eUndefined,
    };
  case image_use::color_attachment:
    return {
      .stage = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
      .access = vk::AccessFlagBits2::eColorAttachmentWrite,
      .layout = vk::ImageLayout::eColorAttachmentOptimal,
    };
  case image_use::color_blend:
    return {
      .stage = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
      .access = vk::AccessFlagBits2::eColorAttachmentRead |
        vk::AccessFlagBits2::eColorAttachmentWrite,
      .layout = vk::ImageLayout::eColorAttachmentOptimal,
    };
  case image_use::depth_attachment:
    return {
      .stage = vk::PipelineStageFlagBits2::eEarlyFragmentTests |
        vk::PipelineStageFlagBits2::eLateFragmentTests,
      .access = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
      .layout = vk::ImageLayout::eDepthAttachmentOptimal,
    };
  case image_use::present:
    return {
      .stage = vk::PipelineStageFlagBits2::eBottomOfPipe,
      .access = {},
      .layout = vk::ImageLayout::ePresentSrcKHR,
    };
  case image_use::sampled_fragment:
    return {
      .stage = vk::PipelineStageFlagBits2::eFragmentShader,
      .access = vk::AccessFlagBits2::eShaderSampledRead,
      .layout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };
  case image_use::transfer_src:
    return {
      .stage = vk::PipelineStageFlagBits2::eCopy,
      .access = vk::AccessFlagBits2::eTransferRead,
      .layout = vk::ImageLayout::eTransferSrcOptimal,
    };
  }
  return {};
}

export struct image_use_transition
{
  vk::Image image {};
  image_use from { image_use::none };
  image_use to {};
  vk::ImageAspectFlags aspect { vk::ImageAspectFlagBits::eColor };
};

export [[nodiscard]] constexpr auto
make_image_barrier(const image_use_transition& transition) -> image_barrier
{
  const image_use_fields dst = fields_for(transition.to);
  const bool first_use = transition.from == image_use::none;
  const image_use_fields src = first_use
    ? image_use_fields {
        .stage = dst.stage,
        .access = {},
        .layout = vk::ImageLayout::eUndefined,
      }
    : fields_for(transition.from);

  return {
    .src_stage = src.stage,
    .src_access = src.access,
    .dst_stage = dst.stage,
    .dst_access = dst.access,
    .old_layout = src.layout,
    .new_layout = dst.layout,
    .image = transition.image,
    .range = {
      .aspectMask = transition.aspect,
      .baseMipLevel = 0U,
      .levelCount = 1U,
      .baseArrayLayer = 0U,
      .layerCount = 1U,
    },
  };
}

// Sadly we can't use std::inplace_vector as return value, because of GCC's ICE.
// Why do we convert size to std::uint32_t???
// The conversion should be only done when interacting with Vulkan API, but in
// the modern C++ std::size_t (std::ssize_t) should be used almost everytime.
export [[nodiscard]] constexpr auto
fill_image_barriers(std::span<const image_use_transition> transitions,
  std::span<image_barrier> out) -> std::uint32_t
{
  const std::uint32_t count =
    static_cast<std::uint32_t>(std::min(transitions.size(), out.size()));
  for (auto index : std::views::indices(count))
  {
    out[ index ] = make_image_barrier(transitions[ index ]);
  }
  return count;
}

export void
record_image_use_transitions(vk::raii::CommandBuffer& command_buffer,
  std::span<const image_use_transition> transitions)
{
  std::array<image_barrier, 8> storage {};
  const std::uint32_t count =
    fill_image_barriers(transitions, std::span { storage });
  record_barriers(command_buffer,
    std::span { storage.data(), static_cast<std::size_t>(count) });
}

export enum class buffer_use : std::uint8_t {
  none,
  storage_compute_write,
  storage_compute_read,
  storage_vertex_read,
  storage_fragment_read,
  uniform_fragment_read,
  transfer_src,
  transfer_dst,
  vertex,
  index,
  indirect,
};

export struct buffer_use_fields
{
  vk::PipelineStageFlags2 stage {};
  vk::AccessFlags2 access {};
};

export [[nodiscard]] constexpr auto
fields_for(buffer_use use) -> buffer_use_fields
{
  switch (use)
  {
  case buffer_use::none: return {};
  case buffer_use::storage_compute_write:
    return {
      .stage = vk::PipelineStageFlagBits2::eComputeShader,
      .access = vk::AccessFlagBits2::eShaderStorageWrite,
    };
  case buffer_use::storage_compute_read:
    return {
      .stage = vk::PipelineStageFlagBits2::eComputeShader,
      .access = vk::AccessFlagBits2::eShaderStorageRead,
    };
  case buffer_use::storage_vertex_read:
    return {
      .stage = vk::PipelineStageFlagBits2::eVertexShader,
      .access = vk::AccessFlagBits2::eShaderStorageRead,
    };
  case buffer_use::storage_fragment_read:
    return {
      .stage = vk::PipelineStageFlagBits2::eFragmentShader,
      .access = vk::AccessFlagBits2::eShaderStorageRead,
    };
  case buffer_use::uniform_fragment_read:
    return {
      .stage = vk::PipelineStageFlagBits2::eFragmentShader,
      .access = vk::AccessFlagBits2::eUniformRead,
    };
  case buffer_use::transfer_src:
    return {
      .stage = vk::PipelineStageFlagBits2::eCopy,
      .access = vk::AccessFlagBits2::eTransferRead,
    };
  case buffer_use::transfer_dst:
    return {
      .stage = vk::PipelineStageFlagBits2::eCopy,
      .access = vk::AccessFlagBits2::eTransferWrite,
    };
  case buffer_use::vertex:
    return {
      .stage = vk::PipelineStageFlagBits2::eVertexAttributeInput,
      .access = vk::AccessFlagBits2::eVertexAttributeRead,
    };
  case buffer_use::index:
    return {
      .stage = vk::PipelineStageFlagBits2::eIndexInput,
      .access = vk::AccessFlagBits2::eIndexRead,
    };
  case buffer_use::indirect:
    return {
      .stage = vk::PipelineStageFlagBits2::eDrawIndirect,
      .access = vk::AccessFlagBits2::eIndirectCommandRead,
    };
  }
  return {};
}

export struct buffer_use_transition
{
  vk::Buffer buffer {};
  buffer_use from { buffer_use::none };
  buffer_use to {};
  vk::DeviceSize offset { 0UZ };
  vk::DeviceSize size { vk::WholeSize };
};

export [[nodiscard]] constexpr auto
make_buffer_barrier(const buffer_use_transition& transition) -> buffer_barrier
{
  const auto dst = fields_for(transition.to);
  const bool first_use = transition.from == buffer_use::none;
  const auto src = first_use
    ? buffer_use_fields { .stage = dst.stage, .access = {} }
    : fields_for(transition.from);
  return {
    .src_stage = src.stage,
    .src_access = src.access,
    .dst_stage = dst.stage,
    .dst_access = dst.access,
    .buffer = transition.buffer,
    .offset = transition.offset,
    .size = transition.size,
  };
}

export void
record_buffer_use_transitions(vk::raii::CommandBuffer& command_buffer,
  std::span<const buffer_use_transition> transitions)
{
  std::array<buffer_barrier, 8> storage {};
  const auto count =
    static_cast<std::uint32_t>(std::min(transitions.size(), storage.size()));
  for (auto index : std::views::indices(count))
  {
    storage[ index ] = make_buffer_barrier(transitions[ index ]);
  }
  record_barriers(command_buffer,
    std::span { storage.data(), static_cast<std::size_t>(count) });
}

export class image_use_tracker
{
public:
  auto
  transition(vk::raii::CommandBuffer& command_buffer, vk::Image image,
    image_use to, vk::ImageAspectFlags aspect) -> bool
  {
    image_use from { image_use::none };
    for (auto index : std::views::indices(count_))
    {
      if (images_[ index ] == image)
      {
        from = std::exchange(uses_[ index ], to);
        const image_use_transition transition {
          .image = image,
          .from = from,
          .to = to,
          .aspect = aspect,
        };
        record_image_use_transitions(
          command_buffer, std::span { &transition, 1UZ });
        return true;
      }
    }

    if (count_ >= images_.size()) { return false; }

    images_[ count_ ] = image;
    uses_[ count_ ] = to;
    ++count_;
    const image_use_transition transition {
      .image = image,
      .from = image_use::none,
      .to = to,
      .aspect = aspect,
    };
    record_image_use_transitions(
      command_buffer, std::span { &transition, 1UZ });
    return true;
  }

  void
  reset()
  { count_ = 0U; }

private:
  std::array<vk::Image, 16> images_ {};
  std::array<image_use, 16> uses_ {};
  std::uint32_t count_ {};
};

} // namespace vkpp
