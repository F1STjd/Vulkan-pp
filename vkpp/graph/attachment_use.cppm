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
  depth_attachment,
  present,
  sampled_fragment,
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
// the modern C++ std::size_t (std::ssize_t) should be used almost everythime.
export [[nodiscard]] constexpr auto
fill_image_barriers(std::span<const image_use_transition> transitions,
  std::span<image_barrier> out) -> std::uint32_t
{
  const std::uint32_t count =
    static_cast<std::uint32_t>(std::min(transitions.size(), out.size()));
  for (auto index : std::views::iota(0U, count))
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

} // namespace vkpp
