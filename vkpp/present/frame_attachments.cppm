module;

#include "error/vk_error_config.hpp"

export module vkpp.frame_attachments;

import std;
import vulkan;

import vkpp.device;
import vkpp.graph;
import vkpp.image;
import vkpp.swapchain;
import vkpp.error;

namespace vkpp
{
using namespace std::string_view_literals;

export enum class color_sink : std::uint8_t {
  presentable,
  sampled,
  transfer_src,
};

export struct frame_attachments_create_info
{
  vk::Extent2D extent {};
  vk::Format color_format {};
  vk::SampleCountFlagBits samples { vk::SampleCountFlagBits::e1 };
  color_sink sink { color_sink::sampled };
};

export class frame_attachments
{
public:
  frame_attachments() = default;

  [[nodiscard]] static auto
  create(
    device_context& device, const frame_attachments_create_info& create_info)
    -> std::expected<frame_attachments, error_t>
  {
    frame_attachments_create_info args = create_info;
    if (args.extent.width == 0U) { args.extent.width = 1U; }
    if (args.extent.height == 0U) { args.extent.height = 1U; }

    image_resource<> color {};
    image_resource<> depth {};
    image_resource<> resolve {};

    const bool msaa = args.samples != vk::SampleCountFlagBits::e1;
    if (msaa)
    {
      auto made = make_image_resource<image_kind::color>(device.allocator(),
        device.device(),
        {
          .extent = args.extent,
          .format = args.color_format,
          .samples = args.samples,
        });
      if (!made) { return std::unexpected { std::move(made).error() }; }
      color = std::move(*made);
    }

    auto depth_format = find_depth_attachment_format(device.physical_device());
    if (!depth_format)
    {
      return std::unexpected { std::move(depth_format).error() };
    }
    auto made_depth = make_image_resource<image_kind::depth>(device.allocator(),
      device.device(),
      {
        .extent = args.extent,
        .format = *depth_format,
        .samples = args.samples,
      });
    if (!made_depth)
    {
      return std::unexpected { std::move(made_depth).error() };
    }
    depth = std::move(*made_depth);

    if (args.sink == color_sink::sampled)
    {
      auto made = make_image_resource<image_kind::color_sampled>(
        device.allocator(), device.device(),
        {
          .extent = args.extent,
          .format = args.color_format,
          .samples = vk::SampleCountFlagBits::e1,
        });
      if (!made) { return std::unexpected { std::move(made).error() }; }
      resolve = std::move(*made);
    }
    else if (args.sink == color_sink::transfer_src)
    {
      auto made = make_image_resource<image_kind::color_transfer>(
        device.allocator(), device.device(),
        {
          .extent = args.extent,
          .format = args.color_format,
          .samples = vk::SampleCountFlagBits::e1,
        });
      if (!made) { return std::unexpected { std::move(made).error() }; }
      resolve = std::move(*made);
    }

    return frame_attachments {
      std::move(color),
      std::move(depth),
      std::move(resolve),
      args,
    };
  }

  [[nodiscard]] auto
  recreate(device_context& device, vk::Extent2D extent)
    -> std::expected<void, error_t>
  {
    create_info_.extent = extent;
    return create(device, create_info_)
      .transform([ this ](frame_attachments&& next) -> void
        { *this = std::move(next); });
  }

  [[nodiscard]] auto
  extent() const -> vk::Extent2D
  { return create_info_.extent; }

  [[nodiscard]] auto
  sink() const -> color_sink
  { return create_info_.sink; }

  [[nodiscard]] auto
  color() -> image_resource<>&
  { return color_; }

  [[nodiscard]] auto
  depth() -> image_resource<>&
  { return depth_; }

  [[nodiscard]] auto
  resolve() -> image_resource<>&
  { return resolve_; }

  [[nodiscard]] auto
  last_color_use() const -> image_use
  {
    switch (create_info_.sink)
    {
    case color_sink::presentable : return image_use::color_attachment;
    case color_sink::sampled     : return image_use::sampled_fragment;
    case color_sink::transfer_src: return image_use::transfer_src;
    }
    return image_use::none;
  }

  [[nodiscard]] auto
  color_attachment_info(vk::ClearValue clear_value,
    std::optional<vk::ImageView> presentable_view = {}) const
    -> vk::RenderingAttachmentInfo
  {
    const bool msaa = create_info_.samples != vk::SampleCountFlagBits::e1;
    const vk::ImageView one_x = create_info_.sink == color_sink::presentable
      ? presentable_view.value_or(vk::ImageView {})
      : *resolve_.view();
    const vk::ImageView color_view = msaa ? *color_.view() : one_x;
    return {
      .imageView = color_view,
      .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
      .resolveMode = msaa ? vk::ResolveModeFlagBits::eAverage
                          : vk::ResolveModeFlagBits::eNone,
      .resolveImageView = msaa ? one_x : vk::ImageView {},
      .resolveImageLayout =
        msaa ? vk::ImageLayout::eColorAttachmentOptimal : vk::ImageLayout {},
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eStore,
      .clearValue = clear_value,
    };
  }

  [[nodiscard]] auto
  depth_attachment_info(vk::ClearValue clear_value) const
    -> vk::RenderingAttachmentInfo
  {
    return {
      .imageView = *depth_.view(),
      .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eDontCare,
      .clearValue = clear_value,
    };
  }

  void
  record_begin_uses(image_use_tracker& tracker,
    vk::raii::CommandBuffer& command_buffer,
    vk::Image presentable_image = {}) const
  {
    const bool msaa = create_info_.samples != vk::SampleCountFlagBits::e1;
    if (msaa)
    {
      tracker.transition(command_buffer, color_.image(),
        image_use::color_attachment, vk::ImageAspectFlagBits::eColor);
    }
    tracker.transition(command_buffer, depth_.image(),
      image_use::depth_attachment, vk::ImageAspectFlagBits::eDepth);
    const vk::Image one_x = create_info_.sink == color_sink::presentable
      ? presentable_image
      : resolve_.image();
    tracker.transition(command_buffer, one_x, image_use::color_attachment,
      vk::ImageAspectFlagBits::eColor);
  }

  void
  record_after_store(
    image_use_tracker& tracker, vk::raii::CommandBuffer& command_buffer) const
  {
    if (create_info_.sink == color_sink::presentable) { return; }
    tracker.transition(command_buffer, resolve_.image(), last_color_use(),
      vk::ImageAspectFlagBits::eColor);
  }

private:
  frame_attachments(image_resource<>&& color, image_resource<>&& depth,
    image_resource<>&& resolve, frame_attachments_create_info create_info)
  : color_ { std::move(color) }, depth_ { std::move(depth) },
    resolve_ { std::move(resolve) }, create_info_ { create_info }
  {}

  image_resource<> color_ {};
  image_resource<> depth_ {};
  image_resource<> resolve_ {};
  frame_attachments_create_info create_info_ {};
};

} // namespace vkpp
