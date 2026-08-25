module;

#include "error/vk_error_config.hpp"

export module vkpp.query;

import std;
import vulkan;

import vkpp.error;

namespace vkpp
{
using namespace std::string_view_literals;

export class timestamp_ring
{
public:
  timestamp_ring() = default;

  [[nodiscard]] static auto
  create(const vk::raii::Device& device,
    const vk::raii::PhysicalDevice& physical_device,
    std::uint32_t frames_in_flight, std::uint32_t queries_per_frame)
    -> std::expected<timestamp_ring, error_t>
  {
    const auto properties = physical_device.getProperties();
    if (properties.limits.timestampComputeAndGraphics != vk::True)
    {
      return std::unexpected {
        app_error {
          .kind = app_error_kind::no_supported_format,
          .detail = "timestamp npt supported on all graphics/compute queues"sv,
        },
      };
    }

    const vk::QueryPoolCreateInfo query_pool_info {
      .queryType = vk::QueryType::eTimestamp,
      .queryCount = frames_in_flight * queries_per_frame,
    };
    return UTILS_VK(device.createQueryPool(query_pool_info),
      ^^vk::raii::Device::createQueryPool)
      .transform(
        [ & ](vk::raii::QueryPool&& pool) -> timestamp_ring
        {
          timestamp_ring ring {};
          ring.pool_ = std::move(pool);
          ring.timestamp_period_ = properties.limits.timestampPeriod;
          ring.queries_per_frame_ = queries_per_frame;
          ring.pool_.reset(0U, frames_in_flight * queries_per_frame);
          return ring;
        });
  }

  void
  write(vk::raii::CommandBuffer& command_buffer, std::uint32_t frame_slot,
    std::uint32_t query, vk::PipelineStageFlagBits2 stage) const
  {
    command_buffer.writeTimestamp2(
      stage, *pool_, frame_slot * queries_per_frame_ + query);
  }

  [[nodiscard]] auto
  read_and_reset_frame_ns(std::uint32_t frame_slot)
    -> std::expected<std::vector<double>, error_t>
  {
    // some differences in plan. ticks is no container
    const auto first = frame_slot * queries_per_frame_;
    auto [ result, ticks ] = pool_.getResults<std::uint64_t>(first,
      queries_per_frame_, queries_per_frame_ * sizeof(std::uint64_t),
      sizeof(std::uint64_t), vk::QueryResultFlagBits::e64);
    if (result == vk::Result::eNotReady)
    {
      return std::unexpected {
        app_error {
          .kind = app_error_kind::invalid_argument,
          .detail = "timestamp slots read before frame completion"sv,
        },
      };
    }
    pool_.reset(first, queries_per_frame_);

    std::vector<double> nanoseconds(ticks.size());
    for (auto index : std::views::indices(ticks.size()))
    {
      nanoseconds[ index ] = static_cast<double>(ticks[ index ]) *
        static_cast<double>(timestamp_period_);
    }
    return nanoseconds;
  }

private:
  vk::raii::QueryPool pool_ { nullptr };
  float timestamp_period_ { 0.0F };
  std::uint32_t queries_per_frame_ { 0U };
};

} // namespace vkpp
