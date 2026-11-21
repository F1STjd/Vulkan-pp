export module vkpp.query;

import std;
import vulkan;

import vkpp.error;
import vkpp.diagnostics;

namespace vkpp
{

[[nodiscard]]
constexpr
auto
valid_bit_mask(std::uint32_t valid_bits) -> std::uint64_t
{
  constexpr auto digits = std::numeric_limits<std::uint64_t>::digits;
  if (valid_bits >= digits) { return ~std::uint64_t {}; }
  return (std::uint64_t {1} << valid_bits) - 1U;
}

export class timestamp_ring
{
public:
  timestamp_ring() = default;

  timestamp_ring(
    vk::raii::QueryPool&& pool,
    float                 timestamp_period,
    std::uint32_t         queries_per_frame,
    std::uint32_t         timestamp_valid_bits)
  : pool_ {std::move(pool)},
    timestamp_period_ {timestamp_period},
    queries_per_frame_ {queries_per_frame},
    valid_bit_mask_ {valid_bit_mask(timestamp_valid_bits)}
  {}

  [[nodiscard]]
  static
  auto
  create(
    const vk::raii::Device&         device,
    const vk::raii::PhysicalDevice& physical_device,
    std::uint32_t                   frames_in_flight,
    std::uint32_t queries_per_frame) -> std::expected<timestamp_ring, error_t>
  {
    const auto properties = physical_device.getProperties();
    if (properties.limits.timestampComputeAndGraphics != vk::True)
    {
      return std::unexpected {
        make_app_error(app_error_code::feature_not_supported),
      };
    }

    const vk::QueryPoolCreateInfo query_pool_info {
      .queryType  = vk::QueryType::eTimestamp,
      .queryCount = frames_in_flight * queries_per_frame,
    };
    return map_vk_error(device.createQueryPool(query_pool_info), std::nullopt)
      .transform([ & ](vk::raii::QueryPool&& pool) -> timestamp_ring {
        pool.reset(0U, frames_in_flight * queries_per_frame);
        return timestamp_ring {
          std::move(pool),
          properties.limits.timestampPeriod,
          queries_per_frame,
          std::numeric_limits<std::uint64_t>::digits,
        };
      });
  }

  [[nodiscard]]
  static
  auto
  create_elapsed(
    const vk::raii::Device& device,
    float                   timestamp_period,
    std::uint32_t           timestamp_valid_bits,
    bool reset_on_create) -> std::expected<timestamp_ring, error_t>
  {
    vk::QueryPoolCreateInfo query_pool_info {
      .queryType  = vk::QueryType::eTimestamp,
      .queryCount = elapsed_query_count,
    };
    if (reset_on_create)
    {
      query_pool_info.flags = vk::QueryPoolCreateFlagBits::eResetKHR;
    }
    return map_vk_error(device.createQueryPool(query_pool_info), std::nullopt)
      .transform([ & ](vk::raii::QueryPool&& pool) -> timestamp_ring {
        return timestamp_ring {
          std::move(pool),
          timestamp_period,
          elapsed_query_count,
          timestamp_valid_bits,
        };
      });
  }

  void
  reset_elapsed(vk::raii::CommandBuffer& command_buffer) const
  { command_buffer.resetQueryPool(*pool_, 0U, elapsed_query_count); }

  [[nodiscard]]
  auto
  read_elapsed()
    -> std::expected<std::chrono::duration<double, std::nano>, error_t>
  {
    auto [ result, timestamps ] = pool_.getResults<std::uint64_t>(
      0U,
      elapsed_query_count,
      elapsed_query_count * sizeof(std::uint64_t),
      sizeof(std::uint64_t),
      vk::QueryResultFlagBits::e64);
    if (
      result != vk::Result::eSuccess ||
      timestamps.size() != elapsed_query_count)
    {
      return std::unexpected {make_app_error(app_error_code::invalid_state)};
    }
    const auto elapsed = (timestamps[ 1 ] - timestamps[ 0 ]) & valid_bit_mask_;
    return std::chrono::duration<double, std::nano> {
      static_cast<double>(elapsed) * static_cast<double>(timestamp_period_),
    };
  }

  void
  write(
    vk::raii::CommandBuffer&   command_buffer,
    std::uint32_t              frame_slot,
    std::uint32_t              query,
    vk::PipelineStageFlagBits2 stage) const
  {
    command_buffer.writeTimestamp2(
      stage, *pool_, frame_slot * queries_per_frame_ + query);
  }

  [[nodiscard]]
  auto
  read_and_reset_frame_ns(std::uint32_t frame_slot)
    -> std::expected<std::vector<double>, error_t>
  {
    const auto first       = frame_slot * queries_per_frame_;
    auto [ result, ticks ] = pool_.getResults<std::uint64_t>(
      first,
      queries_per_frame_,
      queries_per_frame_ * sizeof(std::uint64_t),
      sizeof(std::uint64_t),
      vk::QueryResultFlagBits::e64);
    if (result == vk::Result::eNotReady)
    {
      return std::unexpected {
        make_app_error(app_error_code::invalid_state),
      };
    }
    pool_.reset(first, queries_per_frame_);

    std::vector<double> nanoseconds(ticks.size());
    for (auto index : std::views::indices(ticks.size()))
    {
      nanoseconds[ index ] =
        static_cast<double>(ticks[ index ]) *
        static_cast<double>(timestamp_period_);
    }
    return nanoseconds;
  }

private:
  static constexpr std::uint32_t elapsed_query_count {2U};

  vk::raii::QueryPool pool_ {nullptr};
  float               timestamp_period_ {0.0F};
  std::uint32_t       queries_per_frame_ {0U};
  std::uint64_t       valid_bit_mask_ {
    valid_bit_mask(std::numeric_limits<std::uint64_t>::digits),
  };
};

} // namespace vkpp
