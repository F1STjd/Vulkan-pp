export module vkpp.gpu:workspace;

import std;
import vulkan;
import fiat;

import vkpp.error;
import vkpp.device;
import vkpp.buffer;

namespace vkpp::gpu
{

export class context;

export using workspace_sizes = fiat::workspace_sizes;

export enum class workspace_region : std::uint8_t
{
  table,
  ping_pong,
  scratch,
};

inline constexpr std::array workspace_regions {
  workspace_region::table,
  workspace_region::ping_pong,
  workspace_region::scratch,
};

inline constexpr vk::DeviceSize region_alignment {16U};

[[nodiscard]]
constexpr
auto
aligned_region_bytes(std::size_t bytes) -> vk::DeviceSize
{
  const auto padded =
    std::saturating_add(vk::DeviceSize {bytes}, region_alignment - 1U);
  return padded - padded % region_alignment;
}

[[nodiscard]]
constexpr
auto
region_bounds(const workspace_sizes& sizes) -> std::array<vk::DeviceSize, 4UZ>
{
  const auto                      regions = sizes.regions();
  std::array<vk::DeviceSize, 4UZ> bounds {};
  for (const auto index : std::views::indices(regions.size()))
  {
    bounds[ index + 1UZ ] = std::saturating_add(
      bounds[ index ], aligned_region_bytes(regions[ index ]));
  }
  return bounds;
}

[[nodiscard]]
constexpr
auto
workspace_bytes(const workspace_sizes& sizes) -> vk::DeviceSize
{ return region_bounds(sizes).back(); }

[[nodiscard]]
constexpr
auto
covers(const workspace_sizes& available, const workspace_sizes& required)
  -> bool
{
  return std::ranges::equal(
    available.regions(), required.regions(), std::ranges::greater_equal {});
}

export class workspace
{
public:
  [[nodiscard]]
  auto
  sizes() const noexcept -> const workspace_sizes&
  { return sizes_; }

  [[nodiscard]]
  auto
  buffer() const noexcept -> const device_address_buffer&
  { return buffer_; }

  [[nodiscard]]
  auto
  offset(workspace_region region) const noexcept -> vk::DeviceSize
  { return region_bounds(sizes_)[ std::to_underlying(region) ]; }

  [[nodiscard]]
  auto
  address(workspace_region region) const noexcept -> vk::DeviceAddress
  { return base_address_ + offset(region); }

private:
  friend class context;

  workspace(
    device_address_buffer&& buffer,
    const workspace_sizes&  sizes,
    vk::DeviceAddress       base_address)
  : buffer_ {std::move(buffer)}, sizes_(sizes), base_address_(base_address)
  {}

  device_address_buffer buffer_;
  workspace_sizes       sizes_;
  vk::DeviceAddress     base_address_ {};
};

template<class Stage>
using region_addresses =
  std::array<vk::DeviceSize, fiat::workspace_of<Stage>.region_count()>;

template<class Stage>
[[nodiscard]]
auto
bind_workspace(const workspace& available)
  -> std::expected<region_addresses<Stage>, error_t>
{
  constexpr auto required = fiat::workspace_of<Stage>;
  if (!covers(available.sizes(), required))
  {
    return std::unexpected {make_app_error(app_error_code::invalid_data_size)};
  }
  region_addresses<Stage> addresses {};
  auto                    next = addresses.begin();
  for (const auto region : workspace_regions)
  {
    if (required.regions()[ std::to_underlying(region) ] != 0UZ)
    {
      *next++ = available.address(region);
    }
  }
  return addresses;
}

static_assert(workspace_bytes({.scratch_bytes = 4'104UZ}) == 4'112U);
static_assert(
  region_bounds({.table_bytes = 1UZ, .scratch_bytes = 8UZ}) ==
  std::array<vk::DeviceSize, 4UZ> {0U, 16U, 16U, 32U});
static_assert(
  workspace_bytes({
    .table_bytes   = std::numeric_limits<std::size_t>::max(),
    .scratch_bytes = std::numeric_limits<std::size_t>::max(),
  }) == std::numeric_limits<vk::DeviceSize>::max());
static_assert(covers({.scratch_bytes = 4'104UZ}, {.scratch_bytes = 4'104UZ}));
static_assert(!covers({.scratch_bytes = 8UZ}, {.scratch_bytes = 4'104UZ}));

} // namespace vkpp::gpu
