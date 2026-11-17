export module vkpp.integer;

import std;

namespace vkpp
{

static_assert(
  std::numeric_limits<std::size_t>::max() >=
    std::numeric_limits<std::uint64_t>::max(),
  "for now vkpp requires a 64-bit host");

[[noreturn]]
inline
void
narrowing_failed() noexcept
{ std::abort(); }

export
template<std::integral To, std::integral From>
  requires (!std::same_as<To, From>)
[[nodiscard]]
constexpr auto
narrow(From value) noexcept -> To
{
  if (!std::in_range<To>(value)) { narrowing_failed(); }
  return static_cast<To>(value);
}

export
template<std::integral To, std::integral From>
  requires (!std::same_as<To, From>)
[[nodiscard]]
constexpr auto
try_narrow(From value) noexcept -> std::optional<To>
{
  if (!std::in_range<To>(value)) { return std::nullopt; }
  return static_cast<To>(value);
}

export
template<std::unsigned_integral Integer>
  requires (sizeof(Integer) >= sizeof(unsigned))
[[nodiscard]]
constexpr auto
devide_rounding_up(Integer dividend, Integer divisor) noexcept -> Integer
{ return dividend / divisor + (dividend % divisor != 0U ? 1U : 0U); }

export
template<std::ranges::sized_range Range>
[[nodiscard]]
constexpr auto
vulkan_count(const Range& range) noexcept -> std::uint32_t
{ return narrow<std::uint32_t>(std::ranges::size(range)); }

} // namespace vkpp
