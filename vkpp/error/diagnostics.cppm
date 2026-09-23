export module vkpp.diagnostics;

import std;
import vulkan;

import vkpp.error;

namespace vkpp
{

export enum class diagnostic_severity : std::uint8_t {
  off,
  error,
  warning,
  info,
  verbose,
};

export struct diagnostic_record
{
  diagnostic_severity severity { diagnostic_severity::error };
  std::optional<error_t> error {};
  std::source_location location {};
  std::array<char, 256> text {};
  std::uint16_t text_size {};
  bool truncated { false };
};

export class diagnostic_buffer
{
public:
  diagnostic_buffer(
    std::span<diagnostic_record> storage, diagnostic_severity threshhold)
  : storage_ { storage }, threshhold_ { threshhold }
  {}

  [[nodiscard]] auto
  threshold() const -> diagnostic_severity
  { return threshhold_; }

  [[nodiscard]] auto
  size() const -> std::size_t
  { return count_; }

  [[nodiscard]] auto
  dropped() const -> std::size_t
  { return dropped_; }

  template<class... Args>
  void
  report(diagnostic_severity severity, std::optional<error_t> error,
    std::source_location location, std::format_string<Args...> fmt,
    Args&&... args)
  {
    if (threshhold_ == diagnostic_severity::off) { return; }
    if (severity > threshhold_) { return; }
    if (storage_.empty()) { return; }

    diagnostic_record record {
      .severity = severity,
      .error = error,
      .location = location,
    };
    auto [ out, count ] = std::format_to_n(
      record.text.begin(), 255, fmt, std::forward<Args>(args)...);
    (void)out;
    const auto written = static_cast<std::size_t>(count);
    record.text_size = static_cast<std::uint16_t>(std::min(written, 255UZ));
    record.truncated = written > 255UZ;
    record.text[ record.text_size ] = '\0';

    const std::scoped_lock loc { mutex_ };
    if (count_ < storage_.size())
    {
      storage_[ count++ ] = std::move(record);
      return;
    }
    storage_[ head_ ] = std::move(record);
    head_ = (head_ + 1UZ) % storage_.size();
    ++dropped_;
  }

  [[nodiscard]] auto
  copy_and_clear(std::span<diagnostic_record> destination) -> std::size_t
  {
    const std::scoped_lock lock { mutex_ };
    const auto n = std::min(count_, destination.size());
    if (count_ < storage_.size() || head_ == 0UZ)
    {
      std::ranges::copy_n(
        storage_.begin(), static_cast<std::ptrdiff_t>(n), destination.begin());
    }
    else
    {
      const auto first = storage_.size() - head_;
      const auto copy_first = std::min(first, n);
      std::ranges::copy_n(storage_.begin() + static_cast<std::ptrdiff_t>(head_),
        static_cast<std::ptrdiff_t>(copy_first), destination.begin());
      if (copy_first < n)
      {
        std::ranges::copy_n(storage_.begin(),
          static_cast<std::ptrdiff_t>(n - copy_first),
          destination.begin() + static_cast<std::ptrdiff_t>(copy_first));
      }
    }
    count_ = 0UZ;
    head_ = 0UZ;
    return n;
  }

private:
  std::span<diagnostic_record> storage_ {};
  diagnostic_severity threshhold_ { diagnostic_severity::error };
  std::size_t count_ {};
  std::size_t head_ {};
  std::size_t dropped_ {};
  std::mutex mutex_ {};
};

export template<typename T>
auto
map_vk_error(std::expected<T, vk::Result>&& result,
  std::optional<diagnostic_buffer&> diagnostics = {},
  std::source_location location = std::source_location::current())
  -> std::expected<T, error_t>
{
  if (result)
  {
    if constexpr (std::is_void_v<T>) { return {}; }
    else { return std::move(*result); }
  }
  const error_t error = make_vk_error(result.error());
  if (diagnostics.has_value())
  {
    diagnostics->report(diagnostic_severity::error, error, location,
      "vulkan call failed: {}", message(error));
  }
  return std::unexpected { error };
}

} // namespace vkpp
