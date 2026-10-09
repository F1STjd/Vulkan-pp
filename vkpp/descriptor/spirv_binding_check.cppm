export module vkpp.descriptor.spirv_binding_check;

import std;
import vulkan;

import vkpp.error;
import vkpp.descriptor.layout_decl;

namespace vkpp
{

inline constexpr std::uint32_t spirv_op_decorate { 71U };
inline constexpr std::uint32_t spirv_decoration_binding { 33U };
inline constexpr std::uint32_t spirv_decoration_descriptor_set { 34U };
inline constexpr std::uint32_t spirv_opcode_mask { 0xFFFFU };
inline constexpr std::uint32_t spirv_word_count_shift { 16U };

export template<class Layout>
auto
validate_spirv_bindings(std::span<const std::uint32_t> spirv_words, Layout)
  -> std::expected<void, error_t>
{
  struct partial
  {
    std::optional<std::uint32_t> set {};
    std::optional<std::uint32_t> binding {};
  };
  std::unordered_map<std::uint32_t, partial> by_id {};

  std::size_t i { 5UZ }; // skip SPIR-V header (5 words)
  while (i < spirv_words.size())
  {
    const std::uint32_t header = spirv_words[i];
    const std::uint32_t wc = header >> spirv_word_count_shift;
    const std::uint32_t op = header & spirv_opcode_mask;
    if (wc == 0 || i + wc > spirv_words.size())
    {
      return std::unexpected { make_app_error(app_error_code::invalid_state) };
    }
    if (op == spirv_op_decorate && wc >= 4U)
    {
      const std::uint32_t target_id = spirv_words[i + 1U];
      const std::uint32_t decoration = spirv_words[i + 2U];
      const std::uint32_t literal = spirv_words[i + 3U];
      auto& slot = by_id[target_id];
      if (decoration == spirv_decoration_descriptor_set) { slot.set = literal; }
      else if (decoration == spirv_decoration_binding) { slot.binding = literal; }
    }
    i += wc;
  }

  std::vector<classic_binding_id> found {};
  found.reserve(by_id.size());
  for (const auto& [_, p] : by_id)
  {
    if (!p.set || !p.binding) { continue; }
    found.push_back({ .set = *p.set, .binding = *p.binding });
  }

  constexpr auto table = Layout::assignment_table();

  auto row_matches = [&](const binding_assignment& row) -> bool
  {
    for (const auto& f : found)
    {
      if (f.set == row.set && f.binding == row.binding) { return true; }
    }
    return false;
  };
  auto found_matches_row = [&](const classic_binding_id& f) -> bool
  {
    for (const auto& row : table)
    {
      if (f.set == row.set && f.binding == row.binding) { return true; }
    }
    return false;
  };

  for (const auto& f : found)
  {
    if (!found_matches_row(f))
    {
      return std::unexpected { make_app_error(app_error_code::invalid_state) };
    }
  }
  for (const auto& row : table)
  {
    if (!row_matches(row))
    {
      return std::unexpected { make_app_error(app_error_code::invalid_state) };
    }
  }
  return {};
}

export template<class Layout>
auto
validate_spirv_bindings(std::span<const std::uint32_t> spirv_words, Layout,
  vk::ShaderStageFlags expected_stages) -> std::expected<void, error_t>
{
  if (spirv_words.size() < 5U)
  {
    return std::unexpected { make_app_error(app_error_code::invalid_state) };
  }

  struct partial
  {
    std::optional<std::uint32_t> set {};
    std::optional<std::uint32_t> binding {};
  };
  std::unordered_map<std::uint32_t, partial> by_id {};

  std::size_t i { 5U };
  while (i < spirv_words.size())
  {
    const std::uint32_t header = spirv_words[i];
    const std::uint32_t wc = header >> spirv_word_count_shift;
    const std::uint32_t op = header & spirv_opcode_mask;
    if (wc == 0U || i + wc > spirv_words.size())
    {
      return std::unexpected { make_app_error(app_error_code::invalid_state) };
    }
    if (op == spirv_op_decorate && wc >= 4U)
    {
      const std::uint32_t target_id = spirv_words[i + 1U];
      const std::uint32_t decoration = spirv_words[i + 2U];
      const std::uint32_t literal = spirv_words[i + 3U];
      auto& slot = by_id[target_id];
      if (decoration == spirv_decoration_descriptor_set) { slot.set = literal; }
      else if (decoration == spirv_decoration_binding) { slot.binding = literal; }
    }
    i += wc;
  }

  std::vector<classic_binding_id> found {};
  for (const auto& [_, p] : by_id)
  {
    if (p.set && p.binding)
    { found.push_back({ .set = *p.set, .binding = *p.binding }); }
  }

  constexpr auto table = Layout::assignment_table();
  for (const auto& f : found)
  {
    bool ok = false;
    for (const auto& row : table)
    {
      if (f.set == row.set && f.binding == row.binding) { ok = true; break; }
    }
    if (!ok)
    {
      return std::unexpected { make_app_error(app_error_code::invalid_state) };
    }
  }
  for (const auto& row : table)
  {
    if ((row.stage_flags & expected_stages) == vk::ShaderStageFlags{})
    {
      continue;
    }
    bool ok = false;
    for (const auto& f : found)
    {
      if (f.set == row.set && f.binding == row.binding) { ok = true; break; }
    }
    if (!ok)
    {
      return std::unexpected { make_app_error(app_error_code::invalid_state) };
    }
  }
  return {};
}

} // namespace vkpp