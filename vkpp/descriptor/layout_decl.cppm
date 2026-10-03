export module vkpp.descriptor.layout_decl;

import std;
import vulkan;

import vkpp.capabilities;

namespace vkpp
{

export template<class Tag, class T>
struct push_field
{
  using tag = Tag;
  using value_type = T;
  static_assert(std::is_trivially_copyable_v<T>);
};

template<class... Fields>
consteval auto
push_offsets_array() -> std::array<std::uint32_t, sizeof...(Fields)>
{
  std::array<std::uint32_t, sizeof...(Fields)> offsets {};
  std::uint32_t cursor {};
  std::size_t index {};
  auto consider = [ & ]<class Field>()
  {
    constexpr auto align =
      static_cast<std::uint32_t>(alignof(typename Field::value_type));
    constexpr auto size =
      static_cast<std::uint32_t>(sizeof(typename Field::value_type));
    cursor = (cursor + align - 1U) / align * align;
    offsets[ index++ ] = cursor;
    cursor += size;
  };
  (consider.template operator()<Fields>(), ...);
  return offsets;
}

template<class... Fields>
consteval auto
push_total_bytes() -> std::uint32_t
{
  if constexpr (sizeof...(Fields) == 0UZ) { return 0U; }
  else
  {
    constexpr auto offsets = push_offsets_array<Fields...>();
    using last = typename std::tuple_element_t<sizeof...(Fields) - 1UZ,
      std::tuple<Fields...>>;
    return offsets.back() +
      static_cast<std::uint32_t>(sizeof(typename last::value_type));
  }
}

export template<class FrameSlotTag, class... Fields>
struct push_layout
{
  using frame_slot_tag = FrameSlotTag;
  static constexpr std::size_t field_count { sizeof...(Fields) };
  static constexpr std::array<std::uint32_t, sizeof...(Fields)> offsets {
    push_offsets_array<Fields...>()
  };
  static constexpr std::uint32_t total_bytes { push_total_bytes<Fields...>() };
  static_assert((std::is_same_v<FrameSlotTag, typename Fields::tag> || ...),
    "FrameSlotTag must name one of the push fields");

  template<class Tag>
  static consteval auto
  offset_of() -> std::uint32_t
  {
    std::uint32_t found { std::numeric_limits<std::uint32_t>::max() };
    std::size_t index {};
    auto consider = [ & ]<class Field>()
    {
      if constexpr (std::is_same_v<Tag, typename Field::tag>)
      {
        found = offsets[ index ];
      }
      ++index;
    };
    (consider.template operator()<Fields>(), ...);
    if (found == std::numeric_limits<std::uint32_t>::max()) { throw; }
    return found;
  }
};

export enum class logical_resource_role : std::uint8_t {
  frame_uniform_ring,
  material_storage,
  draw_storage,
  bindless_cis,
  ibl_cis,
  compute_cis,
};

export template<logical_resource_role... Roles>
struct logical_resource_pack
{
  static constexpr std::size_t count { sizeof...(Roles) };
  static constexpr std::array<logical_resource_role, sizeof...(Roles)> roles {
    Roles...
  };

  static consteval auto
  index_of(logical_resource_role role) -> std::size_t
  {
    auto found = std::ranges::find(roles, role);
    if (found == roles.end()) { throw; }
    return std::ranges::distance(roles.begin(), found);
  }
};

export struct classic_binding_id
{
  std::uint32_t set {};
  std::uint32_t binding {};
};

export consteval auto
classic_binding_for(logical_resource_role role) -> classic_binding_id
{
  switch (role)
  {
  case logical_resource_role::frame_uniform_ring:
    return { .set = 0U, .binding = 0U };
  case logical_resource_role::material_storage:
    return { .set = 0U, .binding = 1U };
  case logical_resource_role::draw_storage: return { .set = 0U, .binding = 3U };
  case logical_resource_role::bindless_cis: return { .set = 1U, .binding = 0U };
  case logical_resource_role::ibl_cis     : return { .set = 2U, .binding = 0U };
  case logical_resource_role::compute_cis : return { .set = 0U, .binding = 4U };
  }
  throw;
}

} // namespace vkpp
