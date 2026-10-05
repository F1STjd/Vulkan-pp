export module vkpp.descriptor.layout_decl;

import std;
import vulkan;

import vkpp.capabilities;

namespace vkpp
{

namespace detail
{

export template<class>
inline constexpr bool always_false_v { false };

}

export template<class T>
struct shader_layout_traits
{};

template<>
struct shader_layout_traits<std::uint32_t>
{
  static constexpr std::uint32_t align { 4U };
  static constexpr std::uint32_t size { 4U };
};

template<>
struct shader_layout_traits<std::uint64_t>
{
  static constexpr std::uint32_t align { 8U };
  static constexpr std::uint32_t size { 8U };
};

export template<class T>
  requires(std::is_trivially_copyable_v<T> && sizeof(T) == alignof(T) &&
    (sizeof(T) == 1UZ || sizeof(T) == 2UZ || sizeof(T) == 4UZ ||
      sizeof(T) == 8UZ) &&
    !std::is_same_v<T, std::uint32_t> && !std::is_same_v<T, std::uint64_t>)
struct shader_layout_traits<T>
{
  static constexpr std::uint32_t align {
    static_cast<std::uint32_t>(alignof(T)),
  };
  static constexpr std::uint32_t size {
    static_cast<std::uint32_t>(sizeof(T)),
  };
};

export template<class U>
struct shader_std430_vec3
{
  U x {};
  U y {};
  U z {};
};

template<class U>
struct shader_layout_traits<shader_std430_vec3<U>>
{
  static constexpr std::uint32_t align { 16U };
  static constexpr std::uint32_t size { 12U };
};

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
      shader_layout_traits<typename Field::value_type>::align;
    constexpr auto size =
      shader_layout_traits<typename Field::value_type>::size;
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
    const auto raw =
      offsets.back() + shader_layout_traits<typename last::value_type>::size;
    return (raw + 3U) / 4U * 4U;
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
  static_assert(total_bytes <= 128,
    "push_layout: exceeds portable push-constants minimum (128 bytes)");
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
    if (found == std::numeric_limits<std::uint32_t>::max())
    {
      static_assert(
        detail::always_false_v<Tag>, "push_layout: tag not in field pack");
    }
    return found;
  }
};

export struct classic_binding_id
{
  std::uint32_t set {};
  std::uint32_t binding {};
};

} // namespace vkpp
