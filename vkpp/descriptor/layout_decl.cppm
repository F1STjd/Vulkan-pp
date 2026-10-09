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

inline void
layout_error([[maybe_unused]] const char* message)
{}

} // namespace detail

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

template<class T>
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
  [[maybe_unused]] auto consider = [ & ]<class Field>()
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
    [[maybe_unused]] auto consider = [ & ]<class Field>()
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
      detail::layout_error("push_layout: tag not in field pack");
    }
    return found;
  }
};

export struct classic_binding_id
{
  std::uint32_t set {};
  std::uint32_t binding {};
};

export enum class resource_scope : std::uint8_t {
  persistent,
  bindless,
  per_frame,
  per_pass,
};

export consteval auto
slot_of(resource_scope scope) -> std::uint32_t
{
  switch (scope)
  {
  case resource_scope::persistent: return 0U;
  case resource_scope::bindless  : return 1U;
  case resource_scope::per_frame : return 2U;
  case resource_scope::per_pass  : return 3U;
  }
  detail::layout_error("resource_scope: unknown scope");
  return 0U;
}

export template<resource_scope Scope, class... Decls>
struct scope
{
  static constexpr resource_scope value = Scope;
  using declarations = std::tuple<Decls...>;
  static constexpr std::size_t count = sizeof...(Decls);
};

export template<std::uint32_t N>
struct pin
{
  static constexpr std::uint32_t value { N };
};

export template<vk::ShaderStageFlagBits... Flags>
struct stages
{
  static constexpr vk::ShaderStageFlags value {
    (vk::ShaderStageFlags {} | ... | vk::ShaderStageFlags { Flags }),
  };
};

export template<std::uint32_t N>
struct variable_count
{
  static constexpr std::uint32_t value { N };
};

export template<std::uint32_t N>
struct array_count
{
  static constexpr std::uint32_t value { N };
};

export enum class resource_declaration_kind : std::uint8_t {
  storage,
  dynamic_uniform,
  cis,
  cis_table,
};

export template<class T, class... Traits>
struct storage_resource
{
  using tag = T;
  using value_type = T;
  using traits_tuple = std::tuple<Traits...>;
  static constexpr resource_declaration_kind kind {
    resource_declaration_kind::storage
  };
};

export template<class T, class... Traits>
struct dynamic_uniform_resource
{
  using tag = T;
  using value_type = T;
  using traits_tuple = std::tuple<Traits...>;
  static constexpr resource_declaration_kind kind {
    resource_declaration_kind::dynamic_uniform
  };
};

export template<class T, class... Traits>
struct cis_resource
{
  using tag = T;
  using value_type = T;
  using traits_tuple = std::tuple<Traits...>;
  static constexpr resource_declaration_kind kind {
    resource_declaration_kind::cis
  };
};

export template<class T, class... Traits>
struct cis_table_resource
{
  using tag = T;
  using value_type = T;
  using traits_tuple = std::tuple<Traits...>;
  static constexpr resource_declaration_kind kind {
    resource_declaration_kind::cis_table
  };
};

export struct binding_assignment
{
  std::uint32_t tag_index {};
  resource_scope scope {};
  std::uint32_t set {};
  std::uint32_t binding {};
  vk::DescriptorType descriptor_type {};
  vk::ShaderStageFlags stage_flags {};
  std::uint32_t count {};
  bool dynamic {};
  bool update_after_bind {};
};

namespace detail
{

template<class Needle, class... Traits>
inline constexpr bool has_trait_v = (std::is_same_v<Needle, Traits> || ...);

template<class>
struct is_pin : std::false_type
{};

template<std::uint32_t N>
struct is_pin<pin<N>> : std::true_type
{};

template<class>
struct is_variable_count : std::false_type
{};

template<std::uint32_t N>
struct is_variable_count<variable_count<N>> : std::true_type
{};

template<class>
struct is_array_count : std::false_type
{};

template<std::uint32_t N>
struct is_array_count<array_count<N>> : std::true_type
{};

template<class... Traits>
consteval auto
find_pin() -> std::optional<std::uint32_t>
{
  std::optional<std::uint32_t> out {};
  [[maybe_unused]] auto consider = [ & ]<class Trait>
  {
    if constexpr (is_pin<Trait>::value) { out = Trait::value; }
  };
  (consider.template operator()<Traits>(), ...);
  return out;
}

template<class... Traits>
consteval auto
find_variable_count() -> std::optional<std::uint32_t>
{
  std::optional<std::uint32_t> out {};
  [[maybe_unused]] auto consider = [ & ]<class Trait>
  {
    if constexpr (is_variable_count<Trait>::value) { out = Trait::value; }
  };
  (consider.template operator()<Traits>(), ...);
  return out;
}

template<class... Traits>
consteval auto
find_array_count() -> std::optional<std::uint32_t>
{
  std::optional<std::uint32_t> out {};
  [[maybe_unused]] auto consider = [ & ]<class Trait>
  {
    if constexpr (is_array_count<Trait>::value) { out = Trait::value; }
  };
  (consider.template operator()<Traits>(), ...);
  return out;
}

template<class...>
struct stages_detector
{
  static constexpr vk::ShaderStageFlags value { vk::ShaderStageFlagBits::eAll };
};

template<vk::ShaderStageFlagBits... Flags, class... Rest>
struct stages_detector<stages<Flags...>, Rest...>
{
  static constexpr vk::ShaderStageFlags value { stages<Flags...>::value };
};

template<class Head, class... Rest>
struct stages_detector<Head, Rest...> : stages_detector<Rest...>
{};

template<class... Traits>
consteval auto
find_stages() -> vk::ShaderStageFlags
{ return stages_detector<Traits...>::value; }

template<class Tuple>
struct trait_queries;

template<class... Traits>
struct trait_queries<std::tuple<Traits...>>
{
  static consteval auto
  variable_count() -> std::optional<std::uint32_t>
  { return find_variable_count<Traits...>(); }

  static consteval auto
  array_count() -> std::optional<std::uint32_t>
  { return find_array_count<Traits...>(); }

  static consteval auto
  stages() -> vk::ShaderStageFlags
  { return find_stages<Traits...>(); }

  static consteval auto
  pin() -> std::optional<std::uint32_t>
  { return find_pin<Traits...>(); }
};

struct kind_meta_t
{
  vk::DescriptorType descriptor_type {};
  bool dynamic {};
  bool update_after_bind {};
  vk::DescriptorBindingFlags binding_flags {};
};

consteval auto
kind_meta(resource_declaration_kind kind) -> kind_meta_t
{
  switch (kind)
  {
  case resource_declaration_kind::storage:
    return {
      .descriptor_type = vk::DescriptorType::eStorageBuffer,
      .dynamic = false,
      .update_after_bind = false,
      .binding_flags = {},
    };
  case resource_declaration_kind::dynamic_uniform:
    return {
      .descriptor_type = vk::DescriptorType::eUniformBufferDynamic,
      .dynamic = true,
      .update_after_bind = false,
      .binding_flags = {},
    };
  case resource_declaration_kind::cis:
    return {
      .descriptor_type = vk::DescriptorType::eCombinedImageSampler,
      .dynamic = false,
      .update_after_bind = false,
      .binding_flags = {},
    };
  case resource_declaration_kind::cis_table:
    return {
      .descriptor_type = vk::DescriptorType::eCombinedImageSampler,
      .dynamic = false,
      .update_after_bind = true,
      .binding_flags = vk::DescriptorBindingFlagBits::eUpdateAfterBind |
        vk::DescriptorBindingFlagBits::eVariableDescriptorCount |
        vk::DescriptorBindingFlagBits::ePartiallyBound,
    };
  }
  return {};
}

template<class... Tags>
struct tag_list
{
  static constexpr std::size_t size { sizeof...(Tags) };
  using tuple = std::tuple<Tags...>;
  template<std::size_t I>
  using at = std::tuple_element_t<I, tuple>;
};

template<resource_scope S, class Decl>
struct decl_in_scope
{
  using declaration = Decl;
  static constexpr resource_scope scope_value = S;
};

template<class Acc, class... ScopeOrDecl>
struct flatten_scoped_decls;

template<class... Acc>
struct flatten_scoped_decls<std::tuple<Acc...>>
{
  using type = std::tuple<Acc...>;
};

template<class... Acc, resource_scope S, class... Decls, class... Rest>
struct flatten_scoped_decls<std::tuple<Acc...>, scope<S, Decls...>, Rest...>
{
  using type = typename flatten_scoped_decls<
    std::tuple<Acc..., decl_in_scope<S, Decls>...>, Rest...>::type;
};

template<class Acc, class... ScopeOrDecl>
struct flatten_tags;

template<class... AccTags>
struct flatten_tags<tag_list<AccTags...>>
{
  using type = tag_list<AccTags...>;
};

template<class... AccTags, resource_scope S, class... Decls, class... Rest>
struct flatten_tags<tag_list<AccTags...>, scope<S, Decls...>, Rest...>
{
  using type =
    typename flatten_tags<tag_list<AccTags..., typename Decls::tag...>,
      Rest...>::type;
};

template<class Tags, class Needle, std::size_t I = 0>
consteval auto
tag_index_of() -> std::size_t
{
  if constexpr (I >= Tags::size) { return static_cast<std::size_t>(-1); }
  else if constexpr (std::is_same_v<typename Tags::template at<I>, Needle>)
  {
    return I;
  }
  else
  {
    return tag_index_of<Tags, Needle, I + 1>();
  }
}

} // namespace detail

export template<class... ScopeOrDecl>
struct resource_layout
{
  static constexpr std::uint32_t set_count { 4U };

  using scoped_decls =
    typename detail::flatten_scoped_decls<std::tuple<>, ScopeOrDecl...>::type;
  using tags =
    typename detail::flatten_tags<detail::tag_list<>, ScopeOrDecl...>::type;

  static consteval auto
  resource_count() -> std::size_t
  { return std::tuple_size_v<scoped_decls>; }

  static consteval auto
  assignment_table()
    -> std::array<binding_assignment, std::tuple_size_v<scoped_decls>>
  {
    std::array<binding_assignment, std::tuple_size_v<scoped_decls>> table;
    std::array<std::uint32_t, set_count> next_binding {};
    std::array<bool, set_count> scope_has_uab {};
    std::array<bool, set_count> scope_has_dynamic {};
    std::array<std::array<bool, 64>, set_count> occupied {};

    auto fill_one = [ & ]<std::size_t I>()
    {
      using SD = std::tuple_element_t<I, scoped_decls>;
      using Decl = typename SD::declaration;
      using TraitsTuple = typename Decl::traits_tuple;
      constexpr resource_scope sc = SD::scope_value;
      constexpr std::uint32_t set = slot_of(sc);
      constexpr auto meta = detail::kind_meta(Decl::kind);
      constexpr auto pin_out = detail::trait_queries<TraitsTuple>::pin();
      constexpr auto var_opt =
        detail::trait_queries<TraitsTuple>::variable_count();
      constexpr auto arr_opt =
        detail::trait_queries<TraitsTuple>::array_count();
      constexpr auto stage_flags = detail::trait_queries<TraitsTuple>::stages();

      if constexpr (Decl::kind == resource_declaration_kind::cis_table)
      {
        static_assert(sc == resource_scope::bindless,
          "resource_layout: cis_table_resource only in bindless");
        static_assert(var_opt.has_value(),
          "resource_layout: cis_table_resource needs variable_count");
      }

      std::uint32_t binding { 0U };
      if constexpr (pin_out.has_value()) { binding = *pin_out; }
      else
      {
        binding = next_binding[ set ];
        while (binding < 64U && occupied[ set ][ binding ])
        {
          ++binding;
        }
      }
      if (binding >= 64U || occupied[ set ][ binding ])
      {
        detail::layout_error(
          "resource_layout: pin collision or binding exhausted in scope");
        return;
      }
      occupied[ set ][ binding ] = true;
      if constexpr (meta.update_after_bind) { scope_has_uab[ set ] = true; }
      if constexpr (meta.dynamic) { scope_has_dynamic[ set ] = true; }
      if (scope_has_uab[ set ] && scope_has_dynamic[ set ])
      {
        detail::layout_error(
          "resource_layout: UpdateAfterBind and dynamic in same scope");
      }

      const std::uint32_t count = var_opt ? *var_opt : arr_opt ? *arr_opt : 1U;

      table[ I ] = binding_assignment {
        .tag_index = static_cast<std::uint32_t>(I),
        .scope = sc,
        .set = set,
        .binding = binding,
        .descriptor_type = meta.descriptor_type,
        .stage_flags = stage_flags,
        .count = count,
        .dynamic = meta.dynamic,
        .update_after_bind = meta.update_after_bind,
      };
      next_binding[ set ] = std::max(next_binding[ set ], binding + 1U);
    };

    [ & ]<std::size_t... Is>(std::index_sequence<Is...>)
    { (fill_one.template operator()<Is>(), ...); }(
      std::make_index_sequence<std::tuple_size_v<scoped_decls>> {});

    [ & ]<std::size_t... Is>(std::index_sequence<Is...>)
    {
      auto check_i = [ & ]<std::size_t I>()
      {
        auto check_j = [ & ]<std::size_t J>()
        {
          if constexpr (I < J)
          {
            static_assert(!std::is_same_v<typename tags::template at<I>,
                            typename tags::template at<J>>,
              "resource_layout: duplicate tag");
          }
        };
        (check_j.template operator()<Is>(), ...);
      };
      (check_i.template operator()<Is>(), ...);
    }(std::make_index_sequence<tags::size> {});

    return table;
  }

  template<class Tag>
  static consteval auto
  resource_index_of() -> std::size_t
  {
    constexpr auto idx = detail::tag_index_of<tags, Tag>();
    static_assert(
      idx != static_cast<std::size_t>(-1), "resource_layout: tag not declared");
    return idx;
  }

  template<class Tag>
  static consteval auto
  binding_of() -> classic_binding_id
  {
    constexpr auto table = assignment_table();
    constexpr auto idx = resource_index_of<Tag>();
    return classic_binding_id {
      .set = table[ idx ].set,
      .binding = table[ idx ].binding,
    };
  }

  template<class Tag>
  static consteval auto
  dynamic_offset_index() -> std::uint32_t
  {
    constexpr auto table = assignment_table();
    constexpr auto idx = resource_index_of<Tag>();
    constexpr auto target = table[ idx ];
    static_assert(target.dynamic, "resource_layout: tag is not dynamic");
    return static_cast<std::uint32_t>(std::ranges::count_if(table,
      [ &target ](const binding_assignment& assignment)
      {
        return assignment.dynamic &&
          (assignment.set < target.set ||
            (assignment.set == target.set &&
              assignment.binding < target.binding));
      }));
  }

  static consteval auto
  dynamic_offset_count() -> std::uint32_t
  {
    constexpr auto table = assignment_table();
    return static_cast<std::uint32_t>(std::ranges::count_if(table,
      [](const binding_assignment& assignment) -> bool
      { return assignment.dynamic; }));
  }
};

namespace detail::layout_selftest
{

struct material
{};
struct draw_ssbo
{};
struct ibl
{};
struct bindless_textures
{};
struct frame_ubo
{};
struct histogram_image
{};
struct pinned
{};
struct after_pin
{};

using shape = resource_layout<
  scope<resource_scope::persistent, storage_resource<material>,
    storage_resource<draw_ssbo>, cis_resource<ibl>>,
  scope<resource_scope::bindless,
    cis_table_resource<bindless_textures, variable_count<1024U>>>,
  scope<resource_scope::per_frame, dynamic_uniform_resource<frame_ubo>>,
  scope<resource_scope::per_pass,
    cis_resource<histogram_image, stages<vk::ShaderStageFlagBits::eCompute>>>>;

consteval auto
is_at(classic_binding_id id, std::uint32_t set, std::uint32_t binding) -> bool
{ return id.set == set && id.binding == binding; }

static_assert(is_at(shape::binding_of<material>(), 0U, 0U));
static_assert(is_at(shape::binding_of<draw_ssbo>(), 0U, 1U));
static_assert(is_at(shape::binding_of<ibl>(), 0U, 2U));
static_assert(is_at(shape::binding_of<bindless_textures>(), 1U, 0U));
static_assert(is_at(shape::binding_of<frame_ubo>(), 2U, 0U));
static_assert(is_at(shape::binding_of<histogram_image>(), 3U, 0U));

constexpr auto f1st_table = shape::assignment_table();
static_assert(
  f1st_table[ shape::resource_index_of<bindless_textures>() ].count == 1024U);
static_assert(f1st_table[ shape::resource_index_of<material>() ].count == 1U);
static_assert(
  f1st_table[ shape::resource_index_of<histogram_image>() ].stage_flags ==
  vk::ShaderStageFlags { vk::ShaderStageFlagBits::eCompute });
static_assert(f1st_table[ shape::resource_index_of<material>() ].stage_flags ==
  vk::ShaderStageFlags { vk::ShaderStageFlagBits::eAll });

static_assert(shape::dynamic_offset_count() == 1U);
static_assert(shape::dynamic_offset_index<frame_ubo>() == 0U);

using mixed_shape = resource_layout<scope<resource_scope::persistent,
  storage_resource<pinned, pin<3U>, array_count<4U>,
    stages<vk::ShaderStageFlagBits::eVertex,
      vk::ShaderStageFlagBits::eFragment>>,
  storage_resource<after_pin>>>;
constexpr auto mixed_table = mixed_shape::assignment_table();
static_assert(is_at(mixed_shape::binding_of<pinned>(), 0U, 3U));
static_assert(is_at(mixed_shape::binding_of<after_pin>(), 0U, 4U));
static_assert(mixed_table[ 0 ].count == 4U);
static_assert(mixed_table[ 0 ].stage_flags ==
  (vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment));
static_assert(mixed_shape::dynamic_offset_count() == 0U);

} // namespace detail::layout_selftest

} // namespace vkpp
