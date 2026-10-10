export module vkpp.descriptor.layout_decl;

import std;
import vulkan;

import vkpp.capabilities;

namespace vkpp
{

namespace detail
{

inline
void
layout_error(const char*)
{}

template<std::size_t Count>
consteval
auto
index_of_true(const std::array<bool, Count>& matches) -> std::size_t
{
  return std::ranges::distance(
    matches.begin(), std::ranges::find(matches, true));
}

} // namespace detail

export template<class T>
struct shader_layout_traits {};

template<>
struct shader_layout_traits<std::uint32_t>
{
  static constexpr std::uint32_t align {4U};
  static constexpr std::uint32_t size {4U};
};

template<>
struct shader_layout_traits<std::uint64_t>
{
  static constexpr std::uint32_t align {8U};
  static constexpr std::uint32_t size {8U};
};

template<class T>
  requires (
    std::is_trivially_copyable_v<T> &&
    sizeof(T) == alignof(T) &&
    (sizeof(T) == 1UZ || sizeof(T) == 2UZ || sizeof(T) == 4UZ || sizeof(T) == 8UZ) &&
    !std::is_same_v<T, std::uint32_t> &&
    !std::is_same_v<T, std::uint64_t>)
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
  static constexpr std::uint32_t align {16U};
  static constexpr std::uint32_t size {12U};
};

export template<class Tag, class T>
struct push_field
{
  using tag        = Tag;
  using value_type = T;
  static_assert(std::is_trivially_copyable_v<T>);
};

template<class... Fields>
consteval
auto
push_offsets_array() -> std::array<std::uint32_t, sizeof...(Fields)>
{
  std::array<std::uint32_t, sizeof...(Fields)> offsets {};
  std::uint32_t                                cursor {};
  std::size_t                                  index {};
  [[maybe_unused]]
  auto consider = [ & ]<class Field>() {
    constexpr auto align =
      shader_layout_traits<typename Field::value_type>::align;
    constexpr auto size = shader_layout_traits<typename Field::value_type>::size;
    cursor              = (cursor + align - 1U) / align * align;
    offsets[ index++ ]  = cursor;
    cursor             += size;
  };
  (consider.template operator ()<Fields>(), ...);
  return offsets;
}

template<class... Fields>
consteval
auto
push_total_bytes() -> std::uint32_t
{
  if constexpr (sizeof...(Fields) == 0UZ) { return 0U; }
  else
  {
    constexpr auto offsets = push_offsets_array<Fields...>();
    using last             = typename std::
      tuple_element_t<sizeof...(Fields) - 1UZ, std::tuple<Fields...>>;
    const auto raw =
      offsets.back() + shader_layout_traits<typename last::value_type>::size;
    return (raw + 3U) / 4U * 4U;
  }
}

export template<class FrameSlotTag, class... Fields>
struct push_layout
{
  using frame_slot_tag = FrameSlotTag;
  static constexpr std::size_t field_count {sizeof...(Fields)};
  static constexpr std::array<std::uint32_t, sizeof...(Fields)> offsets {
    push_offsets_array<Fields...>()
  };
  static constexpr std::uint32_t total_bytes {push_total_bytes<Fields...>()};
  static_assert(
    total_bytes <= 128,
    "push_layout: exceeds portable push-constants minimum (128 bytes)");
  static_assert(
    (std::is_same_v<FrameSlotTag, typename Fields::tag> || ...),
    "FrameSlotTag must name one of the push fields");

  template<class Tag>
  static consteval
  auto
  offset_of() -> std::uint32_t
  {
    constexpr std::size_t position {
      detail::index_of_true(
        std::array<bool, sizeof...(Fields)> {
          std::is_same_v<Tag, typename Fields::tag>...,
        }),
    };
    static_assert(
      position < sizeof...(Fields), "push_layout: tag not in the field pack");
    return offsets[ position ];
  }
};

export struct classic_binding_id
{
  std::uint32_t set {};
  std::uint32_t binding {};
};

export enum class resource_scope : std::uint8_t
{
  persistent,
  bindless,
  per_frame,
  per_pass,
};

export constexpr
auto
slot_of(resource_scope scope) -> std::uint32_t
{
  switch (scope)
  {
  case resource_scope::persistent : return 0U;
  case resource_scope::bindless   : return 1U;
  case resource_scope::per_frame  : return 2U;
  case resource_scope::per_pass   : return 3U;
  }
  detail::layout_error("resource_scope: unknown scope");
  return 0U;
}

namespace detail
{

template<resource_scope Scope, class Declaration>
struct scoped_declaration
{
  using declaration = Declaration;
  static constexpr resource_scope scope_value {Scope};
};

} // namespace detail

export template<resource_scope Scope, class... Declarations>
struct scope
{
  static constexpr resource_scope value {Scope};
  using scoped_declarations =
    std::tuple<detail::scoped_declaration<Scope, Declarations>...>;
};

export template<std::uint32_t N>
struct pin
{
  static constexpr std::uint32_t value {N};
};

export template<vk::ShaderStageFlagBits... Flags>
struct stages
{
  static constexpr vk::ShaderStageFlags value {
    (vk::ShaderStageFlags {} | ... | vk::ShaderStageFlags {Flags}),
  };
};

export template<std::uint32_t N>
struct variable_count
{
  static constexpr std::uint32_t value {N};
};

export template<std::uint32_t N>
struct array_count
{
  static constexpr std::uint32_t value {N};
};

export enum class resource_declaration_kind : std::uint8_t
{
  storage,
  dynamic_uniform,
  cis,
  cis_table,
};

namespace detail
{

template<template<std::uint32_t> class Wrapper, class Trait>
inline constexpr bool is_value_trait_v {false};

template<template<std::uint32_t> class Wrapper, std::uint32_t Value>
inline constexpr bool is_value_trait_v<Wrapper, Wrapper<Value>> {true};

template<class Trait>
inline constexpr bool is_stages_v {false};

template<vk::ShaderStageFlagBits... Flags>
inline constexpr bool is_stages_v<stages<Flags...>> {true};

template<template<std::uint32_t> class Wrapper, class... Traits>
consteval
auto
find_value_trait() -> std::optional<std::uint32_t>
{
  std::optional<std::uint32_t> found {};
  [[maybe_unused]]
  auto consider = [ & ]<class Trait> {
    if constexpr (is_value_trait_v<Wrapper, Trait>) { found = Trait::value; }
  };
  (consider.template operator ()<Traits>(), ...);
  return found;
}

template<class... Traits>
consteval
auto
find_stages() -> vk::ShaderStageFlags
{
  vk::ShaderStageFlags found {vk::ShaderStageFlagBits::eAll};
  [[maybe_unused]]
  auto consider = [ & ]<class Trait> {
    if constexpr (is_stages_v<Trait>) { found = Trait::value; }
  };
  (consider.template operator ()<Traits>(), ...);
  return found;
}

template<resource_declaration_kind Kind, class Tag, class... Traits>
struct resource_declaration
{
  static_assert(
    ((is_value_trait_v<pin, Traits> ||
       is_value_trait_v<variable_count, Traits> ||
       is_value_trait_v<array_count, Traits> ||
       is_stages_v<Traits>) &&
      ...),
    "resource declaration: unknown trait");

  using tag = Tag;
  static constexpr resource_declaration_kind    kind {Kind};
  static constexpr std::optional<std::uint32_t> pinned_binding {
    find_value_trait<pin, Traits...>(),
  };
  static constexpr std::optional<std::uint32_t> variable_count_value {
    find_value_trait<variable_count, Traits...>(),
  };
  static constexpr std::optional<std::uint32_t> array_count_value {
    find_value_trait<array_count, Traits...>(),
  };
  static constexpr vk::ShaderStageFlags stage_flags {
    find_stages<Traits...>(),
  };
};

} // namespace detail

template<class Tag, class... Traits>
struct storage_resource :
detail::resource_declaration<resource_declaration_kind::storage, Tag, Traits...> {
};

template<class Tag, class... Traits>
struct dynamic_uniform_resource :
detail::resource_declaration<
  resource_declaration_kind::dynamic_uniform,
  Tag,
  Traits...
> {};

template<class Tag, class... Traits>
struct cis_resource :
detail::resource_declaration<resource_declaration_kind::cis, Tag, Traits...> {};

template<class Tag, class... Traits>
struct cis_table_resource :
detail::
  resource_declaration<resource_declaration_kind::cis_table, Tag, Traits...> {};

export struct binding_assignment
{
  std::uint32_t              tag_index {};
  resource_scope             scope {};
  std::uint32_t              set {};
  std::uint32_t              binding {};
  vk::DescriptorType         descriptor_type {};
  vk::ShaderStageFlags       stage_flags {};
  std::uint32_t              count {};
  bool                       dynamic {};
  bool                       update_after_bind {};
  vk::DescriptorBindingFlags binding_flags {};
};

namespace detail
{

struct kind_meta_t
{
  vk::DescriptorType         descriptor_type {};
  bool                       dynamic {};
  bool                       update_after_bind {};
  vk::DescriptorBindingFlags binding_flags {};
};

consteval
auto
kind_meta(resource_declaration_kind kind) -> kind_meta_t
{
  switch (kind)
  {
  case resource_declaration_kind::storage :
    return {
      .descriptor_type   = vk::DescriptorType::eStorageBuffer,
      .dynamic           = false,
      .update_after_bind = false,
      .binding_flags     = {},
    };
  case resource_declaration_kind::dynamic_uniform :
    return {
      .descriptor_type   = vk::DescriptorType::eUniformBufferDynamic,
      .dynamic           = true,
      .update_after_bind = false,
      .binding_flags     = {},
    };
  case resource_declaration_kind::cis :
    return {
      .descriptor_type   = vk::DescriptorType::eCombinedImageSampler,
      .dynamic           = false,
      .update_after_bind = false,
      .binding_flags     = {},
    };
  case resource_declaration_kind::cis_table :
    return {
      .descriptor_type   = vk::DescriptorType::eCombinedImageSampler,
      .dynamic           = false,
      .update_after_bind = true,
      .binding_flags =
        vk::DescriptorBindingFlagBits::ePartiallyBound |
        vk::DescriptorBindingFlagBits::eUpdateAfterBind |
        vk::DescriptorBindingFlagBits::eVariableDescriptorCount |
        vk::DescriptorBindingFlagBits::eUpdateUnusedWhilePending,
    };
  }
  return {};
}

template<class Tag, class ScopedDeclarations>
consteval
auto
tag_position() -> std::size_t
{
  return std::apply(
    []<class... Scoped>(Scoped...) -> std::size_t {
      return index_of_true(
        std::array<bool, sizeof...(Scoped)> {
          std::is_same_v<Tag, typename Scoped::declaration::tag>...,
        });
    },
    ScopedDeclarations {});
}

template<class ScopedDeclarations>
consteval
auto
tags_unique() -> bool
{
  return std::apply(
    []<class... Scoped>(Scoped...) -> bool {
      std::size_t position {};
      return (
        (tah_position<typename Scoped::declaration::tag, ScopedDeclarations>() ==
          position++) &&
        ...);
    },
    ScopedDeclarations {});
}

} // namespace detail

export template<class... Scopes>
struct resource_layout
{
  static constexpr std::uint32_t set_count {4U};

  using scoped_declarations = decltype(std::tuple_cat(
    std::declval<typename Scopes::scoped_declarations>()...));

  static_assert(
    detail::tags_unique<scoped_declarations>(),
    "resource_layout: duplicate tag");

  static consteval
  auto
  resource_count() -> std::size_t
  { return std::tuple_size_v<scoped_declarations>; }

  static consteval
  auto
  assignment_table()
    -> std::array<binding_assignment, std::tuple_size_v<scoped_declarations>>
  {
    std::array<binding_assignment, std::tuple_size_v<scoped_declarations>> table;
    std::array<std::uint32_t, set_count>        next_binding {};
    std::array<bool, set_count>                 scope_has_update_after_bind {};
    std::array<bool, set_count>                 scope_has_dynamic {};
    std::array<std::array<bool, 64>, set_count> occupied {};

    auto assign = [ & ]<std::size_t Index>() {
      using scoped      = std::tuple_element_t<Index, scoped_declarations>;
      using declaration = typename scoped::declaration;
      constexpr resource_scope      declared_scope {scoped::scope_value};
      constexpr std::uint32_t       set {slot_of(declared_scope)};
      constexpr detail::kind_meta_t meta {detail::kind_meta(declaration::kind)};

      if constexpr (declaration::kind == resource_declaration_kind::cis_table)
      {
        static_assert(
          declared_scope == resource_scope::bindless,
          "resource_layout: cis_table_resource only in bindless");
        static_assert(
          declaration::variable_count_value.has_value(),
          "resource_layout: cis_table_resource needs variable_count");
      }

      std::uint32_t binding {};
      if constexpr (declaration::pinned_binding.has_value())
      {
        binding = *declaration::pinned_binding;
      }
      else
      {
        binding = next_binding[ set ];
        while (binding < 64U && occupied[ set ][ binding ]) { ++binding; }
      }
      if (binding >= 64U || occupied[ set ][ binding ])
      {
        detail::layout_error(
          "resource_layout: pin collision or binding exhausted in scope");
        return;
      }
      occupied[ set ][ binding ] = true;
      if constexpr (meta.update_after_bind)
      {
        scope_has_update_after_bind[ set ] = true;
      }
      if constexpr (meta.dynamic) { scope_has_dynamic[ set ] = true; }
      if (scope_has_update_after_bind[ set ] && scope_has_dynamic[ set ])
      {
        detail::layout_error(
          "resource_layout: UpdateAfterBind and dynamic in same scope");
      }

      table[ Index ] = binding_assignment {
        .tag_index       = static_cast<std::uint32_t>(Index),
        .scope           = declared_scope,
        .set             = set,
        .binding         = binding,
        .descriptor_type = meta.descriptor_type,
        .stage_flags     = declaration::stage_flags,
        .count           = declaration::variable_count_value.value_or(
          declaration::array_count_value.value_or(1U)),
        .dynamic           = meta.dynamic,
        .update_after_bind = meta.update_after_bind,
        .binding_flags     = meta.binding_flags
      };
      next_binding[ set ] = std::max(next_binding[ set ], binding + 1U);
    };

    [ & ]<std::size_t... Indices>(std::index_sequence<Indices...>) {
      (assign.template operator ()<Indices>(), ...);
    }(std::make_index_sequence<std::tuple_size_v<scoped_declarations>> {});
    return table;
  }

  template<class Tag>
  static consteval
  auto
  resource_index_of() -> std::size_t
  {
    constexpr std::size_t position =
      detail::tag_position<Tag, scoped_declarations>();
    static_assert(
      position < std::tuple_size_v<scoped_declarations>,
      "resource_layout: tag not declared");
    return position;
  }

  template<class Tag>
  static constexpr binding_assignment assignment_of {
    assignment_table()[ resource_index_of<Tag>() ],
  };

  template<class Tag>
  using declaration_of = typename std::
    tuple_element_t<resource_index_of<Tag>(), scoped_declarations>::declaration;

  template<class Tag>
  static consteval
  auto
  binding_of() -> classic_binding_id
  {
    return classic_binding_id {
      .set     = assignment_of<Tag>.set,
      .binding = assignment_of<Tag>.binding,
    };
  }

  template<class Tag>
  static consteval
  auto
  dynamic_offset_index() -> std::uint32_t
  {
    constexpr binding_assignment target {assignment_of<Tag>};
    static_assert(target.dynamic, "resource_layout: tag is not dynamic");
    constexpr auto table = assignment_table();
    return static_cast<std::uint32_t>(std::ranges::count_if(
      table, [ &target ](const binding_assignment& assignment) -> bool {
        return assignment.dynamic &&
               (assignment.set < target.set ||
                 (assignment.set == target.set &&
                   assignment.binding < target.binding));
      }));
  }

  static consteval
  auto
  dynamic_offset_count() -> std::uint32_t
  {
    constexpr auto table = assignment_table();
    return static_cast<std::uint32_t>(std::ranges::count_if(
      table, [](const binding_assignment& assignment) -> bool {
        return assignment.dynamic;
      }));
  }
};

export template<class Layout>
concept declared_resource_layout = requires {
  { Layout::set_count } -> std::convertible_to<std::uint32_t>;
  Layout::assignment_table();
  typename Layout::scoped_declarations;
};

} // namespace vkpp
