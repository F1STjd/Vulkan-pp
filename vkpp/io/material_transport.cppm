module;

#include <cstddef>

export module vkpp.io.material_transport;

import std;

import vkpp.error;
import vkpp.diagnostics;
import vkpp.io.types;

namespace vkpp::gltf
{

export struct material_transport_spec
{
  bool transmission {false};
  bool clearcoat {false};
};

export template<material_transport_spec Spec>
struct material_transport_traits;

inline constexpr material_transport_spec material_transport_core_spec {
  .transmission = false,
  .clearcoat    = false,
};

inline constexpr material_transport_spec material_transport_transmission_spec {
  .transmission = true,
  .clearcoat    = false,
};

inline constexpr material_transport_spec material_transport_clearcoat_spec {
  .transmission = false,
  .clearcoat    = true,
};

inline constexpr material_transport_spec
  material_transport_transmission_clearcoat_spec {
    .transmission = true,
    .clearcoat    = true,
};

export struct alignas(16) material_record_core
{
  std::array<float, 4> base_color_factor {1.0F, 1.0F, 1.0F, 1.0F};
  std::array<float, 4> emissive_factor_and_metallic {0.0F, 0.0F, 0.0F, 1.0F};
  float                roughness_factor {1.0F};
  float                normal_scale {1.0F};
  float                occlusion_strength {1.0F};
  float                alpha_cutoff {0.5F};
  std::uint32_t        base_color_slot {0U};
  std::uint32_t        metallic_roughness_slot {0U};
  std::uint32_t        normal_slot {0U};
  std::uint32_t        occlusion_slot {0U};
  std::uint32_t        emissive_slot {0U};
  std::uint32_t        alpha_mode {0U};
  std::uint32_t        texture_presence_mask {0U};
  std::uint32_t        pad0 {0U};
};
static_assert(sizeof(material_record_core) == 80UZ);
static_assert(alignof(material_record_core) == 16UZ);
static_assert(std::is_trivially_copyable_v<material_record_core>);
static_assert(offsetof(material_record_core, base_color_factor) == 0UZ);
static_assert(
  offsetof(material_record_core, emissive_factor_and_metallic) == 16UZ);
static_assert(offsetof(material_record_core, roughness_factor) == 32UZ);
static_assert(offsetof(material_record_core, normal_scale) == 36UZ);
static_assert(offsetof(material_record_core, occlusion_strength) == 40UZ);
static_assert(offsetof(material_record_core, alpha_cutoff) == 44UZ);
static_assert(offsetof(material_record_core, base_color_slot) == 48UZ);
static_assert(offsetof(material_record_core, metallic_roughness_slot) == 52UZ);
static_assert(offsetof(material_record_core, normal_slot) == 56UZ);
static_assert(offsetof(material_record_core, occlusion_slot) == 60UZ);
static_assert(offsetof(material_record_core, emissive_slot) == 64UZ);
static_assert(offsetof(material_record_core, alpha_mode) == 68UZ);
static_assert(offsetof(material_record_core, texture_presence_mask) == 72UZ);
static_assert(offsetof(material_record_core, pad0) == 76UZ);

export struct alignas(16) material_record_transmission
{
  std::array<float, 4> base_color_factor {1.0F, 1.0F, 1.0F, 1.0F};
  std::array<float, 4> emissive_factor_and_metallic {0.0F, 0.0F, 0.0F, 1.0F};
  float                roughness_factor {1.0F};
  float                normal_scale {1.0F};
  float                occlusion_strength {1.0F};
  float                alpha_cutoff {0.5F};
  std::uint32_t        base_color_slot {0U};
  std::uint32_t        metallic_roughness_slot {0U};
  std::uint32_t        normal_slot {0U};
  std::uint32_t        occlusion_slot {0U};
  std::uint32_t        emissive_slot {0U};
  std::uint32_t        alpha_mode {0U};
  std::uint32_t        texture_presence_mask {0U};
  float                transmission_factor {0.0F};
  std::uint32_t        transmission_slot {0U};
  std::uint32_t        pad0 {0U};
  std::uint32_t        pad1 {0U};
  std::uint32_t        pad2 {0U};
};
static_assert(sizeof(material_record_transmission) == 96UZ);
static_assert(alignof(material_record_transmission) == 16UZ);
static_assert(std::is_trivially_copyable_v<material_record_transmission>);
static_assert(
  offsetof(material_record_transmission, texture_presence_mask) == 72UZ);
static_assert(
  offsetof(material_record_transmission, transmission_factor) == 76UZ);
static_assert(offsetof(material_record_transmission, transmission_slot) == 80UZ);

export struct alignas(16) material_record_clearcoat
{
  std::array<float, 4> base_color_factor {1.0F, 1.0F, 1.0F, 1.0F};
  std::array<float, 4> emissive_factor_and_metallic {0.0F, 0.0F, 0.0F, 1.0F};
  float                roughness_factor {1.0F};
  float                normal_scale {1.0F};
  float                occlusion_strength {1.0F};
  float                alpha_cutoff {0.5F};
  std::uint32_t        base_color_slot {0U};
  std::uint32_t        metallic_roughness_slot {0U};
  std::uint32_t        normal_slot {0U};
  std::uint32_t        occlusion_slot {0U};
  std::uint32_t        emissive_slot {0U};
  std::uint32_t        alpha_mode {0U};
  std::uint32_t        texture_presence_mask {0U};
  float                clearcoat_factor {0.0F};
  float                clearcoat_roughness_factor {0.0F};
  float                clearcoat_normal_scale {1.0F};
  std::uint32_t        clearcoat_slot {0U};
  std::uint32_t        clearcoat_roughness_slot {0U};
  std::uint32_t        clearcoat_normal_slot {0U};
  std::uint32_t        pad0 {0U};
  std::uint32_t        pad1 {0U};
  std::uint32_t        pad2 {0U};
};
static_assert(sizeof(material_record_clearcoat) == 112UZ);
static_assert(alignof(material_record_clearcoat) == 16UZ);
static_assert(std::is_trivially_copyable_v<material_record_clearcoat>);
static_assert(
  offsetof(material_record_clearcoat, texture_presence_mask) == 72UZ);
static_assert(offsetof(material_record_clearcoat, clearcoat_factor) == 76UZ);
static_assert(
  offsetof(material_record_clearcoat, clearcoat_roughness_factor) == 80UZ);
static_assert(
  offsetof(material_record_clearcoat, clearcoat_normal_scale) == 84UZ);
static_assert(offsetof(material_record_clearcoat, clearcoat_slot) == 88UZ);
static_assert(
  offsetof(material_record_clearcoat, clearcoat_roughness_slot) == 92UZ);
static_assert(
  offsetof(material_record_clearcoat, clearcoat_normal_slot) == 96UZ);

export struct alignas(16) material_record_transmission_clearcoat
{
  std::array<float, 4> base_color_factor {1.0F, 1.0F, 1.0F, 1.0F};
  std::array<float, 4> emissive_factor_and_metallic {0.0F, 0.0F, 0.0F, 1.0F};
  float                roughness_factor {1.0F};
  float                normal_scale {1.0F};
  float                occlusion_strength {1.0F};
  float                alpha_cutoff {0.5F};
  std::uint32_t        base_color_slot {0U};
  std::uint32_t        metallic_roughness_slot {0U};
  std::uint32_t        normal_slot {0U};
  std::uint32_t        occlusion_slot {0U};
  std::uint32_t        emissive_slot {0U};
  std::uint32_t        alpha_mode {0U};
  std::uint32_t        texture_presence_mask {0U};
  float                transmission_factor {0.0F};
  std::uint32_t        transmission_slot {0U};
  float                clearcoat_factor {0.0F};
  float                clearcoat_roughness_factor {0.0F};
  float                clearcoat_normal_scale {1.0F};
  std::uint32_t        clearcoat_slot {0U};
  std::uint32_t        clearcoat_roughness_slot {0U};
  std::uint32_t        clearcoat_normal_slot {0U};
  std::uint32_t        pad0 {0U};
};
static_assert(sizeof(material_record_transmission_clearcoat) == 112UZ);
static_assert(alignof(material_record_transmission_clearcoat) == 16UZ);
static_assert(
  std::is_trivially_copyable_v<material_record_transmission_clearcoat>);
static_assert(
  offsetof(material_record_transmission_clearcoat, texture_presence_mask) ==
  72UZ);
static_assert(
  offsetof(material_record_transmission_clearcoat, transmission_factor) == 76UZ);
static_assert(
  offsetof(material_record_transmission_clearcoat, transmission_slot) == 80UZ);
static_assert(
  offsetof(material_record_transmission_clearcoat, clearcoat_factor) == 84UZ);
static_assert(
  offsetof(material_record_transmission_clearcoat, clearcoat_roughness_factor) ==
  88UZ);
static_assert(
  offsetof(material_record_transmission_clearcoat, clearcoat_normal_scale) ==
  92UZ);
static_assert(
  offsetof(material_record_transmission_clearcoat, clearcoat_slot) == 96UZ);
static_assert(
  offsetof(material_record_transmission_clearcoat, clearcoat_roughness_slot) ==
  100UZ);
static_assert(
  offsetof(material_record_transmission_clearcoat, clearcoat_normal_slot) ==
  104UZ);
static_assert(offsetof(material_record_transmission_clearcoat, pad0) == 108UZ);

template<>
struct material_transport_traits<material_transport_core_spec>
{
  using record_type = material_record_core;
  static constexpr std::size_t record_size {sizeof(record_type)};
  static constexpr std::size_t offset_base_color_factor {0UZ};
  static constexpr std::size_t offset_emissive_factor_and_metallic {16UZ};
  static constexpr std::size_t offset_roughness_factor {32UZ};
  static constexpr std::size_t offset_normal_scale {36UZ};
  static constexpr std::size_t offset_occlusion_strength {40UZ};
  static constexpr std::size_t offset_alpha_cutoff {44UZ};
  static constexpr std::size_t offset_base_color_slot {48UZ};
  static constexpr std::size_t offset_metallic_roughness_slot {52UZ};
  static constexpr std::size_t offset_normal_slot {56UZ};
  static constexpr std::size_t offset_occlusion_slot {60UZ};
  static constexpr std::size_t offset_emissive_slot {64UZ};
  static constexpr std::size_t offset_alpha_mode {68UZ};
  static constexpr std::size_t offset_texture_presence_mask {72UZ};
};

template<>
struct material_transport_traits<material_transport_transmission_spec>
{
  using record_type = material_record_transmission;
  static constexpr std::size_t record_size {sizeof(record_type)};
  static constexpr std::size_t offset_base_color_factor {0UZ};
  static constexpr std::size_t offset_emissive_factor_and_metallic {16UZ};
  static constexpr std::size_t offset_roughness_factor {32UZ};
  static constexpr std::size_t offset_normal_scale {36UZ};
  static constexpr std::size_t offset_occlusion_strength {40UZ};
  static constexpr std::size_t offset_alpha_cutoff {44UZ};
  static constexpr std::size_t offset_base_color_slot {48UZ};
  static constexpr std::size_t offset_metallic_roughness_slot {52UZ};
  static constexpr std::size_t offset_normal_slot {56UZ};
  static constexpr std::size_t offset_occlusion_slot {60UZ};
  static constexpr std::size_t offset_emissive_slot {64UZ};
  static constexpr std::size_t offset_alpha_mode {68UZ};
  static constexpr std::size_t offset_texture_presence_mask {72UZ};
  static constexpr std::size_t offset_transmission_factor {76UZ};
  static constexpr std::size_t offset_transmission_slot {80UZ};
};

template<>
struct material_transport_traits<material_transport_clearcoat_spec>
{
  using record_type                        = material_record_clearcoat;
  static constexpr std::size_t record_size = sizeof(record_type);
  static constexpr std::size_t offset_base_color_factor            = 0UZ;
  static constexpr std::size_t offset_emissive_factor_and_metallic = 16UZ;
  static constexpr std::size_t offset_roughness_factor             = 32UZ;
  static constexpr std::size_t offset_normal_scale                 = 36UZ;
  static constexpr std::size_t offset_occlusion_strength           = 40UZ;
  static constexpr std::size_t offset_alpha_cutoff                 = 44UZ;
  static constexpr std::size_t offset_base_color_slot              = 48UZ;
  static constexpr std::size_t offset_metallic_roughness_slot      = 52UZ;
  static constexpr std::size_t offset_normal_slot                  = 56UZ;
  static constexpr std::size_t offset_occlusion_slot               = 60UZ;
  static constexpr std::size_t offset_emissive_slot                = 64UZ;
  static constexpr std::size_t offset_alpha_mode                   = 68UZ;
  static constexpr std::size_t offset_texture_presence_mask        = 72UZ;
  static constexpr std::size_t offset_clearcoat_factor             = 76UZ;
  static constexpr std::size_t offset_clearcoat_roughness_factor   = 80UZ;
  static constexpr std::size_t offset_clearcoat_normal_scale       = 84UZ;
  static constexpr std::size_t offset_clearcoat_slot               = 88UZ;
  static constexpr std::size_t offset_clearcoat_roughness_slot     = 92UZ;
  static constexpr std::size_t offset_clearcoat_normal_slot        = 96UZ;
};

template<>
struct material_transport_traits<material_transport_transmission_clearcoat_spec>
{
  using record_type = material_record_transmission_clearcoat;
  static constexpr std::size_t record_size {sizeof(record_type)};
  static constexpr std::size_t offset_base_color_factor {0UZ};
  static constexpr std::size_t offset_emissive_factor_and_metallic {16UZ};
  static constexpr std::size_t offset_roughness_factor {32UZ};
  static constexpr std::size_t offset_normal_scale {36UZ};
  static constexpr std::size_t offset_occlusion_strength {40UZ};
  static constexpr std::size_t offset_alpha_cutoff {44UZ};
  static constexpr std::size_t offset_base_color_slot {48UZ};
  static constexpr std::size_t offset_metallic_roughness_slot {52UZ};
  static constexpr std::size_t offset_normal_slot {56UZ};
  static constexpr std::size_t offset_occlusion_slot {60UZ};
  static constexpr std::size_t offset_emissive_slot {64UZ};
  static constexpr std::size_t offset_alpha_mode {68UZ};
  static constexpr std::size_t offset_texture_presence_mask {72UZ};
  static constexpr std::size_t offset_transmission_factor {76UZ};
  static constexpr std::size_t offset_transmission_slot {80UZ};
  static constexpr std::size_t offset_clearcoat_factor {84UZ};
  static constexpr std::size_t offset_clearcoat_roughness_factor {88UZ};
  static constexpr std::size_t offset_clearcoat_normal_scale {92UZ};
  static constexpr std::size_t offset_clearcoat_slot {96UZ};
  static constexpr std::size_t offset_clearcoat_roughness_slot {100UZ};
  static constexpr std::size_t offset_clearcoat_normal_slot {104UZ};
};

export template<material_transport_spec Spec>
struct material_transport_cpu
{
  std::vector<typename material_transport_traits<Spec>::record_type> records {};
  std::uint32_t default_material_index {};
};

template<class DiagnosticsPolicy>
[[nodiscard]]
auto
resolve_texture_use_slot(
  const std::optional<texture_use_cpu>& use,
  std::span<const texture_role_slots>   texture_slots,
  std::uint32_t                         mask_bit,
  std::uint32_t&                        mask,
  std::optional<diagnostic_buffer&>     diagnostics,
  std::source_location location = std::source_location::current())
  -> std::expected<std::uint32_t, error_t>
{
  if (!use.has_value()) { return 0U; }
  if (use->texture_index >= texture_slots.size())
  {
    if constexpr (DiagnosticsPolicy::enabled)
    {
      if (diagnostics)
      {
        diagnostics->report(
          diagnostic_severity::error,
          make_app_error(app_error_code::out_of_range),
          location,
          "material texture_index {} out of range (slots={})",
          use->texture_index,
          texture_slots.size());
      }
    }
    return std::unexpected {make_app_error(app_error_code::out_of_range)};
  }
  if (use->texcoord_index != 0U)
  {
    if constexpr (DiagnosticsPolicy::enabled)
    {
      if (diagnostics)
      {
        diagnostics->report(
          diagnostic_severity::error,
          make_app_error(app_error_code::unsupported_operation),
          location,
          "TEXCOORD_{} unsupported; transport consumes TEXCOORD_0 only",
          use->texcoord_index);
      }
    }
    return std::unexpected {
      make_app_error(app_error_code::unsupported_operation),
    };
  }
  const auto& slots = texture_slots[ use->texture_index ];
  const auto& slot =
    use->color_space == image_color_space::srgb ? slots.srgb : slots.linear;
  if (!slot.has_value())
  {
    if constexpr (DiagnosticsPolicy::enabled)
    {
      if (diagnostics)
      {
        diagnostics->report(
          diagnostic_severity::error,
          make_app_error(app_error_code::invalid_state),
          location,
          "texture {} role {} not registered",
          use->texture_index,
          static_cast<unsigned>(use->color_space));
      }
    }
    return std::unexpected {make_app_error(app_error_code::invalid_state)};
  }
  mask |= (1U << mask_bit);
  return *slot;
}

template<material_transport_spec Spec, class DiagnosticsPolicy>
[[nodiscard]]
auto
extension_bags_supported(
  const material_cpu&               material,
  std::optional<diagnostic_buffer&> diagnostics,
  std::source_location              location = std::source_location::current())
  -> std::expected<void, error_t>
{
  const auto reject =
    [ & ](bool bad, std::string_view what) -> std::expected<void, error_t> {
    if (!bad) { return {}; }
    if constexpr (DiagnosticsPolicy::enabled)
    {
      if (diagnostics)
      {
        diagnostics->report(
          diagnostic_severity::error,
          make_app_error(app_error_code::unsupported_operation),
          location,
          "material transport spec disables {}",
          what);
      }
    }
    return std::unexpected {
      make_app_error(app_error_code::unsupported_operation),
    };
  };

  if constexpr (!Spec.transmission)
  {
    if (
      auto result = reject(
        material.transmission.factor != 0.0F ||
          material.transmission.texture.has_value(),
        "transmission");
      !result)
    {
      return result;
    }
  }
  if constexpr (!Spec.clearcoat)
  {
    const bool clearcoat_used =
      material.clearcoat.factor != 0.0F ||
      material.clearcoat.roughness_factor != 0.0F ||
      material.clearcoat.texture.has_value() ||
      material.clearcoat.roughness_texture.has_value() ||
      material.clearcoat.normal_texture.has_value() ||
      material.clearcoat.normal_scale != 1.0F;
    if (auto result = reject(clearcoat_used, "clearcoat"); !result)
    {
      return result;
    }
  }
  return {};
}

template<typename Record, class DiagnosticsPolicy>
[[nodiscard]]
auto
fill_common_fields(
  Record&                             record,
  const material_cpu&                 material,
  std::span<const texture_role_slots> texture_slots,
  std::optional<diagnostic_buffer&> diagnostics) -> std::expected<void, error_t>
{
  record.base_color_factor            = material.base_color_factor;
  record.emissive_factor_and_metallic = {
    material.emissive_factor[ 0 ],
    material.emissive_factor[ 1 ],
    material.emissive_factor[ 2 ],
    material.metallic_factor,
  };
  record.roughness_factor   = material.roughness_factor;
  record.normal_scale       = material.normal_scale;
  record.occlusion_strength = material.occlusion_strength;
  record.alpha_cutoff       = material.alpha_cutoff;
  record.alpha_mode         = static_cast<std::uint32_t>(material.alpha_mode);
  record.texture_presence_mask = 0U;

  auto base_color_slot = resolve_texture_use_slot<DiagnosticsPolicy>(
    material.base_color_texture,
    texture_slots,
    0U,
    record.texture_presence_mask,
    diagnostics);
  if (!base_color_slot) { return std::unexpected {base_color_slot.error()}; }
  record.base_color_slot = *base_color_slot;

  auto metallic_roughness_slot = resolve_texture_use_slot<DiagnosticsPolicy>(
    material.metallic_roughness_texture,
    texture_slots,
    1U,
    record.texture_presence_mask,
    diagnostics);
  if (!metallic_roughness_slot)
  {
    return std::unexpected {metallic_roughness_slot.error()};
  }
  record.metallic_roughness_slot = *metallic_roughness_slot;

  auto normal_slot = resolve_texture_use_slot<DiagnosticsPolicy>(
    material.normal_texture,
    texture_slots,
    2U,
    record.texture_presence_mask,
    diagnostics);
  if (!normal_slot) { return std::unexpected {normal_slot.error()}; }
  record.normal_slot = *normal_slot;

  auto occlusion_slot = resolve_texture_use_slot<DiagnosticsPolicy>(
    material.occlusion_texture,
    texture_slots,
    3U,
    record.texture_presence_mask,
    diagnostics);
  if (!occlusion_slot) { return std::unexpected {occlusion_slot.error()}; }
  record.occlusion_slot = *occlusion_slot;

  auto emissive_slot = resolve_texture_use_slot<DiagnosticsPolicy>(
    material.emissive_texture,
    texture_slots,
    4U,
    record.texture_presence_mask,
    diagnostics);
  if (!emissive_slot) { return std::unexpected {emissive_slot.error()}; }
  record.emissive_slot = *emissive_slot;
  return {};
}

template<material_transport_spec Spec, class DiagnosticsPolicy>
[[nodiscard]]
auto
pack_material_records_impl(
  std::span<const material_cpu>       materials,
  std::span<const texture_role_slots> texture_slots,
  std::optional<diagnostic_buffer&>   diagnostics)
  -> std::expected<material_transport_cpu<Spec>, error_t>
{
  using traits      = material_transport_traits<Spec>;
  using record_type = typename traits::record_type;

  material_transport_cpu<Spec> out {};
  out.records.reserve(materials.size() + 1UZ);

  for (const auto& material : materials)
  {
    if (
      auto supported = extension_bags_supported<Spec, DiagnosticsPolicy>(
        material, diagnostics);
      !supported)
    {
      return std::unexpected {supported.error()};
    }

    record_type record {};
    if (
      auto filled = fill_common_fields<record_type, DiagnosticsPolicy>(
        record, material, texture_slots, diagnostics);
      !filled)
    {
      return std::unexpected {filled.error()};
    }

    if constexpr (Spec.transmission)
    {
      record.transmission_factor = material.transmission.factor;
      auto transmission          = resolve_texture_use_slot<DiagnosticsPolicy>(
        material.transmission.texture,
        texture_slots,
        5U,
        record.texture_presence_mask,
        diagnostics);
      if (!transmission) { return std::unexpected {transmission.error()}; }
      record.transmission_slot = *transmission;
    }
    if constexpr (Spec.clearcoat)
    {
      record.clearcoat_factor           = material.clearcoat.factor;
      record.clearcoat_roughness_factor = material.clearcoat.roughness_factor;
      record.clearcoat_normal_scale     = material.clearcoat.normal_scale;
      auto clearcoat = resolve_texture_use_slot<DiagnosticsPolicy>(
        material.clearcoat.texture,
        texture_slots,
        6U,
        record.texture_presence_mask,
        diagnostics);
      if (!clearcoat) { return std::unexpected {clearcoat.error()}; }
      record.clearcoat_slot    = *clearcoat;
      auto clearcoat_roughness = resolve_texture_use_slot<DiagnosticsPolicy>(
        material.clearcoat.roughness_texture,
        texture_slots,
        7U,
        record.texture_presence_mask,
        diagnostics);
      if (!clearcoat_roughness)
      {
        return std::unexpected {clearcoat_roughness.error()};
      }
      record.clearcoat_roughness_slot = *clearcoat_roughness;
      auto clearcoat_normal = resolve_texture_use_slot<DiagnosticsPolicy>(
        material.clearcoat.normal_texture,
        texture_slots,
        8U,
        record.texture_presence_mask,
        diagnostics);
      if (!clearcoat_normal)
      {
        return std::unexpected {clearcoat_normal.error()};
      }
      record.clearcoat_normal_slot = *clearcoat_normal;
    }

    out.records.push_back(record);
  }

  out.records.push_back(record_type {});
  out.default_material_index = static_cast<std::uint32_t>(materials.size());
  return out;
}

export
template<material_transport_spec Spec, class DiagnosticsPolicy>
[[nodiscard]]
auto
pack_material_records(
  std::span<const material_cpu>       materials,
  std::span<const texture_role_slots> texture_slots,
  diagnostic_buffer&                  diagnostics)
  -> std::expected<material_transport_cpu<Spec>, error_t>
  requires (DiagnosticsPolicy::enabled)
{
  return pack_material_records_impl<Spec, DiagnosticsPolicy>(
    materials, texture_slots, diagnostics);
}

export
template<material_transport_spec Spec, class DiagnosticsPolicy>
[[nodiscard]]
auto
pack_material_records(
  std::span<const material_cpu>       materials,
  std::span<const texture_role_slots> texture_slots)
  -> std::expected<material_transport_cpu<Spec>, error_t>
  requires (!DiagnosticsPolicy::enabled)
{
  return pack_material_records_impl<Spec, DiagnosticsPolicy>(
    materials, texture_slots, std::nullopt);
}

} // namespace vkpp::gltf
