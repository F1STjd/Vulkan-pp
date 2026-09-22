module;

#include <vulkan/vulkan_core.h>

export module vkpp.error;

import std;
import vulkan;

namespace vkpp
{

export enum class error_domain : std::uint8_t {
  application,
  vulkan,
  vma,
  fastgltf,
  ktx,
};

export enum class app_error_code : std::int32_t {
  file_open,
  file_read,
  file_write,
  image_decode,
  unsupported_image_format,
  model_parse,
  unsupported_model_data,
  missing_instance_extension,
  missing_validation_layer,
  surface_creation,
  surface_not_presentable,
  no_suitable_gpu,
  no_graphics_present_queue,
  no_supported_format,
  no_memory_type,
  mapping_failed,
  capacity_exhausted,
  missing_required_argument,
  invalid_count,
  out_of_range,
  invalid_data_size,
  invalid_state,
  unsupported_operation,
  feature_not_supported,
  feature_not_enabled,
  corrupt_persistent_data,
};

export struct error_t
{
  error_domain domain {};
  std::int32_t code {};
};

static_assert(std::is_trivially_copyable_v<error_t>);
static_assert(sizeof(error_t) <= 8UZ);

export [[nodiscard]] constexpr auto
make_app_error(app_error_code code) -> error_t
{
  return error_t {
    .domain = error_domain::application,
    .code = std::to_underlying(code),
  };
}

export [[nodiscard]] constexpr auto
make_vk_error(vk::Result result) -> error_t
{
  return error_t {
    .domain = error_domain::vulkan,
    .code = static_cast<std::int32_t>(result),
  };
}

export [[nodiscard]] constexpr auto
make_vma_error(VkResult result) -> error_t
{
  return error_t {
    .domain = error_domain::vma,
    .code = static_cast<std::int32_t>(result),
  };
}

[[nodiscard]] constexpr auto
app_error_message(app_error_code code) -> std::string_view
{
  switch (code)
  {
  case app_error_code::file_open   : return "file_open";
  case app_error_code::file_read   : return "file_read";
  case app_error_code::file_write  : return "file_write";
  case app_error_code::image_decode: return "image_decode";
  case app_error_code::unsupported_image_format:
    return "unsupported_image_format";
  case app_error_code::model_parse           : return "model_parse";
  case app_error_code::unsupported_model_data: return "unsupported_model_data";
  case app_error_code::missing_instance_extension:
    return "missing_instance_extension";
  case app_error_code::missing_validation_layer:
    return "missing_validation_layer";
  case app_error_code::surface_creation: return "surface_creation";
  case app_error_code::surface_not_presentable:
    return "surface_not_presentable";
  case app_error_code::no_suitable_gpu: return "no_suitable_gpu";
  case app_error_code::no_graphics_present_queue:
    return "no_graphics_present_queue";
  case app_error_code::no_supported_format: return "no_supported_format";
  case app_error_code::no_memory_type     : return "no_memory_type";
  case app_error_code::mapping_failed     : return "mapping_failed";
  case app_error_code::capacity_exhausted : return "capacity_exhausted";
  case app_error_code::missing_required_argument:
    return "missing_required_argument";
  case app_error_code::invalid_count        : return "invalid_count";
  case app_error_code::out_of_range         : return "out_of_range";
  case app_error_code::invalid_data_size    : return "invalid_data_size";
  case app_error_code::invalid_state        : return "invalid_state";
  case app_error_code::unsupported_operation: return "unsupported_operation";
  case app_error_code::feature_not_supported: return "feature_not_supported";
  case app_error_code::feature_not_enabled  : return "feature_not_enabled";
  case app_error_code::corrupt_persistent_data:
    return "corrupt_persistent_data";
  }
  return "unknown_app_error_code";
}

export [[nodiscard]] auto
message(error_t error) -> std::string
{
  switch (error.domain)
  {

  case error_domain::application:
    return std::format(
      "app:{}", app_error_message(static_cast<app_error_code>(error.code)));
  case error_domain::vulkan:
    return std::format(
      "vulkan:{}", vk::to_string(static_cast<vk::Result>(error.code)));
  case error_domain::vma:
    return std::format(
      "vma:{}", vk::to_string(static_cast<vk::Result>(error.code)));
  case error_domain::fastgltf: return std::format("fastgltf:{}", error.code);
  case error_domain::ktx     : return std::format("ktx:{}", error.code);
  }
  return std::format("unknown_domain:{}", error.code);
}

export template<typename T>
constexpr auto
map_vk_error(std::expected<T, vk::Result>&& result) -> std::expected<T, error_t>
{
  return std::move(result).transform_error(
    [](vk::Result result) -> error_t { return make_vk_error(result); });
}

#if defined(__cpp_impl_reflection) && __cpp_impl_reflection
export template<std::meta::info Fn, typename T>
constexpr auto
map_vk_error(std::expected<T, vk::Result>&& result) -> std::expected<T, error_t>
{ return map_vk_error(std::move(result)); }
#endif

} // namespace vkpp
