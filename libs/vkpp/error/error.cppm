export module vkpp.error;

import std;
import vulkan;

namespace vkpp
{

export enum class app_error_kind {
  file_open,
  image_decode,
  model_parse,
  missing_instance_extension,
  missing_validation_layer,
  surface_creation,
  no_suitable_gpu,
  no_graphics_present_queue,
  no_supported_format,
  no_memory_type,
};

constexpr auto
to_string(app_error_kind kind) -> std::string_view
{
#if __cpp_impl_reflection
  return std::define_static_string(std::meta::identifier_of(^^kind));

#else
  static constexpr std::array reflected_error_kind {
    "file_open",
    "image_decode",
    "model_parse",
    "missing_instance_extension",
    "missing_validation_layer",
    "surface_creation",
    "no_suitable_gpu",
    "no_graphics_present_queue",
    "no_supported_format",
    "no_memory_type",
  };

  return reflected_error_kind[ std::to_underlying(kind) ];
#endif
}

export struct app_error
{
  app_error_kind kind {};
  std::string_view detail;

  constexpr auto
  message() const -> std::string
  {
    return std::format(
      "Error of type: {}, was returned.\nDetailed message: {}.",
      to_string(kind), detail);
  }
};

export struct vk_error
{
  std::string_view function {};
  std::string_view type {};
  vk::Result result {};

  constexpr auto
  message() const -> std::string
  {
    return std::format(
      "{}::{}() returned {}", type, function, vk::to_string(result));
  }
};

export using error_t = std::variant<vk_error, app_error>;

// Todo: Konrad - later change to format_to/append, so it can coexist with one
// logging buffer
export constexpr auto
message(const error_t& error) -> std::string
{
  return std::visit(
    [](const auto& value) -> std::string { return value.message(); }, error);
}

#if __cpp_impl_reflection

template<std::meta::info Fn>
inline constexpr std::string_view vk_fn_name =
  std::define_static_string(std::meta::identifier_of(Fn));

template<std::meta::info Fn>
inline constexpr std::string_view vk_type_name = std::define_static_string(
  std::meta::display_string_of(std::meta::parent_of(Fn)));

export template<std::meta::info Fn, typename T>
constexpr auto
map_vk_error(std::expected<T, vk::Result>&& result) -> std::expected<T, error_t>
{
  return std::move(result).transform_error(
    [](vk::Result error)
    {
      return vk_error {
        .function = vk_fn_name<Fn>,
        .type = vk_type_name<Fn>,
        .result = error,
      };
    });
}

#else

export template<typename T>
constexpr auto
map_vk_error(std::expected<T, vk::Result>&& result) -> std::expected<T, error_t>
{
  return std::move(result).transform_error(
    [](vk::Result error) { return vk_error { .result = error }; });
}

#endif

} // namespace vkpp
