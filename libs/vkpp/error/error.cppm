export module vkpp.error;

import std;
import vulkan;

export namespace vkpp
{

struct vk_error
{
  vk::Result result {};
  std::string_view function {};
  std::string_view type {};

  auto
  message() const -> std::string
  {
    return std::format(
      "{}::{}() returned {}", type, function, vk::to_string(result));
  }
};

#if __cpp_impl_reflection

template<std::meta::info Fn>
inline constexpr std::string_view vk_fn_name =
  std::define_static_string(std::meta::identifier_of(Fn));

template<std::meta::info Fn>
inline constexpr std::string_view vk_type_name = std::define_static_string(
  std::meta::display_string_of(std::meta::parent_of(Fn)));

template<std::meta::info Fn, typename T>
auto
map_vk_error(std::expected<T, vk::Result>&& result)
  -> std::expected<T, vk_error>
{
  return std::move(result).transform_error(
    [](vk::Result error)
    {
      return vk_error {
        .result = error,
        .function = vk_fn_name<Fn>,
        .type = vk_type_name<Fn>,
      };
    });
}

#else

template<typename T>
auto
map_vk_error(std::expected<T, vk::Result>&& result)
  -> std::expected<T, vk_error>
{
  return std::move(result).transform_error(
    [](vk::Result error) { return vk_error { .result = error }; });
}

#endif

}; // namespace vkpp
