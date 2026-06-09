export module utils;

import std;
import vulkan;

export namespace utils
{

#if __cpp_impl_reflection

template<std::meta::info Fn>
consteval auto
vk_error_type_name()
{ return std::meta::display_string_of(std::meta::parent_of(Fn)); }

template<std::meta::info Fn>
consteval auto
vk_error_fn_name()
{ return std::meta::identifier_of(Fn); }

template<std::meta::info Fn, typename T>
auto
map_vk_error(std::expected<T, vk::Result>&& result)
  -> std::expected<T, std::string>
{
  constexpr auto type_name = vk_error_type_name<Fn>();
  constexpr auto fn_name = vk_error_fn_name<Fn>();
  return std::move(result).transform_error(
    [ type_name, fn_name ](vk::Result error)
    {
      return std::format("{}::{}() returned error: {}", type_name, fn_name,
        vk::to_string(error));
    });
}

#else

template<typename T>
auto
map_vk_error(std::expected<T, vk::Result>&& result)
  -> std::expected<T, std::string>
{
  return std::move(result).transform_error(
    [](vk::Result error) -> std::string { return vk::to_string(error); });
}

#endif

}; // namespace utils