#pragma once

#if defined(__cpp_impl_reflection) && __cpp_impl_reflection
#define UTILS_VK(expr, fn) ::vkpp::map_vk_error<fn>(expr)
#else
#define UTILS_VK(expr, fn) ::vkpp::map_vk_error(expr)
#endif
