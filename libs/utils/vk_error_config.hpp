#pragma once

#if __cpp_impl_reflection
#define UTILS_VK(expr, fn) ::utils::map_vk_error<fn>(expr)
#else
#define UTILS_VK(expr, fn) ::utils::map_vk_error(expr)
#endif