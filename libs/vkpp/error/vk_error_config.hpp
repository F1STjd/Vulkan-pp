#pragma once

#if __cpp_impl_reflection // NOLINT
#define UTILS_VK(expr, fn) ::vkpp::map_vk_error<fn>(expr)
#else
#define UTILS_VK(expr, fn) ::vkpp::map_vk_error(expr)
#endif
