get_target_property(_vk_inc Vulkan::Headers INTERFACE_INCLUDE_DIRECTORIES)
set(_vulkan_cppm "${_vk_inc}/vulkan/vulkan.cppm")
set(_vulkan_video_cppm "${_vk_inc}/vulkan/vulkan_video.cppm")

add_library(vulkan_cxx_module STATIC)
target_sources(vulkan_cxx_module
  PUBLIC FILE_SET CXX_MODULES
    BASE_DIRS "${_vk_inc}"
    FILES "${_vulkan_cppm}" "${_vulkan_video_cppm}")

target_compile_features(vulkan_cxx_module PUBLIC cxx_std_26)
target_compile_definitions(vulkan_cxx_module
  PUBLIC
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=0
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
    VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS=1
    VULKAN_HPP_CXX_MODULE_EXPERIMENTAL_WARNING=" "
    VULKAN_HPP_USE_STD_EXPECTED=1
    VULKAN_HPP_NO_EXCEPTIONS=1)
target_link_libraries(vulkan_cxx_module PUBLIC Vulkan::Vulkan)
set_target_properties(vulkan_cxx_module PROPERTIES CXX_MODULE_STD ON)
vkpp_silence_third_party(vulkan_cxx_module)

add_library(Vulkan::cppm ALIAS vulkan_cxx_module)
