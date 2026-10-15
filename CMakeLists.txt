cmake_minimum_required(VERSION 4.0)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")

project(vkpp LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ProjectWarnings)
# include(ProjectReflection) # reflection is not used for now
include(ProjectDiagnostics)

find_package(Vulkan REQUIRED)
find_package(fastgltf CONFIG REQUIRED)
find_package(Ktx CONFIG REQUIRED)
if(NOT TARGET Vulkan::Headers)
  find_package(VulkanHeaders REQUIRED)
endif()
include(TutorialDependencies)
include(TinyObjLoader)
include(VulkanCxxModule)
include(VmaCxxModule)
include(GlmCxxModule)

add_subdirectory(vkpp)

# Root compile_commands.json is only useful for Clang IDE builds.
# GCC builds produce .gcm files that clangd cannot consume.
if(PROJECT_IS_TOP_LEVEL AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  add_custom_target(copy_compile_commands ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_BINARY_DIR}/compile_commands.json"
      "${CMAKE_SOURCE_DIR}/compile_commands.json")
endif()
