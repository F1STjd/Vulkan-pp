cmake_minimum_required(VERSION 4.0)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "451f2fe2-a8a2-47c3-bc32-94786d8fc91b")

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)
set(CMAKE_CXX_MODULE_STD ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(tutorial-final LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(ProjectWarnings)
include(ProjectReflection)
include(ProjectDiagnostics)
include(ProjectContracts)

find_package(Vulkan REQUIRED)
find_package(SFML 3 REQUIRED COMPONENTS Window Graphics System)
if(NOT TARGET Vulkan::Headers)
  find_package(VulkanHeaders REQUIRED)
endif()
include(VulkanCxxModule)
include(TutorialDependencies)
include(SlangShaders)

set(SHADER_SLANG_SOURCES ${CMAKE_SOURCE_DIR}/shaders/shader.slang)
add_slang_shader_target(slang_shaders SOURCES ${SHADER_SLANG_SOURCES})

add_subdirectory(libs/utils)
add_subdirectory(libs/lbn)
add_subdirectory(src)

# Root compile_commands.json is only useful for Clang IDE builds.
# GCC builds produce .gcm files that clangd cannot consume.
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  add_custom_target(copy_compile_commands ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_BINARY_DIR}/compile_commands.json"
      "${CMAKE_SOURCE_DIR}/compile_commands.json")
endif()
