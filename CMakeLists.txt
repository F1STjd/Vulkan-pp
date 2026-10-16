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
