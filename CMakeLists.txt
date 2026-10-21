cmake_minimum_required(VERSION 4.0)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")

project(vkpp LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ProjectWarnings)
# include(ProjectReflection) # reflection is not used for now
if(PROJECT_IS_TOP_LEVEL)
  include(ProjectDiagnostics)
endif()

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

option(
  VKPP_WITH_GPU
  "Build vkpp::gpu (fiat kernels; needs spirv-as / spirv-val)"
  OFF)
if(VKPP_WITH_GPU)
  include(FetchContent)
  if(
    NOT DEFINED FETCHCONTENT_SOURCE_DIR_FIAT
    AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../fiat/CMakeLists.txt")
    set(FETCHCONTENT_SOURCE_DIR_FIAT "${CMAKE_CURRENT_SOURCE_DIR}/../fiat")
  endif()
  FetchContent_Declare(
    fiat
    GIT_REPOSITORY git@github.com:F1STjd/fiat.git
    GIT_TAG        ce83a4e76c65abd671240a7f6b819ff9c43049da)
  set(FIAT_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(fiat)

  add_library(vkpp_gpu STATIC)
  add_library(vkpp::gpu ALIAS vkpp_gpu)
  target_sources(vkpp_gpu
    PUBLIC FILE_SET CXX_MODULES
      BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/vkpp
      FILES vkpp/gpu/gpu.cppm)
  target_compile_features(vkpp_gpu PUBLIC cxx_std_26)
  set_target_properties(vkpp_gpu PROPERTIES CXX_MODULE_STD ON)
  target_link_libraries(vkpp_gpu PUBLIC vkpp::vkpp fiat::fiat)
  vkpp_enable_warnings(vkpp_gpu)
endif()
