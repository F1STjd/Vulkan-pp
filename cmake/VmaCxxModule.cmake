# Vulkan Memory Allocator for vkpp. Provides the `Vulkan::vma` target.
#
# ============================================================================
# WHY THE DEFAULT IS THE C API, NOT `import vk_mem_alloc;`
# ============================================================================
# The VMA-Hpp wrapper advertises `import vk_mem_alloc;`. It was evaluated end to
# end against THIS project's toolchain (GCC 16.1.0) and config, and the module
# path is currently a dead end here for two independent reasons:
#
#   1. GCC's P1689 module scanner rejects the wrapper's private module fragment
#      (`module : private;`, an MSVC-visibility workaround). Fixable: strip that
#      block + drive GCC with -fmodules instead of CMake's default -fmodules-ts.
#      (Both are automated in the opt-in module branch below.)
#
#   2. THE HARD BLOCKER: this project builds Vulkan-Hpp with
#      VULKAN_HPP_USE_STD_EXPECTED=1 (foundational to vkpp's vk_error /
#      map_vk_error design). VMA-Hpp 3.3.0's RAII layer does not support that
#      flag - it is absent from the wrapper's tested-flags matrix - and its
#      `raii::detail::Converter` tries to construct
#      `std::expected<vma::raii::Allocator, vk::Result>` by forwarding the raii
#      handle's 4 constructor arguments, which std::expected has no ctor for:
#          error: no matching function for call to
#          'std::expected<vma::raii::Allocator, vk::Result>::expected(
#             const vk::raii::Device&, std::expected<vma::Allocator, vk::Result>,
#             const vk::AllocationCallbacks*&, const ...DeviceDispatcher*)'
#      Dropping USE_STD_EXPECTED would fix VMA but break vkpp's error model, so
#      it is not an option. Revisit `import vk_mem_alloc;` if/when upstream
#      supports USE_STD_EXPECTED (track the YaaZ/VulkanMemoryAllocator-Hpp repo).
#
# RECOMMENDED PATH (default): consume the VMA *C* API from MSYS2
# (mingw-w64-x86_64-vulkan-memory-allocator, .../include/vma/vk_mem_alloc.h) and
# hand-roll a thin gpu_image/gpu_buffer RAII wrapper inside the vkpp.memory:vma
# partition - include <vk_mem_alloc.h> in that unit's global module fragment,
# exactly like stb/tinyobj are consumed today. No third-party module, no macro
# conflict, no network, and full control over the RAII shape. The MSYS2 C VMA is
# built against the same MSYS2 Vulkan headers this project uses (VK_HEADER_VERSION
# 350), so there is no version skew.
# ============================================================================

option(VKPP_VMA_USE_MODULE
  "Opt in to the VMA-Hpp C++ module (import vk_mem_alloc). Currently blocked by \
VMA-Hpp 3.3.0's RAII layer vs VULKAN_HPP_USE_STD_EXPECTED; kept for the future." OFF)

# --- The VMA C header is needed in BOTH modes ---------------------------------
find_path(VMA_C_INCLUDE_DIR
  NAMES vk_mem_alloc.h
  PATH_SUFFIXES vma
  HINTS
    "$ENV{VULKAN_SDK}/include"
    "D:/msys2/mingw64/include"
  DOC "Directory containing VMA's vk_mem_alloc.h (MSYS2: .../include/vma)")

if(NOT VMA_C_INCLUDE_DIR)
  message(FATAL_ERROR
    "VMA C header (vk_mem_alloc.h) not found. Install the MSYS2 package "
    "mingw-w64-x86_64-vulkan-memory-allocator, or set VMA_C_INCLUDE_DIR.")
endif()

# ============================================================================
# DEFAULT: C-API interface target (no fetch, no third-party compilation)
# ============================================================================
if(NOT VKPP_VMA_USE_MODULE)
  add_library(vma_iface INTERFACE)
  target_include_directories(vma_iface SYSTEM INTERFACE "${VMA_C_INCLUDE_DIR}")
  # Vulkan::cppm gives the `vulkan` module, the matching Vulkan-Hpp macros, and
  # the Vulkan loader (Vulkan::Vulkan) that VMA's static function pointers call.
  target_link_libraries(vma_iface INTERFACE Vulkan::cppm)
  add_library(Vulkan::vma ALIAS vma_iface)
  return()
endif()

# ============================================================================
# OPT-IN: VMA-Hpp module (import vk_mem_alloc) - see blocker #2 above
# ============================================================================
include(FetchContent)
FetchContent_Declare(vma_hpp
  GIT_REPOSITORY https://github.com/YaaZ/VulkanMemoryAllocator-Hpp.git
  GIT_TAG        v3.3.0+3                  # upstream 3.3.0 tags use +N suffixes
  GIT_SUBMODULES ""                        # do not pull the bundled C VMA
  GIT_SHALLOW    TRUE
  # A source subdir that does not exist makes FetchContent_MakeAvailable populate
  # the sources WITHOUT calling add_subdirectory() (we only want include/*).
  SOURCE_SUBDIR  _do_not_configure_vma_hpp)
FetchContent_MakeAvailable(vma_hpp)
set(_vma_hpp_include "${vma_hpp_SOURCE_DIR}/include")

# Strip the private module fragment (blocker #1) inline after populate, so the
# path is known and FetchContent's step stamps are undisturbed. Idempotent.
set(_vma_cppm "${_vma_hpp_include}/vk_mem_alloc.cppm")
file(READ "${_vma_cppm}" _vma_src)
string(FIND "${_vma_src}" "\nmodule : private;" _vma_priv_idx)
if(_vma_priv_idx GREATER -1)
  string(SUBSTRING "${_vma_src}" 0 ${_vma_priv_idx} _vma_src)
  file(WRITE "${_vma_cppm}" "${_vma_src}\n")
  message(STATUS "VMA-Hpp: stripped private module fragment for GCC scanner.")
endif()
unset(_vma_src)

# Drive GCC with -fmodules (CMake hardcodes the legacy -fmodules-ts, under which
# GCC also rejects the private fragment). These are ordinary variables; rewriting
# them here (include() shares the caller scope) affects every module target.
string(REPLACE "-fmodules-ts" "-fmodules"
  CMAKE_CXX_SCANDEP_SOURCE "${CMAKE_CXX_SCANDEP_SOURCE}")
string(REPLACE "-fmodules-ts" "-fmodules"
  CMAKE_CXX_MODULE_MAP_FLAG "${CMAKE_CXX_MODULE_MAP_FLAG}")

add_library(vma_cxx_module STATIC)
target_sources(vma_cxx_module
  PUBLIC FILE_SET CXX_MODULES
    BASE_DIRS "${_vma_hpp_include}"
    FILES "${_vma_cppm}")

target_include_directories(vma_cxx_module SYSTEM PUBLIC
  "${_vma_hpp_include}"   # vk_mem_alloc.hpp
  "${VMA_C_INCLUDE_DIR}") # vk_mem_alloc.h (MSYS2)

target_compile_features(vma_cxx_module PUBLIC cxx_std_26)
set_target_properties(vma_cxx_module PROPERTIES CXX_MODULE_STD ON)
target_link_libraries(vma_cxx_module PUBLIC Vulkan::cppm)

add_library(Vulkan::vma ALIAS vma_cxx_module)
