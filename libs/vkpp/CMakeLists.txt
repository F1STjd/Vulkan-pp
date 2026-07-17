# The vkpp library (namespace vkpp). Successor to the old `utils` target.
#
# The full directory/file skeleton already exists (Phase 0.1). Module interface
# units are added to the CXX_MODULES FILE_SET below *as they become populated*
# (per the tutorial plan), so the project grows into its final shape instead of
# being reorganized midway. Modules for later phases are listed but commented
# out; uncomment each one when you start implementing it.

add_library(vkpp STATIC)

# Public include dir so headers placed under libs/vkpp (e.g. a future
# error/vk_error_config.hpp holding the UTILS_VK macro) are visible to consumers.
target_include_directories(vkpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# tinyobjloader implementation is a plain TU (no `import std;`); see the file.
target_sources(vkpp PRIVATE io/tiny_obj_loader_impl.cpp memory/vma_impl.cpp)

target_sources(vkpp
  PUBLIC FILE_SET CXX_MODULES
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    FILES
      # --- Foundation: error + io + mesh (migrated from old `utils`) ---
      error/error.cppm          # module vkpp.error
      io/channels.cppm          # module vkpp.io.channels
      io/load.cppm              # module vkpp.io
      mesh/vertex.cppm          # module vkpp.vertex

      # --- Phase 0.4 Allocator policy (see plan 0.4.0-0.4.4 for full specs) ---
      # Three SIBLING modules (vkpp.memory, vkpp.memory.raw, vkpp.memory.vma),
      # NOT partitions: partitions cannot import their own primary interface,
      # but raw/vma need the concept from vkpp.memory. Same precedent as
      # vkpp.io / vkpp.io.channels.
      # vma_allocator includes the VMA *C* header (<vk_mem_alloc.h>) in its
      # global module fragment and hand-rolls RAII; add `memory/vma_impl.cpp`
      # (the single VMA_IMPLEMENTATION TU) to the PRIVATE sources above and
      # `Vulkan::vma` to target_link_libraries below when implementing it.
      # `import vk_mem_alloc;` is blocked here by VMA-Hpp's RAII layer vs
      # VULKAN_HPP_USE_STD_EXPECTED - see cmake/VmaCxxModule.cmake.
      memory/allocator.cppm
      memory/raw_allocator.cppm
      memory/vma_allocator.cppm

      # --- Phase 0.5 / 0.6 Composition primitives + factory ---
      resources/image.cppm    # module vkpp.image
      resources/buffer.cppm   # module vkpp.buffer

      # --- Phase 1 Class decomposition ---
      core/instance_context.cppm   # module vkpp.instance
      core/device_context.cppm     # module vkpp.device
      present/swapchain.cppm       # module vkpp.swapchain
      present/frame.cppm           # module vkpp.frame
      command/command.cppm         # module vkpp.command

      # --- Phase 2 Textures ---
      # resources/texture.cppm  # module vkpp.texture

      # --- Phase 3 Descriptors ---
      # descriptor/descriptor.cppm   # module vkpp.descriptor

      # --- Phase 4 Pipelines ---
      # pipeline/pipeline.cppm       # module vkpp.pipeline
)

target_compile_features(vkpp PUBLIC cxx_std_26)
set_target_properties(vkpp PROPERTIES CXX_MODULE_STD ON)

# Surface creation from an SFML window currently lives in the app layer, so vkpp
# deliberately has no SFML dependency. Phase 1 must decide whether the app keeps
# passing a vk::SurfaceKHR handle into vkpp (preferred) or vkpp gains SFML here.
target_link_libraries(vkpp
  PUBLIC glm::glm glm::cppm Tutorial::stb Vulkan::cppm
  PRIVATE Tutorial::tinyobjloader Vulkan::vma)

# io/load.cppm builds model/texture paths from these (same as the old `utils`).
file(TO_CMAKE_PATH "${MODEL_OUTPUT_DIRECTORY}" _model_directory)
string(REPLACE "\\" "/" _model_directory "${_model_directory}")
file(TO_CMAKE_PATH "${TEXTURE_OUTPUT_DIRECTORY}" _texture_directory)
string(REPLACE "\\" "/" _texture_directory "${_texture_directory}")
target_compile_definitions(vkpp PRIVATE
  MODEL_DIRECTORY="${_model_directory}/"
  TEXTURE_DIRECTORY="${_texture_directory}/")
