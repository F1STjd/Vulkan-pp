# The vkpp library (namespace vkpp). Successor to the old `utils` target.
#
# The full directory/file skeleton already exists (Phase 0.1). Module interface
# units are added to the CXX_MODULES FILE_SET below *as they become populated*
# (per the tutorial plan), so the project grows into its final shape instead of
# being reorganized midway. Modules for later phases are listed but commented
# out; uncomment each one when you start implementing it.

add_library(vkpp STATIC)

# Public include dir so headers placed under vkpp (e.g. a future
# error/vk_error_config.hpp holding the UTILS_VK macro) are visible to consumers.
target_include_directories(vkpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_sources(vkpp PRIVATE
  io/tiny_obj_loader_impl.cpp
  memory/vma_impl.cpp)

target_sources(vkpp
  PUBLIC FILE_SET CXX_MODULES
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    FILES
      error/error.cppm          # module vkpp.error
      io/channels.cppm          # module vkpp.io.channels
      io/load.cppm              # module vkpp.io
      io/shader.cppm            # module vkpp.io.shader
      io/types.cppm             # module vkpp.io.types
      io/mesh_tags.cppm
      io/mesh_gltf.cppm
      io/mesh_obj.cppm
      io/image_tags.cppm
      io/image_png.cppm
      io/image_ktx2.cppm

      mesh/vertex.cppm          # module vkpp.vertex

      memory/allocator.cppm # module vkpp.memory.allocator
      memory/raw_allocator.cppm # module vkpp.memory.raw
      memory/vma_allocator.cppm # module vkpp.memory.vma
      memory/virtual_block.cppm # module vkpp.memory.virtual_block

      resources/image.cppm    # module vkpp.image
      resources/buffer.cppm   # module vkpp.buffer
      resources/buffer_upload.cppm  # module vkpp.buffer.upload
      resources/buffer_arena.cppm # module vkpp.buffer.arena
      resources/buffer_arena_upload.cppm # module vkpp.buffer.arena.upload

      core/instance_context.cppm   # module vkpp.instance
      core/device_context.cppm     # module vkpp.device

      present/swapchain.cppm       # module vkpp.swapchain
      present/frame.cppm           # module vkpp.frame

      command/command.cppm         # module vkpp.command

      resources/texture.cppm  # module vkpp.texture
      resources/texture_upload.cppm  # module vkpp.texture.upload
      resources/sampler.cppm  # module vkpp.sampler

      sync/barrier.cppm      # module vkpp.barrier
      sync/semaphore.cppm    # module vkpp.semaphore

      graph/attachment_use.cppm  # module vkpp.graph.attachment_use

      descriptor/descriptor.cppm   # module vkpp.descriptor

      pipeline/pipeline.cppm       # module vkpp.pipeline
      pipeline/compute.cppm       # module vkpp.pipeline.compute
)

target_compile_features(vkpp PUBLIC cxx_std_26)
set_target_properties(vkpp PROPERTIES CXX_MODULE_STD ON)

target_link_libraries(vkpp
  PUBLIC glm::glm glm::cppm Tutorial::stb Vulkan::cppm fastgltf::fastgltf KTX::ktx
  PRIVATE Tutorial::tinyobjloader Vulkan::vma)

