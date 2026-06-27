add_library(utils STATIC)
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# tinyobjloader implementation is a plain TU (no `import std;`); see the file.
target_sources(utils PRIVATE load/tiny_obj_loader_impl.cpp)
target_sources(utils
  PUBLIC FILE_SET CXX_MODULES
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    FILES
      load/channels.cppm
      load/load.cppm
      utils/utils.cppm
      vertex.cppm)
target_compile_features(utils PUBLIC cxx_std_26)
set_target_properties(utils PROPERTIES CXX_MODULE_STD ON)
target_link_libraries(utils
  PUBLIC glm::glm glm::cppm Tutorial::stb Vulkan::cppm
  PRIVATE Tutorial::tinyobjloader)

file(TO_CMAKE_PATH "${MODEL_OUTPUT_DIRECTORY}" _model_directory)
string(REPLACE "\\" "/" _model_directory "${_model_directory}")
file(TO_CMAKE_PATH "${TEXTURE_OUTPUT_DIRECTORY}" _texture_directory)
string(REPLACE "\\" "/" _texture_directory "${_texture_directory}")
target_compile_definitions(utils PRIVATE
  MODEL_DIRECTORY="${_model_directory}/"
  TEXTURE_DIRECTORY="${_texture_directory}/")
