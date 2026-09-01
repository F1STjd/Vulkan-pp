add_executable(f1st main.cpp)

target_sources(f1st
  PUBLIC FILE_SET CXX_MODULES
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    FILES
      app.cppm
      uniform_buffer.cppm
)

target_sources(f1st PRIVATE
  imgui_platform.cpp
  imgui_platform.hpp
)

target_compile_features(f1st PRIVATE cxx_std_26)
set_target_properties(f1st PROPERTIES
  CXX_MODULE_STD ON
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

target_link_libraries(f1st PRIVATE
  vkpp SFML::Window glm::cppm Vulkan::cppm imgui project_warnings)
target_link_libraries(f1st PRIVATE
  $<$<PLATFORM_ID:Windows>:stdc++exp>)

add_dependencies(f1st slang_shaders)

# Contracts: the renderer uses -fcontracts (GNU only), same as the old `lbn`
# target. contracts_config.hpp / contracts_handler.cpp live beside the app.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_sources(f1st PRIVATE contracts_handler.cpp)
endif()

# Shader/texture/model directories for constructing asset paths at call sites
# (SHADER_DIRECTORY was on the old `lbn` target; MODEL/TEXTURE mirror `utils`).
file(TO_CMAKE_PATH "${SHADER_OUTPUT_DIRECTORY}" _shader_directory)
string(REPLACE "\\" "/" _shader_directory "${_shader_directory}")
file(TO_CMAKE_PATH "${TEXTURE_OUTPUT_DIRECTORY}" _texture_directory)
string(REPLACE "\\" "/" _texture_directory "${_texture_directory}")
file(TO_CMAKE_PATH "${MODEL_OUTPUT_DIRECTORY}" _model_directory)
string(REPLACE "\\" "/" _model_directory "${_model_directory}")
target_compile_definitions(f1st PRIVATE
  SHADER_DIRECTORY="${_shader_directory}/"
  TEXTURE_DIRECTORY="${_texture_directory}/"
  MODEL_DIRECTORY="${_model_directory}/")

