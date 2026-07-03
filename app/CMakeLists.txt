# The f1st application (namespace f1st). Successor to the old `src` executable
# (`tutorial-final`) plus the renderer that used to live in the `lbn` target.
#
# app/app.cppm (module f1st.app) is the renderer monolith ported from the old
# lbn.cppm; app/uniform_buffer.cppm (module f1st.uniform_buffer) and the
# contracts files also moved here. Phase 1 splits the monolith into vkpp
# subsystems. main.cpp must define main() and construct f1st::app.

add_executable(f1st main.cpp)

target_sources(f1st
  PUBLIC FILE_SET CXX_MODULES
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    FILES
      app.cppm             # module f1st.app        (the renderer wiring; Phase 1)
      uniform_buffer.cppm) # module f1st.uniform_buffer (UBO layout matches shader)

target_compile_features(f1st PRIVATE cxx_std_26)
set_target_properties(f1st PROPERTIES
  CXX_MODULE_STD ON
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

target_link_libraries(f1st PRIVATE
  vkpp SFML::Window glm::cppm Vulkan::cppm project_warnings)
target_link_libraries(f1st PRIVATE
  $<$<PLATFORM_ID:Windows>:stdc++exp>)

add_dependencies(f1st slang_shaders)

# Contracts: the renderer uses -fcontracts (GNU only), same as the old `lbn`
# target. contracts_config.hpp / contracts_handler.cpp now live in app/.
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
