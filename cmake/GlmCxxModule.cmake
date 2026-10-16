get_target_property(_glm_inc glm::glm-header-only INTERFACE_INCLUDE_DIRECTORIES)

file(READ "${_glm_inc}/glm/glm.cppm" _glm_src)
string(REPLACE "#include \"./glm.hpp\"" "#include <glm/glm.hpp>" _glm_src "${_glm_src}")
string(REPLACE "#include \"./ext.hpp\"" "#include <glm/ext.hpp>" _glm_src "${_glm_src}")
set(_glm_cppm "${CMAKE_CURRENT_BINARY_DIR}/glm_module/glm.cppm")
set(_glm_old "")
if(EXISTS "${_glm_cppm}")
  file(READ "${_glm_cppm}" _glm_old)
endif()
if(NOT _glm_old STREQUAL _glm_src)
  file(WRITE "${_glm_cppm}" "${_glm_src}")
endif()

add_library(glm_cxx_module STATIC)
target_sources(glm_cxx_module
  PUBLIC FILE_SET CXX_MODULES
    BASE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/glm_module"
    FILES "${_glm_cppm}")

target_include_directories(glm_cxx_module SYSTEM PUBLIC "${_glm_inc}")

target_compile_definitions(glm_cxx_module
  PUBLIC
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    GLM_ENABLE_EXPERIMENTAL)
target_compile_features(glm_cxx_module PUBLIC cxx_std_26)
target_link_libraries(glm_cxx_module PUBLIC glm::glm-header-only)
set_target_properties(glm_cxx_module PROPERTIES CXX_MODULE_STD ON)
vkpp_silence_third_party(glm_cxx_module)

add_library(glm::cppm ALIAS glm_cxx_module)
