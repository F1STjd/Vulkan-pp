get_target_property(_glm_inc glm::glm-header-only INTERFACE_INCLUDE_DIRECTORIES)
set(_glm_cppm "${_glm_inc}/glm/glm.cppm")

add_library(glm_cxx_module STATIC)
target_sources(glm_cxx_module
  PUBLIC FILE_SET CXX_MODULES
    BASE_DIRS "${_glm_inc}"
    FILES "${_glm_cppm}")

target_compile_features(glm_cxx_module PUBLIC cxx_std_26)
target_link_libraries(glm_cxx_module PUBLIC glm::glm-header-only)
set_target_properties(glm_cxx_module PROPERTIES CXX_MODULE_STD ON)

add_library(glm::cppm ALIAS glm_cxx_module)
