add_library(utils STATIC)
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_sources(utils
  PUBLIC FILE_SET CXX_MODULES
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    FILES
      load/load.cppm
      utils/utils.cppm)
target_compile_features(utils PUBLIC cxx_std_26)
set_target_properties(utils PROPERTIES CXX_MODULE_STD ON)
target_link_libraries(utils PUBLIC glm::glm Tutorial::stb Vulkan::cppm)
