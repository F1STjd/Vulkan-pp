function(add_slang_shader_target TARGET)
  cmake_parse_arguments(SHADER "" "" "SOURCES" ${ARGN})

  if(NOT SHADER_SOURCES)
    message(FATAL_ERROR "add_slang_shader_target(${TARGET}): SOURCES is required")
  endif()

  set(SHADERS_DIR "${CMAKE_SOURCE_DIR}/shaders")
  set(SHADER_OUTPUT_DIRECTORY "${SHADERS_DIR}" PARENT_SCOPE)

  set(ENTRY_POINTS -entry vertex_main -entry fragment_main)

  add_custom_command(
    OUTPUT ${SHADERS_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
  )

  add_custom_command(
    OUTPUT ${SHADERS_DIR}/slang.spv
    COMMAND ${SLANGC_EXECUTABLE}
      ${SHADER_SOURCES}
      -target spirv
      -profile spirv_1_4
      -emit-spirv-directly
      -fvk-use-entrypoint-name
      ${ENTRY_POINTS}
      -o slang.spv
    WORKING_DIRECTORY ${SHADERS_DIR}
    DEPENDS ${SHADERS_DIR} ${SHADER_SOURCES}
    COMMENT "Compiling Slang shaders"
    VERBATIM
  )

  add_custom_target(${TARGET} DEPENDS ${SHADERS_DIR}/slang.spv)
endfunction()
