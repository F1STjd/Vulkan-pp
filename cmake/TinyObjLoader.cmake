# Fetches tiny_obj_loader.h from the tinyobjloader release branch at build time.
# https://github.com/tinyobjloader/tinyobjloader

set(TINYOBJLOADER_DIR "${CMAKE_SOURCE_DIR}/third-party/tinyobjloader")
set(TINYOBJLOADER_HEADER "${TINYOBJLOADER_DIR}/tiny_obj_loader.h")

file(MAKE_DIRECTORY "${TINYOBJLOADER_DIR}")

add_custom_target(fetch_tinyobjloader ALL
  COMMAND ${CMAKE_COMMAND}
    -DTINYOBJLOADER_DIR=${TINYOBJLOADER_DIR}
    -DCMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}
    -P ${CMAKE_CURRENT_LIST_DIR}/scripts/fetch_tinyobjloader.cmake
  BYPRODUCTS
    "${TINYOBJLOADER_HEADER}"
    "${TINYOBJLOADER_DIR}/.release_sha"
  COMMENT "Checking tinyobjloader release branch for updates"
  VERBATIM)

add_library(tinyobjloader INTERFACE)
add_library(Tutorial::tinyobjloader ALIAS tinyobjloader)
target_include_directories(tinyobjloader SYSTEM INTERFACE "${TINYOBJLOADER_DIR}")
add_dependencies(tinyobjloader fetch_tinyobjloader)
