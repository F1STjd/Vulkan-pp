# Build-time sync of tiny_obj_loader.h from the tinyobjloader release branch.
# Invoked via: cmake -P fetch_tinyobjloader.cmake

cmake_minimum_required(VERSION 3.19)

if(NOT TINYOBJLOADER_DIR)
  message(FATAL_ERROR "TINYOBJLOADER_DIR is not set")
endif()
if(NOT CMAKE_BINARY_DIR)
  message(FATAL_ERROR "CMAKE_BINARY_DIR is not set")
endif()

set(_header "${TINYOBJLOADER_DIR}/tiny_obj_loader.h")
set(_sha_file "${TINYOBJLOADER_DIR}/.release_sha")
set(_raw_url "https://raw.githubusercontent.com/tinyobjloader/tinyobjloader/release/tiny_obj_loader.h")
set(_api_url "https://api.github.com/repos/tinyobjloader/tinyobjloader/commits/release")

file(MAKE_DIRECTORY "${TINYOBJLOADER_DIR}")

set(_local_sha "")
if(EXISTS "${_sha_file}")
  file(READ "${_sha_file}" _local_sha)
  string(STRIP "${_local_sha}" _local_sha)
endif()

set(_api_response "${CMAKE_BINARY_DIR}/_tinyobjloader_api.json")
file(DOWNLOAD
  "${_api_url}"
  "${_api_response}"
  STATUS _api_status
  TLS_VERIFY ON)

list(GET _api_status 0 _api_code)
if(NOT _api_code EQUAL 0)
  if(EXISTS "${_header}")
    message(WARNING
      "Could not check tinyobjloader release branch (HTTP ${_api_code}); using existing header.")
    return()
  endif()
  list(GET _api_status 1 _api_msg)
  message(FATAL_ERROR
    "Could not fetch tinyobjloader release info (HTTP ${_api_code}: ${_api_msg}) "
    "and no local tiny_obj_loader.h exists.")
endif()

file(READ "${_api_response}" _api_json)
string(JSON _remote_sha GET "${_api_json}" sha)

if(_remote_sha STREQUAL _local_sha AND EXISTS "${_header}")
  message(STATUS "tinyobjloader is up to date (${_remote_sha})")
  return()
endif()

message(STATUS "Updating tinyobjloader from release branch (${_remote_sha})...")

set(_header_tmp "${CMAKE_BINARY_DIR}/_tinyobjloader.h.tmp")
file(DOWNLOAD
  "${_raw_url}"
  "${_header_tmp}"
  STATUS _hdr_status
  TLS_VERIFY ON)

list(GET _hdr_status 0 _hdr_code)
if(NOT _hdr_code EQUAL 0)
  list(GET _hdr_status 1 _hdr_msg)
  message(FATAL_ERROR "Failed to download tiny_obj_loader.h: ${_hdr_msg}")
endif()

file(RENAME "${_header_tmp}" "${_header}")
file(WRITE "${_sha_file}" "${_remote_sha}\n")
message(STATUS "tinyobjloader updated to ${_remote_sha}")
