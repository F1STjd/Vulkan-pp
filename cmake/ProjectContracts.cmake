include(CheckCXXCompilerFlag)

option(ENABLE_CXX_CONTRACTS "Enable C++26 contract annotations (-fcontracts)" ON)

if(NOT ENABLE_CXX_CONTRACTS)
  return()
endif()

set(_contract_candidates -fcontracts)

foreach(_flag IN LISTS _contract_candidates)
  string(MAKE_C_IDENTIFIER "${_flag}" _id)
  check_cxx_compiler_flag("${_flag}" HAS_${_id})
  if(HAS_${_id})
    add_compile_options("${_flag}")
  endif()
endforeach()

if(ENABLE_CXX_CONTRACTS AND NOT HAS__fcontracts)
  message(WARNING
    "ENABLE_CXX_CONTRACTS is ON but ${CMAKE_CXX_COMPILER_ID} does not support -fcontracts; "
    "contract annotations will not compile.")
endif()
