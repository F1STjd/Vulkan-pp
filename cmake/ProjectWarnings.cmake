include(CheckCXXCompilerFlag)

option(ENABLE_WEFFCPP "Enable -Weffc++ (known false positives)" OFF)
option(ENABLE_GCC_ANALYZER "Enable -fanalyzer (slow)" OFF)
option(WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

set(_warn_candidates
  -Wpedantic
  -pedantic-errors
  -Wall
  -Wextra
  -Wshadow
  -Wconversion
  -Wold-style-cast
  -Wcast-qual
  -Wcast-align
  -Wsign-promo
  -Wdouble-promotion
  -Wnull-dereference
  -Wzero-as-null-pointer-constant
  -Wuninitialized
  -Wstrict-overflow=4
  -Wlogical-op
  -Wuseless-cast
  -Wduplicated-branches
  -Wduplicated-cond
  -Wmismatched-tags
  -Wextra-semi
  -Wsuggest-override
  -Wsuggest-final-types
  -Wsuggest-final-methods
  -Wnoexcept
  -Wctor-dtor-privacy
  -Wstrict-null-sentinel
  -Wredundant-tags
  -Wnrvo
  -Walloc-zero
  -Warith-conversion
  -Wtrampolines
  -Wunsafe-loop-optimizations
  -Wmissing-noreturn
  -Wno-pedantic-ms-format
  # Extra static analysis (probed; skipped if unsupported)
  -Wformat=2
  -Wmaybe-uninitialized
  -Wmultistatement-macros
  -Wnon-virtual-dtor
  -Wdelete-non-virtual-dtor
  -Wpessimizing-move
  -Wrange-loop-construct
  -Wshift-overflow=2
  -Wuse-after-free=3
  -Wno-free-nonheap-object # GCC false positive (-Wfree-nonheap-object) on std::expected chains
  -Wstringop-overflow=4
  -Warray-bounds=2
  -Wvla
  -Wcatch-value=3
  -Wfloat-equal
  -Woverloaded-virtual
  -Wplacement-new=2
  -Wundef)

if(ENABLE_WEFFCPP)
  list(APPEND _warn_candidates -Weffc++)
endif()
if(ENABLE_GCC_ANALYZER)
  list(APPEND _warn_candidates
    -fanalyzer
    -Wno-error=analyzer-symbol-too-complex
    -Wno-error=analyzer-too-complex
    -Wno-error=analyzer-exposure-through-uninit-copy
    -fdiagnostics-path-format=separate-events)
endif()
if(WARNINGS_AS_ERRORS)
  list(APPEND _warn_candidates -Werror)
endif()

add_library(project_warnings INTERFACE)
foreach(_flag IN LISTS _warn_candidates)
  string(MAKE_C_IDENTIFIER "${_flag}" _id)
  check_cxx_compiler_flag("${_flag}" HAS_${_id})
  if(HAS_${_id})
    target_compile_options(project_warnings INTERFACE "${_flag}")
  endif()
endforeach()
