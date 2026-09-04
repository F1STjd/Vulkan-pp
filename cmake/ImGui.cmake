include_guard(GLOBAL)
include(FetchContent)

FetchContent_Declare(imgui_upstream
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.92.9-docking
  GIT_SHALLOW TRUE
  GIT_SUBMODULES ""
  SOURCE_SUBDIR _do_not_configure_imgui
)
FetchContent_MakeAvailable(imgui_upstream)

add_library(imgui STATIC
  "${imgui_upstream_SOURCE_DIR}/imgui.cpp"
  "${imgui_upstream_SOURCE_DIR}/imgui_draw.cpp"
  "${imgui_upstream_SOURCE_DIR}/imgui_tables.cpp"
  "${imgui_upstream_SOURCE_DIR}/imgui_widgets.cpp"
  "${imgui_upstream_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp"
)
target_include_directories(imgui PUBLIC "${imgui_upstream_SOURCE_DIR}")
target_compile_definitions(imgui PUBLIC
  "IMGUI_IMPL_VULKAN_VULKAN_HAS_DYNAMIC_RENDERING"
)
target_link_libraries(imgui PUBLIC Vulkan::Vulkan)