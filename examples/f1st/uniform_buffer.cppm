module;

#include <cstddef> // I need just offsetof, so importing std would be an overkill

export module f1st.uniform_buffer;

import glm;

namespace f1st
{

// TODO: Konrad - Be explicit about alignment, create more alignment helpers
// inside the ::alignment namespace for all the GLM types and nested structures.
// See:
// https://docs.vulkan.org/spec/latest/chapters/interfaces.html#interfaces-resources-layout
namespace alignment
{
static constexpr auto scalar { 4U };
static constexpr auto vec2 { 8U };
static constexpr auto vec3 { 16U };
static constexpr auto vec4 { 16U };
static constexpr auto mat4 { 16U };
// ...
} // namespace alignment

export struct uniform_buffer_object
{
  alignas(alignment::mat4) glm::mat4 model;
  alignas(alignment::mat4) glm::mat4 view;
  alignas(alignment::mat4) glm::mat4 projection;
  alignas(alignment::vec3) glm::vec3 light_direction;
  alignas(alignment::vec3) glm::vec3 light_color;
  alignas(alignment::vec3) glm::vec3 camera_position;
  alignas(alignment::scalar) float exposure { 1.0F };
};
static_assert(offsetof(uniform_buffer_object, light_direction) == 192UZ);
static_assert(offsetof(uniform_buffer_object, light_color) == 208UZ);
static_assert(offsetof(uniform_buffer_object, camera_position) == 224UZ);
static_assert(offsetof(uniform_buffer_object, exposure) == 236UZ);
static_assert(sizeof(uniform_buffer_object) == 240UZ);

} // namespace f1st
