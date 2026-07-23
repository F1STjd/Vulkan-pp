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
static constexpr auto mat4 { 16U };
} // namespace alignment

export struct uniform_buffer_object
{
  alignas(alignment::mat4) glm::mat4 model;
  alignas(alignment::mat4) glm::mat4 view;
  alignas(alignment::mat4) glm::mat4 projection;
};

} // namespace f1st
