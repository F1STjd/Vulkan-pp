module;

#include <cstddef>

export module vertex;

import glm;
import std;
import vulkan;

namespace lbn
{
export struct vertex
{
  glm::vec2 position;
  glm::vec3 color;

  static consteval auto
  get_binding_description() -> vk::VertexInputBindingDescription
  {
    return vk::VertexInputBindingDescription {
      .binding = 0,
      .stride = sizeof(vertex),
      .inputRate = vk::VertexInputRate::eVertex,
    };
  }

  static consteval auto
  get_attribute_descriptions()
    -> std::array<vk::VertexInputAttributeDescription, 2>
  {
    return std::array {
      vk::VertexInputAttributeDescription {
        .location = 0,
        .binding = 0,
        .format = vk::Format::eR32G32Sfloat,
        .offset = offsetof(vertex, position),
      },
      vk::VertexInputAttributeDescription {
        .location = 1,
        .binding = 0,
        .format = vk::Format::eR32G32B32Sfloat,
        .offset = offsetof(vertex, color),
      },
    };
  }
};

export constexpr std::array vertices {
  vertex {
    .position = { 0.0F, -1.0F },
    .color = { 1.0F, 0.0F, 0.0F },
  },
  vertex {
    .position = { 1.0F, 1.0F },
    .color = { 0.0F, 1.0F, 0.0F },
  },
  vertex {
    .position = { -1.0F, 1.0F },
    .color = { 0.0F, 0.0F, 1.0F },
  },
};
} // namespace lbn