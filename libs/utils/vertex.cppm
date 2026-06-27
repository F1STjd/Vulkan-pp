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
  glm::vec3 position;
  glm::vec3 color;
  glm::vec2 texture_coordinates;

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
    -> std::array<vk::VertexInputAttributeDescription, 3>
  {
    return std::array {
      vk::VertexInputAttributeDescription {
        .location = 0,
        .binding = 0,
        .format = vk::Format::eR32G32B32Sfloat,
        .offset = offsetof(vertex, position),
      },
      vk::VertexInputAttributeDescription {
        .location = 1,
        .binding = 0,
        .format = vk::Format::eR32G32B32Sfloat,
        .offset = offsetof(vertex, color),
      },
      vk::VertexInputAttributeDescription {
        .location = 2,
        .binding = 0,
        .format = vk::Format::eR32G32Sfloat,
        .offset = offsetof(vertex, texture_coordinates),
      },
    };
  }

  auto
  operator==(const vertex& rhs) const -> bool
  {
    return position == rhs.position && color == rhs.color &&
      texture_coordinates == rhs.texture_coordinates;
  }
};
} // namespace lbn

namespace std
{
template<>
struct hash<lbn::vertex>
{
  size_t
  operator()(const lbn::vertex& vertex) const
  {
    return ((hash<glm::vec3>()(vertex.position) ^
              (hash<glm::vec3>()(vertex.color) << 1)) >>
             1) ^
      (hash<glm::vec2>()(vertex.texture_coordinates) << 1);
  }
};
} // namespace std
