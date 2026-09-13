module;

#include "error/vk_error_config.hpp"

export module vkpp.pipeline.shader_object;

import std;
import vulkan;

import vkpp.error;

namespace vkpp
{
using namespace std::string_view_literals;

export struct shader_stage_object_create_info
{
  vk::ShaderStageFlagBits stage { vk::ShaderStageFlagBits::eVertex };
  vk::ShaderStageFlags next_stage {};
  std::span<const std::byte> spirv {};
  const char* entry { "main" };
  std::span<const vk::DescriptorSetLayout> set_layouts {};
  std::span<const vk::PushConstantRange> push_constant_ranges {};
};

export class shader_stage_object
{
public:
  shader_stage_object() = default;

  explicit shader_stage_object(vk::raii::ShaderEXT&& shader)
  : shader_ { std::move(shader) }
  {}

  [[nodiscard]] static auto
  create(const vk::raii::Device& device,
    const shader_stage_object_create_info& create_info)
    -> std::expected<shader_stage_object, error_t>
  {
    if (create_info.spirv.empty())
    {
      return std::unexpected {
        app_error {
          .kind = app_error_kind::invalid_argument,
          .detail = "shader_stage_object spirv empty"sv,
        },
      };
    }
    const vk::ShaderCreateInfoEXT info {
      .stage = create_info.stage,
      .nextStage = create_info.next_stage,
      .codeType = vk::ShaderCodeTypeEXT::eSpirv,
      .codeSize = create_info.spirv.size_bytes(),
      .pCode = create_info.spirv.data(),
      .pName = create_info.entry,
      .setLayoutCount =
        static_cast<std::uint32_t>(create_info.set_layouts.size()),
      .pSetLayouts = create_info.set_layouts.data(),
      .pushConstantRangeCount =
        static_cast<std::uint32_t>(create_info.push_constant_ranges.size()),
      .pPushConstantRanges = create_info.push_constant_ranges.data(),
    };
    return UTILS_VK(
      device.createShaderEXT(info), ^^vk::raii::Device::createShaderEXT)
      .transform([](vk::raii::ShaderEXT&& shader) -> shader_stage_object
        { return shader_stage_object { std::move(shader) }; });
  }

  [[nodiscard]] auto
  get(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.shader_); }

  [[nodiscard]] auto
  handle() const -> vk::ShaderEXT
  { return *shader_; }

private:
  vk::raii::ShaderEXT shader_ { nullptr };
};

} // namespace vkpp
