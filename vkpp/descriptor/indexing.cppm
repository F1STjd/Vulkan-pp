module;

#include "error/vk_error_config.hpp"

export module vkpp.descriptor.indexing;

import std;
import vulkan;

import vkpp.error;

namespace vkpp
{
using namespace std::string_view_literals;

export struct bindless_table_create_info
{
  std::uint32_t capacity { 1024U };
  vk::ShaderStageFlags stages { vk::ShaderStageFlagBits::eFragment };
};

export class bindless_table
{
public:
  bindless_table() = default;

  bindless_table(vk::raii::DescriptorSetLayout&& layout,
    vk::raii::DescriptorPool&& pool, vk::DescriptorSet set,
    std::uint32_t capacity)
  : layout_ { std::move(layout) }, pool_ { std::move(pool) }, set_ { set },
    capacity_ { capacity }
  {}

  [[nodiscard]] static auto
  create(const vk::raii::Device& device,
    const bindless_table_create_info& create_info)
    -> std::expected<bindless_table, error_t>
  {
    const vk::DescriptorSetLayoutBinding binding {
      .binding = 0U,
      .descriptorType = vk::DescriptorType::eCombinedImageSampler,
      .descriptorCount = create_info.capacity,
      .stageFlags = create_info.stages,
    };
    const vk::DescriptorBindingFlags binding_flags {
      vk::DescriptorBindingFlagBits::ePartiallyBound |
      vk::DescriptorBindingFlagBits::eUpdateAfterBind |
      vk::DescriptorBindingFlagBits::eUpdateUnusedWhilePending |
      vk::DescriptorBindingFlagBits::eVariableDescriptorCount
    };
    const vk::StructureChain layout_chain {
      vk::DescriptorSetLayoutCreateInfo {
        .flags = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool,
        .bindingCount = 1U,
        .pBindings = &binding,
      },
      vk::DescriptorSetLayoutBindingFlagsCreateInfo {
        .bindingCount = 1U,
        .pBindingFlags = &binding_flags,
      },
    };

    return UTILS_VK(device.createDescriptorSetLayout(
                      layout_chain.get<vk::DescriptorSetLayoutCreateInfo>()),
      ^^vk::raii::Device::createDescriptorSetLayout)
      .and_then(
        [ & ](vk::raii::DescriptorSetLayout&& layout)
          -> std::expected<bindless_table, error_t>
        {
          const vk::DescriptorPoolSize pool_size {
            .type = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = create_info.capacity,
          };
          const vk::DescriptorPoolCreateInfo pool_info {
            .flags = vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind,
            .maxSets = 1U,
            .poolSizeCount = 1U,
            .pPoolSizes = &pool_size,
          };
          return UTILS_VK(device.createDescriptorPool(pool_info),
            ^^vk::raii::Device::createDescriptorPool)
            .and_then(
              [ &, layout = std::move(layout) ](
                vk::raii::DescriptorPool&& pool) mutable
                -> std::expected<bindless_table, error_t>
              {
                const std::uint32_t variable_count { create_info.capacity };
                const vk::StructureChain allocate_chain {
                  vk::DescriptorSetAllocateInfo {
                    .descriptorPool = *pool,
                    .descriptorSetCount = 1U,
                    .pSetLayouts = &*layout,
                  },
                  vk::DescriptorSetVariableDescriptorCountAllocateInfo {
                    .descriptorSetCount = 1U,
                    .pDescriptorCounts = &variable_count,
                  },
                };

                return UTILS_VK(
                  device.allocateDescriptorSets(
                    allocate_chain.get<vk::DescriptorSetAllocateInfo>()),
                  ^^vk::raii::Device::allocateDescriptorSets)
                  .transform(
                    [ & ](std::vector<vk::raii::DescriptorSet>&& sets) mutable
                      -> bindless_table
                    {
                      return bindless_table {
                        std::move(layout),
                        std::move(pool),
                        sets.front().release(),
                        create_info.capacity,
                      };
                    });
              });
        });
  }

  [[nodiscard]] auto
  acquire_index() -> std::optional<std::uint32_t>
  {
    if (next_index_ >= capacity_) { return std::nullopt; }
    return next_index_++;
  }

  void
  write(const vk::raii::Device& device, std::uint32_t index,
    vk::Sampler sampler, vk::ImageView view) const
  {
    const vk::DescriptorImageInfo image_info {
      .sampler = sampler,
      .imageView = view,
      .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };
    const vk::WriteDescriptorSet write {
      .dstSet = set_,
      .dstBinding = 0U,
      .dstArrayElement = index,
      .descriptorCount = 1U,
      .descriptorType = vk::DescriptorType::eCombinedImageSampler,
      .pImageInfo = &image_info,
    };
    device.updateDescriptorSets(write, nullptr);
  }

  [[nodiscard]] auto
  layout() const -> const vk::raii::DescriptorSetLayout&
  { return layout_; }

  [[nodiscard]] auto
  set() const -> vk::DescriptorSet
  { return set_; }

private:
  vk::raii::DescriptorSetLayout layout_ { nullptr };
  vk::raii::DescriptorPool pool_ { nullptr };
  vk::DescriptorSet set_ {};
  std::uint32_t capacity_ { 0U };
  std::uint32_t next_index_ { 0U };
};

} // namespace vkpp
