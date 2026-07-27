module;

#include "error/vk_error_config.hpp"

export module vkpp.descriptor;

import std;
import vulkan;

import vkpp.error;

namespace vkpp
{

// why duplicate vk:: struct?
export struct descriptor_set_layout_binding
{
  std::uint32_t binding {};
  vk::DescriptorType type {};
  std::uint32_t count { 1U };
  vk::ShaderStageFlags stages {};
};

export auto
make_descriptor_set_layout(const vk::raii::Device& device,
  std::span<const descriptor_set_layout_binding> bindings)
  -> std::expected<vk::raii::DescriptorSetLayout, error_t>
{
  // insn't pointer valid when passed as static array in f1st?
  // why then copy?
  std::vector<vk::DescriptorSetLayoutBinding> native_bindings;
  native_bindings.reserve(bindings.size());
  for (const descriptor_set_layout_binding& binding : bindings)
  {
    native_bindings.push_back({
      .binding = binding.binding,
      .descriptorType = binding.type,
      .descriptorCount = binding.count,
      .stageFlags = binding.stages,
      .pImmutableSamplers = nullptr,
    });
  }
  const vk::DescriptorSetLayoutCreateInfo create_info {
    .bindingCount = static_cast<std::uint32_t>(native_bindings.size()),
    .pBindings = native_bindings.data(),
  };
  return UTILS_VK(device.createDescriptorSetLayout(create_info),
    ^^vk::raii::Device::createDescriptorSetLayout);
}

export class descriptor_pool
{
public:
  [[nodiscard]] static auto
  create(const vk::raii::Device& device, std::uint32_t max_sets,
    std::span<const vk::DescriptorPoolSize> pool_sizes,
    vk::DescriptorPoolCreateFlags flags = {})
    -> std::expected<descriptor_pool, error_t>
  {
    const vk::DescriptorPoolCreateInfo info {
      .flags = flags,
      .maxSets = max_sets,
      .poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size()),
      .pPoolSizes = pool_sizes.data(),
    };
    return UTILS_VK(device.createDescriptorPool(info),
      ^^vk::raii::Device::createDescriptorPool)
      .transform([](vk::raii::DescriptorPool&& pool) -> descriptor_pool
        { return { .pool_ = std::move(pool) }; });
  }

  [[nodiscard]] auto
  allocate(const vk::raii::Device& device,
    const vk::raii::DescriptorSetLayout& layout, std::uint32_t count)
    -> std::expected<std::vector<vk::raii::DescriptorSet>, error_t>
  {
    std::vector layouts(count, *layout);
    const vk::DescriptorSetAllocateInfo info {
      .descriptorPool = *pool_,
      .descriptorSetCount = count,
      .pSetLayouts = layouts.data(),
    };
    return UTILS_VK(device.allocateDescriptorSets(info),
      ^^vk::raii::Device::allocateDescriptorSets);
  }

  [[nodiscard]] auto
  reset() -> std::expected<void, error_t>
  { return UTILS_VK(pool_.reset(), ^^vk::raii::DescriptorPool::reset); }

  [[nodiscard]] auto
  gandle(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.pool_); }

  // I'd rather leave the member public, because of inplace init when creating
  //  private:
public:
  vk::raii::DescriptorPool pool_ { nullptr };
};

export [[nodiscard]] auto
pool_sizes_for(std::span<const descriptor_set_layout_binding> bindings,
  std::uint32_t set_count) -> std::vector<vk::DescriptorPoolSize>
{
  struct type_count
  {
    vk::DescriptorType type {};
    std::uint32_t count {};
  };
  std::vector<type_count> counts_by_type;
  for (const auto& binding : bindings)
  {
    auto existing =
      std::ranges::find(counts_by_type, binding.type, &type_count::type);
    const std::uint32_t added = binding.count * set_count;
    if (existing == counts_by_type.end())
    {
      counts_by_type.push_back({
        .type = binding.type,
        .count = added,
      });
    }
    else
    {
      existing->count += added;
    }
  }
  std::vector<vk::DescriptorPoolSize> pool_sizes;
  pool_sizes.reserve(counts_by_type.size());
  for (const auto& element : counts_by_type)
  {
    pool_sizes.push_back({
      .type = element.type,
      .descriptorCount = element.count,
    });
  }

  return pool_sizes;
}

export void
update_descriptor_sets(const vk::raii::Device& device,
  std::span<const vk::WriteDescriptorSet> writes)
{ device.updateDescriptorSets(writes, {}); }

export void
write_ubo_and_combined_image(const vk::raii::Device& device,
  vk::DescriptorSet destination, vk::Buffer ubo, vk::DeviceSize ubo_range,
  vk::Sampler sampler, vk::ImageView view,
  vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal)
{
  const vk::DescriptorBufferInfo buffer_info {
    .buffer = ubo,
    .offset = 0U,
    .range = ubo_range,
  };
  const vk::DescriptorImageInfo image_info {
    .sampler = sampler,
    .imageView = view,
    .imageLayout = layout,
  };
  const std::array writes {
    vk::WriteDescriptorSet {
      .dstSet = destination,
      .dstBinding = 0U,
      .dstArrayElement = 0U,
      .descriptorCount = 1U,
      .descriptorType = vk::DescriptorType::eUniformBuffer,
      .pBufferInfo = &buffer_info,
    },
    vk::WriteDescriptorSet {
      .dstSet = destination,
      .dstBinding = 1U,
      .dstArrayElement = 0U,
      .descriptorCount = 1U,
      .descriptorType = vk::DescriptorType::eCombinedImageSampler,
      .pImageInfo = &image_info,
    },
  };
  update_descriptor_sets(device, writes);
}

} // namespace vkpp
