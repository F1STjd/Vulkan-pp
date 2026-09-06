module;

#include "error/vk_error_config.hpp"

export module vkpp.descriptor;

import std;
import vulkan;

import vkpp.error;

namespace vkpp
{

export auto
make_descriptor_set_layout(const vk::raii::Device& device,
  std::span<const vk::DescriptorSetLayoutBinding> bindings)
  -> std::expected<vk::raii::DescriptorSetLayout, error_t>
{
  const vk::DescriptorSetLayoutCreateInfo create_info {
    .bindingCount = static_cast<std::uint32_t>(bindings.size()),
    .pBindings = bindings.data(),
  };
  return UTILS_VK(device.createDescriptorSetLayout(create_info),
    ^^vk::raii::Device::createDescriptorSetLayout);
}

export class descriptor_pool
{
public:
  descriptor_pool() = default;
  explicit descriptor_pool(vk::raii::DescriptorPool&& pool)
  : pool_ { std::move(pool) }
  {}

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
        { return descriptor_pool { std::move(pool) }; });
  }

  [[nodiscard]] auto
  allocate(const vk::raii::Device& device,
    const vk::raii::DescriptorSetLayout& layout, std::uint32_t count)
    -> std::expected<std::vector<vk::DescriptorSet>, error_t>
  {
    std::vector layouts(count, *layout);
    const vk::DescriptorSetAllocateInfo info {
      .descriptorPool = *pool_,
      .descriptorSetCount = count,
      .pSetLayouts = layouts.data(),
    };
    return UTILS_VK(device.allocateDescriptorSets(info),
      ^^vk::raii::Device::allocateDescriptorSets)
      .transform(
        [](std::vector<vk::raii::DescriptorSet>&& owned)
        {
          std::vector<vk::DescriptorSet> handles;
          handles.reserve(owned.size());
          for (vk::raii::DescriptorSet& set : owned)
          {
            handles.push_back(set.release());
          }
          return handles;
        });
  }

  [[nodiscard]] auto
  reset() -> std::expected<void, error_t>
  { return UTILS_VK(pool_.reset(), ^^vk::raii::DescriptorPool::reset); }

  [[nodiscard]] auto
  handle(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.pool_); }

private:
  vk::raii::DescriptorPool pool_ { nullptr };
};

export [[nodiscard]] auto
pool_sizes_for(std::span<const vk::DescriptorSetLayoutBinding> bindings,
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
    auto existing = std::ranges::find(
      counts_by_type, binding.descriptorType, &type_count::type);
    const std::uint32_t added = binding.descriptorCount * set_count;
    if (existing == counts_by_type.end())
    {
      counts_by_type.push_back({
        .type = binding.descriptorType,
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

export struct descriptor_set_arena_create_info
{
  const vk::raii::Device& device;
  std::span<const vk::DescriptorSetLayoutBinding> bindings {};
  std::uint32_t set_count { 1U };
  vk::DescriptorPoolCreateFlags pool_flags {};
  std::span<const vk::DescriptorPoolSize> pool_sizes_escape {};
};

export class descriptor_set_arena
{
public:
  descriptor_set_arena() = default;

  descriptor_set_arena(vk::raii::DescriptorSetLayout&& layout,
    descriptor_pool&& pool, std::vector<vk::DescriptorSet>&& sets)
  : layout_ { std::move(layout) }, pool_ { std::move(pool) },
    sets_ { std::move(sets) }
  {}

  [[nodiscard]] auto
  layout() const -> const vk::raii::DescriptorSetLayout&
  { return layout_; }

  [[nodiscard]] auto
  sets() const -> std::span<const vk::DescriptorSet>
  { return sets_; }

  [[nodiscard]] auto
  set(std::uint32_t index) const -> vk::DescriptorSet
  { return sets_[ index ]; }

  [[nodiscard]] static auto
  create(const descriptor_set_arena_create_info& create_info)
    -> std::expected<descriptor_set_arena, error_t>
  {
    return make_descriptor_set_layout(create_info.device, create_info.bindings)
      .and_then(
        [ & ](vk::raii::DescriptorSetLayout&& layout)
          -> std::expected<descriptor_set_arena, error_t>
        {
          std::vector<vk::DescriptorPoolSize> derived;
          std::span<const vk::DescriptorPoolSize> sizes =
            create_info.pool_sizes_escape;
          if (sizes.empty())
          {
            derived =
              pool_sizes_for(create_info.bindings, create_info.set_count);
            sizes = derived;
          }
          return descriptor_pool::create(create_info.device,
            create_info.set_count, sizes, create_info.pool_flags)
            .and_then(
              [ &, layout = std::move(layout) ](descriptor_pool&& pool) mutable
                -> std::expected<descriptor_set_arena, error_t>
              {
                return pool
                  .allocate(create_info.device, layout, create_info.set_count)
                  .transform(
                    [ &, layout = std::move(layout), pool = std::move(pool) ](
                      std::vector<vk::DescriptorSet>&& sets) mutable
                      -> descriptor_set_arena
                    {
                      return descriptor_set_arena {
                        std::move(layout),
                        std::move(pool),
                        std::move(sets),
                      };
                    });
              });
        });
  }

private:
  vk::raii::DescriptorSetLayout layout_ { nullptr };
  descriptor_pool pool_ {};
  std::vector<vk::DescriptorSet> sets_ {};
};

export void
write_buffer_descriptor(const vk::raii::Device& device, vk::DescriptorSet set,
  std::uint32_t binding, vk::DescriptorType type, vk::Buffer buffer,
  vk::DeviceSize offset, vk::DeviceSize range)
{
  const vk::DescriptorBufferInfo buffer_info {
    .buffer = buffer,
    .offset = offset,
    .range = range,
  };
  const vk::WriteDescriptorSet write {
    .dstSet = set,
    .dstBinding = binding,
    .dstArrayElement = 0U,
    .descriptorCount = 1U,
    .descriptorType = type,
    .pBufferInfo = &buffer_info,
  };
  update_descriptor_sets(device, std::span { &write, 1UZ });
}

export void
write_uniform_buffer(const vk::raii::Device& device, vk::DescriptorSet set,
  std::uint32_t binding, vk::Buffer buffer, vk::DeviceSize range,
  vk::DeviceSize offset = 0UZ)
{
  write_buffer_descriptor(device, set, binding,
    vk::DescriptorType::eUniformBuffer, buffer, offset, range);
}

export void
write_storage_buffer(const vk::raii::Device& device, vk::DescriptorSet set,
  std::uint32_t binding, vk::Buffer buffer, vk::DeviceSize range,
  vk::DeviceSize offset = 0UZ)
{
  write_buffer_descriptor(device, set, binding,
    vk::DescriptorType::eStorageBuffer, buffer, offset, range);
}

export void
write_combined_image_sampler(const vk::raii::Device& device,
  vk::DescriptorSet set, std::uint32_t binding, vk::Sampler sampler,
  vk::ImageView view,
  vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal)
{
  const vk::DescriptorImageInfo image_info {
    .sampler = sampler,
    .imageView = view,
    .imageLayout = layout,
  };
  const vk::WriteDescriptorSet write {
    .dstSet = set,
    .dstBinding = binding,
    .dstArrayElement = 0U,
    .descriptorCount = 1U,
    .descriptorType = vk::DescriptorType::eCombinedImageSampler,
    .pImageInfo = &image_info,
  };
  update_descriptor_sets(device, std::span { &write, 1UZ });
}

export void
write_ubo_and_combined_image(const vk::raii::Device& device,
  vk::DescriptorSet destination, vk::Buffer ubo, vk::DeviceSize ubo_range,
  vk::Sampler sampler, vk::ImageView view,
  vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal)
{
  write_uniform_buffer(device, destination, 0U, ubo, ubo_range);
  write_combined_image_sampler(device, destination, 1U, sampler, view, layout);
}

} // namespace vkpp
