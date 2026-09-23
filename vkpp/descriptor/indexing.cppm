export module vkpp.descriptor.indexing;

import std;
import vulkan;

import vkpp.error;
import vkpp.diagnostics;
import vkpp.memory;
import vkpp.memory.vma;
import vkpp.device;

namespace vkpp
{
using namespace std::string_view_literals;

export enum class descriptor_table_backend : std::uint8_t {
  classic,
  heap,
};

export struct bindless_table_create_info
{
  std::uint32_t capacity { 1024U };
  vk::ShaderStageFlags stages { vk::ShaderStageFlagBits::eFragment };
};

export template<descriptor_table_backend Backend>
class bindless_table;

template<>
class bindless_table<descriptor_table_backend::classic>
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

    return map_vk_error(
      device.createDescriptorSetLayout(
        layout_chain.get<vk::DescriptorSetLayoutCreateInfo>()),
      std::nullopt)
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
          return map_vk_error(
            device.createDescriptorPool(pool_info), std::nullopt)
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

                return map_vk_error(
                  device.allocateDescriptorSets(
                    allocate_chain.get<vk::DescriptorSetAllocateInfo>()),
                  std::nullopt)
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
    if (!free_list_.empty())
    {
      const auto index = free_list_.back();
      free_list_.pop_back();
      return index;
    }
    if (next_index_ >= capacity_) { return std::nullopt; }
    return next_index_++;
  }

  void
  release_index(
    std::uint32_t index, std::uint64_t earliest_reuse_timeline_value)
  { pending_retire_.emplace_back(index, earliest_reuse_timeline_value); }

  void
  retire(std::uint64_t completed_timeline_value)
  {
    const auto split = std::ranges::stable_partition(pending_retire_,
      [ completed_timeline_value ](
        const std::pair<std::uint32_t, std::uint64_t>& e) -> bool
      { return e.second > completed_timeline_value; });
    for (const auto& entry : split)
    {
      free_list_.push_back(entry.first);
    }
    pending_retire_.erase(split.begin(), split.end());
  }

  [[nodiscard]] auto
  write(const vk::raii::Device& device, std::uint32_t index,
    vk::Sampler sampler, vk::ImageView view) const
    -> std::expected<void, error_t>
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
    return {};
  }

  [[nodiscard]] auto
  register_combined_image_sampler(
    const vk::raii::Device& device, vk::Sampler sampler, vk::ImageView view)
    -> std::expected<std::uint32_t, error_t>
  {
    const auto index = acquire_index();
    if (!index)
    {
      return std::unexpected {
        make_app_error(app_error_code::capacity_exhausted),
      };
    }
    return write(device, *index, sampler, view)
      .transform([ index ] -> std::uint32_t { return *index; });
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
  std::vector<std::uint32_t> free_list_ {};
  std::vector<std::pair<std::uint32_t, std::uint64_t>> pending_retire_ {};
};

template<>
class bindless_table<descriptor_table_backend::heap>
{
public:
  bindless_table() = default;

  bindless_table(vma_policy::buffer_handle&& sampler_heap,
    vma_policy::buffer_handle&& resource_heap,
    vk::DeviceAddress sampler_heap_address,
    vk::DeviceAddress resource_heap_address, vk::DeviceSize sampler_heap_size,
    vk::DeviceSize resource_heap_size, vk::DeviceSize sampler_descriptor_size,
    vk::DeviceSize image_descriptor_size,
    vk::DeviceSize sampler_reserved_offset,
    vk::DeviceSize sampler_reserved_size,
    vk::DeviceSize resource_reserved_offset,
    vk::DeviceSize resource_reserved_size, std::uint32_t capacity,
    vk::DescriptorSetAndBindingMappingEXT mapping)
  : sampler_heap_ { std::move(sampler_heap) },
    resource_heap_ { std::move(resource_heap) },
    sampler_heap_address_ { sampler_heap_address },
    resource_heap_address_ { resource_heap_address },
    sampler_heap_size_ { sampler_heap_size },
    resource_heap_size_ { resource_heap_size },
    sampler_descriptor_size_ { sampler_descriptor_size },
    image_descriptor_size_ { image_descriptor_size },
    sampler_reserved_offset_ { sampler_reserved_offset },
    sampler_reserved_size_ { sampler_reserved_size },
    resource_reserved_offset_ { resource_reserved_offset },
    resource_reserved_size_ { resource_reserved_size }, capacity_ { capacity },
    mapping_ { mapping }
  {}

  [[nodiscard]] static auto
  create(const vk::raii::Device& device,
    const vk::raii::PhysicalDevice& physical, vma_policy& allocator,
    const bindless_table_create_info& create_info)
    -> std::expected<bindless_table, error_t>
  {
    if (!descriptor_heap_supported(physical))
    {
      return std::unexpected {
        make_app_error(app_error_code::feature_not_supported),
      };
    }

    const auto features = physical.getFeatures2<vk::PhysicalDeviceFeatures2,
      vk::PhysicalDeviceVulkan12Features>();
    if (features.get<vk::PhysicalDeviceVulkan12Features>()
          .bufferDeviceAddress != vk::True)
    {
      return std::unexpected {
        make_app_error(app_error_code::feature_not_supported),
      };
    }

    const auto properties =
      physical.getProperties2<vk::PhysicalDeviceProperties2,
        vk::PhysicalDeviceDescriptorHeapPropertiesEXT>();
    const auto& heap_porperties =
      properties.get<vk::PhysicalDeviceDescriptorHeapPropertiesEXT>();

    const auto sampler_descriptor_size =
      physical.getDescriptorSizeEXT(vk::DescriptorType::eSampler);
    const auto image_descriptor_size =
      physical.getDescriptorSizeEXT(vk::DescriptorType::eSampledImage);

    const auto sampler_payload =
      align_up(sampler_descriptor_size * create_info.capacity,
        heap_porperties.samplerDescriptorAlignment);
    const auto resource_payload =
      align_up(image_descriptor_size * create_info.capacity,
        heap_porperties.imageDescriptorAlignment);

    const auto sampler_reserved = heap_porperties.minSamplerHeapReservedRange;
    const auto resource_reserved = heap_porperties.minResourceHeapReservedRange;

    const auto sampler_size = align_up(
      sampler_payload + sampler_reserved, heap_porperties.samplerHeapAlignment);
    const auto resource_size = align_up(resource_payload + resource_reserved,
      heap_porperties.resourceHeapAlignment);

    if (sampler_size > heap_porperties.maxSamplerHeapSize ||
      resource_size > heap_porperties.maxResourceHeapSize)
    {
      return std::unexpected {
        make_app_error(app_error_code::capacity_exhausted),
      };
    }

    const vk::BufferUsageFlags heap_usage =
      vk::BufferUsageFlagBits::eDescriptorHeapEXT |
      vk::BufferUsageFlagBits::eShaderDeviceAddress;

    return allocator
      .create_buffer(
        {
          .size = sampler_size,
          .usage = heap_usage,
        },
        memory_intent::cpu_to_gpu)
      .and_then(
        [ & ](vma_policy::buffer_handle&& sampler_heap)
          -> std::expected<bindless_table, error_t>
        {
          return allocator
            .create_buffer(
              {
                .size = resource_size,
                .usage = heap_usage,
              },
              memory_intent::cpu_to_gpu)
            .and_then(
              [ &, sampler_heap = std::move(sampler_heap) ](
                vma_policy::buffer_handle&& resource_heap) mutable
                -> std::expected<bindless_table, error_t>
              {
                if (sampler_heap.mapped() == nullptr ||
                  resource_heap.mapped() == nullptr)
                {
                  return std::unexpected {
                    make_app_error(app_error_code::invalid_state),
                  };
                }

                const auto sampler_address = device.getBufferAddress({
                  .buffer = sampler_heap.get(),
                });
                const auto resource_address = device.getBufferAddress({
                  .buffer = resource_heap.get(),
                });

                const vk::DescriptorMappingSourceConstantOffsetEXT
                  constant_offset {
                    .heapOffset = 0U,
                    .heapArrayStride =
                      static_cast<std::uint32_t>(image_descriptor_size),
                    .samplerHeapOffset = 0U,
                    .samplerHeapArrayStride =
                      static_cast<std::uint32_t>(sampler_descriptor_size),
                  };

                const vk::DescriptorSetAndBindingMappingEXT mapping {
                  .descriptorSet = 0U,
                  .firstBinding = 0U,
                  .bindingCount = 1U,
                  .resourceMask =
                    vk::SpirvResourceTypeFlagBitsEXT::eCombinedSampledImage,
                  .source =
                    vk::DescriptorMappingSourceEXT::eHeapWithConstantOffset,
                  .sourceData = { constant_offset },
                };

                return bindless_table {
                  std::move(sampler_heap),
                  std::move(resource_heap),
                  sampler_address,
                  resource_address,
                  sampler_size,
                  resource_size,
                  sampler_descriptor_size,
                  image_descriptor_size,
                  sampler_size - sampler_reserved,
                  sampler_reserved,
                  resource_size - resource_reserved,
                  resource_reserved,
                  create_info.capacity,
                  mapping,
                };
              });
        });
  }

  [[nodiscard]] auto
  acquire_index() -> std::optional<std::uint32_t>
  {
    if (!free_list_.empty())
    {
      const auto index = free_list_.back();
      free_list_.pop_back();
      return index;
    }
    if (next_index_ >= capacity_) { return std::nullopt; }
    return next_index_++;
  }

  void
  release_index(
    std::uint32_t index, std::uint64_t earliest_reuse_timeline_value)
  { pending_retire_.emplace_back(index, earliest_reuse_timeline_value); }

  [[nodiscard]] auto
  write(const vk::raii::Device& device, std::uint32_t index,
    const vk::SamplerCreateInfo& sampler_create_info,
    const vk::ImageViewCreateInfo& image_view_create_info,
    vk::ImageLayout image_layout = vk::ImageLayout::eShaderReadOnlyOptimal)
    const -> std::expected<void, error_t>
  {
    if (index >= capacity_)
    {
      return std::unexpected { make_app_error(app_error_code::out_of_range) };
    }

    auto* const sampler_base = static_cast<std::byte*>(sampler_heap_.mapped());
    auto* const resource_base =
      static_cast<std::byte*>(resource_heap_.mapped());

    if (sampler_base == nullptr || resource_base == nullptr)
    {
      return std::unexpected { make_app_error(app_error_code::invalid_state) };
    }

    const vk::HostAddressRangeEXT sampler_range {
      .address = sampler_base + index * sampler_descriptor_size_,
      .size = static_cast<std::uint32_t>(sampler_descriptor_size_),
    };
    const vk::HostAddressRangeEXT resource_range {
      .address = resource_base + index * image_descriptor_size_,
      .size = static_cast<std::uint32_t>(image_descriptor_size_),
    };

    const vk::ImageDescriptorInfoEXT image_descriptor {
      .pView = &image_view_create_info,
      .layout = image_layout,
    };
    const vk::ResourceDescriptorInfoEXT resource_info {
      .type = vk::DescriptorType::eSampledImage,
      .data = &image_descriptor,
    };

    return map_vk_error(
      device.writeSamplerDescriptorsEXT({ sampler_create_info },
        {
          sampler_range,
        }),
      std::nullopt)
      .and_then(
        [ & ] -> std::expected<void, error_t>
        {
          return map_vk_error(
            device.writeResourceDescriptorsEXT({ resource_info },
              {
                resource_range,
              }),
            std::nullopt);
        });
  }

  [[nodiscard]] auto
  register_combined_image_sampler(const vk::raii::Device& device,
    const vk::SamplerCreateInfo& sampler_create_info,
    const vk::ImageViewCreateInfo& image_view_create_info,
    vk::ImageLayout image_layout = vk::ImageLayout::eShaderReadOnlyOptimal)
    -> std::expected<std::uint32_t, error_t>
  {
    const auto index = acquire_index();
    if (!index)
    {
      return std::unexpected {
        make_app_error(app_error_code::capacity_exhausted),
      };
    }
    return write(
      device, *index, sampler_create_info, image_view_create_info, image_layout)
      .transform([ index ] -> std::uint32_t { return *index; });
  }

  void
  bind(vk::raii::CommandBuffer& command_buffer) const
  {
    const vk::BindHeapInfoEXT sampler_info {
      .heapRange = {
        .address = sampler_heap_address_,
        .size = sampler_heap_size_,
      },
      .reservedRangeOffset = sampler_reserved_offset_,
      .reservedRangeSize = sampler_reserved_size_,
    };
    const vk::BindHeapInfoEXT resource_info {
      .heapRange = {
        .address = resource_heap_address_,
        .size = resource_heap_size_,
      },
      .reservedRangeOffset = resource_reserved_offset_,
      .reservedRangeSize = resource_reserved_size_,
    };
    command_buffer.bindSamplerHeapEXT(sampler_info);
    command_buffer.bindResourceHeapEXT(resource_info);
  }

  void
  rebind_shader_mapping(
    std::uint32_t descriptor_set, std::uint32_t first_binding = 0U)
  {
    mapping_.descriptorSet = descriptor_set;
    mapping_.firstBinding = first_binding;
  }

  [[nodiscard]] auto
  shader_mapping_info() const -> vk::ShaderDescriptorSetAndBindingMappingInfoEXT
  {
    return vk::ShaderDescriptorSetAndBindingMappingInfoEXT {
      .mappingCount = 1U,
      .pMappings = &mapping_,
    };
  }

  [[nodiscard]] auto
  mapping() const -> const vk::DescriptorSetAndBindingMappingEXT&
  { return mapping_; }

private:
  [[nodiscard]] static constexpr auto
  align_up(vk::DeviceSize value, vk::DeviceSize alignment) -> vk::DeviceSize
  {
    if (alignment <= 1UZ) { return value; }
    return (value + alignment - 1UZ) / alignment * alignment;
  }

  vma_policy::buffer_handle sampler_heap_ {};
  vma_policy::buffer_handle resource_heap_ {};
  vk::DeviceAddress sampler_heap_address_ {};
  vk::DeviceAddress resource_heap_address_ {};
  vk::DeviceSize sampler_heap_size_ {};
  vk::DeviceSize resource_heap_size_ {};
  vk::DeviceSize sampler_descriptor_size_ {};
  vk::DeviceSize image_descriptor_size_ {};
  vk::DeviceSize sampler_reserved_offset_ {};
  vk::DeviceSize sampler_reserved_size_ {};
  vk::DeviceSize resource_reserved_offset_ {};
  vk::DeviceSize resource_reserved_size_ {};
  std::uint32_t capacity_ {};
  std::uint32_t next_index_ {};
  std::vector<std::uint32_t> free_list_ {};
  std::vector<std::pair<std::uint32_t, std::uint64_t>> pending_retire_ {};
  mutable vk::DescriptorSetAndBindingMappingEXT mapping_ {};
};

export using classic_bindless_table =
  bindless_table<descriptor_table_backend::classic>;

export using heap_bindless_table =
  bindless_table<descriptor_table_backend::heap>;

} // namespace vkpp
