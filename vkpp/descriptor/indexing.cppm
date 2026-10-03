export module vkpp.descriptor.indexing;

import std;
import vulkan;

import vkpp.error;
export import vkpp.capabilities;
import vkpp.diagnostics;
import vkpp.memory;
import vkpp.memory.vma;
import vkpp.device;
import vkpp.buffer;
import vkpp.descriptor.layout_decl;

namespace vkpp
{
using namespace std::string_view_literals;

export struct descriptor_heap_arena_create_info
{
  std::uint32_t bindless_capacity { 1024U };
  std::uint32_t frames_in_flight { 2U };
};

export class descriptor_heap_arena
{
public:
  descriptor_heap_arena() = default;

  descriptor_heap_arena(vma_policy::buffer_handle&& sampler_heap,
    vma_policy::buffer_handle&& resource_heap,
    vk::DeviceAddress sampler_heap_address,
    vk::DeviceAddress resource_heap_address, vk::DeviceSize sampler_heap_size,
    vk::DeviceSize resource_heap_size, vk::DeviceSize sampler_descriptor_size,
    vk::DeviceSize image_descriptor_size, vk::DeviceSize buffer_descriptor_size,
    vk::DeviceSize uniform_descriptor_size,
    vk::DeviceSize sampler_reserved_offset,
    vk::DeviceSize sampler_reserved_size,
    vk::DeviceSize resource_reserved_offset,
    vk::DeviceSize resource_reserved_size, std::uint32_t bindless_capacity,
    std::uint32_t frames_in_flight, vk::DeviceSize resource_uniform_base,
    vk::DeviceSize resource_storage_b1_base,
    vk::DeviceSize resource_storage_b3_base,
    vk::DeviceSize resource_bindless_base, vk::DeviceSize resource_ibl_base,
    vk::DeviceSize resource_compute_base, vk::DeviceSize sampler_bindless_base,
    vk::DeviceSize sampler_ibl_base, vk::DeviceSize sampler_compute_base,
    std::array<vk::DescriptorSetAndBindingMappingEXT, 6> mappings)
  : sampler_heap_ { std::move(sampler_heap) },
    resource_heap_ { std::move(resource_heap) },
    sampler_heap_address_ { sampler_heap_address },
    resource_heap_address_ { resource_heap_address },
    sampler_heap_size_ { sampler_heap_size },
    resource_heap_size_ { resource_heap_size },
    sampler_descriptor_size_ { sampler_descriptor_size },
    image_descriptor_size_ { image_descriptor_size },
    buffer_descriptor_size_ { buffer_descriptor_size },
    uniform_descriptor_size_ { uniform_descriptor_size },
    sampler_reserved_offset_ { sampler_reserved_offset },
    sampler_reserved_size_ { sampler_reserved_size },
    resource_reserved_offset_ { resource_reserved_offset },
    resource_reserved_size_ { resource_reserved_size },
    bindless_capacity_ { bindless_capacity },
    frames_in_flight_ { frames_in_flight },
    resource_uniform_base_ { resource_uniform_base },
    resource_storage_b1_base_ { resource_storage_b1_base },
    resource_storage_b3_base_ { resource_storage_b3_base },
    resource_bindless_base_ { resource_bindless_base },
    resource_ibl_base_ { resource_ibl_base },
    resource_compute_base_ { resource_compute_base },
    sampler_bindless_base_ { sampler_bindless_base },
    sampler_ibl_base_ { sampler_ibl_base },
    sampler_compute_base_ { sampler_compute_base }, mappings_ { mappings }
  {}

  template<class PushLayout, class Regions>
  [[nodiscard]] static auto
  create(const vk::raii::Device& device,
    const vk::raii::PhysicalDevice& physical, vma_policy& allocator,
    const selected_device_capabilities& capabilities,
    const descriptor_heap_arena_create_info& create_info,
    std::optional<diagnostic_buffer&> diagnostics = {})
    -> std::expected<descriptor_heap_arena, error_t>
    requires(PushLayout::field_count >= 1UZ && Regions::count >= 1UZ)
  {
    static_assert(Regions::count == 6UZ,
      "descriptor_heap_arena mappings_ is fixed ad 6 until generalized");

    if (!capabilities.descriptor_heap_enabled)
    {
      return std::unexpected {
        make_app_error(app_error_code::feature_not_enabled),
      };
    }

    const auto properties =
      physical.getProperties2<vk::PhysicalDeviceProperties2,
        vk::PhysicalDeviceDescriptorHeapPropertiesEXT>();
    const auto& heap_props =
      properties.get<vk::PhysicalDeviceDescriptorHeapPropertiesEXT>();

    if (PushLayout::total_bytes > heap_props.maxPushDataSize)
    {
      return std::unexpected {
        make_app_error(app_error_code::invalid_data_size),
      };
    }

    const auto sampler_descriptor_size =
      physical.getDescriptorSizeEXT(vk::DescriptorType::eSampler);
    const auto image_descriptor_size =
      physical.getDescriptorSizeEXT(vk::DescriptorType::eSampledImage);
    const auto uniform_descriptor_size =
      physical.getDescriptorSizeEXT(vk::DescriptorType::eUniformBuffer);
    const auto storage_descriptor_size =
      physical.getDescriptorSizeEXT(vk::DescriptorType::eStorageBuffer);

    const auto align_up = [](vk::DeviceSize value,
                            vk::DeviceSize alignment) -> vk::DeviceSize
    {
      return alignment <= 1UZ
        ? value
        : (value + alignment - 1UZ) / alignment * alignment;
    };

    constexpr vk::DeviceSize unset { ~0ULL };
    auto resource_uniform_base { unset };
    auto resource_storage_b1_base { unset };
    auto resource_storage_b3_base { unset };
    auto resource_bindless_base { unset };
    auto resource_ibl_base { unset };
    auto resource_compute_base { unset };
    auto sampler_bindless_base { unset };
    auto sampler_ibl_base { unset };
    auto sampler_compute_base { unset };

    vk::DeviceSize resource_cursor {};
    vk::DeviceSize sampler_cursor {};
    std::array<vk::DescriptorSetAndBindingMappingEXT, Regions::count>
      mappings {};

    for (auto index : std::views::indices(Regions::count))
    {
      const logical_resource_role role = Regions::roles[ index ];
      const auto binding = classic_binding_for(role);

      switch (role)
      {
      case logical_resource_role::frame_uniform_ring:
      {
        resource_uniform_base = resource_cursor;
        const vk::DescriptorMappingSourcePushIndexEXT uniform_push {
          .heapOffset = static_cast<std::uint32_t>(resource_uniform_base),
          .pushOffset = PushLayout::template offset_of<
            typename PushLayout::frame_slot_tag>(),
          .heapIndexStride =
            static_cast<std::uint32_t>(uniform_descriptor_size),
          .heapArrayStride =
            static_cast<std::uint32_t>(uniform_descriptor_size),
        };
        mappings[ index ] = vk::DescriptorSetAndBindingMappingEXT {
          .descriptorSet = binding.set,
          .firstBinding = binding.binding,
          .bindingCount = 1U,
          .resourceMask = vk::SpirvResourceTypeFlagBitsEXT::eUniformBuffer,
          .source = vk::DescriptorMappingSourceEXT::eHeapWithPushIndex,
          .sourceData = { uniform_push },
        };
        resource_cursor = align_up(resource_cursor +
            uniform_descriptor_size * create_info.frames_in_flight,
          heap_props.bufferDescriptorAlignment);
        break;
      }
      case logical_resource_role::material_storage:
      {
        resource_storage_b1_base = resource_cursor;
        const vk::DescriptorMappingSourceConstantOffsetEXT storage {
          .heapOffset = static_cast<std::uint32_t>(resource_storage_b1_base),
          .heapArrayStride =
            static_cast<std::uint32_t>(storage_descriptor_size),
        };
        mappings[ index ] = vk::DescriptorSetAndBindingMappingEXT {
          .descriptorSet = binding.set,
          .firstBinding = binding.binding,
          .bindingCount = 1U,
          .resourceMask =
            vk::SpirvResourceTypeFlagBitsEXT::eReadOnlyStorageBuffer,
          .source = vk::DescriptorMappingSourceEXT::eHeapWithConstantOffset,
          .sourceData = { storage },
        };
        resource_cursor = align_up(resource_cursor + storage_descriptor_size,
          heap_props.bufferDescriptorAlignment);
        break;
      }
      case logical_resource_role::draw_storage:
      {
        resource_storage_b3_base = resource_cursor;
        const vk::DescriptorMappingSourceConstantOffsetEXT storage {
          .heapOffset = static_cast<std::uint32_t>(resource_storage_b3_base),
          .heapArrayStride =
            static_cast<std::uint32_t>(storage_descriptor_size),
        };
        mappings[ index ] = vk::DescriptorSetAndBindingMappingEXT {
          .descriptorSet = binding.set,
          .firstBinding = binding.binding,
          .bindingCount = 1U,
          .resourceMask =
            vk::SpirvResourceTypeFlagBitsEXT::eReadOnlyStorageBuffer,
          .source = vk::DescriptorMappingSourceEXT::eHeapWithConstantOffset,
          .sourceData = { storage },
        };
        resource_cursor = align_up(resource_cursor + storage_descriptor_size,
          heap_props.bufferDescriptorAlignment);
        break;
      }
      case logical_resource_role::bindless_cis:
      {
        resource_bindless_base = resource_cursor;
        sampler_bindless_base = sampler_cursor;
        const vk::DescriptorMappingSourceConstantOffsetEXT cis {
          .heapOffset = static_cast<std::uint32_t>(resource_bindless_base),
          .heapArrayStride = static_cast<std::uint32_t>(image_descriptor_size),
          .samplerHeapOffset =
            static_cast<std::uint32_t>(sampler_bindless_base),
          .samplerHeapArrayStride =
            static_cast<std::uint32_t>(sampler_descriptor_size),
        };
        mappings[ index ] = vk::DescriptorSetAndBindingMappingEXT {
          .descriptorSet = binding.set,
          .firstBinding = binding.binding,
          .bindingCount = 1U,
          .resourceMask =
            vk::SpirvResourceTypeFlagBitsEXT::eCombinedSampledImage,
          .source = vk::DescriptorMappingSourceEXT::eHeapWithConstantOffset,
          .sourceData = { cis },
        };
        resource_cursor = align_up(resource_cursor +
            image_descriptor_size * create_info.bindless_capacity,
          heap_props.imageDescriptorAlignment);
        sampler_cursor = align_up(sampler_cursor +
            sampler_descriptor_size * create_info.bindless_capacity,
          heap_props.samplerDescriptorAlignment);
        break;
      }
      case logical_resource_role::ibl_cis:
      {
        resource_ibl_base = resource_cursor;
        sampler_ibl_base = sampler_cursor;
        const vk::DescriptorMappingSourceConstantOffsetEXT cis {
          .heapOffset = static_cast<std::uint32_t>(resource_ibl_base),
          .heapArrayStride = static_cast<std::uint32_t>(image_descriptor_size),
          .samplerHeapOffset = static_cast<std::uint32_t>(sampler_ibl_base),
          .samplerHeapArrayStride =
            static_cast<std::uint32_t>(sampler_descriptor_size),
        };
        mappings[ index ] = vk::DescriptorSetAndBindingMappingEXT {
          .descriptorSet = binding.set,
          .firstBinding = binding.binding,
          .bindingCount = 1U,
          .resourceMask =
            vk::SpirvResourceTypeFlagBitsEXT::eCombinedSampledImage,
          .source = vk::DescriptorMappingSourceEXT::eHeapWithConstantOffset,
          .sourceData = { cis },
        };
        resource_cursor = align_up(resource_cursor + image_descriptor_size,
          heap_props.imageDescriptorAlignment);
        sampler_cursor = align_up(sampler_cursor + sampler_descriptor_size,
          heap_props.samplerDescriptorAlignment);
        break;
      }
      case logical_resource_role::compute_cis:
      {
        resource_compute_base = resource_cursor;
        sampler_compute_base = sampler_cursor;
        const vk::DescriptorMappingSourceConstantOffsetEXT cis {
          .heapOffset = static_cast<std::uint32_t>(resource_compute_base),
          .heapArrayStride = static_cast<std::uint32_t>(image_descriptor_size),
          .samplerHeapOffset = static_cast<std::uint32_t>(sampler_compute_base),
          .samplerHeapArrayStride =
            static_cast<std::uint32_t>(sampler_descriptor_size),
        };
        mappings[ index ] = vk::DescriptorSetAndBindingMappingEXT {
          .descriptorSet = binding.set,
          .firstBinding = binding.binding,
          .bindingCount = 1U,
          .resourceMask =
            vk::SpirvResourceTypeFlagBitsEXT::eCombinedSampledImage,
          .source = vk::DescriptorMappingSourceEXT::eHeapWithConstantOffset,
          .sourceData = { cis },
        };
        resource_cursor = align_up(resource_cursor + image_descriptor_size,
          heap_props.imageDescriptorAlignment);
        sampler_cursor = align_up(sampler_cursor + sampler_descriptor_size,
          heap_props.samplerDescriptorAlignment);
        break;
      }
      }
    }

    if (resource_uniform_base == unset || resource_storage_b1_base == unset ||
      resource_storage_b3_base == unset || resource_bindless_base == unset ||
      resource_ibl_base == unset || resource_compute_base == unset ||
      sampler_bindless_base == unset || sampler_ibl_base == unset ||
      sampler_compute_base == unset)
    {
      return std::unexpected { make_app_error(app_error_code::invalid_state) };
    }

    const auto resource_reserved = heap_props.minResourceHeapReservedRange;
    const auto resource_payload = resource_cursor;
    const auto resource_size = align_up(
      resource_payload + resource_reserved, heap_props.resourceHeapAlignment);
    const auto resource_reserved_offset = resource_size - resource_reserved;

    const auto sampler_reserved = heap_props.minSamplerHeapReservedRange;
    const auto sampler_payload = sampler_cursor;
    const auto sampler_size = align_up(
      sampler_payload + sampler_reserved, heap_props.samplerHeapAlignment);
    const auto sampler_reserved_offset = sampler_size - sampler_reserved;

    if (resource_size > heap_props.maxResourceHeapSize ||
      sampler_size > heap_props.maxSamplerHeapSize)
    {
      return std::unexpected {
        make_app_error(app_error_code::capacity_exhausted),
      };
    }

    const vk::BufferUsageFlags heap_usage =
      vk::BufferUsageFlagBits::eDescriptorHeapEXT |
      vk::BufferUsageFlagBits::eShaderDeviceAddress;

    return allocator
      .create_buffer({ .size = sampler_size, .usage = heap_usage },
        memory_intent::cpu_to_gpu)
      .and_then(
        [ & ](vma_policy::buffer_handle&& sampler_heap)
          -> std::expected<descriptor_heap_arena, error_t>
        {
          return allocator
            .create_buffer({ .size = resource_size, .usage = heap_usage },
              memory_intent::cpu_to_gpu)
            .and_then(
              [ &, sampler_heap = std::move(sampler_heap) ](
                vma_policy::buffer_handle&& resource_heap) mutable
                -> std::expected<descriptor_heap_arena, error_t>
              {
                if (sampler_heap.mapped() == nullptr ||
                  resource_heap.mapped() == nullptr)
                {
                  return std::unexpected {
                    make_app_error(app_error_code::mapping_failed),
                  };
                }
                const auto sampler_address =
                  device.getBufferAddress({ .buffer = sampler_heap.get() });
                const auto resource_address =
                  device.getBufferAddress({ .buffer = resource_heap.get() });
                return descriptor_heap_arena {
                  std::move(sampler_heap),
                  std::move(resource_heap),
                  sampler_address,
                  resource_address,
                  sampler_size,
                  resource_size,
                  sampler_descriptor_size,
                  image_descriptor_size,
                  storage_descriptor_size,
                  uniform_descriptor_size,
                  sampler_reserved_offset,
                  sampler_reserved,
                  resource_reserved_offset,
                  resource_reserved,
                  create_info.bindless_capacity,
                  create_info.frames_in_flight,
                  resource_uniform_base,
                  resource_storage_b1_base,
                  resource_storage_b3_base,
                  resource_bindless_base,
                  resource_ibl_base,
                  resource_compute_base,
                  sampler_bindless_base,
                  sampler_ibl_base,
                  sampler_compute_base,
                  std::move(mappings),
                };
              });
        });
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
    command_buffer.bindSamplerHeapEXT(resource_info);
  }

  [[nodiscard]] auto
  mapping() const -> std::span<const vk::DescriptorSetAndBindingMappingEXT>
  { return mappings_; }

  [[nodiscard]] auto
  shader_and_mapping_info() const
    -> vk::ShaderDescriptorSetAndBindingMappingInfoEXT
  {
    return vk::ShaderDescriptorSetAndBindingMappingInfoEXT {
      .mappingCount = static_cast<std::uint32_t>(mappings_.size()),
      .pMappings = mappings_.data(),
    };
  }

  [[nodiscard]] auto
  acquire_bindless_index() -> std::optional<std::uint32_t>
  {
    if (!free_list_.empty())
    {
      const auto index = free_list_.back();
      free_list_.pop_back();
      return index;
    }
    if (next_bindless_index_ >= bindless_capacity_) { return std::nullopt; }
    return next_bindless_index_++;
  }

  void
  release_bindless_index(
    std::uint32_t index, std::uint64_t earliest_reuse_timeline_value)
  { pending_retire_.emplace_back(index, earliest_reuse_timeline_value); }

  void
  retire(std::uint64_t completed_timeline_value)
  {
    const auto split = std::ranges::stable_partition(pending_retire_,
      [ completed_timeline_value ](
        const std::pair<std::uint32_t, std::uint64_t>& entry) -> bool
      { return entry.second > completed_timeline_value; });
    for (const auto& entry : split)
    {
      free_list_.push_back(entry.first);
    }
    pending_retire_.erase(split.begin(), split.end());
  }

  [[nodiscard]] auto
  write_bindless_cis(const vk::raii::Device& device, std::uint32_t index,
    const vk::SamplerCreateInfo& sampler_info,
    const vk::ImageViewCreateInfo& view_info,
    vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal)
    -> std::expected<void, error_t>
  {
    if (index >= bindless_capacity_)
    {
      return std::unexpected { make_app_error(app_error_code::out_of_range) };
    }
    auto* const sampler_base = static_cast<std::byte*>(sampler_heap_.mapped());
    auto* const resource_base =
      static_cast<std::byte*>(resource_heap_.mapped());
    if (sampler_base == nullptr || resource_base == nullptr)
    {
      return std::unexpected { make_app_error(app_error_code::mapping_failed) };
    }
    const vk::HostAddressRangeEXT sampler_range {
      .address = sampler_base + sampler_bindless_base_ +
        index * sampler_descriptor_size_,
      .size = static_cast<std::uint32_t>(sampler_descriptor_size_),
    };
    const vk::HostAddressRangeEXT resource_range {
      .address = resource_base + resource_bindless_base_ +
        index * image_descriptor_size_,
      .size = static_cast<std::uint32_t>(image_descriptor_size_),
    };
    const vk::ImageDescriptorInfoEXT image_descriptor {
      .pView = &view_info,
      .layout = layout,
    };
    const vk::ResourceDescriptorInfoEXT resource_info {
      .type = vk::DescriptorType::eSampledImage,
      .data = &image_descriptor,
    };
    return map_vk_error(
      device.writeSamplerDescriptorsEXT({ sampler_info }, { sampler_range }))
      .and_then(
        [ & ]() -> std::expected<void, error_t>
        {
          return map_vk_error(device.writeResourceDescriptorsEXT(
            { resource_info }, { resource_range }));
        });
  }

  template<buffer_kind Kind, class T>
  [[nodiscard]] auto
  write_uniform_slot(const vk::raii::Device& device, std::uint32_t frame_index,
    const buffer_view<Kind, T, addressed>& view) -> std::expected<void, error_t>
  {
    if (frame_index >= frames_in_flight_)
    {
      return std::unexpected { make_app_error(app_error_code::out_of_range) };
    }
    auto* const resource_base =
      static_cast<std::byte*>(resource_heap_.mapped());
    if (resource_base == nullptr)
    {
      return std::unexpected { make_app_error(app_error_code::mapping_failed) };
    }
    const vk::DeviceAddressRangeEXT address_range {
      .address = view.device_address(),
      .size = view.size(),
    };
    const vk::ResourceDescriptorInfoEXT resource_info {
      .type = vk::DescriptorType::eUniformBuffer,
      .data = &address_range,
    };
    const vk::HostAddressRangeEXT host_range {
      .address = resource_base + resource_uniform_base_ +
        frame_index * uniform_descriptor_size_,
      .size = static_cast<std::uint32_t>(uniform_descriptor_size_),
    };
    return map_vk_error(
      device.writeResourceDescriptorsEXT({ resource_info }, { host_range }));
  }

  template<buffer_kind Kind, class T>
  [[nodiscard]] auto
  write_storage_binding(const vk::raii::Device& device, std::uint32_t binding,
    const buffer_view<Kind, T, addressed>& view) -> std::expected<void, error_t>
  {
    const vk::DeviceSize base = //
      (binding == 1U)   ? resource_storage_b1_base_
      : (binding == 3U) ? resource_storage_b3_base_
                        : vk::DeviceSize { ~0ULL };
    if (base == vk::DeviceSize { ~0ULL })
    {
      return std::unexpected { make_app_error(app_error_code::out_of_range) };
    }
    auto* const resource_base =
      static_cast<std::byte*>(resource_heap_.mapped());
    if (resource_base == nullptr)
    {
      return std::unexpected { make_app_error(app_error_code::mapping_failed) };
    }
    const vk::DeviceAddressRangeEXT address_range {
      .address = view.device_address(),
      .size = view.size(),
    };
    const vk::ResourceDescriptorInfoEXT resource_info {
      .type = vk::DescriptorType::eStorageBuffer,
      .data = &address_range,
    };
    const vk::HostAddressRangeEXT host_range {
      .address = resource_base + base,
      .size = static_cast<std::uint32_t>(buffer_descriptor_size_),
    };
    return map_vk_error(
      device.writeResourceDescriptorsEXT({ resource_info }, { host_range }));
  }

  enum class fixed_cis_region : std::uint8_t
  {
    ibl,
    compute
  };

  [[nodiscard]] auto
  write_fixed_cis(const vk::raii::Device& device, fixed_cis_region region,
    const vk::SamplerCreateInfo& sampler_info,
    const vk::ImageViewCreateInfo& view_info,
    vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal)
    -> std::expected<void, error_t>
  {
    const vk::DeviceSize sampler_base_off = (region == fixed_cis_region::ibl)
      ? sampler_ibl_base_
      : sampler_compute_base_;
    const vk::DeviceSize resource_base_off = (region == fixed_cis_region::ibl)
      ? resource_ibl_base_
      : resource_compute_base_;
    auto* const sampler_base = static_cast<std::byte*>(sampler_heap_.mapped());
    auto* const resource_base =
      static_cast<std::byte*>(resource_heap_.mapped());
    if (sampler_base == nullptr || resource_base == nullptr)
    {
      return std::unexpected { make_app_error(app_error_code::mapping_failed) };
    }
    const vk::HostAddressRangeEXT sampler_range {
      .address = sampler_base + sampler_base_off,
      .size = static_cast<std::uint32_t>(sampler_descriptor_size_),
    };
    const vk::HostAddressRangeEXT resource_range {
      .address = resource_base + resource_base_off,
      .size = static_cast<std::uint32_t>(image_descriptor_size_),
    };
    const vk::ImageDescriptorInfoEXT image_descriptor {
      .pView = &view_info,
      .layout = layout,
    };
    const vk::ResourceDescriptorInfoEXT resource_info {
      .type = vk::DescriptorType::eSampledImage,
      .data = &image_descriptor,
    };
    return map_vk_error(
      device.writeSamplerDescriptorsEXT({ sampler_info }, { sampler_range }))
      .and_then(
        [ & ]() -> std::expected<void, error_t>
        {
          return map_vk_error(device.writeResourceDescriptorsEXT(
            { resource_info }, { resource_range }));
        });
  }

private:
  vma_policy::buffer_handle sampler_heap_ {};
  vma_policy::buffer_handle resource_heap_ {};
  vk::DeviceAddress sampler_heap_address_ {};
  vk::DeviceAddress resource_heap_address_ {};
  vk::DeviceSize sampler_heap_size_ {};
  vk::DeviceSize resource_heap_size_ {};
  vk::DeviceSize sampler_descriptor_size_ {};
  vk::DeviceSize image_descriptor_size_ {};
  vk::DeviceSize buffer_descriptor_size_ {};
  vk::DeviceSize uniform_descriptor_size_ {};
  vk::DeviceSize sampler_reserved_offset_ {};
  vk::DeviceSize sampler_reserved_size_ {};
  vk::DeviceSize resource_reserved_offset_ {};
  vk::DeviceSize resource_reserved_size_ {};
  std::uint32_t bindless_capacity_ {};
  std::uint32_t frames_in_flight_ {};
  std::uint32_t next_bindless_index_ {};
  std::vector<std::uint32_t> free_list_ {};
  std::vector<std::pair<std::uint32_t, std::uint64_t>> pending_retire_ {};
  vk::DeviceSize resource_uniform_base_ {};
  vk::DeviceSize resource_storage_b1_base_ {};
  vk::DeviceSize resource_storage_b3_base_ {};
  vk::DeviceSize resource_bindless_base_ {};
  vk::DeviceSize resource_ibl_base_ {};
  vk::DeviceSize resource_compute_base_ {};
  vk::DeviceSize sampler_bindless_base_ {};
  vk::DeviceSize sampler_ibl_base_ {};
  vk::DeviceSize sampler_compute_base_ {};
  std::array<vk::DescriptorSetAndBindingMappingEXT, 6> mappings_ {};
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
  register_combined_image_sampler(const vk::raii::Device& device,
    [[maybe_unused]] const vk::SamplerCreateInfo& sampler_info,
    [[maybe_unused]] const vk::ImageViewCreateInfo& image_view_info,
    vk::Sampler sampler, vk::ImageView view)
    -> std::expected<std::uint32_t, error_t>
  { return register_combined_image_sampler(device, sampler, view); }

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

export using classic_bindless_table =
  bindless_table<descriptor_table_backend::classic>;

template<>
class bindless_table<descriptor_table_backend::heap>
{
public:
  bindless_table() = default;

  explicit bindless_table(descriptor_heap_arena& arena) : arena_ { arena } {};

  [[nodiscard]] auto
  register_combined_image_sampler(const vk::raii::Device& device,
    const vk::SamplerCreateInfo& sampler_create_info,
    const vk::ImageViewCreateInfo& image_view_create_info,
    [[maybe_unused]] vk::Sampler sampler, [[maybe_unused]] vk::ImageView view)
    -> std::expected<std::uint32_t, error_t>
  {
    if (!arena_.has_value())
    {
      return std::unexpected {
        make_app_error(app_error_code::invalid_state),
      };
    }
    const auto index = arena_->acquire_bindless_index();
    if (!index)
    {
      return std::unexpected {
        make_app_error(app_error_code::capacity_exhausted),
      };
    }
    return arena_
      ->write_bindless_cis(
        device, *index, sampler_create_info, image_view_create_info)
      .transform([ index ] -> std::uint32_t { return *index; });
  }

  void
  release_index(
    std::uint32_t index, std::uint64_t earliest_reuse_timeline_value)
  { arena_->release_bindless_index(index, earliest_reuse_timeline_value); }

  void
  retire(std::uint64_t completed_timeline_value)
  { arena_->retire(completed_timeline_value); }

private:
  std::optional<descriptor_heap_arena&> arena_ {};
};

export using heap_bindless_table =
  bindless_table<descriptor_table_backend::heap>;

} // namespace vkpp
