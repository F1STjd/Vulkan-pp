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
    std::uint32_t frames_in_flight,
    std::vector<vk::DeviceSize> resource_region_bases,
    std::vector<vk::DeviceSize> sampler_region_bases,
    std::vector<vk::DescriptorSetAndBindingMappingEXT> mappings)
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
    resource_region_bases_ { std::move(resource_region_bases) },
    sampler_region_bases_ { std::move(sampler_region_bases) },
    mappings_ { std::move(mappings) }
  {}

  template<class PushLayout, class Layout, class DiagnosticsPolicy>
  [[nodiscard]] static auto
  create(const vk::raii::Device& device,
    const vk::raii::PhysicalDevice& physical, vma_policy& allocator,
    const selected_device_capabilities& capabilities,
    const descriptor_heap_arena_create_info& create_info,
    [[maybe_unused]] diagnostic_buffer& diagnostics)
    -> std::expected<descriptor_heap_arena, error_t>
    requires(PushLayout::field_count >= 1UZ && Layout::set_count == 4UZ &&
      DiagnosticsPolicy::enabled)
  {
    return create_impl<PushLayout, Layout>(
      device, physical, allocator, capabilities, create_info);
  }

  template<class PushLayout, class Layout, class DiagnosticsPolicy>
  [[nodiscard]] static auto
  create(const vk::raii::Device& device,
    const vk::raii::PhysicalDevice& physical, vma_policy& allocator,
    const selected_device_capabilities& capabilities,
    const descriptor_heap_arena_create_info& create_info)
    -> std::expected<descriptor_heap_arena, error_t>
    requires(PushLayout::field_count >= 1UZ && Layout::set_count == 4UZ &&
      !DiagnosticsPolicy::enabled)
  {
    return create_impl<PushLayout, Layout>(
      device, physical, allocator, capabilities, create_info);
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

  template<class Layout, class Tag>
  [[nodiscard]] auto
  write_bindless_cis(const vk::raii::Device& device, std::uint32_t index,
    const vk::SamplerCreateInfo& sampler_info,
    const vk::ImageViewCreateInfo& view_info,
    vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal)
    -> std::expected<void, error_t>
  {
    static_assert(
      std::tuple_element_t<Layout::template resource_index_of<Tag>(),
        typename Layout::scoped_decls>::declaration::kind ==
        resource_declaration_kind::cis_table,
      "write_bindless_cis: Tag is not cis_table");
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
      .address = sampler_base + sampler_region_base_of<Layout, Tag>() +
        index * sampler_descriptor_size_,
      .size = static_cast<std::uint32_t>(sampler_descriptor_size_),
    };
    const vk::HostAddressRangeEXT resource_range {
      .address = resource_base + region_base_of<Layout, Tag>() +
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

  template<class Layout, class Tag, buffer_kind Kind, class T>
  [[nodiscard]] auto
  write_uniform(const vk::raii::Device& device, std::uint32_t frame_index,
    const buffer_view<Kind, T, addressed>& view) -> std::expected<void, error_t>
  {
    static_assert(
      std::tuple_element_t<Layout::template resource_index_of<Tag>(),
        typename Layout::scoped_decls>::declaration::kind ==
        resource_declaration_kind::dynamic_uniform,
      "write_uniform: Tag is not dynamic_uniform");
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
    const vk::DeviceSize base =
      region_base_of<Layout, Tag>() + frame_index * uniform_descriptor_size_;
    const vk::HostAddressRangeEXT host_range {
      .address = resource_base + base,
      .size = static_cast<std::uint32_t>(uniform_descriptor_size_),
    };
    return map_vk_error(
      device.writeResourceDescriptorsEXT({ resource_info }, { host_range }));
  }

  template<class Layout, class Tag, buffer_kind Kind, class T>
  [[nodiscard]] auto
  write_storage(const vk::raii::Device& device,
    const buffer_view<Kind, T, addressed>& view) -> std::expected<void, error_t>
  {
    static_assert(
      std::is_same_v<
        typename std::tuple_element_t<Layout::template resource_index_of<Tag>(),
          typename Layout::scoped_decls>::declaration::tag,
        Tag> &&
        std::tuple_element_t<Layout::template resource_index_of<Tag>(),
          typename Layout::scoped_decls>::declaration::kind ==
          resource_declaration_kind::storage,
      "write_storage: Tag/kind/T mismatch");
    const vk::DeviceSize base = region_base_of<Layout, Tag>();
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

  template<class Layout, class Tag>
  [[nodiscard]] auto
  write_cis(const vk::raii::Device& device,
    const vk::SamplerCreateInfo& sampler_info,
    const vk::ImageViewCreateInfo& view_info,
    vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal)
    -> std::expected<void, error_t>
  {
    static_assert(
      std::tuple_element_t<Layout::template resource_index_of<Tag>(),
        typename Layout::scoped_decls>::declaration::kind ==
        resource_declaration_kind::cis,
      "write_cis: Tag is not cis");
    const vk::DeviceSize sampler_base_off =
      sampler_region_base_of<Layout, Tag>();
    const vk::DeviceSize resource_base_off = resource_base_of<Layout, Tag>();
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
  template<class PushLayout, class Layout>
  [[nodiscard]] static auto
  create_impl(const vk::raii::Device& device,
    const vk::raii::PhysicalDevice& physical, vma_policy& allocator,
    const selected_device_capabilities& capabilities,
    const descriptor_heap_arena_create_info& create_info)
    -> std::expected<descriptor_heap_arena, error_t>
    requires(PushLayout::field_count >= 1UZ && Layout::set_count == 4U)
  {
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

    constexpr auto table = Layout::assignment_table();
    constexpr std::size_t region_count = table.size();
    static_assert(region_count == Layout::resource_count());
    std::vector<vk::DeviceSize> resource_region_bases(region_count, ~0UZ);
    std::vector<vk::DeviceSize> sampler_region_bases(region_count, ~0UZ);
    std::vector<vk::DescriptorSetAndBindingMappingEXT> mappings(region_count);

    vk::DeviceSize resource_cursor {};
    vk::DeviceSize sampler_cursor {};
    for (auto index : std::views::indices(region_count))
    {
      const auto& row = table[ index ];
      const resource_declaration_kind kind = row.dynamic
        ? resource_declaration_kind::dynamic_uniform
        : row.update_after_bind ? resource_declaration_kind::cis_table
        : (row.descriptor_type == vk::DescriptorType::eStorageBuffer)
        ? resource_declaration_kind::storage
        : resource_declaration_kind::cis;
      const auto mask = spirv_resource_mask_for(kind);
      if (kind == resource_declaration_kind::dynamic_uniform)
      {
        resource_region_bases[ index ] = resource_cursor;
        const vk::DescriptorMappingSourcePushIndexEXT uniform_push {
          .heapOffset =
            static_cast<std::uint32_t>(resource_region_bases[ index ]),
          .pushOffset =
            PushLayout::template offset_of<typename PushLayout::frame_slot_tag>,
          .heapIndexStride =
            static_cast<std::uint32_t>(uniform_descriptor_size),
          .heapArrayStride =
            static_cast<std::uint32_t>(uniform_descriptor_size),
        };
        mappings[ index ] = vk::DescriptorSetAndBindingMappingEXT {
          .descriptorSet = row.set,
          .firstBinding = row.binding,
          .bindingCount = 1U,
          .resourceMask = mask,
          .source = vk::DescriptorMappingSourceEXT::eHeapWithPushIndex,
          .sourceData = { uniform_push },
        };
        resource_cursor = align_up(resource_cursor +
            uniform_descriptor_size * create_info.frames_in_flight,
          heap_props.bufferDescriptorAlignment);
      }
      else if (kind == resource_declaration_kind::storage)
      {
        resource_region_bases[ index ] = resource_cursor;
        const vk::DescriptorMappingSourceConstantOffsetEXT storage {
          .heapOffset =
            static_cast<std::uint32_t>(resource_region_bases[ index ]),
          .heapArrayStride =
            static_cast<std::uint32_t>(storage_descriptor_size),
        };
        mappings[ index ] = vk::DescriptorSetAndBindingMappingEXT {
          .descriptorSet = row.set,
          .firstBinding = row.binding,
          .bindingCount = 1U,
          .resourceMask = mask,
          .source = vk::DescriptorMappingSourceEXT::eHeapWithConstantOffset,
          .sourceData = { storage },
        };
        resource_cursor = align_up(resource_cursor + storage_descriptor_size,
          heap_props.bufferDescriptorAlignment);
      }
      else
      {
        resource_region_bases[ index ] = resource_cursor;
        sampler_region_bases[ index ] = sampler_cursor;
        const std::uint32_t cis_count =
          (kind == resource_declaration_kind::cis_table)
          ? create_info.bindless_capacity
          : row.count;
        const vk::DescriptorMappingSourceConstantOffsetEXT cis {
          .heapOffset =
            static_cast<std::uint32_t>(resource_region_bases[ index ]),
          .heapArrayStride = static_cast<std::uint32_t>(image_descriptor_size),
          .samplerHeapOffset =
            static_cast<std::uint32_t>(sampler_region_bases[ index ]),
          .samplerHeapArrayStride =
            static_cast<std::uint32_t>(sampler_descriptor_size),
        };
        mappings[ index ] = vk::DescriptorSetAndBindingMappingEXT {
          .descriptorSet = row.set,
          .firstBinding = row.binding,
          .bindingCount = 1U,
          .resourceMask = mask,
          .source = vk::DescriptorMappingSourceEXT::eHeapWithConstantOffset,
          .sourceData = { cis },
        };
        resource_cursor =
          align_up(resource_cursor + image_descriptor_size * cis_count,
            heap_props.imageDescriptorAlignment);
        sampler_cursor =
          align_up(sampler_cursor + sampler_descriptor_size * cis_count,
            heap_props.samplerDescriptorAlignment);
      }
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
                vma_policy::buffer_handle&& resource_heap)
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
                  std::move(resource_region_bases),
                  std::move(sampler_region_bases),
                  std::move(mappings),
                };
              });
        });
  }

  template<class Layout, class Tag>
  [[nodiscard]] auto
  region_base_of() const -> vk::DeviceSize
  {
    constexpr std::size_t idx = Layout::template resource_index_of<Tag>();
    return resource_region_bases_[ idx ];
  }

  template<class Layout, class Tag>
  [[nodiscard]] auto
  sampler_region_base_of() const -> vk::DeviceSize
  {
    constexpr std::size_t idx = Layout::template resource_index_of<Tag>();
    return sampler_region_bases_[ idx ];
  }

  [[nodiscard]] static constexpr auto
  spirv_resource_mask_for(resource_declaration_kind kind)
    -> vk::SpirvResourceTypeFlagBitsEXT
  {
    switch (kind)
    {
    case resource_declaration_kind::storage:
      return vk::SpirvResourceTypeFlagBitsEXT::eReadOnlyStorageBuffer;
    case resource_declaration_kind::dynamic_uniform:
      return vk::SpirvResourceTypeFlagBitsEXT::eUniformBuffer;
    case resource_declaration_kind::cis:
    case resource_declaration_kind::cis_table:
      return vk::SpirvResourceTypeFlagBitsEXT::eCombinedSampledImage;
    }
    std::unreachable();
  }

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
  std::vector<vk::DeviceSize> resource_region_bases_ {};
  std::vector<vk::DeviceSize> sampler_region_bases_ {};
  std::vector<vk::DescriptorSetAndBindingMappingEXT> mappings_ {};
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

export using classic_bindless_table =
  bindless_table<descriptor_table_backend::classic>;

template<>
class bindless_table<descriptor_table_backend::heap>
{
public:
  bindless_table() = default;

  explicit bindless_table(descriptor_heap_arena& arena) : arena_ { arena } {};

  template<class Layout, class Tag>
  [[nodiscard]] auto
  register_combined_image_sampler(const vk::raii::Device& device,
    const vk::SamplerCreateInfo& sampler_create_info,
    const vk::ImageViewCreateInfo& image_view_create_info)
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
      ->template write_bindless_cis<Layout, Tag>(
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
