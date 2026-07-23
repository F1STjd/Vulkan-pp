module;
#include "error/vk_error_config.hpp"
export module vkpp.memory.raw;

import std;
import vulkan;
import vkpp.error;
import vkpp.memory;

namespace vkpp
{
using namespace std::string_view_literals;

export struct raw_image_handle
{
  vk::raii::DeviceMemory memory { nullptr };
  vk::raii::Image image { nullptr };

  [[nodiscard]]
  auto
  get() const -> vk::Image
  { return *image; }
};

export struct raw_buffer_handle
{
  vk::raii::DeviceMemory memory { nullptr };
  vk::raii::Buffer buffer { nullptr };
  void* mapped_p {};

  [[nodiscard]]
  auto
  get() const -> vk::Buffer
  { return *buffer; }

  [[nodiscard]]
  auto
  mapped() const -> void*
  { return mapped_p; }
};

export class raw_policy
{
public:
  using image_handle = raw_image_handle;
  using buffer_handle = raw_buffer_handle;

  raw_policy() = default;
  raw_policy(const vk::raii::PhysicalDevice& physical_device,
    const vk::raii::Device& device)
  : physical_device_ { &physical_device }, device_ { &device }
  {}

  [[nodiscard]]
  auto
  create_image(const vk::ImageCreateInfo& image_info,
    memory_intent intent) const -> std::expected<image_handle, error_t>
  {
    image_handle handle {};
    vk::DeviceSize memory_size {};

    return UTILS_VK(
      device_->createImage(image_info), ^^vk::raii::Device::createImage)
      .and_then(
        [ & ](vk::raii::Image&& image)
        {
          handle.image = std::move(image);
          const auto requirements = handle.image.getMemoryRequirements();
          memory_size = requirements.size;
          return find_memory_type(
            requirements.memoryTypeBits, to_properties(intent));
        })
      .and_then(
        [ & ](std::uint32_t memory_type)
        {
          const vk::MemoryAllocateInfo allocate_info {
            .allocationSize = memory_size,
            .memoryTypeIndex = memory_type,
          };
          return UTILS_VK(device_->allocateMemory(allocate_info),
            ^^vk::raii::Device::allocateMemory);
        })
      .and_then(
        [ & ](vk::raii::DeviceMemory&& memory)
        {
          handle.memory = std::move(memory);
          return UTILS_VK(handle.image.bindMemory(*handle.memory, 0ULL),
            ^^vk::raii::Image::bindMemory);
        })
      .transform([ & ] { return std::move(handle); });
  }

  [[nodiscard]]
  auto
  create_buffer(const vk::BufferCreateInfo& buffer_info,
    memory_intent intent) const -> std::expected<buffer_handle, error_t>
  {
    buffer_handle handle {};
    vk::DeviceSize memory_size {};

    return UTILS_VK(
      device_->createBuffer(buffer_info), ^^vk::raii::Device::createBuffer)
      .and_then(
        [ & ](vk::raii::Buffer&& buffer)
        {
          handle.buffer = std::move(buffer);
          const auto requirements = handle.buffer.getMemoryRequirements();
          memory_size = requirements.size;
          return find_memory_type(
            requirements.memoryTypeBits, to_properties(intent));
        })
      .and_then(
        [ & ](std::uint32_t memory_type)
        {
          const vk::MemoryAllocateInfo allocate_info {
            .allocationSize = memory_size,
            .memoryTypeIndex = memory_type,
          };
          return UTILS_VK(device_->allocateMemory(allocate_info),
            ^^vk::raii::Device::allocateMemory);
        })
      .and_then(
        [ & ](vk::raii::DeviceMemory&& memory)
        {
          handle.memory = std::move(memory);
          return UTILS_VK(handle.buffer.bindMemory(*handle.memory, 0ULL),
            ^^vk::raii::Buffer::bindMemory);
        })
      .and_then(
        [ & ] -> std::expected<void, error_t>
        {
          if (intent != memory_intent::gpu_only)
          {
            return UTILS_VK(handle.memory.mapMemory(0ULL, vk::WholeSize),
              ^^vk::raii::DeviceMemory::mapMemory)
              .transform([ & ](void* mapped) { handle.mapped_p = mapped; });
          }
          return {};
        })
      .transform([ & ] { return std::move(handle); });
  }

private:
  [[nodiscard]]
  static constexpr auto
  to_properties(memory_intent intent) -> vk::MemoryPropertyFlags
  {
    switch (intent)
    {
    case memory_intent::gpu_only:
      return vk::MemoryPropertyFlagBits::eDeviceLocal;
    case memory_intent::cpu_to_gpu:
    case memory_intent::gpu_to_cpu:
      return vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent;
    }
    std::unreachable();
  }

  auto
  find_memory_type(
    std::uint32_t type_filter, vk::MemoryPropertyFlags properties) const
    -> std::expected<std::uint32_t, vkpp::error_t>
  {
    const auto available_properties = physical_device_->getMemoryProperties();
    const auto memory_types =
      std::views::iota(0U, available_properties.memoryTypeCount);
    auto memory_type_it = std::ranges::find_if(memory_types,
      [ type_filter, properties, &available_properties ](
        std::uint32_t memory_type) -> bool
      {
        return (type_filter & (1U << memory_type)) &&
          (available_properties.memoryTypes[ memory_type ].propertyFlags &
            properties) == properties;
      });
    if (memory_type_it == memory_types.end())
    {
      return std::unexpected {
        vkpp::app_error {
          .kind = vkpp::app_error_kind::no_memory_type,
          .detail = "Failed to find suitable memory type"sv,
        },
      };
    }

    return *memory_type_it;
  }

  // Pointers nor references, so the policy stays movable
  const vk::raii::PhysicalDevice* physical_device_ {};
  const vk::raii::Device* device_ {};
};

static_assert(device_allocator<raw_policy>);

}; // namespace vkpp
