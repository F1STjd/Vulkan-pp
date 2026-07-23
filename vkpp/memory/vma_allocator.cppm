module;
#include <vk_mem_alloc.h>
export module vkpp.memory.vma;

import std;
import vulkan;
import vkpp.error;
import vkpp.memory;

namespace vkpp
{

[[nodiscard]]
constexpr auto
vma_error(std::string_view function, VkResult result) -> error_t
{
  return vk_error {
    .function = function,
    .type = "vma",
    .result = static_cast<vk::Result>(result),
  };
}

export class gpu_image
{
public:
  gpu_image() = default;
  gpu_image(VmaAllocator allocator, vk::Image image, VmaAllocation allocation)
  : allocator_ { allocator }, image_ { image }, allocation_ { allocation }
  {}

  gpu_image(const gpu_image&) = delete (
    "There is no reason to copy initialise vulkan image\n"
    "Each image should be unique");
  auto
  operator=(const gpu_image&) -> gpu_image& = delete ( //
    "There is no reason to copy assign vulkan image\n"
    "Each image should be unique");

  gpu_image(gpu_image&& other) noexcept
  : allocator_ { std::exchange(other.allocator_, {}) },
    image_ { std::exchange(other.image_, {}) },
    allocation_ { std::exchange(other.allocation_, {}) }
  {}

  auto
  operator=(gpu_image&& other) noexcept -> gpu_image&
  {
    if (this != &other)
    {
      destroy();
      allocator_ = std::exchange(other.allocator_, {});
      image_ = std::exchange(other.image_, {});
      allocation_ = std::exchange(other.allocation_, {});
    }
    return *this;
  }

  ~gpu_image() { destroy(); }

  [[nodiscard]]
  auto
  get() const -> vk::Image
  { return image_; }

private:
  auto
  destroy() noexcept -> void
  {
    if (allocation_ != nullptr)
    {
      vmaDestroyImage(allocator_, static_cast<VkImage>(image_), allocation_);
    }
  }

  VmaAllocator allocator_ {};
  vk::Image image_ {};
  VmaAllocation allocation_ {};
};

export class gpu_buffer
{
public:
  gpu_buffer() = default;
  gpu_buffer(VmaAllocator allocator, vk::Buffer buffer,
    VmaAllocation allocation, void* mapped)
  : allocator_ { allocator }, buffer_ { buffer }, allocation_ { allocation },
    mapped_p { mapped }
  {}

  gpu_buffer(const gpu_buffer&) = delete (
    "There is no reason to copy initialise vulkan buffer\n"
    "Each buffer should be unique");
  auto
  operator=(const gpu_buffer&) -> gpu_buffer& = delete ( //
    "There is no reason to copy assign vulkan buffer\n"
    "Each buffer should be unique");

  gpu_buffer(gpu_buffer&& other) noexcept
  : allocator_ { std::exchange(other.allocator_, {}) },
    buffer_ { std::exchange(other.buffer_, {}) },
    allocation_ { std::exchange(other.allocation_, {}) },
    mapped_p { std::exchange(other.mapped_p, {}) }
  {}

  auto
  operator=(gpu_buffer&& other) noexcept -> gpu_buffer&
  {
    if (this != &other)
    {
      destroy();
      allocator_ = std::exchange(other.allocator_, {});
      buffer_ = std::exchange(other.buffer_, {});
      allocation_ = std::exchange(other.allocation_, {});
      mapped_p = std::exchange(other.mapped_p, {});
    }
    return *this;
  }

  ~gpu_buffer() { destroy(); }

  [[nodiscard]]
  auto
  get() const -> vk::Buffer
  { return buffer_; }

  [[nodiscard]]
  auto
  mapped() const -> void*
  { return mapped_p; }

private:
  auto
  destroy() noexcept -> void
  {
    if (allocation_ != nullptr)
    {
      vmaDestroyBuffer(allocator_, static_cast<VkBuffer>(buffer_), allocation_);
    }
  }

  VmaAllocator allocator_ {};
  vk::Buffer buffer_ {};
  VmaAllocation allocation_ {};
  void* mapped_p {};
};

export class vma_policy
{
public:
  using image_handle = gpu_image;
  using buffer_handle = gpu_buffer;

  vma_policy() = default;

  vma_policy(const vma_policy&) = delete (
    "There is no reason to copy initialise vma_policy");
  auto
  operator=(const vma_policy&)
    -> vma_policy& = delete ("There is no reason to copy assign vma_policy");

  vma_policy(vma_policy&& other) noexcept
  : allocator_ { std::exchange(other.allocator_, {}) }
  {}

  auto
  operator=(vma_policy&& other) noexcept -> vma_policy&
  {
    if (this != &other)
    {
      destroy();
      allocator_ = std::exchange(other.allocator_, {});
    }
    return *this;
  }

  ~vma_policy() { destroy(); }

  [[nodiscard]]
  static auto
  create(vk::Instance instance, vk::PhysicalDevice physical_device,
    vk::Device device, std::uint32_t api_version)
    -> std::expected<vma_policy, error_t>
  {
    const VmaAllocatorCreateInfo create_info {
      .physicalDevice = static_cast<VkPhysicalDevice>(physical_device),
      .device = static_cast<VkDevice>(device),
      .instance = static_cast<VkInstance>(instance),
      .vulkanApiVersion = api_version,
    };

    VmaAllocator allocator {};
    if (const auto result = vmaCreateAllocator(&create_info, &allocator);
      result != VK_SUCCESS)
    {
      return std::unexpected {
        vma_error("vmaCreateAllocator", result),
      };
    }
    return vma_policy { allocator };
  }

  [[nodiscard]]
  auto
  create_image(const vk::ImageCreateInfo& image_info, memory_intent intent)
    -> std::expected<image_handle, error_t>
  {
    const VkImageCreateInfo& c_image_info = image_info;
    const VmaAllocationCreateInfo allocation_info = to_allocation_info(intent);

    VkImage image {};
    VmaAllocation allocation {};
    if (const auto result = vmaCreateImage(allocator_, &c_image_info,
          &allocation_info, &image, &allocation, nullptr);
      result != VK_SUCCESS)
    {
      return std::unexpected {
        vma_error("vmaCreateImage", result),
      };
    }
    return image_handle {
      allocator_,
      vk::Image { image },
      allocation,
    };
  }

  [[nodiscard]]
  auto
  create_buffer(const vk::BufferCreateInfo& buffer_info, memory_intent intent)
    -> std::expected<buffer_handle, error_t>
  {
    const VkBufferCreateInfo& c_buffer_info = buffer_info;
    const VmaAllocationCreateInfo allocation_info = to_allocation_info(intent);

    VkBuffer buffer {};
    VmaAllocation allocation {};
    VmaAllocationInfo info {};
    if (const auto result = vmaCreateBuffer(allocator_, &c_buffer_info,
          &allocation_info, &buffer, &allocation, &info);
      result != VK_SUCCESS)
    {
      return std::unexpected {
        vma_error("vmaCreateBuffer", result),
      };
    }
    return buffer_handle {
      allocator_,
      vk::Buffer { buffer },
      allocation,
      info.pMappedData,
    };
  }

private:
  explicit vma_policy(VmaAllocator allocator) : allocator_ { allocator } {}

  auto
  destroy() noexcept -> void
  { vmaDestroyAllocator(allocator_); }

  [[nodiscard]]
  static constexpr auto
  to_allocation_info(memory_intent intent) -> VmaAllocationCreateInfo
  {
    switch (intent)
    {
    case memory_intent::gpu_only:
      return { .usage = VMA_MEMORY_USAGE_AUTO };
    case memory_intent::cpu_to_gpu:
      return {
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
          VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
      };
    case memory_intent::gpu_to_cpu:
      return {
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
          VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
      };
    }
    std::unreachable();
  }

  VmaAllocator allocator_ {};
};

static_assert(device_allocator<vma_policy>);

}; // namespace vkpp
