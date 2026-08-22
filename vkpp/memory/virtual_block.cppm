module;

#include <vma/vk_mem_alloc.h>

export module vkpp.memory.virtual_block;

import std;
import vulkan;

import vkpp.error;

namespace vkpp
{
using namespace std::string_view_literals;

export struct virtual_slice
{
  VmaVirtualAllocation allocation { VK_NULL_HANDLE };
  vk::DeviceSize offset { 0UZ };
  vk::DeviceSize size { 0UZ };
};

export class virtual_block
{
public:
  virtual_block() = default;

  virtual_block(const virtual_block&) = delete;

  auto
  operator=(const virtual_block&) -> virtual_block& = delete;

  virtual_block(virtual_block&& other) noexcept
  : block_ { std::exchange(other.block_, nullptr) }
  {}

  auto
  operator=(virtual_block&& other) noexcept -> virtual_block&
  {
    if (this != &other)
    {
      destroy();
      block_ = std::exchange(other.block_, nullptr);
    }
    return *this;
  };

  ~virtual_block() { destroy(); }

  [[nodiscard]] static auto
  create(vk::DeviceSize size) -> std::expected<virtual_block, error_t>
  {
    const VmaVirtualBlockCreateInfo create_info { .size = size };
    virtual_block out {};

    if (vmaCreateVirtualBlock(&create_info, &out.block_) != VK_SUCCESS)
    {
      return std::unexpected {
        app_error {
          .kind = app_error_kind::invalid_argument,
          .detail = "vmaCreateVirtualBlock failed"sv,
        },
      };
    }
    return out;
  }

  [[nodiscard]] auto
  allocate(vk::DeviceSize size, vk::DeviceSize alignment = 0UZ)
    -> std::expected<virtual_slice, error_t>
  {
    const VmaVirtualAllocationCreateInfo allocation_info {
      .size = size,
      .alignment = alignment,
    };
    virtual_slice slice { .size = size };

    if (vmaVirtualAllocate(block_, &allocation_info, &slice.allocation,
          &slice.offset) != VK_SUCCESS)
    {
      return std::unexpected {
        app_error {
          .kind = app_error_kind::arena_exhausted,
          .detail = "virtual_block_is_full"sv,
        },
      };
    }
    return slice;
  }

  void
  free(virtual_slice& slice) noexcept
  {
    vmaVirtualFree(block_, slice.allocation);
    slice = {};
  }

private:
  void
  destroy() noexcept
  {
    if (block_ != nullptr)
    {
      vmaClearVirtualBlock(block_);
      vmaDestroyVirtualBlock(block_);
      block_ = nullptr;
    }
  }

  VmaVirtualBlock block_ { VK_NULL_HANDLE };
};

} // namespace vkpp
