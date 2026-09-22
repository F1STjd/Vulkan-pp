module;

#include "error/vk_error_config.hpp"

export module vkpp.buffer.stage_pool;

import std;
import vulkan;

export import vkpp.buffer;
import vkpp.memory;
import vkpp.memory.vma;
import vkpp.memory.virtual_block;
import vkpp.error;

namespace vkpp
{
using namespace std::string_view_literals;

export struct stage_allocation
{
  vk::DeviceSize offset {};
  vk::DeviceSize size {};
  void* mapped {};
  virtual_slice slice {};
};

export class stage_pool
{
public:
  stage_pool() = default;

  stage_pool(staging_buffer&& buffer, virtual_block&& block)
  : buffer_ { std::move(buffer) }, block_ { std::move(block) }
  {}

  template<device_allocator Alloc = vma_policy>
  [[nodiscard]] static auto
  create(Alloc& allocator, vk::DeviceSize capacity)
    -> std::expected<stage_pool, error_t>
  {
    return staging_buffer::create(allocator, capacity)
      .and_then(
        [ & ](staging_buffer&& buffer) -> std::expected<stage_pool, error_t>
        {
          if (buffer.mapped() == nullptr)
          {
            return std::unexpected {
              make_app_error(app_error_code::mapping_failed),
            };
          }
          return virtual_block::create(capacity).transform(
            [ buffer = std::move(buffer) ](
              virtual_block&& block) mutable -> stage_pool
            { return stage_pool { std::move(buffer), std::move(block) }; });
        });
  }

  [[nodiscard]] auto
  allocate(vk::DeviceSize size, vk::DeviceSize alignment)
    -> std::expected<stage_allocation, error_t>
  {
    return block_.allocate(size, alignment)
      .and_then(
        [ this, size ](
          virtual_slice&& slice) -> std::expected<stage_allocation, error_t>
        {
          return stage_allocation {
            .offset = slice.offset,
            .size = size,
            .mapped = static_cast<std::byte*>(buffer_.mapped()) + slice.offset,
            .slice = std::move(slice),
          };
        })
      .or_else(
        [](error_t&& err) -> std::expected<stage_allocation, error_t>
        {
          if (err.domain == error_domain::application &&
            err.code == std::to_underlying(app_error_code::capacity_exhausted))
          {
            return std::unexpected {
              make_app_error(app_error_code::capacity_exhausted),
            };
          }
          return std::unexpected { std::move(err) };
        });
  }

  void
  free(stage_allocation& allocation) noexcept
  {
    block_.free(allocation.slice);
    allocation = {};
  }

  [[nodiscard]] auto
  buffer() const -> vk::Buffer
  { return buffer_.buffer(); }

private:
  staging_buffer buffer_ {};
  virtual_block block_ {};
};

} // namespace vkpp
