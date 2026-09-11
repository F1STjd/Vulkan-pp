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
              app_error {
                .kind = app_error_kind::mapping_failed,
                .detail = "stage_pool: staging map returned nullptr"sv,
              },
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
          if (auto* app = std::get_if<app_error>(&err);
            app != nullptr && app->kind == app_error_kind::arena_exhausted)
          {
            return std::unexpected {
              app_error {
                .kind = app_error_kind::arena_exhausted,
                .detail = "stage_pool exhausted"sv,
              },
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
