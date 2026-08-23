module;

export module vkpp.buffer.arena;

import std;
import vulkan;

export import vkpp.buffer;
import vkpp.memory;
import vkpp.memory.vma;
import vkpp.memory.virtual_block;
import vkpp.error;

namespace vkpp
{

export template<device_allocator Alloc = vma_policy>
class buffer_arena
{
public:
  buffer_arena() = default;

  buffer_arena(buffer_resource<Alloc>&& buffer, virtual_block&& block)
  : buffer_ { std::move(buffer) }, block_ { std::move(block) }
  {}

  [[nodiscard]] auto
  buffer() -> vk::Buffer
  { return buffer_.buffer(); }

  [[nodiscard]] auto
  allocate(vk::DeviceSize size, vk::DeviceSize alignment = 0UZ)
    -> std::expected<virtual_slice, error_t>
  { return block_.allocate(size, alignment); }

  void
  free(virtual_slice& slice) noexcept
  { block_.free(slice); }

private:
  buffer_resource<Alloc> buffer_ {};
  virtual_block block_;
};

export template<device_allocator Alloc = vma_policy>
[[nodiscard]] auto
make_buffer_arena(Alloc& allocator, vk::DeviceSize size,
  vk::BufferUsageFlags usgae, memory_intent intent)
  -> std::expected<buffer_arena<Alloc>, error_t>
{
  return make_buffer_resource(
    allocator, size, usgae | vk::BufferUsageFlagBits::eTransferDst, intent)
    .and_then(
      [ & ](vkpp::buffer_resource<Alloc>&& buffer)
        -> std::expected<buffer_arena<Alloc>, error_t>
      {
        return virtual_block::create(size).transform(
          [ & ](virtual_block&& block) -> buffer_arena<Alloc>
          { return { std::move(buffer), std::move(block) }; });
      });
}

} // namespace vkpp
