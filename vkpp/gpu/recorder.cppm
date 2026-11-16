export module vkpp.gpu:recorder;

import std;
import vulkan;
import fiat;

import vkpp.error;
import vkpp.capabilities;
import vkpp.command;
import vkpp.command.record;
import vkpp.pipeline.compute;

namespace vkpp::gpu
{

export class context;

export template<fiat::scalar Element>
class device_array;

inline constexpr vk::MemoryBarrier2 compute_write_to_any_read {
  .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
  .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
  .dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands,
  .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
};

inline constexpr vk::MemoryBarrier2 compute_reuse {
  .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
  .srcAccessMask =
    vk::AccessFlagBits2::eShaderStorageRead |
    vk::AccessFlagBits2::eShaderStorageWrite,
  .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
  .dstAccessMask =
    vk::AccessFlagBits2::eShaderStorageRead |
    vk::AccessFlagBits2::eShaderStorageWrite,
};

inline constexpr vk::MemoryBarrier2 compute_write_to_copy_read {
  .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
  .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
  .dstStageMask  = vk::PipelineStageFlagBits2::eCopy,
  .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
};

inline constexpr vk::MemoryBarrier2 copy_write_to_compute {
  .srcStageMask  = vk::PipelineStageFlagBits2::eCopy,
  .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
  .dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
  .dstAccessMask =
    vk::AccessFlagBits2::eShaderStorageRead |
    vk::AccessFlagBits2::eShaderStorageWrite,
};

inline constexpr vk::MemoryBarrier2 transfer_write_to_compute {
  .srcStageMask  = vk::PipelineStageFlagBits2::eAllTransfer,
  .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
  .dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
  .dstAccessMask =
    vk::AccessFlagBits2::eShaderStorageRead |
    vk::AccessFlagBits2::eShaderStorageWrite,
};

inline constexpr vk::MemoryBarrier2 copy_write_to_host_read {
  .srcStageMask  = vk::PipelineStageFlagBits2::eCopy,
  .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
  .dstStageMask  = vk::PipelineStageFlagBits2::eHost,
  .dstAccessMask = vk::AccessFlagBits2::eHostRead,
};

inline
void
record_barrier(
  vk::raii::CommandBuffer& command_buffer, const vk::MemoryBarrier2& barrier)
{
  command_buffer.pipelineBarrier2({
    .memoryBarrierCount = 1U,
    .pMemoryBarriers    = &barrier,
  });
}

template<
  std::same_as<context>                    Context,
  std::invocable<vk::raii::CommandBuffer&> Record
>
  requires std::same_as<
    std::invoke_result_t<Record&, vk::raii::CommandBuffer&>,
    std::expected<void, error_t>
  >
[[nodiscard]]
auto
submit_and_wait(Context& context, Record&& record)
  -> std::expected<void, error_t>
{
  single_time_submit submit {
    context.pool(),
    context.device().device(),
    context.device().graphics_queue(),
  };
  return submit.begin()
    .and_then([ & ] -> std::expected<void, error_t> {
      return std::invoke(record, submit.command_buffer());
    })
    .and_then([ & ] -> std::expected<void, error_t> {
      return submit.end_and_submit(upload::wait_idle);
    });
}

template<std::same_as<context> Context, std::invocable<std::uint32_t> MakePush>
void
record_chunks(
  vk::raii::CommandBuffer& command_buffer,
  const Context&           context,
  const compute_pipeline&  compute,
  std::uint32_t            count,
  MakePush&&               make_push)
{
  command_buffer.bindPipeline(
    vk::PipelineBindPoint::eCompute, *compute.pipeline());
  const std::uint32_t group_size = context.group_size();
  const auto          stride     = static_cast<std::uint32_t>(std::min(
    std::uint64_t {context.max_group_count()} * group_size,
    std::uint64_t {count}));
  for (
    const auto first : std::views::indices(count) | std::views::stride(stride))
  {
    const auto first_element = static_cast<std::uint32_t>(first);
    const auto elements      = std::min(count - first_element, stride);
    push_compute<descriptor_table_backend::classic>(
      command_buffer, compute, make_push(first_element));
    dispatch(command_buffer, (elements + group_size - 1U) / group_size);
  }
}

[[nodiscard]]
inline
auto
check_distinct_addresses(std::span<const std::uint64_t> addresses)
  -> std::expected<void, error_t>
{
  for (const auto position : std::views::indices(addresses.size()))
  {
    if (
      std::ranges::contains(
        addresses.subspan(position + 1UZ), addresses[ position ]))
    {
      return std::unexpected {
        make_app_error(app_error_code::invalid_data_size),
      };
    }
  }
  return {};
}

template<class Ports>
struct port_traits;

template<class... Ports>
struct port_traits<std::tuple<Ports...>>
{
  using elements = std::tuple<typename Ports::element_type...>;

  static constexpr std::array accesses {Ports::access...};
};

template<fiat::scalar Element, class Stage>
using stage_ports = port_traits<typename Stage::template ports<Element>>;

using dispatch_words = std::array<
  std::uint32_t,
  fiat::skeleton_push_bytes / fiat::dispatch_push::word_bytes
>;

template<fiat::scalar Element, fiat::archetype_stage<Element> Stage>
using port_addresses =
  std::array<std::uint64_t, fiat::skeleton_push_of<Stage, Element>.ports>;

constexpr
auto
word_at(std::uint32_t offset) -> std::size_t
{ return offset / fiat::dispatch_push::word_bytes; }

template<fiat::scalar Element, fiat::archetype_stage<Element> Stage>
[[nodiscard]]
constexpr
auto
fill_dispatch_push(
  const Stage&                          stage,
  const port_addresses<Element, Stage>& addresses,
  std::uint32_t                         count) -> dispatch_words
{
  constexpr auto layout = fiat::skeleton_push_of<Stage, Element>;
  dispatch_words words {};
  for (const auto port_index : std::views::indices(layout.ports))
  {
    const auto address  = addresses[ port_index ];
    const auto word     = word_at(layout.address_offset(port_index));
    words[ word ]       = static_cast<std::uint32_t>(address);
    words[ word + 1UZ ] = static_cast<std::uint32_t>(address >> 32U);
  }
  words[ word_at(layout.count_offset()) ] = count;
  std::apply(
    [ & ]<class... Arguments>(const Arguments&... argument) {
      const std::array<std::uint32_t, sizeof...(Arguments)> packed {
        std::bit_cast<std::uint32_t>(argument)...,
      };
      std::ranges::copy(
        packed, words.begin() + word_at(layout.argument_offset()));
    },
    stage.arguments);
  return words;
}

template<
  fiat::scalar                   Element,
  fiat::archetype_stage<Element> Stage,
  std::same_as<context>          Context,
  fiat::scalar... PortElements
>
  requires std::same_as<
    std::tuple<PortElements...>,
    typename stage_ports<Element, Stage>::elements
  >
[[nodiscard]]
auto
record_kernel(
  vk::raii::CommandBuffer& command_buffer,
  Context&                 context,
  const Stage&             stage,
  const device_array<PortElements>&... ports) -> std::expected<void, error_t>
{
  static_assert(
    Stage::dispatches.size() == 1UZ &&
      Stage::dispatches.front().extent == fiat::dispatch_extent::per_element,
    "record_kernel records one per_element dispatch");

  const std::array    sizes {ports.size()...};
  const std::uint32_t count = sizes.front();
  if (std::ranges::any_of(sizes, [ & ](std::uint32_t size) -> bool {
        return size != count;
      }))
  {
    return std::unexpected {make_app_error(app_error_code::invalid_data_size)};
  }

  const std::array bound {
    ports.addressed_view(context.device())
      .transform([](const auto& view) -> std::uint64_t {
        return view.device_address();
      })...,
  };
  port_addresses<Element, Stage> addresses {};
  port_addresses<Element, Stage> written {};
  std::size_t                    written_count {};
  for (const auto port_index : std::views::indices(addresses.size()))
  {
    if (!bound[ port_index ])
    {
      return std::unexpected {bound[ port_index ].error()};
    }
    addresses[ port_index ] = *bound[ port_index ];
    if (
      stage_ports<Element, Stage>::accesses[ port_index ] !=
      fiat::port_access::read)
    {
      written[ written_count++ ] = addresses[ port_index ];
    }
  }
  if (
    auto distinct =
      check_distinct_addresses(std::span {written}.first(written_count));
    !distinct)
  {
    return distinct;
  }

  auto kernel = context.template kernel<Element, Stage>();
  if (!kernel) { return std::unexpected {kernel.error()}; }

  constexpr auto layout = fiat::skeleton_push_of<Stage, Element>;
  auto           words  = fill_dispatch_push<Element>(stage, addresses, count);
  record_chunks(
    command_buffer, context, kernel->get(), count, [ & ](std::uint32_t first) {
      words[ word_at(layout.first_offset()) ] = first;
      return words;
    });
  return {};
}

} // namespace vkpp::gpu
