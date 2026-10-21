export module vkpp.gpu;

import std;
import vulkan;
import fiat;

import vkpp.error;
import vkpp.diagnostics;
import vkpp.capabilities;
import vkpp.device;
import vkpp.buffer;
import vkpp.command;
import vkpp.command.record;
import vkpp.pipeline.compute;
export import vkpp.cstring_view;

namespace vkpp::gpu
{

export template<fiat::scalar Element, class Operation>
  requires fiat::traceable<Operation, Element>
class transform_kernel
{
public:
  static constexpr auto words =
    fiat::compile<Element>(fiat::transform(Operation {}));

  [[nodiscard]]
  static
  auto
  create(
    const device_context&             device,
    const vk::raii::PipelineCache&    cache       = {nullptr},
    std::uint32_t                     group_size  = 256U,
    std::optional<diagnostic_buffer&> diagnostics = {})
    -> std::expected<transform_kernel, error_t>
  {
    const auto& enabled = device.selected_capabilities().enabled;
    if (!enabled.maintenance4 || !enabled.buffer_device_address)
    {
      return std::unexpected {make_app_error(
        app_error_code::feature_not_enabled)};
    }

    const auto limits  = device.physical_device().getProperties().limits;
    const auto clamped = std::clamp(
      group_size,
      1U,
      std::min(
        limits.maxComputeWorkGroupSize[ 0 ],
        limits.maxComputeWorkGroupInvocations));
    const std::array specialization {
      specialization_constant {
        .id    = fiat::group_size_constant_id,
        .value = clamped,
      },
    };

    return make_compute_pipeline(
      device.device(),
      compute_pipeline_runtime_args {
        .push_constant_size = sizeof(fiat::transform_push),
      },
      compute_shader {
        .words          = words,
        .entry          = cstring_view {fiat::transform_entry},
        .specialization = specialization,
      },
      cache,
      diagnostics)
      .transform([ & ](compute_pipeline&& pipeline) -> transform_kernel {
        return transform_kernel {
          device,
          std::move(pipeline),
          clamped,
          limits.maxComputeWorkGroupCount[ 0 ],
        };
      });
  }

  [[nodiscard]]
  auto
  record(
    vk::raii::CommandBuffer&     command_buffer,
    const device_address_buffer& input,
    const device_address_buffer& output,
    std::uint32_t                count) const -> std::expected<void, error_t>
  {
    const auto bytes = static_cast<vk::DeviceSize>(count) * sizeof(Element);
    if (count == 0U || input.size() < bytes || output.size() < bytes)
    {
      return std::unexpected {
        make_app_error(app_error_code::invalid_data_size),
      };
    }

    auto input_view =
      input.template addressed_view<Element>(device_, 0UZ, bytes);
    auto output_view =
      output.template addressed_view<Element>(device_, 0UZ, bytes);
    if (!input_view) { return std::unexpected {input_view.error()}; }
    if (!output_view) { return std::unexpected {output_view.error()}; }

    command_buffer.bindPipeline(
      vk::PipelineBindPoint::eCompute, *pipeline_.pipeline());

    const auto stride = static_cast<std::uint32_t>(std::min<std::uint64_t>(
      std::uint64_t {max_group_count_} * group_size_, count));
    for (
      const auto first :
      std::views::indices(count) | std::views::stride(stride))
    {
      const std::uint32_t elements = std::min(count - first, stride);
      push_compute<descriptor_table_backend::classic>(
        command_buffer,
        pipeline_,
        fiat::transform_push {
          .input  = input_view->device_address(),
          .output = output_view->device_address(),
          .count  = count,
          .first  = first,
        });
      dispatch(command_buffer, (elements + group_size_ - 1U) / group_size_);
    }
    return {};
  }

  [[nodiscard]]
  auto
  group_size() const -> std::uint32_t
  { return group_size_; }

private:
  transform_kernel(
    const device_context& device,
    compute_pipeline&&    pipeline,
    std::uint32_t         group_size,
    std::uint32_t         max_group_count)
  : device_ {device},
    pipeline_ {std::move(pipeline)},
    group_size_ {group_size},
    max_group_count_ {max_group_count}
  {}

  const device_context& device_;
  compute_pipeline      pipeline_;
  std::uint32_t         group_size_ {};
  std::uint32_t         max_group_count_ {};
};

export
template<fiat::scalar Element, class Operation>
  requires fiat::traceable<Operation, Element>
[[nodiscard]]
auto
transform(
  const device_context&        device,
  command_pool&                pool,
  const device_address_buffer& input,
  const device_address_buffer& output,
  std::uint32_t                count,
  Operation,
  std::optional<diagnostic_buffer&> diagnostics = {})
  -> std::expected<void, error_t>
{
  auto kernel = transform_kernel<Element, Operation>::create(
    device, {nullptr}, 256U, diagnostics);
  if (!kernel) { return std::unexpected {kernel.error()}; }

  single_time_submit submit {pool, device.device(), device.graphics_queue()};
  return submit.begin()
    .and_then([ & ] -> std::expected<void, error_t> {
      return kernel->record(submit.command_buffer(), input, output, count);
    })
    .and_then([ & ] -> std::expected<void, error_t> {
      const vk::MemoryBarrier2 written {
        .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
      };
      submit.command_buffer().pipelineBarrier2(
        vk::DependencyInfo {
          .memoryBarrierCount = 1U,
          .pMemoryBarriers    = &written,
        });
      return submit.end_and_submit(upload::wait_idle);
    });
}

} // namespace vkpp::gpu
