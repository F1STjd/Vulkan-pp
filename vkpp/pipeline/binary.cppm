export module vkpp.pipeline.binary;

import std;
import vulkan;

import vkpp.error;
import vkpp.diagnostics;

namespace vkpp
{
using namespace std::string_view_literals;

export struct pipeline_binary_blob
{
  vk::PipelineBinaryKeyKHR key {};
  std::vector<std::uint8_t> data {};
};

export class pipeline_binaries
{
public:
  pipeline_binaries() = default;

  explicit pipeline_binaries(
    std::vector<vk::raii::PipelineBinaryKHR>&& binaries)
  : binaries_ { std::move(binaries) }, handles_ { std::from_range,
      binaries_ |
        std::views::transform(
          [](const vk::raii::PipelineBinaryKHR& binary) -> vk::PipelineBinaryKHR
          { return *binary; }) }
  {}

  [[nodiscard]] static auto
  capture(const vk::raii::Device& device, vk::Pipeline pipeline)
    -> std::expected<pipeline_binaries, error_t>
  {
    const vk::PipelineBinaryCreateInfoKHR create_info {
      .pipeline = pipeline,
    };
    return map_vk_error(
      device.createPipelineBinariesKHR(create_info), std::nullopt)
      .and_then(
        [ & ](std::vector<vk::raii::PipelineBinaryKHR>&& binaries)
          -> std::expected<pipeline_binaries, error_t>
        {
          const vk::ReleaseCapturedPipelineDataInfoKHR release_info {
            .pipeline = pipeline,
          };
          return map_vk_error(
            device.releaseCapturedPipelineDataKHR(release_info), std::nullopt)
            .transform(
              [ binaries = std::move(binaries) ] mutable -> pipeline_binaries
              { return pipeline_binaries { std::move(binaries) }; });
        });
  }

  [[nodiscard]] static auto
  create(
    const vk::raii::Device& device, std::span<const pipeline_binary_blob> blobs)
    -> std::expected<pipeline_binaries, error_t>
  {
    if (blobs.empty())
    {
      return std::unexpected {
        make_app_error(app_error_code::missing_required_argument),
      };
    }

    std::vector<vk::PipelineBinaryKeyKHR> keys(blobs.size());
    std::vector<vk::PipelineBinaryDataKHR> data_infos(blobs.size());
    for (auto index : std::views::indices(blobs.size()))
    {
      keys[ index ] = blobs[ index ].key;
      data_infos[ index ] = vk::PipelineBinaryDataKHR {
        .dataSize = blobs[ index ].data.size(),
        .pData = const_cast<void*>(
          static_cast<const void*>(blobs[ index ].data.data())),
      };
    }
    const vk::PipelineBinaryKeysAndDataKHR keys_and_data {
      .binaryCount = static_cast<std::uint32_t>(blobs.size()),
      .pPipelineBinaryKeys = keys.data(),
      .pPipelineBinaryData = data_infos.data(),
    };
    const vk::PipelineBinaryCreateInfoKHR create_info {
      .pKeysAndDataInfo = &keys_and_data,
    };

    return map_vk_error(
      device.createPipelineBinariesKHR(create_info), std::nullopt)
      .transform([](std::vector<vk::raii::PipelineBinaryKHR>&& binaries)
                   -> pipeline_binaries
        { return pipeline_binaries { std::move(binaries) }; });
  }

  [[nodiscard]] auto
  extract_blobs(const vk::raii::Device& device)
    -> std::expected<std::vector<pipeline_binary_blob>, error_t>
  {
    std::vector<pipeline_binary_blob> blobs {};
    blobs.reserve(binaries_.size());
    for (const auto& binary : binaries_)
    {
      const vk::PipelineBinaryDataInfoKHR info {
        .pipelineBinary = *binary,
      };
      auto got =
        map_vk_error(device.getPipelineBinaryDataKHR(info), std::nullopt);
      if (!got) { return std::unexpected { std::move(got).error() }; }
      auto [ key, data ] = std::move(*got);
      blobs.push_back(pipeline_binary_blob {
        .key = key,
        .data = std::move(data),
      });
    }
    return blobs;
  }

  [[nodiscard]] auto
  info() const -> vk::PipelineBinaryInfoKHR
  {
    return vk::PipelineBinaryInfoKHR {
      .binaryCount = static_cast<std::uint32_t>(handles_.size()),
      .pPipelineBinaries = handles_.data(),
    };
  }

  [[nodiscard]] auto
  empty() const -> bool
  { return binaries_.empty(); }

private:
  std::vector<vk::raii::PipelineBinaryKHR> binaries_ {};
  std::vector<vk::PipelineBinaryKHR> handles_ {};
};

} // namespace vkpp
