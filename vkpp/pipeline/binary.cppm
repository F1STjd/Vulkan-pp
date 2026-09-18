module;

#include "error/vk_error_config.hpp"

export module vkpp.pipeline.binary;

import std;
import vulkan;

import vkpp.error;

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
    return UTILS_VK(device.createPipelineBinariesKHR(create_info),
      ^^vk::raii::Device::createPipelineBinariesKHR)
      .and_then(
        [ & ](std::vector<vk::raii::PipelineBinaryKHR>&& binaries)
          -> std::expected<pipeline_binaries, error_t>
        {
          const vk::ReleaseCapturedPipelineDataInfoKHR release_info {
            .pipeline = pipeline,
          };
          return UTILS_VK(device.releaseCapturedPipelineDataKHR(release_info),
            ^^vk::raii::Device::releaseCapturedPipelineDataKHR)
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
        app_error {
          .kind = app_error_kind::invalid_argument,
          .detail = "pipeline_binaries::create: empty blobs"sv,
        },
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

    return UTILS_VK(device.createPipelineBinariesKHR(create_info),
      ^^vk::raii::Device::createPipelineBinariesKHR)
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
      auto got = UTILS_VK(device.getPipelineBinaryDataKHR(info),
        ^^vk::raii::Device::getPipelineBinaryDataKHR);
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

export [[nodiscard]] auto
save_pipeline_binary_file(const std::filesystem::path& path,
  std::span<const pipeline_binary_blob> blobs) -> std::expected<void, error_t>
{
  std::ofstream output { path, std::ios::binary | std::ios::trunc };
  if (!output)
  {
    return std::unexpected {
      app_error {
        .kind = app_error_kind::file_open,
        .detail = "save_pipeline_binary_file: open failed"sv,
      },
    };
  }

  const auto count = static_cast<std::uint32_t>(blobs.size());
  output.write(reinterpret_cast<const char*>(&count), sizeof(count));
  for (const auto& blob : blobs)
  {
    const auto key_size = blob.key.keySize;
    output.write(reinterpret_cast<const char*>(&key_size), sizeof(key_size));
    output.write(reinterpret_cast<const char*>(blob.key.key.data()),
      vk::MaxPipelineBinaryKeySizeKHR);
    const auto data_size = blob.data.size();
    output.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
    if (data_size != 0U)
    {
      output.write(reinterpret_cast<const char*>(blob.data.data()),
        static_cast<std::streamsize>(data_size));
    }
  }
  if (!output)
  {
    return std::unexpected {
      app_error {
        .kind = app_error_kind::file_open,
        .detail = "save_pipeline_binary_file: write failed"sv,
      },
    };
  }
  return {};
}

export [[nodiscard]] auto
load_pipeline_binary_file(const std::filesystem::path& path)
  -> std::expected<std::vector<pipeline_binary_blob>, error_t>
{
  if (!std::filesystem::exists(path)) { return {}; }
  std::ifstream input { path, std::ios::binary };
  if (!input)
  {
    return std::unexpected {
      app_error {
        .kind = app_error_kind::file_open,
        .detail = "load_pipeline_binary_file: open failed"sv,
      },
    };
  }

  std::uint32_t count {};
  input.read(reinterpret_cast<char*>(&count), sizeof(count));
  if (!input) { return {}; }
  std::vector<pipeline_binary_blob> blobs(count);
  for (auto& blob : blobs)
  {
    std::uint32_t key_size {};
    input.read(reinterpret_cast<char*>(&key_size), sizeof(key_size));
    std::array<std::uint8_t, vk::MaxPipelineBinaryKeySizeKHR> key_bytes {};
    input.read(reinterpret_cast<char*>(key_bytes.data()),
      vk::MaxPipelineBinaryKeySizeKHR);
    blob.key = vk::PipelineBinaryKeyKHR {
      .keySize = key_size,
      .key = key_bytes,
    };
    std::uint64_t data_size {};
    input.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
    blob.data.resize(static_cast<std::size_t>(data_size));
    if (data_size != 0U)
    {
      input.read(reinterpret_cast<char*>(blob.data.data()),
        static_cast<std::streamsize>(data_size));
    }
    if (!input)
    {
      return std::unexpected {
        app_error {
          .kind = app_error_kind::file_open,
          .detail = "load_pipeline_binary_file: read failed"sv,
        },
      };
    }
  }
  return blobs;
}

} // namespace vkpp
