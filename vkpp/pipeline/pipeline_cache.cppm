module;

#include "error/vk_error_config.hpp"

export module vkpp.pipeline.cache;

import std;
import vulkan;

import vkpp.error;

namespace vkpp
{
using namespace std::string_view_literals;

export [[nodiscard]] auto
pipeline_cache_header_matches(std::span<const std::byte> blob,
  const vk::PhysicalDeviceProperties& properties) -> bool
{
  if (blob.size() < sizeof(vk::PipelineCacheHeaderVersionOne)) { return false; }
  vk::PipelineCacheHeaderVersionOne header {};
  std::memcpy(&header, blob.data(), sizeof(header));
  if (header.headerSize < sizeof(vk::PipelineCacheHeaderVersionOne))
  {
    return false;
  }
  if (header.headerVersion != vk::PipelineCacheHeaderVersion::eOne)
  {
    return false;
  }
  if (header.vendorID != properties.vendorID ||
    header.deviceID != properties.deviceID)
  {
    return false;
  }
  return std::ranges::equal(
    header.pipelineCacheUUID, properties.pipelineCacheUUID);
}

export [[nodiscard]] auto
load_pipeline_cache_file(const std::filesystem::path& path)
  -> std::expected<std::vector<std::byte>, error_t>
{
  if (!std::filesystem::exists(path)) { return {}; }
  std::ifstream input { path, std::ios::binary };
  if (!input)
  {
    return std::unexpected {
      app_error {
        .kind = app_error_kind::file_open,
        .detail = "load_pipeline_cache_file: open failed"sv,
      },
    };
  }

  input.seekg(0, std::ios::end);
  const auto end = input.tellg();
  input.seekg(0, std::ios::beg);
  if (end <= 0) { return {}; }

  std::vector<std::byte> bytes(static_cast<std::size_t>(end));
  input.read(reinterpret_cast<char*>(bytes.data()), end);
  if (!input)
  {
    return std::unexpected {
      app_error {
        .kind = app_error_kind::file_open,
        .detail = "load_pipeline_cache_file: read failed"sv,
      },
    };
  }

  return bytes;
}

export [[nodiscard]] auto
save_pipeline_cache_file(const std::filesystem::path& path,
  std::span<const std::byte> bytes) -> std::expected<void, error_t>
{
  std::ofstream output { path, std::ios::binary | std::ios::trunc };
  if (!output)
  {
    return std::unexpected {
      app_error {
        .kind = app_error_kind::file_open,
        .detail = "load_pipeline_cache_file: open failed"sv,
      },
    };
  }

  output.write(reinterpret_cast<const char*>(bytes.data()),
    static_cast<std::streamsize>(bytes.size()));
  if (!output)
  {
    return std::unexpected {
      app_error {
        .kind = app_error_kind::file_open,
        .detail = "load_pipeline_cache_file: write failed"sv,
      },
    };
  }

  return {};
}

export class pipeline_cache
{
public:
  pipeline_cache() = default;

  explicit pipeline_cache(vk::raii::PipelineCache&& cache)
  : cache_ { std::move(cache) }
  {}

  [[nodiscard]] static auto
  create(const vk::raii::Device& device)
    -> std::expected<pipeline_cache, error_t>
  { return create(device, {}); }

  [[nodiscard]] static auto
  create(const vk::raii::Device& device, std::span<const std::byte> initial)
    -> std::expected<pipeline_cache, error_t>
  {
    const vk::PipelineCacheCreateInfo create_info {
      .initialDataSize = initial.size(),
      .pInitialData = initial.data(),
    };
    return UTILS_VK(device.createPipelineCache(create_info),
      ^^vk::raii::Device::createPipelineCache)
      .transform([](vk::raii::PipelineCache&& cache) -> pipeline_cache
        { return pipeline_cache { std::move(cache) }; });
  }

  [[nodiscard]] auto
  get() const -> const vk::raii::PipelineCache&
  { return cache_; }

  [[nodiscard]] auto
  native_handle() const -> vk::PipelineCache
  { return get(); }

  [[nodiscard]] auto
  data() const -> std::expected<std::vector<std::uint8_t>, error_t>
  { return UTILS_VK(cache_.getData(), ^^vk::raii::PipelineCache::getData); }

private:
  vk::raii::PipelineCache cache_ { nullptr };
};

} // namespace vkpp
