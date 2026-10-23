export module vkpp.pipeline.cache;

import std;
import vulkan;

import vkpp.error;
import vkpp.diagnostics;

namespace vkpp
{

export [[nodiscard]]
auto
unique_sibling_path(const std::filesystem::path& path, std::string_view suffix)
  -> std::filesystem::path
{
  auto       random = std::random_device {};
  const auto value  = (std::uint64_t {random()} << 32U) | random();
  auto       sibling {path};
  sibling += std::format(".{:016x}{}", value, suffix);
  return sibling;
}

export [[nodiscard]]
auto
replace_file(
  const std::filesystem::path& temporary_path,
  const std::filesystem::path& path) -> std::expected<void, error_t>
{
  const auto      backup_path = unique_sibling_path(path, ".bak");
  std::error_code error {};
  if (std::filesystem::exists(path, error))
  {
    std::filesystem::rename(path, backup_path, error);
    if (error)
    {
      std::filesystem::remove(temporary_path, error);
      return std::unexpected {make_app_error(app_error_code::file_write)};
    }
  }
  std::filesystem::rename(temporary_path, path, error);
  if (error)
  {
    std::error_code restore_error {};
    std::filesystem::rename(backup_path, path, restore_error);
    std::filesystem::remove(temporary_path, restore_error);
    return std::unexpected {make_app_error(app_error_code::file_write)};
  }
  std::filesystem::remove(backup_path, error);
  return {};
}

export [[nodiscard]]
auto
pipeline_cache_header_matches(
  std::span<const std::byte>          blob,
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
  if (
    header.vendorID != properties.vendorID ||
    header.deviceID != properties.deviceID)
  {
    return false;
  }
  return std::ranges::equal(
    header.pipelineCacheUUID, properties.pipelineCacheUUID);
}

export [[nodiscard]]
auto
load_pipeline_cache_file(const std::filesystem::path& path)
  -> std::expected<std::vector<std::byte>, error_t>
{
  if (!std::filesystem::exists(path)) { return {}; }
  std::ifstream input {path, std::ios::binary};
  if (!input)
  {
    return std::unexpected {make_app_error(app_error_code::file_open)};
  }

  input.seekg(0, std::ios::end);
  const auto end = input.tellg();
  input.seekg(0, std::ios::beg);
  if (end <= 0) { return {}; }

  std::vector<std::byte> bytes(static_cast<std::size_t>(end));
  input.read(reinterpret_cast<char*>(bytes.data()), end);
  if (!input)
  {
    return std::unexpected {make_app_error(app_error_code::file_read)};
  }
  return bytes;
}

export [[nodiscard]]
auto
save_pipeline_cache_file(
  const std::filesystem::path& path, std::span<const std::byte> bytes)
  -> std::expected<void, error_t>
{
  const auto temporary_path = unique_sibling_path(path, ".tmp");
  {
    std::ofstream output {temporary_path, std::ios::binary | std::ios::trunc};
    if (!output)
    {
      return std::unexpected {make_app_error(app_error_code::file_open)};
    }

    output.write(
      reinterpret_cast<const char*>(bytes.data()),
      static_cast<std::streamsize>(bytes.size()));
    if (!output)
    {
      return std::unexpected {make_app_error(app_error_code::file_write)};
    }
  }
  return replace_file(temporary_path, path);
}

export class pipeline_cache
{
public:
  pipeline_cache() = default;

  explicit pipeline_cache(vk::raii::PipelineCache&& cache)
  : cache_ {std::move(cache)}
  {}

  [[nodiscard]]
  static
  auto
  create(const vk::raii::Device& device)
    -> std::expected<pipeline_cache, error_t>
  { return create(device, {}); }

  [[nodiscard]]
  static
  auto
  create(const vk::raii::Device& device, std::span<const std::byte> initial)
    -> std::expected<pipeline_cache, error_t>
  {
    const vk::PipelineCacheCreateInfo create_info {
      .initialDataSize = initial.size(),
      .pInitialData    = initial.data(),
    };
    return map_vk_error(device.createPipelineCache(create_info), std::nullopt)
      .transform([](vk::raii::PipelineCache&& cache) -> pipeline_cache {
        return pipeline_cache {std::move(cache)};
      });
  }

  [[nodiscard]]
  auto
  get() const -> const vk::raii::PipelineCache&
  { return cache_; }

  [[nodiscard]]
  auto
  native_handle() const -> vk::PipelineCache
  { return get(); }

  [[nodiscard]]
  auto
  data() const -> std::expected<std::vector<std::uint8_t>, error_t>
  { return map_vk_error(cache_.getData(), std::nullopt); }

private:
  vk::raii::PipelineCache cache_ {nullptr};
};

} // namespace vkpp
