export module vkpp.pipeline.persistence;

import std;
import vulkan;

import vkpp.error;
import vkpp.diagnostics;
import vkpp.capabilities;
import vkpp.pipeline.binary;
import vkpp.pipeline.cache;

namespace vkpp
{
using namespace std::string_view_literals;

inline constexpr std::uint32_t pipeline_binary_store_schema_version { 1U };
inline constexpr std::array pipeline_binary_store_magic {
  'V',
  'K',
  'P',
  'P',
  'B',
  'I',
  'N',
};

[[nodiscard]] inline auto
write_le_u32(std::ostream& out, std::uint32_t value) -> bool
{
  const auto bytes = std::bit_cast<std::array<char, 4UZ>>(
    std::endian::native == std::endian::big ? std::byteswap(value) : value);
  out.write(bytes.data(), bytes.size());
  return static_cast<bool>(out);
}

[[nodiscard]] inline auto
write_le_u64(std::ostream& out, std::uint64_t value) -> bool
{
  const auto bytes = std::bit_cast<std::array<char, 8UZ>>(
    std::endian::native == std::endian::big ? std::byteswap(value) : value);
  out.write(bytes.data(), bytes.size());
  return static_cast<bool>(out);
}

[[nodiscard]] inline auto
read_le_u32(std::istream& in, std::uint32_t& value) -> bool
{
  std::array<char, 4UZ> bytes {};
  in.read(bytes.data(), bytes.size());
  value = std::bit_cast<std::uint32_t>(
    std::endian::native == std::endian::big ? std::byteswap(value) : value);
  return true;
}

[[nodiscard]] inline auto
read_le_u64(std::istream& in, std::uint64_t& value) -> bool
{
  std::array<char, 8UZ> bytes {};
  in.read(bytes.data(), bytes.size());
  value = std::bit_cast<std::uint64_t>(
    std::endian::native == std::endian::big ? std::byteswap(value) : value);
  return true;
}

[[nodiscard]] inline auto
keys_equal(const vk::PipelineBinaryKeyKHR& lhs,
  const vk::PipelineBinaryKeyKHR& rhs) -> bool
{
  return lhs.keySize == rhs.keySize &&
    std::ranges::equal(std::span { lhs.key }.first(lhs.keySize),
      std::span { rhs.key }.first(rhs.keySize));
}

export struct pipeline_binary_store_entry
{
  vk::PipelineBinaryKeyKHR pipeline_key {};
  std::vector<vk::PipelineBinaryKeyKHR> ordered_binary_keys {};
};

export class pipeline_binary_store
{
public:
  pipeline_binary_store() = default;

  explicit pipeline_binary_store(vk::PipelineBinaryKeyKHR global_key)
  : global_key_ { global_key }
  {}

  [[nodiscard]] auto
  global_key() const -> const vk::PipelineBinaryKeyKHR&
  { return global_key_; }

  [[nodiscard]] auto
  empty() const -> bool
  { return entries_.empty() && blobs_.empty(); }

  [[nodiscard]] auto
  find_ordered_keys(const vk::PipelineBinaryKeyKHR& pipeline_key) const
    -> std::optional<std::span<const vk::PipelineBinaryKeyKHR>>
  {
    const auto it = std::ranges::find_if(entries_,
      [ & ](const pipeline_binary_store_entry& entry) -> bool
      { return keys_equal(entry.pipeline_key, pipeline_key); });
    if (it == entries_.end()) { return std::nullopt; }
    return std::span { it->ordered_binary_keys };
  }

  [[nodiscard]] auto
  blob_for(const vk::PipelineBinaryKeyKHR& binary_key) const
    -> std::optional<std::span<const std::uint8_t>>
  {
    const auto it = std::ranges::find_if(blobs_,
      [ & ](const pipeline_binary_blob& blob) -> bool
      { return keys_equal(blob.key, binary_key); });
    if (it == blobs_.end()) { return std::nullopt; }
    return std::span { it->data };
  }

  [[nodiscard]] auto
  collect_blobs_for_pipeline(const vk::PipelineBinaryKeyKHR& pipeline_key) const
    -> std::expected<std::vector<pipeline_binary_blob>, error_t>
  {
    const auto keys = find_ordered_keys(pipeline_key);
    if (!keys)
    {
      return std::unexpected { make_app_error(app_error_code::out_of_range) };
    }
    std::vector<pipeline_binary_blob> out {};
    out.reserve(keys->size());
    for (const auto& key : *keys)
    {
      const auto bytes = blob_for(key);
      if (!bytes)
      {
        return std::unexpected {
          make_app_error(app_error_code::corrupt_persistent_data),
        };
      }
      out.push_back((pipeline_binary_blob {
        .key = key,
        .data = std::vector<std::uint8_t> { bytes->begin(), bytes->end() },
      }));
    }
    return out;
  }

  [[nodiscard]] auto
  upsert_capture(const vk::PipelineBinaryKeyKHR& pipeline_key,
    std::span<const pipeline_binary_blob> captured,
    std::optional<diagnostic_buffer&> diagnostics = {})
    -> std::expected<void, error_t>
  {
    if (captured.empty())
    {
      if (diagnostics)
      {
        diagnostics->report(diagnostic_severity::warning, std::nullopt,
          std::source_location::current(),
          "pipeline_binary_store: empty capture; not persisting entry");
      }
      return {};
    }
    std::vector<vk::PipelineBinaryKeyKHR> ordered {};
    ordered.reserve(captured.size());
    for (const auto& blob : captured)
    {
      ordered.push_back(blob.key);
      const auto existing = std::ranges::find_if(blobs_,
        [ & ](const pipeline_binary_blob& b) -> bool
        { return keys_equal(b.key, blob.key); });
      if (existing != blobs_.end())
      {
        if (!std::ranges::equal(existing->data, blob.data))
        {
          if (diagnostics)
          {
            diagnostics->report(diagnostic_severity::warning,
              make_app_error(app_error_code::corrupt_persistent_data),
              std::source_location::current(),
              "pipeline_binary_store: duplicate binary key with different "
              "bytes");
          }
          return std::unexpected {
            make_app_error(app_error_code::corrupt_persistent_data),
          };
        }
      }
      else
      {
        blobs_.push_back(blob);
      }
    }
    const auto entry_it = std::ranges::find_if(entries_,
      [ & ](const pipeline_binary_store_entry& entry) -> bool
      { return keys_equal(entry.pipeline_key, pipeline_key); });
    if (entry_it != entries_.end())
    {
      entry_it->ordered_binary_keys = std::move(ordered);
    }
    else
    {
      entries_.push_back(pipeline_binary_store_entry {
        .pipeline_key = pipeline_key,
        .ordered_binary_keys = std::move(ordered),
      });
    }
    return {};
  }

  [[nodiscard]] static auto
  load(const std::filesystem::path& path,
    const vk::PipelineBinaryKeyKHR& expected_global_key,
    std::optional<diagnostic_buffer&> diagnostics = {})
    -> std::expected<pipeline_binary_store, error_t>
  {
    if (!std::filesystem::exists(path))
    {
      return pipeline_binary_store { expected_global_key };
    }
    std::ifstream input { path, std::ios::binary };
    if (!input)
    {
      return std::unexpected { make_app_error(app_error_code::file_open) };
    }
    const auto discard = [ & ](std::string_view why) -> pipeline_binary_store
    {
      if (diagnostics)
      {
        diagnostics->report(diagnostic_severity::warning,
          make_app_error(app_error_code::corrupt_persistent_data),
          std::source_location::current(),
          "pipeline_binary_store: discarding ({})", why);
      }
      return pipeline_binary_store { expected_global_key };
    };

    std::array<char, 8UZ> magic {};
    input.read(magic.data(), 8LL);
    if (!input || !std::ranges::equal(magic, pipeline_binary_store_magic))
    {
      return discard("bad magic");
    }

    std::uint32_t version {};
    if (!read_le_u32(input, version) ||
      version != pipeline_binary_store_schema_version)
    {
      return discard("bad schema version");
    }

    vk::PipelineBinaryKeyKHR global {};
    if (!read_le_u32(input, global.keySize) ||
      global.keySize > vk::MaxPipelineBinaryKeySizeKHR)
    {
      return discard("bad global keySize");
    }

    input.read(reinterpret_cast<char*>(global.key.data()),
      vk::MaxPipelineBinaryKeySizeKHR);
    if (!input || !keys_equal(global, expected_global_key))
    {
      return discard("global key mismatch");
    }

    const auto begin_pos = input.tellg();
    input.seekg(0, std::ios::end);
    const auto end_pos = input.tellg();
    if (!input || begin_pos < 0 || end_pos < begin_pos)
    {
      return discard("size query failed");
    }
    input.seekg(begin_pos);
    auto remaining = static_cast<std::uint64_t>(end_pos - begin_pos);

    auto need = [ & ](std::uint64_t n) -> bool
    {
      if (n > remaining) { return false; }
      remaining -= n;
      return true;
    };

    std::uint32_t entry_count {};
    if (!need(4) || !read_le_u32(input, entry_count))
    {
      return discard("entry_count");
    }
    pipeline_binary_store store { expected_global_key };
    store.entries_.resize(entry_count);
    for (auto& entry : store.entries_)
    {
      if (!need(4) || !read_le_u32(input, entry.pipeline_key.keySize) ||
        entry.pipeline_key.keySize > vk::MaxPipelineBinaryKeySizeKHR)
      {
        return discard("pipeline keySize");
      }
      if (!need(vk::MaxPipelineBinaryKeySizeKHR))
      {
        return discard("pipeline key bytes");
      }
      input.read(reinterpret_cast<char*>(entry.pipeline_key.key.data()),
        vk::MaxPipelineBinaryKeySizeKHR);
      std::uint32_t ordered_count {};
      if (!need(4) || !read_le_u32(input, ordered_count))
      {
        return discard("ordered_count");
      }
      const auto per_key = 4ULL + vk::MaxPipelineBinaryKeySizeKHR;
      if (ordered_count > remaining / per_key)
      {
        return discard("ordered_count overflow");
      }
      entry.ordered_binary_keys.resize(ordered_count);
      for (auto& key : entry.ordered_binary_keys)
      {
        if (!need(4) || !read_le_u32(input, key.keySize) ||
          key.keySize > vk::MaxPipelineBinaryKeySizeKHR)
        {
          return discard("ordered keySize");
        }
        if (!need(vk::MaxPipelineBinaryKeySizeKHR))
        {
          return discard("ordered key bytes");
        }
        input.read(reinterpret_cast<char*>(key.key.data()),
          vk::MaxPipelineBinaryKeySizeKHR);
      }
    }

    std::uint32_t blob_count {};
    if (!need(4) || !read_le_u32(input, blob_count))
    {
      return discard("blob_count");
    }
    store.blobs_.resize(blob_count);
    for (auto& blob : store.blobs_)
    {
      if (!need(4) || !read_le_u32(input, blob.key.keySize))
      {
        return discard("blob key bytes");
      }
      input.read(reinterpret_cast<char*>(blob.key.key.data()),
        vk::MaxPipelineBinaryKeySizeKHR);
      std::uint64_t data_size {};
      if (!need(8) || !read_le_u64(input, data_size))
      {
        return discard("data_size");
      }
      if (data_size > remaining ||
        data_size >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
      {
        return discard("data_size overflow");
      }
      if (!need(data_size)) { return discard("blob bytes"); }
      blob.data.resize(static_cast<std::size_t>(data_size));
      if (data_size != 0U)
      {
        input.read(reinterpret_cast<char*>(blob.data.data()),
          static_cast<std::streamsize>(data_size));
      }
      if (!input) { return discard("blob read"); }
    }

    for (auto i : std::views::indices(store.blobs_.size()))
    {
      for (auto j : std::views::iota(i + 1UZ, store.blobs_.size()))
      {
        if (keys_equal(store.blobs_[ i ].key, store.blobs_[ j ].key))
        {
          return discard("duplicate blob keys");
        }
      }
    }

    return store;
  }

  [[nodiscard]] auto
  save(const std::filesystem::path& path) const -> std::expected<void, error_t>
  {
    auto tmp { path };
    tmp += ".tmp";
    auto bak { path };
    bak += ".bak";

    {
      std::ofstream output { tmp, std::ios::binary | std::ios::trunc };
      if (!output)
      {
        return std::unexpected { make_app_error(app_error_code::file_open) };
      }
      output.write(pipeline_binary_store_magic.data(), 8);
      if (!write_le_u32(output, pipeline_binary_store_schema_version))
      {
        return std::unexpected { make_app_error(app_error_code::file_write) };
      }
      if (!write_le_u32(output, global_key_.keySize))
      {
        return std::unexpected { make_app_error(app_error_code::file_write) };
      }
      output.write(reinterpret_cast<const char*>(global_key_.key.data()),
        vk::MaxPipelineBinaryKeySizeKHR);
      if (!write_le_u32(output, static_cast<std::uint32_t>(entries_.size())))
      {
        return std::unexpected { make_app_error(app_error_code::file_write) };
      }

      for (const auto& entry : entries_)
      {
        if (!write_le_u32(output, entry.pipeline_key.keySize))
        {
          return std::unexpected { make_app_error(app_error_code::file_write) };
        }
        output.write(
          reinterpret_cast<const char*>(entry.pipeline_key.key.data()),
          vk::MaxPipelineBinaryKeySizeKHR);
        if (!write_le_u32(output,
              static_cast<std::uint32_t>(entry.ordered_binary_keys.size())))
        {
          return std::unexpected { make_app_error(app_error_code::file_write) };
        }
        for (const auto& key : entry.ordered_binary_keys)
        {
          if (!write_le_u32(output, key.keySize))
          {
            return std::unexpected {
              make_app_error(app_error_code::file_write),
            };
          }
          output.write(reinterpret_cast<const char*>(key.key.data()),
            vk::MaxPipelineBinaryKeySizeKHR);
        }
      }

      if (!write_le_u32(output, static_cast<std::uint32_t>(blobs_.size())))
      {
        return std::unexpected { make_app_error(app_error_code::file_write) };
      }
      for (const auto& blob : blobs_)
      {
        if (!write_le_u32(output, blob.key.keySize))
        {
          return std::unexpected { make_app_error(app_error_code::file_write) };
        }
        output.write(reinterpret_cast<const char*>(blob.key.key.data()),
          vk::MaxPipelineBinaryKeySizeKHR);
        if (!write_le_u64(output, static_cast<std::uint64_t>(blob.data.size())))
        {
          return std::unexpected { make_app_error(app_error_code::file_write) };
        }
        if (!blob.data.empty())
        {
          output.write(reinterpret_cast<const char*>(blob.data.data()),
            static_cast<std::streamsize>(blob.data.size()));
        }
      }
      output.flush();
      if (!output)
      {
        return std::unexpected { make_app_error(app_error_code::file_write) };
      }
    }

    std::error_code ec {};
    std::filesystem::remove(bak, ec);
    ec.clear();
    if (std::filesystem::exists(path))
    {
      std::filesystem::rename(path, bak, ec);
      if (ec)
      {
        return std::unexpected { make_app_error(app_error_code::file_write) };
      }
    }
    std::filesystem::rename(tmp, path, ec);
    if (ec)
    {
      if (std::filesystem::exists(bak))
      {
        std::error_code restore_ec {};
        std::filesystem::rename(bak, path, restore_ec);
      }
      return std::unexpected { make_app_error(app_error_code::file_write) };
    }
    std::filesystem::remove(bak, ec);
    return {};
  }

private:
  vk::PipelineBinaryKeyKHR global_key_ {};
  std::vector<pipeline_binary_store_entry> entries_ {};
  std::vector<pipeline_binary_blob> blobs_ {};
};

export [[nodiscard]] auto
query_global_pipeline_binary_key(const vk::raii::Device& device,
  std::optional<diagnostic_buffer&> diagnostics = {},
  std::source_location location = std::source_location::current())
  -> std::expected<vk::PipelineBinaryKeyKHR, error_t>
{
  return map_vk_error(device.getPipelineKeyKHR(nullptr), diagnostics, location);
}

export [[nodiscard]] auto
query_pipeline_binary_key(const vk::raii::Device& device,
  const vk::GraphicsPipelineCreateInfo& graphics_create_info,
  std::optional<diagnostic_buffer&> diagnostics = {},
  std::source_location location = std::source_location::current())
  -> std::expected<vk::PipelineBinaryKeyKHR, error_t>
{
  vk::PipelineCreateInfoKHR wrap {};
  wrap.pNext =
    const_cast<vk::GraphicsPipelineCreateInfo*>(&graphics_create_info);
  return map_vk_error(device.getPipelineKeyKHR(wrap), diagnostics, location);
}

export class pipeline_persistence
{
public:
  pipeline_persistence() = default;

  pipeline_persistence(pipeline_cache&& cache)
  : mode_ { pipeline_persistence_mode::pipeline_cache },
    cache_ { std::move(cache) }
  {}

  pipeline_persistence(
    pipeline_binary_store&& store, std::filesystem::path store_path)
  : mode_ { pipeline_persistence_mode::application_binary },
    store_ { std::move(store) }, store_path_ { std::move(store_path) }
  {}

  explicit pipeline_persistence(pipeline_persistence_mode mode) : mode_ { mode }
  {}

  [[nodiscard]] static auto
  create(const vk::raii::Device& device,
    const vk::PhysicalDeviceProperties& physical_properties,
    const selected_device_capabilities& capabilities,
    const std::filesystem::path& cache_path,
    const std::filesystem::path& binary_store_path,
    std::optional<diagnostic_buffer&> diagnostics = {})
    -> std::expected<pipeline_persistence, error_t>
  {
    using enum pipeline_persistence_mode;
    switch (capabilities.pipeline)
    {
    case pipeline_persistence_mode::pipeline_cache:
      return load_pipeline_cache_file(cache_path)
        .and_then(
          [ & ](std::vector<std::byte>&& bytes)
            -> std::expected<pipeline_persistence, error_t>
          {
            std::span<const std::byte> initial = bytes;
            if (!bytes.empty() &&
              !pipeline_cache_header_matches(bytes, physical_properties))
            {
              initial = {};
            }
            return pipeline_cache::create(device, initial)
              .transform(
                [](class pipeline_cache&& cache) -> pipeline_persistence
                { return pipeline_persistence { std::move(cache) }; });
          });
    case pipeline_persistence_mode::application_binary:
      return query_global_pipeline_binary_key(device, diagnostics)
        .and_then(
          [ & ](vk::PipelineBinaryKeyKHR&& global)
            -> std::expected<pipeline_persistence, error_t>
          {
            return pipeline_binary_store::load(
              binary_store_path, global, diagnostics)
              .transform(
                [ & ](pipeline_binary_store&& store) -> pipeline_persistence
                {
                  return pipeline_persistence {
                    std::move(store),
                    binary_store_path,
                  };
                });
          });
    case pipeline_persistence_mode::internal_binary:
    {
      const auto& props = capabilities.pipeline_binary_properties;
      if (props.pipelineBinaryInternalCache != vk::True ||
        props.pipelineBinaryPrefersInternalCache != vk::True)
      {
        return std::unexpected {
          make_app_error(app_error_code::invalid_state),
        };
      }
      return pipeline_persistence { internal_binary };
    }
    }
    std::unreachable();
  }

  [[nodiscard]] auto
  mode() const -> pipeline_persistence_mode
  { return mode_; }

  [[nodiscard]] auto
  cache() -> pipeline_cache*
  {
    return mode_ == pipeline_persistence_mode::pipeline_cache ? &cache_
                                                              : nullptr;
  }

  [[nodiscard]] auto
  cache() const -> const pipeline_cache*
  {
    return mode_ == pipeline_persistence_mode::pipeline_cache ? &cache_
                                                              : nullptr;
  }

  [[nodiscard]] auto
  cache_for_create() const -> const vk::raii::PipelineCache&
  {
    return mode_ == pipeline_persistence_mode::pipeline_cache ? cache_.get()
                                                              : null_cache_;
  }

  [[nodiscard]] auto
  store() -> pipeline_binary_store*
  {
    return mode_ == pipeline_persistence_mode::application_binary ? &store_
                                                                  : nullptr;
  }

  [[nodiscard]] auto
  save_appication_store_if_dirty() -> std::expected<void, error_t>
  {
    return (mode_ != pipeline_persistence_mode::application_binary ||
             !store_dirty_)
      ? std::expected<void, error_t> {}
      : store_.save(store_path_)
          .transform([ this ] -> void { store_dirty_ = false; });
  }

  void
  mark_store_dirty()
  { store_dirty_ = true; }

private:
  pipeline_persistence_mode mode_ { pipeline_persistence_mode::pipeline_cache };
  pipeline_cache cache_ {};
  pipeline_binary_store store_ {};
  std::filesystem::path store_path_ {};
  bool store_dirty_ { false };
  vk::raii::PipelineCache null_cache_ { nullptr };
};

export [[nodiscard]] auto
make_pipeline_binaries_from_store(const vk::raii::Device& device,
  const pipeline_binary_store& store,
  const vk::PipelineBinaryKeyKHR& pipeline_key,
  std::optional<diagnostic_buffer&> diagnostics = {})
  -> std::expected<std::optional<pipeline_binaries>, error_t>
{
  const auto keys = store.find_ordered_keys(pipeline_key);
  if (!keys) { return {}; }
  auto blobs = store.collect_blobs_for_pipeline(pipeline_key);
  if (!blobs) { return std::unexpected { blobs.error() }; }
  return pipeline_binaries::create(device, *blobs)
    .transform([](pipeline_binaries&& binaries)
      { return std::optional<pipeline_binaries> { std::move(binaries) }; });
}

export [[nodiscard]] auto
try_internal_pipeline_binaries(const vk::raii::Device& device,
  const vk::GraphicsPipelineCreateInfo& graphics_create_info,
  std::optional<diagnostic_buffer&> diagnostics = {},
  std::source_location location = std::source_location::current())
  -> std::expected<std::optional<pipeline_binaries>, error_t>
{
  vk::PipelineCreateInfoKHR wrap {};
  wrap.pNext =
    const_cast<vk::GraphicsPipelineCreateInfo*>(&graphics_create_info);
  const vk::PipelineBinaryCreateInfoKHR create_info {
    .pPipelineCreateInfo = &wrap,
  };
  auto raw = device.createPipelineBinariesKHR(create_info);
  if (!raw && raw.error() == vk::Result::ePipelineBinaryMissingKHR)
  {
    return std::optional<pipeline_binaries> {};
  }
  return map_vk_error(std::move(raw), diagnostics, location)
    .transform(
      [](std::vector<vk::raii::PipelineBinaryKHR>&& binaries)
        -> std::optional<pipeline_binaries>
      {
        return std::optional<pipeline_binaries> {
          pipeline_binaries { std::move(binaries) },
        };
      });
}

} // namespace vkpp
