module;
export module vkpp.io.shader;

import std;

import vkpp.error;

namespace vkpp
{

using namespace std::string_view_literals;

export [[nodiscard]]
auto
load_shader_file(const std::filesystem::path& filename)
  -> std::expected<std::vector<std::uint32_t>, vkpp::error_t>
{
  std::ifstream input_file {filename, std::ios::ate | std::ios::binary};
  if (!input_file.is_open())
  {
    return std::unexpected {make_app_error(app_error_code::file_open)};
  }

  const auto byte_count = static_cast<std::size_t>(input_file.tellg());
  if (byte_count == 0UZ || byte_count % sizeof(std::uint32_t) != 0UZ)
  {
    return std::unexpected {make_app_error(app_error_code::invalid_data_size)};
  }

  std::vector<std::uint32_t> words(byte_count / sizeof(std::uint32_t));
  input_file.seekg(0, std::ios::beg);
  input_file.read(
    reinterpret_cast<char*>(words.data()),
    static_cast<std::streamsize>(std::span {words}.size_bytes()));
  if (!input_file)
  {
    return std::unexpected {make_app_error(app_error_code::file_read)};
  }
  return words;
}

} // namespace vkpp
