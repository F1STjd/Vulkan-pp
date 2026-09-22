module;
export module vkpp.io.shader;

import std;

import vkpp.error;

namespace vkpp
{
using namespace std::string_view_literals;

export [[nodiscard]] auto
load_shader_file(const std::filesystem::path& filename)
  -> std::expected<std::vector<char>, vkpp::error_t>
{
  std::ifstream input_file { filename, std::ios::ate | std::ios::binary };
  if (!input_file.is_open())
  {
    return std::unexpected { make_app_error(app_error_code::file_open) };
  }

  std::vector<char> buffer(input_file.tellg());
  input_file.seekg(0, std::ios::beg);
  input_file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
  return { buffer };
}

} // namespace vkpp
