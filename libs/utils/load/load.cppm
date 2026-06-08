export module load;

import std;

export namespace load
{
[[nodiscard]] constexpr auto
read_shader_file(const std::filesystem::path& filename)
  -> std::expected<std::vector<char>, std::string>
{
  std::ifstream input_file { filename, std::ios::ate | std::ios::binary };
  if (!input_file.is_open())
  {
    return std::expected<std::vector<char>, std::string> {
      std::unexpect,
      "Failed to open shader file",
    };
  }

  std::vector<char> buffer(input_file.tellg());
  input_file.seekg(0, std::ios::beg);
  input_file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
  return { buffer };
}
} // namespace load