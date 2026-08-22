module;
export module vkpp.io.mesh;

import std;
import vkpp.io.types;
import vkpp.error;

namespace vkpp
{
export enum class mesh_file_type : std::uint8_t {
  gltf,
  obj,
};

export template<mesh_file_type Type>
struct mesh_file_traits;

template<>
struct mesh_file_traits<mesh_file_type::gltf>
{
  using result = mesh_cpu;
};

template<>
struct mesh_file_traits<mesh_file_type::obj>
{
  using result = mesh_cpu;
};

export template<mesh_file_type Type>
[[nodiscard]] auto
load_mesh_cpu(const std::filesystem::path& path)
  -> std::expected<typename mesh_file_traits<Type>::result, error_t>;

} // namespace vkpp
