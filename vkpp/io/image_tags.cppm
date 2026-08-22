module;
export module vkpp.io.image;

import std;
import vkpp.io.types;
import vkpp.error;

namespace vkpp
{
export enum class image_file_type : std::uint8_t {
  ktx2, // goto default
  png,  // legacy decoded raster
};

export template<image_file_type Type>
struct image_file_traits;

template<>
struct image_file_traits<image_file_type::png>
{
  using result = host_image;
  using runtime_args = png_load_runtime_args;
};

template<>
struct image_file_traits<image_file_type::ktx2>
{
  using result = host_image_mip_chain;
  using runtime_args = ktx2_load_runtime_args;
};

export template<image_file_type Type>
[[nodiscard]] auto
load_host_image(const std::filesystem::path& path,
  const typename image_file_traits<Type>::runtime_args& runtime_args = {})
  -> std::expected<typename image_file_traits<Type>::result, error_t>;
} // namespace vkpp
