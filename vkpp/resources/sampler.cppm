module;

#include "error/vk_error_config.hpp"

export module vkpp.sampler;

import std;
import vulkan;

import vkpp.error;

namespace vkpp
{
using namespace std::string_view_literals;

export struct sampler_create_info
{
  vk::Filter mag_filter { vk::Filter::eLinear };
  vk::Filter min_filter { vk::Filter::eLinear };
  vk::SamplerMipmapMode mipmap_mode { vk::SamplerMipmapMode::eLinear };
  vk::SamplerAddressMode address_mode_u { vk::SamplerAddressMode::eRepeat };
  vk::SamplerAddressMode address_mode_v { vk::SamplerAddressMode::eRepeat };
  vk::SamplerAddressMode address_mode_w { vk::SamplerAddressMode::eRepeat };
  bool anisotropy_enable { true };
  float min_lod { 0.0F };
  float max_lod { vk::LodClampNone };

  [[nodiscard]] auto
  operator==(const sampler_create_info&) const -> bool = default;
};

export auto
make_sampler(const vk::raii::Device& device,
  const vk::raii::PhysicalDevice& physical,
  const sampler_create_info& create_info)
  -> std::expected<vk::raii::Sampler, error_t>
{
  const auto properties = physical.getProperties();
  const vk::SamplerCreateInfo sampler_info {
    .magFilter = create_info.mag_filter,
    .minFilter = create_info.min_filter,
    .mipmapMode = create_info.mipmap_mode,
    .addressModeU = create_info.address_mode_u,
    .addressModeV = create_info.address_mode_v,
    .addressModeW = create_info.address_mode_w,
    .mipLodBias = 0.0F,
    .anisotropyEnable = create_info.anisotropy_enable ? vk::True : vk::False,
    .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
    .compareEnable = vk::False,
    .compareOp = vk::CompareOp::eAlways,
    .minLod = create_info.min_lod,
    .maxLod = create_info.max_lod,
    .borderColor = vk::BorderColor::eIntOpaqueBlack,
    .unnormalizedCoordinates = vk::False,
  };
  return UTILS_VK(
    device.createSampler(sampler_info), ^^vk::raii::Device::createSampler);
}

export class sampler_cache
{
public:
  [[nodiscard]] static auto
  create(
    const vk::raii::Device& device, const vk::raii::PhysicalDevice& physical)
    -> std::expected<sampler_cache, error_t>
  { return sampler_cache { device, physical }; }

  [[nodiscard]] auto
  get_or_create(const sampler_create_info& create_info)
    -> std::expected<vk::Sampler, error_t>
  {
    for (auto index : std::views::indices(count_))
    {
      if (entries_[ index ].create_info == create_info)
      {
        return *entries_[ index ].sampler;
      }
    }

    if (count_ >= entries_.size())
    {
      return std::unexpected {
        app_error {
          .kind = app_error_kind::invalid_argument,
          .detail = "sampler_cache capacity exhausted"sv,
        },
      };
    }

    return make_sampler(device_, physical_, create_info)
      .transform(
        [ &, this ](vk::raii::Sampler&& sampler) -> vk::Sampler
        {
          entry& destination = entries_[ count_ ];
          destination.create_info = create_info;
          destination.sampler = std::move(sampler);
          ++count_;
          return *destination.sampler;
        });
  }

  [[nodiscard]] auto
  size() const -> std::uint32_t
  { return count_; }

  static constexpr std::size_t capacity { 16UZ };

private:
  struct entry
  {
    sampler_create_info create_info {};
    vk::raii::Sampler sampler { nullptr };
  };

  sampler_cache(
    const vk::raii::Device& device, const vk::raii::PhysicalDevice& physical)
  : device_ { device }, physical_ { physical }
  {}

  const vk::raii::Device& device_;
  const vk::raii::PhysicalDevice& physical_;
  std::array<entry, capacity> entries_ {};
  std::uint32_t count_ { 0U };
};

} // namespace vkpp
