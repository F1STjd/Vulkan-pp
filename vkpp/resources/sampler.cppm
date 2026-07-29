module;

#include "error/vk_error_config.hpp"

export module vkpp.sampler;

import std;
import vulkan;

import vkpp.error;

namespace vkpp
{

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

} // namespace vkpp
