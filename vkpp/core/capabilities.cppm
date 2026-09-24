export module vkpp.capabilities;

import std;
import vulkan;

namespace vkpp
{

export enum class capability_requirement : std::uint8_t {
  disabled,
  preferred,
  required,
};

export enum class descriptor_binding_request : std::uint8_t {
  classic,
  heap,
};

export enum class pipeline_persistence_request : std::uint8_t {
  pipeline_cache,
  pipeline_binary,
};

export enum class pipeline_persistence_mode : std::uint8_t {
  pipeline_cache,
  application_binary,
  internal_binary,
};

export enum class descriptor_table_backend : std::uint8_t {
  classic,
  heap,
};

using enum capability_requirement;

export struct device_feature_requests
{
  capability_requirement sampler_anisotropy { disabled };
  capability_requirement sample_rate_shading { disabled };
  capability_requirement dynamic_rendering { disabled };
  capability_requirement synchronization2 { disabled };
  capability_requirement extended_dynamic_state { disabled };
  capability_requirement timeline_semaphore { disabled };
  capability_requirement host_query_reset { disabled };
  capability_requirement descriptor_indexing { disabled };
  capability_requirement buffer_device_address { disabled };
  capability_requirement graphics_pipeline_library { disabled };
  capability_requirement shader_object { disabled };
};

export struct device_profile
{
  std::string_view name {};
  device_feature_requests features {};
  descriptor_binding_request descriptor { descriptor_binding_request::classic };
  pipeline_persistence_request pipeline {
    pipeline_persistence_request::pipeline_cache
  };
};

export struct physical_device_rank_policy
{
  bool prefer_discrete { true };
};

export struct device_requirements
{
  std::span<const device_profile> profiles {};
  std::span<const char* const> extra_extensions {};
  std::uint32_t min_api_version { vk::ApiVersion14 };
  bool require_present { true };
  bool request_dedicated_transfer { false };
  physical_device_rank_policy rank {};
};

export struct enabled_device_features
{
  bool sampler_anisotropy { false };
  bool sample_rate_shading { false };
  bool dynamic_rendering { false };
  bool synchronization2 { false };
  bool extended_dynamic_state { false };
  bool timeline_semaphore { false };
  bool host_query_reset { false };
  bool descriptor_indexing { false };
  bool buffer_device_address { false };
  bool graphics_pipeline_library { false };
  bool shader_object { false };
  bool descriptor_heap { false };
  bool pipeline_binary { false };
};

export struct selected_device_capabilities
{
  std::string profile_name {};
  std::uint32_t profile_index {};
  descriptor_table_backend descriptor { descriptor_table_backend::classic };
  pipeline_persistence_mode pipeline {
    pipeline_persistence_mode::pipeline_cache
  };
  enabled_device_features enabled {};
  vk::PhysicalDevicePipelineBinaryPropertiesKHR pipeline_binary_properties {};
  bool descriptor_heap_enabled { false };
  bool pipeline_binary_enabled { false };
};

} // namespace vkpp
