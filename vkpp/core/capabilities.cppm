export module vkpp.capabilities;

import std;
import vulkan;

namespace vkpp
{

export enum class capability_requirement : std::uint8_t
{
  disabled,
  preferred,
  required,
};

export enum class descriptor_binding_request : std::uint8_t
{
  classic,
  heap,
};

export enum class pipeline_persistence_request : std::uint8_t
{
  pipeline_cache,
  pipeline_binary,
};

export enum class pipeline_persistence_mode : std::uint8_t
{
  pipeline_cache,
  application_binary,
  internal_binary,
};

export enum class descriptor_table_backend : std::uint8_t
{
  classic,
  heap,
};

using enum capability_requirement;

export struct device_feature_requests
{
  capability_requirement sampler_anisotropy {disabled};
  capability_requirement sample_rate_shading {disabled};
  capability_requirement dynamic_rendering {disabled};
  capability_requirement synchronization2 {disabled};
  capability_requirement extended_dynamic_state {disabled};
  capability_requirement timeline_semaphore {disabled};
  capability_requirement host_query_reset {disabled};
  capability_requirement descriptor_indexing {disabled};
  capability_requirement buffer_device_address {disabled};
  capability_requirement graphics_pipeline_library {disabled};
  capability_requirement shader_object {disabled};
  capability_requirement maintenance4 {disabled};
  capability_requirement vulkan_memory_model {disabled};
  capability_requirement subgroup_size_control {disabled};
  capability_requirement subgroup_extended_types {disabled};
  capability_requirement subgroup_uniform_control_flow {disabled};
  capability_requirement maximal_reconvergence {disabled};
  capability_requirement subgroup_partitioned {disabled};
  capability_requirement cooperative_matrix {disabled};
  capability_requirement integer_dot_product {disabled};
  capability_requirement int64_atomics {disabled};
  capability_requirement float_atomics {disabled};
  capability_requirement workgroup_memory_explicit_layout {disabled};
};

export struct device_profile
{
  std::string_view             name {};
  device_feature_requests      features {};
  descriptor_binding_request   descriptor {descriptor_binding_request::classic};
  pipeline_persistence_request pipeline {
    pipeline_persistence_request::pipeline_cache
  };
};

export struct physical_device_rank_policy
{
  bool prefer_discrete {true};
};

export struct device_requirements
{
  std::span<const device_profile> profiles {};
  std::span<const char* const>    extra_extensions {};
  std::uint32_t                   min_api_version {vk::ApiVersion14};
  bool                            require_present {true};
  bool                            request_dedicated_transfer {false};
  physical_device_rank_policy     rank {};
};

export struct enabled_device_features
{
  bool sampler_anisotropy {false};
  bool sample_rate_shading {false};
  bool dynamic_rendering {false};
  bool synchronization2 {false};
  bool extended_dynamic_state {false};
  bool timeline_semaphore {false};
  bool host_query_reset {false};
  bool descriptor_indexing {false};
  bool buffer_device_address {false};
  bool graphics_pipeline_library {false};
  bool shader_object {false};
  bool descriptor_heap {false};
  bool pipeline_binary {false};
  bool maintenance4 {false};
  bool vulkan_memory_model {false};
  bool subgroup_size_control {false};
  bool subgroup_extended_types {false};
  bool subgroup_uniform_control_flow {false};
  bool maximal_reconvergence {false};
  bool subgroup_partitioned {false};
  bool cooperative_matrix {false};
  bool integer_dot_product {false};
  bool int64_atomics {false};
  bool float_atomics {false};
  bool workgroup_memory_explicit_layout {false};
};

export struct compute_capabilities
{
  std::uint32_t                subgroup_size {};
  std::uint32_t                minimum_subgroup_size {};
  std::uint32_t                maximum_subgroup_size {};
  std::uint32_t                maximum_workgroup_subgroups {};
  vk::ShaderStageFlags         subgroup_stages {};
  vk::SubgroupFeatureFlags     subgroup_operations {};
  vk::ShaderStageFlags         required_subgroup_size_stages {};
  bool                         subgroup_size_control {false};
  bool                         compute_full_subgroups {false};
  bool                         subgroup_extended_types {false};
  bool                         subgroup_uniform_control_flow {false};
  bool                         maximal_reconvergence {false};
  bool                         subgroup_partitioned {false};
  bool                         cooperative_matrix {false};
  bool                         integer_dot_product {false};
  bool                         buffer_int64_atomics {false};
  bool                         shared_int64_atomics {false};
  bool                         buffer_float32_atomic_add {false};
  bool                         shared_float32_atomic_add {false};
  bool                         buffer_float32_atomic_minimum_maximum {false};
  bool                         shared_float32_atomic_minimum_maximum {false};
  bool                         workgroup_memory_explicit_layout {false};
  std::uint32_t                maximum_shared_memory_size {};
  std::uint32_t                maximum_push_constants_size {};
  vk::DeviceSize               maximum_buffer_size {};
  std::array<std::uint32_t, 3> maximum_workgroup_size {};
  std::vector<vk::CooperativeMatrixPropertiesKHR>
    cooperative_matrix_properties {};
  vk::PhysicalDeviceShaderIntegerDotProductProperties
    integer_dot_product_properties {};
};

export struct selected_device_capabilities
{
  std::string               profile_name {};
  std::uint32_t             profile_index {};
  descriptor_table_backend  descriptor {descriptor_table_backend::classic};
  pipeline_persistence_mode pipeline {
    pipeline_persistence_mode::pipeline_cache
  };
  enabled_device_features                       enabled {};
  vk::PhysicalDevicePipelineBinaryPropertiesKHR pipeline_binary_properties {};
  bool                                          descriptor_heap_enabled {false};
  bool                                          pipeline_binary_enabled {false};
  compute_capabilities                          compute {};
};

export
template<class Callable>
decltype(auto)
dispatch_descriptor_backend(
  const selected_device_capabilities& capabilities, Callable&& callable)
{
  switch (capabilities.descriptor)
  {
  case descriptor_table_backend::classic :
    return std::forward<Callable>(callable).template
    operator ()<descriptor_table_backend::classic>();
  case descriptor_table_backend::heap :
    return std::forward<Callable>(callable).template
    operator ()<descriptor_table_backend::heap>();
  }
  std::unreachable();
}

} // namespace vkpp
