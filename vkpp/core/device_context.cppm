export module vkpp.device;

import std;
import vulkan;

import vkpp.instance;
import vkpp.memory.vma;
import vkpp.error;
import vkpp.capabilities;
import vkpp.diagnostics;

namespace vkpp
{
using namespace std::string_view_literals;

// clang-format off
export struct feature_tag
{
  struct sampler_anisotropy {};
  struct sample_rate_shading {};
  struct dynamic_rendering {};
  struct synchronization2 {};
  struct extended_dynamic_state {};
  struct timeline_semaphore {};
  struct host_query_reset {};
  struct descriptor_indexing {};
  struct buffer_device_address {};
  struct graphics_pipeline_library {};
  struct shader_object {};
  struct descriptor_heap {};
  struct pipeline_binary {};
};
// clang-format on

export template<typename Tag>
struct feature_traits;

template<>
struct feature_traits<feature_tag::sampler_anisotropy>
{
  static constexpr auto member = &device_feature_requests::sampler_anisotropy;
};

template<>
struct feature_traits<feature_tag::sample_rate_shading>
{
  static constexpr auto member = &device_feature_requests::sample_rate_shading;
};

template<>
struct feature_traits<feature_tag::dynamic_rendering>
{
  static constexpr auto member = &device_feature_requests::dynamic_rendering;
};

template<>
struct feature_traits<feature_tag::synchronization2>
{
  static constexpr auto member = &device_feature_requests::synchronization2;
};

template<>
struct feature_traits<feature_tag::extended_dynamic_state>
{
  static constexpr auto member =
    &device_feature_requests::extended_dynamic_state;
};

template<>
struct feature_traits<feature_tag::timeline_semaphore>
{
  static constexpr auto member = &device_feature_requests::timeline_semaphore;
};

template<>
struct feature_traits<feature_tag::host_query_reset>
{
  static constexpr auto member = &device_feature_requests::host_query_reset;
};

template<>
struct feature_traits<feature_tag::descriptor_indexing>
{
  static constexpr auto member = &device_feature_requests::descriptor_indexing;
};

template<>
struct feature_traits<feature_tag::buffer_device_address>
{
  static constexpr auto member =
    &device_feature_requests::buffer_device_address;
};

template<>
struct feature_traits<feature_tag::graphics_pipeline_library>
{
  static constexpr auto member =
    &device_feature_requests::graphics_pipeline_library;
};

template<>
struct feature_traits<feature_tag::shader_object>
{
  static constexpr auto member = &device_feature_requests::shader_object;
};

template<>
struct feature_traits<feature_tag::descriptor_heap>
{
  static constexpr std::array extensions {
    vk::EXTDescriptorIndexingExtensionName,
  };
  static constexpr auto prerequisite_member =
    &device_feature_requests::buffer_device_address;
};

template<>
struct feature_traits<feature_tag::pipeline_binary>
{
  static constexpr std::array extensions {
    vk::KHRPipelineBinaryExtensionName,
  };
};

export [[nodiscard]] auto
score_physical_device(const vk::raii::PhysicalDevice& physical_device,
  physical_device_rank_policy policy = {}) -> std::int64_t
{
  const auto properties = physical_device.getProperties();
  const auto memory = physical_device.getMemoryProperties();
  std::int64_t device_local_bytes {};
  for (auto index : std::views::indices(memory.memoryHeapCount))
  {
    if (memory.memoryHeaps[ index ].flags &
      vk::MemoryHeapFlagBits::eDeviceLocal)
    {
      device_local_bytes +=
        static_cast<std::int64_t>(memory.memoryHeaps[ index ].size);
    }
  }
  std::int64_t tier {};
  if (policy.prefer_discrete)
  {
    switch (properties.deviceType)
    {
    case vk::PhysicalDeviceType::eDiscreteGpu:
    {
      tier = 3LL;
      break;
    }
    case vk::PhysicalDeviceType::eIntegratedGpu:
    {
      tier = 2LL;
      break;
    }
    case vk::PhysicalDeviceType::eVirtualGpu:
    {
      tier = 1LL;
      break;
    }
    // case vk::PhysicalDeviceType::eCpu:
    // case vk::PhysicalDeviceType::eOther:
    default:
    {
      tier = 0LL;
      break;
    }
    }
  }
  return (tier << 56) + device_local_bytes;
}

export [[nodiscard]] auto
descriptor_heap_supported(const vk::raii::PhysicalDevice& physical_device)
  -> bool
{
  const bool extension_present =
    physical_device.enumerateDeviceExtensionProperties()
      .transform(
        [](std::span<const vk::ExtensionProperties> available) -> bool
        {
          return std::ranges::any_of(available,
            [](const vk::ExtensionProperties& properties) -> bool
            {
              return std::strcmp(properties.extensionName,
                       vk::EXTDescriptorHeapExtensionName) == 0;
            });
        })
      .value_or(false);
  if (!extension_present) { return false; }

  const auto features =
    physical_device.getFeatures2<vk::PhysicalDeviceFeatures2,
      vk::PhysicalDeviceDescriptorHeapFeaturesEXT>();
  return features.get<vk::PhysicalDeviceDescriptorHeapFeaturesEXT>()
           .descriptorHeap == vk::True;
}

export class device_context
{
public:
  [[nodiscard]] static auto
  create(const instance_context& instance,
    const device_requirements& requirements,
    std::optional<diagnostic_buffer&> diagnostics = {})
    -> std::expected<device_context, error_t>
  {
    if (requirements.profiles.empty())
    {
      return std::unexpected {
        make_app_error(app_error_code::missing_required_argument),
      };
    }

    return map_vk_error(
      instance.instance().enumeratePhysicalDevices(), diagnostics)
      .and_then(
        [ & ](std::vector<vk::raii::PhysicalDevice>&& devices)
          -> std::expected<device_context, error_t>
        {
          const auto surface_ref = requirements.require_present
            ? std::optional { std::cref(instance.surface()) }
            : std::nullopt;

          for (auto profile_index :
            std::views::indices(requirements.profiles.size()))
          {
            const auto& profile = requirements.profiles[ profile_index ];
            auto candidates = devices |
              std::views::filter(
                [ & ](const vk::raii::PhysicalDevice& physical) -> bool
                {
                  if (requirements.require_present)
                  {
                    if (!surface_ref.has_value()) { return false; }
                    if (!find_graphics_present_qf(physical, surface_ref->get())
                          .has_value())
                    {
                      return false;
                    }
                  }
                  return profile_is_viable(
                    physical, profile, requirements, diagnostics);
                });
            auto chosen = std::ranges::max_element(candidates, std::less {},
              [ & ](const vk::raii::PhysicalDevice& physical) -> std::int64_t
              { return score_physical_device(physical, requirements.rank); });
            if (chosen == candidates.end()) { continue; }

            device_context output {};
            output.physical_device_ = std::move(*chosen);

            const auto binary_props =
              output.physical_device_
                .getProperties2<vk::PhysicalDeviceProperties2,
                  vk::PhysicalDevicePipelineBinaryPropertiesKHR>()
                .get<vk::PhysicalDevicePipelineBinaryPropertiesKHR>();

            auto enabled = resolve_enabled_features(
              profile, output.physical_device_, diagnostics);
            const auto mode =
              resolve_pipeline_mode(profile.pipeline, binary_props);
            const auto backend =
              profile.descriptor == descriptor_binding_request::heap
              ? descriptor_table_backend::heap
              : descriptor_table_backend::classic;

            const auto extensions =
              derive_device_extensions(enabled, requirements.extra_extensions);
            const auto feature_chain = make_enable_chain(enabled);

            vk::DevicePipelineBinaryInternalCacheControlKHR cache_control {};
            const void* p_next =
              &feature_chain.get<vk::PhysicalDeviceFeatures2>();
            if (mode == pipeline_persistence_mode::application_binary &&
              binary_props.pipelineBinaryInternalCacheControl == vk::True)
            {
              cache_control.pNext = p_next;
              cache_control.disableInternalCache = vk::True;
              p_next = &cache_control;
            }

            output.graphics_qf_index_ = requirements.require_present
              ? *find_graphics_present_qf(
                  output.physical_device_, instance.surface())
              : *find_graphics_qf(output.physical_device_);

            const float queue_priority { 0.5F };
            std::array<vk::DeviceQueueCreateInfo, 2> queue_create_infos {};
            std::uint32_t queue_create_info_count { 1U };
            queue_create_infos[ 0 ] = vk::DeviceQueueCreateInfo {
              .queueFamilyIndex = output.graphics_qf_index_,
              .queueCount = 1U,
              .pQueuePriorities = &queue_priority,
            };
            output.transfer_qf_index_ = output.graphics_qf_index_;
            if (requirements.request_dedicated_transfer)
            {
              if (const auto transfer_qf =
                    find_dedicated_transfer_qf(output.physical_device_))
              {
                output.transfer_qf_index_ = *transfer_qf;
                queue_create_infos[ 1 ] = vk::DeviceQueueCreateInfo {
                  .queueFamilyIndex = *transfer_qf,
                  .queueCount = 1,
                  .pQueuePriorities = &queue_priority,
                };
                queue_create_info_count = 2U;
              }
            }
            output.min_uniform_buffer_offset_alignment_ =
              output.physical_device_.getProperties()
                .limits.minUniformBufferOffsetAlignment;

            const vk::DeviceCreateInfo device_create_info {
              .pNext = p_next,
              .queueCreateInfoCount = queue_create_info_count,
              .pQueueCreateInfos = queue_create_infos.data(),
              .enabledExtensionCount =
                static_cast<std::uint32_t>(extensions.size()),
              .ppEnabledExtensionNames = extensions.data(),
            };

            return map_vk_error(
              output.physical_device_.createDevice(device_create_info),
              diagnostics)
              .transform(
                [ & ](vk::raii::Device&& device) -> void
                {
                  output.device_ = std::move(device);
                  output.graphics_queue_ =
                    output.device_.getQueue(output.graphics_qf_index_, 0);
                  output.transfer_queue_ =
                    output.device_.getQueue(output.transfer_qf_index_, 0);
                  output.msaa_samples_ =
                    get_max_usable_msaa_count(output.physical_device_);
                })
              .and_then(
                [ & ] -> std::expected<void, error_t>
                {
                  return vma_policy::create(*instance.instance(),
                    output.physical_device_, output.device_,
                    requirements.min_api_version,
                    enabled.buffer_device_address || enabled.descriptor_heap)
                    .transform([ & ](vma_policy&& policy) -> void
                      { output.allocator_ = std::move(policy); });
                })
              .transform(
                [ & ] -> device_context
                {
                  output.selected_ = selected_device_capabilities {
                    .profile_name = std::string { profile.name },
                    .profile_index = static_cast<std::uint32_t>(profile_index),
                    .descriptor = backend,
                    .pipeline = mode,
                    .enabled = enabled,
                    .pipeline_binary_properties = binary_props,
                    .descriptor_heap_enabled = enabled.descriptor_heap,
                    .pipeline_binary_enabled = enabled.pipeline_binary,
                  };
                  return std::move(output);
                });
          }

          return std::unexpected {
            make_app_error(app_error_code::no_suitable_gpu),
          };
        });
  }

  [[nodiscard]] auto
  physical_device(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.physical_device_); }

  [[nodiscard]] auto
  device(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.device_); }

  [[nodiscard]] auto
  graphics_queue(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.graphics_queue_); }

  [[nodiscard]] auto
  transfer_queue(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.transfer_queue_); }

  [[nodiscard]] auto
  allocator(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.allocator_); }

  [[nodiscard]] auto
  graphics_qf_index(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.graphics_qf_index_); }

  [[nodiscard]] auto
  transfer_qf_index(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.transfer_qf_index_); }

  [[nodiscard]] auto
  msaa_samples(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.msaa_samples_); }

  [[nodiscard]] auto
  graphics_pipeline_library_fast_linking() const -> bool
  { return graphics_pipeline_library_fast_linking_; }

  [[nodiscard]] auto
  min_uniform_buffer_offset_alignment(this auto&& self) -> decltype(auto)
  {
    return std::forward_like<decltype(self)>(
      self.min_uniform_buffer_offset_alignment_);
  }

  [[nodiscard]] auto
  selected_capabilities() const -> const selected_device_capabilities&
  { return selected_; }

  [[nodiscard]] auto
  has_dedicated_transfer() const -> bool
  { return transfer_qf_index_ != graphics_qf_index_; }

private:
  using device_feature_chain = vk::StructureChain<vk::PhysicalDeviceFeatures2,
    vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features,
    vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
    vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT,
    vk::PhysicalDeviceShaderObjectFeaturesEXT,
    vk::PhysicalDeviceDescriptorHeapFeaturesEXT,
    vk::PhysicalDevicePipelineBinaryFeaturesKHR>;

  [[nodiscard]] static constexpr auto
  make_enable_chain(const enabled_device_features& enabled)
    -> device_feature_chain
  {
    device_feature_chain chain {};
    const bool device_address =
      enabled.buffer_device_address || enabled.descriptor_heap;

    auto& core = chain.get<vk::PhysicalDeviceFeatures2>().features;
    core.samplerAnisotropy = vk::Bool32 { enabled.sampler_anisotropy };
    core.sampleRateShading = vk::Bool32 { enabled.sample_rate_shading };
    core.shaderInt64 = vk::Bool32 { device_address };

    auto& v12 = chain.get<vk::PhysicalDeviceVulkan12Features>();
    v12.timelineSemaphore = vk::Bool32 { enabled.timeline_semaphore };
    v12.hostQueryReset = vk::Bool32 { enabled.host_query_reset };
    v12.descriptorIndexing = vk::Bool32 { enabled.descriptor_indexing };
    v12.shaderSampledImageArrayNonUniformIndexing =
      vk::Bool32 { enabled.descriptor_indexing };
    v12.runtimeDescriptorArray = vk::Bool32 { enabled.descriptor_indexing };
    v12.descriptorBindingPartiallyBound =
      vk::Bool32 { enabled.descriptor_indexing };
    v12.descriptorBindingSampledImageUpdateAfterBind =
      vk::Bool32 { enabled.descriptor_indexing };
    v12.descriptorBindingUpdateUnusedWhilePending =
      vk::Bool32 { enabled.descriptor_indexing };
    v12.descriptorBindingVariableDescriptorCount =
      vk::Bool32 { enabled.descriptor_indexing };
    v12.bufferDeviceAddress = vk::Bool32 { device_address };

    auto& v13 = chain.get<vk::PhysicalDeviceVulkan13Features>();
    v13.dynamicRendering = vk::Bool32 { enabled.dynamic_rendering };
    v13.synchronization2 = vk::Bool32 { enabled.synchronization2 };

    auto& eds = chain.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
    eds.extendedDynamicState = vk::Bool32 { enabled.extended_dynamic_state };

    chain.get<vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT>()
      .graphicsPipelineLibrary =
      vk::Bool32 { enabled.graphics_pipeline_library };

    chain.get<vk::PhysicalDeviceShaderObjectFeaturesEXT>().shaderObject =
      vk::Bool32 { enabled.shader_object };

    chain.get<vk::PhysicalDeviceDescriptorHeapFeaturesEXT>().descriptorHeap =
      vk::Bool32 { enabled.descriptor_heap };

    chain.get<vk::PhysicalDevicePipelineBinaryFeaturesKHR>().pipelineBinaries =
      vk::Bool32 { enabled.pipeline_binary };

    return chain;
  }

  [[nodiscard]] static auto
  supports_requested_features(const vk::raii::PhysicalDevice& physical_device,
    const device_feature_requests& features) -> bool
  {
    const auto available =
      physical_device.getFeatures2<vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
        vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT,
        vk::PhysicalDeviceShaderObjectFeaturesEXT>();

    const auto& f2 = available.get<vk::PhysicalDeviceFeatures2>().features;
    const auto& v12 = available.get<vk::PhysicalDeviceVulkan12Features>();
    const auto& v13 = available.get<vk::PhysicalDeviceVulkan13Features>();
    const auto& eds =
      available.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();

    if (features.sampler_anisotropy == capability_requirement::required &&
      f2.samplerAnisotropy != vk::True)
    {
      return false;
    }
    if (features.sample_rate_shading == capability_requirement::required &&
      f2.sampleRateShading != vk::True)
    {
      return false;
    }
    if (features.dynamic_rendering == capability_requirement::required &&
      v13.dynamicRendering != vk::True)
    {
      return false;
    }
    if (features.synchronization2 == capability_requirement::required &&
      v13.synchronization2 != vk::True)
    {
      return false;
    }
    if (features.extended_dynamic_state == capability_requirement::required &&
      eds.extendedDynamicState != vk::True)
    {
      return false;
    }
    if (features.timeline_semaphore == capability_requirement::required &&
      v12.timelineSemaphore != vk::True)
    {
      return false;
    }
    if (features.host_query_reset == capability_requirement::required &&
      v12.hostQueryReset != vk::True)
    {
      return false;
    }
    if (features.descriptor_indexing == capability_requirement::required &&
      v12.descriptorIndexing != vk::True)
    {
      return false;
    }
    if (features.buffer_device_address == capability_requirement::required &&
      (v12.bufferDeviceAddress != vk::True || f2.shaderInt64 != vk::True))
    {
      return false;
    }
    if (features.graphics_pipeline_library ==
        capability_requirement::required &&
      available.get<vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT>()
          .graphicsPipelineLibrary != vk::True)
    {
      return false;
    }
    if (features.shader_object == capability_requirement::required &&
      available.get<vk::PhysicalDeviceShaderObjectFeaturesEXT>().shaderObject !=
        vk::True)
    {
      return false;
    }

    return true;
  }

  [[nodiscard]] static auto
  find_graphics_present_qf(const vk::raii::PhysicalDevice& physical_device,
    const vk::raii::SurfaceKHR& surface) -> std::optional<std::uint32_t>
  {
    const auto properties = physical_device.getQueueFamilyProperties();
    for (std::size_t property_index : std::views::indices(properties.size()))
    {
      const bool graphics = static_cast<bool>(
        properties[ property_index ].queueFlags & vk::QueueFlagBits::eGraphics);
      const bool present = static_cast<bool>(
        physical_device.getSurfaceSupportKHR(property_index, surface));
      if (graphics && present) { return property_index; }
    }
    return std::nullopt;
  }

  [[nodiscard]] static auto
  find_graphics_qf(const vk::raii::PhysicalDevice& physical_device)
    -> std::optional<std::uint32_t>
  {
    const auto properties = physical_device.getQueueFamilyProperties();
    for (std::size_t property_index : std::views::indices(properties.size()))
    {
      if (properties[ property_index ].queueFlags &
        vk::QueueFlagBits::eGraphics)
      {
        return property_index;
      }
    }
    return std::nullopt;
  }

  [[nodiscard]] static auto
  find_dedicated_transfer_qf(const vk::raii::PhysicalDevice& physical_device)
    -> std::optional<std::uint32_t>
  {
    const auto properties = physical_device.getQueueFamilyProperties();
    for (std::size_t property_index : std::views::indices(properties.size()))
    {
      const auto flags = properties[ property_index ].queueFlags;
      const bool transfer =
        static_cast<bool>(flags & vk::QueueFlagBits::eTransfer);
      const bool graphics_or_compute = static_cast<bool>(
        flags & (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute));
      if (transfer && !graphics_or_compute) { return property_index; }
    }
    return std::nullopt;
  }

  [[nodiscard]] static auto
  device_extensions_supported(const vk::raii::PhysicalDevice& physical_device,
    std::span<const char* const> required) -> bool
  {
    return physical_device.enumerateDeviceExtensionProperties()
      .transform(
        [ & ](std::span<const vk::ExtensionProperties> available)
        {
          return std::ranges::all_of(required,
            [ & ](const char* name)
            {
              return std::ranges::any_of(available,
                [ & ](const vk::ExtensionProperties& p)
                { return std::strcmp(p.extensionName, name) == 0; });
            });
        })
      .value_or(false);
  }

  [[nodiscard]] static auto
  get_max_usable_msaa_count(const vk::raii::PhysicalDevice& physical_device)
    -> vk::SampleCountFlagBits
  {
    const auto properties = physical_device.getProperties();
    const auto counts = properties.limits.framebufferColorSampleCounts &
      properties.limits.framebufferDepthSampleCounts;

    if (counts & vk::SampleCountFlagBits::e64)
    {
      return vk::SampleCountFlagBits::e64;
    }
    if (counts & vk::SampleCountFlagBits::e32)
    {
      return vk::SampleCountFlagBits::e32;
    }
    if (counts & vk::SampleCountFlagBits::e16)
    {
      return vk::SampleCountFlagBits::e16;
    }
    if (counts & vk::SampleCountFlagBits::e8)
    {
      return vk::SampleCountFlagBits::e8;
    }
    if (counts & vk::SampleCountFlagBits::e4)
    {
      return vk::SampleCountFlagBits::e4;
    }
    if (counts & vk::SampleCountFlagBits::e2)
    {
      return vk::SampleCountFlagBits::e2;
    }

    return vk::SampleCountFlagBits::e1;
  }

  [[nodiscard]] static auto
  resolve_pipeline_mode(pipeline_persistence_request request,
    const vk::PhysicalDevicePipelineBinaryPropertiesKHR& props)
    -> pipeline_persistence_mode
  {
    if (request == pipeline_persistence_request::pipeline_cache)
    {
      return pipeline_persistence_mode::pipeline_cache;
    }
    if (props.pipelineBinaryPrefersInternalCache == vk::True &&
      props.pipelineBinaryInternalCache == vk::True)
    {
      return pipeline_persistence_mode::internal_binary;
    }
    return pipeline_persistence_mode::application_binary;
  }

  [[nodiscard]] static auto
  extension_present(
    const vk::raii::PhysicalDevice& physical, std::string_view name) -> bool
  {
    return physical.enumerateDeviceExtensionProperties()
      .transform(
        [ & ](std::span<const vk::ExtensionProperties> available) -> bool
        {
          return std::ranges::any_of(available,
            [ & ](const vk::ExtensionProperties& properties) -> bool
            { return std::string_view { properties.extensionName } == name; });
        })
      .value_or(false);
  }

  [[nodiscard]] static auto
  profile_is_viable(const vk::raii::PhysicalDevice& physical,
    const device_profile& profile, const device_requirements& requirements,
    std::optional<diagnostic_buffer&> diagnostics) -> bool
  {
    if (physical.getProperties().apiVersion < requirements.min_api_version)
    {
      return false;
    }
    if (!device_extensions_supported(physical, requirements.extra_extensions))
    {
      return false;
    }
    if (!supports_requested_features(physical, profile.features))
    {
      return false;
    }
    if (profile.descriptor == descriptor_binding_request::heap)
    {
      if (!extension_present(physical, vk::EXTDescriptorHeapExtensionName) ||
        !descriptor_heap_supported(physical))
      {
        if (diagnostics.has_value())
        {
          diagnostics->report(diagnostic_severity::info, std::nullopt,
            std::source_location::current(), "profile '{}': heap not supported",
            profile.name);
        }
        return false;
      }
    }
    const auto features = physical.getFeatures2<vk::PhysicalDeviceFeatures2,
      vk::PhysicalDeviceVulkan12Features>();
    if (features.get<vk::PhysicalDeviceVulkan12Features>()
          .bufferDeviceAddress != vk::True ||
      features.get<vk::PhysicalDeviceFeatures2>().features.shaderInt64 !=
        vk::True)
    {
      if (diagnostics.has_value())
      {
        diagnostics->report(diagnostic_severity::info, std::nullopt,
          std::source_location::current(),
          "profile '{}': heap requires bufferDeviceAddress", profile.name);
      }
      return false;
    }

    if (profile.pipeline == pipeline_persistence_request::pipeline_binary)
    {
      if (!extension_present(physical, vk::KHRPipelineBinaryExtensionName))
      {
        if (diagnostics.has_value())
        {
          diagnostics->report(diagnostic_severity::info, std::nullopt,
            std::source_location::current(),
            "profile '{}': pipeline binary extension missing", profile.name);
        }
        return false;
      }
      const auto features = physical.getFeatures2<vk::PhysicalDeviceFeatures2,
        vk::PhysicalDevicePipelineBinaryFeaturesKHR>();
      if (features.get<vk::PhysicalDevicePipelineBinaryFeaturesKHR>()
            .pipelineBinaries != vk::True)
      {
        if (diagnostics.has_value())
        {
          diagnostics->report(diagnostic_severity::info, std::nullopt,
            std::source_location::current(),
            "profile '{}': pipelineBinaries feature missing", profile.name);
        }
        return false;
      }
    }

    if (!requirements.require_present &&
      !find_graphics_qf(physical).has_value())
    {
      return false;
    }

    return true;
  }

  [[nodiscard]] static auto
  resolve_enabled_features(const device_profile& profile,
    const vk::raii::PhysicalDevice& physical,
    std::optional<diagnostic_buffer&> diagnostics) -> enabled_device_features
  {
    enabled_device_features enabled {};

    auto apply = [ & ](capability_requirement requirement, bool supported,
                   std::string_view field_name) -> bool
    {
      if (requirement == capability_requirement::required) { return true; }
      if (requirement == capability_requirement::preferred)
      {
        if (!supported && diagnostics.has_value())
        {
          diagnostics->report(diagnostic_severity::info, std::nullopt,
            std::source_location::current(),
            "profile '{}': preferred {} unavailable", profile.name, field_name);
        }
        return supported;
      }
      return false;
    };

    const auto core = physical.getFeatures2<vk::PhysicalDeviceFeatures2,
      vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features,
      vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
      vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT,
      vk::PhysicalDeviceShaderObjectFeaturesEXT>();
    const auto& f2 = core.get<vk::PhysicalDeviceFeatures2>().features;
    const auto& v12 = core.get<vk::PhysicalDeviceVulkan12Features>();
    const auto& v13 = core.get<vk::PhysicalDeviceVulkan13Features>();
    const auto& eds =
      core.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();

    enabled.sampler_anisotropy = apply(profile.features.sampler_anisotropy,
      f2.samplerAnisotropy == vk::True, "sampler_anisotropy");
    enabled.sample_rate_shading = apply(profile.features.sample_rate_shading,
      f2.sampleRateShading == vk::True, "sample_rate_shading");
    enabled.dynamic_rendering = apply(profile.features.dynamic_rendering,
      v13.dynamicRendering == vk::True, "dynamic_rendering");
    enabled.synchronization2 = apply(profile.features.synchronization2,
      v13.synchronization2 == vk::True, "synchronization2");
    enabled.extended_dynamic_state =
      apply(profile.features.extended_dynamic_state,
        eds.extendedDynamicState == vk::True, "extended_dynamic_state");
    enabled.timeline_semaphore = apply(profile.features.timeline_semaphore,
      v12.timelineSemaphore == vk::True, "timeline_semaphore");
    enabled.host_query_reset = apply(profile.features.host_query_reset,
      v12.hostQueryReset == vk::True, "host_query_reset");
    enabled.descriptor_indexing = apply(profile.features.descriptor_indexing,
      v12.descriptorIndexing == vk::True, "descriptor_indexing");
    enabled.buffer_device_address =
      apply(profile.features.buffer_device_address,
        v12.bufferDeviceAddress == vk::True, "buffer_device_address");
    enabled.graphics_pipeline_library =
      apply(profile.features.graphics_pipeline_library,
        core.get<vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT>()
            .graphicsPipelineLibrary == vk::True,
        "graphics_pipeline_library");
    enabled.shader_object = apply(profile.features.shader_object,
      core.get<vk::PhysicalDeviceShaderObjectFeaturesEXT>().shaderObject ==
        vk::True,
      "shader_object");

    enabled.descriptor_heap =
      profile.descriptor == descriptor_binding_request::heap;
    enabled.pipeline_binary =
      profile.pipeline == pipeline_persistence_request::pipeline_binary;
    if (enabled.descriptor_heap) { enabled.buffer_device_address = true; }
    return enabled;
  }

  [[nodiscard]] static auto
  derive_device_extensions(const enabled_device_features& enabled,
    std::span<const char* const> extra) -> std::vector<const char*>
  {
    std::vector<const char*> names { std::from_range, extra };
    auto push_unique = [ & ](const char* name) -> void
    {
      if (!std::ranges::contains(names, std::string_view { name },
            [](const char* value) -> std::string_view
            { return std::string_view { value }; }))
      {
        names.push_back(name);
      }
    };
    if (enabled.extended_dynamic_state)
    {
      push_unique(vk::EXTExtendedDynamicStateExtensionName);
    }
    if (enabled.graphics_pipeline_library)
    {
      push_unique(vk::KHRPipelineLibraryExtensionName);
      push_unique(vk::EXTGraphicsPipelineLibraryExtensionName);
    }
    if (enabled.shader_object)
    {
      push_unique(vk::EXTShaderObjectExtensionName);
    }
    if (enabled.descriptor_heap)
    {
      push_unique(vk::EXTDescriptorHeapExtensionName);
    }
    if (enabled.pipeline_binary)
    {
      push_unique(vk::KHRPipelineBinaryExtensionName);
    }
    return names;
  }

  vk::raii::PhysicalDevice physical_device_ { nullptr };
  vk::raii::Device device_ { nullptr };
  vk::raii::Queue graphics_queue_ { nullptr };
  vk::raii::Queue transfer_queue_ { nullptr };
  vma_policy allocator_ {};
  std::uint32_t graphics_qf_index_ { ~0U };
  std::uint32_t transfer_qf_index_ { ~0U };
  vk::SampleCountFlagBits msaa_samples_ { vk::SampleCountFlagBits::e1 };
  bool graphics_pipeline_library_fast_linking_ { false };
  // there are 3 bytes of padding here, so possible new free 3 byte member here
  vk::DeviceSize min_uniform_buffer_offset_alignment_ { 1UZ };
  selected_device_capabilities selected_ {};
};

}; // namespace vkpp
