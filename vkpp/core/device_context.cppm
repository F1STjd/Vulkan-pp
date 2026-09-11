module;

#include "error/vk_error_config.hpp"

export module vkpp.device;

import std;
import vulkan;

import vkpp.instance;
import vkpp.memory.vma;
import vkpp.error;

namespace vkpp
{
using namespace std::string_view_literals;

export struct device_feature_requests
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
};

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

export struct physical_device_rank_policy
{
  bool prefer_discrete { true };
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

// require Vulkan 1.4 and every feature vkpp needs to construct/run today
// if gpu is limitted then some workarounds should be searched for
// ^^ will be done in far future
export struct device_requirements
{
  std::span<const char* const> extensions {};
  std::uint32_t min_api_version { vk::ApiVersion14 };
  device_feature_requests features {};
  bool require_present { true };
  bool request_dedicated_transfer { false };
  physical_device_rank_policy rank {};
};

export class device_context
{
public:
  [[nodiscard]] static auto
  create(
    const instance_context& instance, const device_requirements& requirements)
    -> std::expected<device_context, error_t>
  {
    return UTILS_VK(instance.instance().enumeratePhysicalDevices(),
      ^^vk::raii::Instance::enumeratePhysicalDevices)
      .and_then(
        [ & ](std::vector<vk::raii::PhysicalDevice>&& devices)
          -> std::expected<device_context, error_t>
        {
          const auto surface_ref = requirements.require_present
            ? std::optional { std::cref(instance.surface()) }
            : std::nullopt;

          auto suitable_devices = devices |
            std::views::filter(std::bind_back(
              is_suitable, surface_ref, std::cref(requirements)));
          auto device = std::ranges::max_element(suitable_devices, std::less {},
            [ &requirements ](
              const vk::raii::PhysicalDevice& device) -> std::int64_t
            { return score_physical_device(device, requirements.rank); });
          if (device == suitable_devices.end())
          {
            return std::unexpected {
              app_error {
                .kind = app_error_kind::no_suitable_gpu,
                .detail = "No suitable GPU found"sv,
              },
            };
          }

          device_context output {};
          output.physical_device_ = std::move(*device);
          // Todo: Konrad - is_suitable() already computes qf index, maybe there
          // is a way no to repeat this computation
          output.graphics_qf_index_ = requirements.require_present
            ? *find_graphics_present_qf(
                output.physical_device_, instance.surface())
            : *find_graphics_qf(output.physical_device_);

          static constexpr float queue_priority { 0.5F };
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

          const vk::StructureChain feature_chain =
            make_enable_chain(requirements.features);
          const vk::DeviceCreateInfo device_create_info {
            .pNext = &feature_chain.get<vk::PhysicalDeviceFeatures2>(),
            .queueCreateInfoCount = queue_create_info_count,
            .pQueueCreateInfos = queue_create_infos.data(),
            .enabledExtensionCount =
              static_cast<std::uint32_t>(requirements.extensions.size()),
            .ppEnabledExtensionNames = requirements.extensions.data(),
          };

          return UTILS_VK(
            output.physical_device_.createDevice(device_create_info),
            ^^vk::raii::PhysicalDevice::createDevice)
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
              [ & ]() -> std::expected<void, error_t>
              {
                return vma_policy::create(*instance.instance(),
                  output.physical_device_, output.device_,
                  requirements.min_api_version,
                  requirements.features.buffer_device_address)
                  .transform([ & ](vma_policy&& policy) -> void
                    { output.allocator_ = std::move(policy); });
              })
            .transform([ & ]() -> device_context { return std::move(output); });
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
  min_uniform_buffer_offset_alignment(this auto&& self) -> decltype(auto)
  {
    return std::forward_like<decltype(self)>(
      self.min_uniform_buffer_offset_alignment_);
  }

  [[nodiscard]] auto
  has_dedicated_transfer() const -> bool
  { return transfer_qf_index_ != graphics_qf_index_; }

private:
  using device_feature_chain = vk::StructureChain<vk::PhysicalDeviceFeatures2,
    vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features,
    vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>;

  [[nodiscard]] static constexpr auto
  make_enable_chain(const device_feature_requests& requests)
    -> device_feature_chain
  {
    device_feature_chain chain {};

    auto& core = chain.get<vk::PhysicalDeviceFeatures2>().features;
    core.samplerAnisotropy = vk::Bool32 { requests.sampler_anisotropy };
    core.sampleRateShading = vk::Bool32 { requests.sample_rate_shading };
    core.shaderInt64 = vk::Bool32 { requests.buffer_device_address };

    auto& v12 = chain.get<vk::PhysicalDeviceVulkan12Features>();
    v12.timelineSemaphore = vk::Bool32 { requests.timeline_semaphore };
    v12.hostQueryReset = vk::Bool32 { requests.host_query_reset };
    v12.descriptorIndexing = vk::Bool32 { requests.descriptor_indexing };
    v12.shaderSampledImageArrayNonUniformIndexing =
      vk::Bool32 { requests.descriptor_indexing };
    v12.runtimeDescriptorArray = vk::Bool32 { requests.descriptor_indexing };
    v12.descriptorBindingPartiallyBound =
      vk::Bool32 { requests.descriptor_indexing };
    v12.descriptorBindingSampledImageUpdateAfterBind =
      vk::Bool32 { requests.descriptor_indexing };
    v12.descriptorBindingUpdateUnusedWhilePending =
      vk::Bool32 { requests.descriptor_indexing };
    v12.descriptorBindingVariableDescriptorCount =
      vk::Bool32 { requests.descriptor_indexing };
    v12.bufferDeviceAddress = vk::Bool32 { requests.buffer_device_address };

    auto& v13 = chain.get<vk::PhysicalDeviceVulkan13Features>();
    v13.dynamicRendering = vk::Bool32 { requests.dynamic_rendering };
    v13.synchronization2 = vk::Bool32 { requests.synchronization2 };

    auto& eds = chain.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
    eds.extendedDynamicState = vk::Bool32 { requests.extended_dynamic_state };

    return chain;
  }

  [[nodiscard]] static auto
  supports_requested_features(const vk::raii::PhysicalDevice& physical_device,
    const device_feature_requests& requests) -> bool
  {
    const auto available =
      physical_device.getFeatures2<vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();

    const auto& core = available.get<vk::PhysicalDeviceFeatures2>().features;
    const auto& v12 = available.get<vk::PhysicalDeviceVulkan12Features>();
    const auto& v13 = available.get<vk::PhysicalDeviceVulkan13Features>();
    const auto& eds =
      available.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();

    if (requests.sampler_anisotropy && core.samplerAnisotropy != vk::True)
    {
      return false;
    }
    if (requests.sample_rate_shading && core.sampleRateShading != vk::True)
    {
      return false;
    }
    if (requests.timeline_semaphore && v12.timelineSemaphore != vk::True)
    {
      return false;
    }
    if (requests.host_query_reset && v12.hostQueryReset != vk::True)
    {
      return false;
    }
    if (requests.descriptor_indexing)
    {
      if (v12.descriptorIndexing != vk::True ||
        v12.shaderSampledImageArrayNonUniformIndexing != vk::True ||
        v12.runtimeDescriptorArray != vk::True ||
        v12.descriptorBindingPartiallyBound != vk::True ||
        v12.descriptorBindingSampledImageUpdateAfterBind != vk::True ||
        v12.descriptorBindingUpdateUnusedWhilePending != vk::True ||
        v12.descriptorBindingVariableDescriptorCount != vk::True)
      {
        return false;
      }
    }
    if (requests.buffer_device_address)
    {
      if (v12.bufferDeviceAddress != vk::True) { return false; }
      if (core.shaderInt64 != vk::True) { return false; }
    }
    {
      if (requests.dynamic_rendering && v13.dynamicRendering != vk::True)
      {
        return false;
      }
    }
    if (requests.synchronization2 && v13.synchronization2 != vk::True)
    {
      return false;
    }
    if (requests.extended_dynamic_state && eds.extendedDynamicState != vk::True)
    {
      return false;
    }

    return true;
  }

  [[nodiscard]] static auto
  features_extensions_consistent(const device_requirements& requirements)
    -> bool
  {
    if (!requirements.features.extended_dynamic_state) { return true; }
    return std::ranges::contains(requirements.extensions,
      std::string_view { vk::EXTExtendedDynamicStateExtensionName },
      [](const char* ext) { return std::string_view { ext }; });
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
  is_suitable(const vk::raii::PhysicalDevice& physical_device,
    std::optional<std::reference_wrapper<const vk::raii::SurfaceKHR>> surface,
    const device_requirements& requirements) -> bool
  {
    if (physical_device.getProperties().apiVersion <
      requirements.min_api_version)
    {
      return false;
    }
    if (!features_extensions_consistent(requirements)) { return false; }
    if (!device_extensions_supported(physical_device, requirements.extensions))
    {
      return false;
    }
    if (!supports_requested_features(physical_device, requirements.features))
    {
      return false;
    }
    if (requirements.require_present)
    {
      if (!surface.has_value()) { return false; }
      return find_graphics_present_qf(physical_device, surface->get())
        .has_value();
    }
    return find_graphics_qf(physical_device).has_value();
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

  vk::raii::PhysicalDevice physical_device_ { nullptr };
  vk::raii::Device device_ { nullptr };
  vk::raii::Queue graphics_queue_ { nullptr };
  vk::raii::Queue transfer_queue_ { nullptr };
  vma_policy allocator_ {};
  std::uint32_t graphics_qf_index_ { ~0U };
  std::uint32_t transfer_qf_index_ { ~0U };
  vk::SampleCountFlagBits msaa_samples_ { vk::SampleCountFlagBits::e1 };
  // there are 4 bytes of padding here, so possible new free 4 byte member here
  vk::DeviceSize min_uniform_buffer_offset_alignment_ { 1UZ };
};

}; // namespace vkpp
