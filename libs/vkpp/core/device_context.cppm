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

export struct device_requirements
{
  std::span<const char* const> extensions {};
  std::uint32_t min_api_version { vk::ApiVersion13 };

  // TODOD: Konrad - maybe pass these flags/extensions as some struct
  bool sampler_anisotropy { true };
  bool dynamic_rendering { true };
  bool synchronization2 { true };
  bool extended_dynamic_state { true };
  bool sample_rate_shading { true };

  bool require_present { true };
};

export class device_context
{
public:
  // TODO: Konrad - Create additional (transfer) queue for any transfering
  // operations. Help:
  // https://docs.vulkan.org/tutorial/latest/04_Vertex_buffers/02_Staging_buffer.html#_transfer_queue
  [[nodiscard]] static auto
  create(
    const instance_context& instance, const device_requirements& requirements)
    -> std::expected<device_context, error_t>
  {
    return UTILS_VK(instance.instance().enumeratePhysicalDevices(),
      ^^vk::raii::Instance::enumeratePhysicalDevices)
      .and_then(
        [ & ](std::span<const vk::raii::PhysicalDevice> devices)
          -> std::expected<device_context, error_t>
        {
          const auto suitable_device_it = std::ranges::find_if(devices,
            [ & ](const vk::raii::PhysicalDevice& device)
            { return is_suitable(device, instance.surface(), requirements); });
          if (suitable_device_it == devices.end())
          {
            return std::unexpected {
              app_error {
                .kind = app_error_kind::no_suitable_gpu,
                .detail = "No suitable GPU found"sv,
              },
            };
          }

          device_context output {};
          output.physical_device_ = *suitable_device_it;
          output.graphics_qf_index_ = *find_graphics_present_qf(
            output.physical_device_, instance.surface());

          static constexpr float graphics_queue_priority { 0.5F };
          vk::DeviceQueueCreateInfo device_queue_create_info {
            .queueFamilyIndex = output.graphics_qf_index_,
            .queueCount = 1,
            .pQueuePriorities = &graphics_queue_priority,
          };
          const vk::StructureChain feature_chain {
              vk::PhysicalDeviceFeatures2 {
                .features = {
                  .sampleRateShading = vk::Bool32{requirements.sample_rate_shading},
                  .samplerAnisotropy = vk::Bool32{requirements.sampler_anisotropy},
                },
              },
              vk::PhysicalDeviceVulkan13Features {
                .synchronization2 = vk::Bool32{requirements.synchronization2},
                .dynamicRendering = vk::Bool32{requirements.dynamic_rendering},
              },
              vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT {
                .extendedDynamicState = vk::Bool32{requirements.extended_dynamic_state},
              },
            };
          const vk::DeviceCreateInfo device_create_info {
            .pNext = &feature_chain.get<vk::PhysicalDeviceFeatures2>(),
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &device_queue_create_info,
            .enabledExtensionCount =
              static_cast<std::uint32_t>(requirements.extensions.size()),
            .ppEnabledExtensionNames = requirements.extensions.data(),
          };

          return UTILS_VK(
            output.physical_device_.createDevice(device_create_info),
            ^^vk::raii::PhysicalDevice::createDevice)
            .and_then(
              [ & ](vk::raii::Device&& device)
              {
                output.device_ = std::move(device);
                output.graphics_queue_ =
                  output.device_.getQueue(output.graphics_qf_index_, 0);
                output.msaa_samples_ =
                  get_max_usable_msaa_count(output.physical_device_);
                return vma_policy::create(*instance.instance(),
                  output.physical_device_, output.device_,
                  requirements.min_api_version);
              })
            .transform(
              [ & ](vma_policy&& policy) -> device_context
              {
                output.allocator_ = std::move(policy);
                return std::move(output);
              });
        });
  }

  [[nodiscard]] auto
  physical_device() const -> const vk::raii::PhysicalDevice&
  { return physical_device_; }

  [[nodiscard]] auto
  device() const -> const vk::raii::Device&
  { return device_; }

  [[nodiscard]] auto
  graphics_queue() const -> const vk::raii::Queue&
  { return graphics_queue_; }

  // TODO: Konrad - all getters should be rewritten like the one below (use:
  // deducing this)
  [[nodiscard]] auto
  allocator(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.allocator_); }

  [[nodiscard]] auto
  graphics_qf_index() const -> const std::uint32_t
  { return graphics_qf_index_; }

  [[nodiscard]] auto
  msaa_samples() const -> const vk::SampleCountFlagBits
  { return msaa_samples_; }

private:
  [[nodiscard]] static auto
  find_graphics_present_qf(const vk::raii::PhysicalDevice& physical_device,
    const vk::raii::SurfaceKHR& surface) -> std::optional<std::uint32_t>
  {
    const auto properties = physical_device.getQueueFamilyProperties();
    for (std::size_t property_index : std::views::iota(0UZ, properties.size()))
    {
      const bool graphics = static_cast<bool>(
        properties[ property_index ].queueFlags & vk::QueueFlagBits::eGraphics);
      const auto present =
        physical_device.getSurfaceSupportKHR(property_index, surface);
      if (graphics && present) { return property_index; }
    }

    return std::nullopt;
  }

  [[nodiscard]] static auto
  is_suitable(const vk::raii::PhysicalDevice& physical_device,
    const vk::raii::SurfaceKHR& surface,
    const device_requirements& requirements) -> bool
  {

    if (physical_device.getProperties().apiVersion <
      requirements.min_api_version)
    {
      return false;
    }
    if (requirements.require_present &&
      !find_graphics_present_qf(physical_device, surface))
    {
      return false;
    }

    const auto queue_family_properties =
      physical_device.getQueueFamilyProperties();
    const bool supports_graphics = std::ranges::any_of(queue_family_properties,
      [](const auto& qf_property)
      {
        return (static_cast<bool>(
          qf_property.queueFlags & vk::QueueFlagBits::eGraphics));
      });

    const bool supports_required_device_extensions =
      physical_device.enumerateDeviceExtensionProperties()
        .transform(
          [ & ](std::span<const vk::ExtensionProperties>
              available_device_extensions)
          {
            return std::ranges::all_of(requirements.extensions,
              [ &available_device_extensions ](
                const auto& required_device_extension)
              {
                return std::ranges::any_of(available_device_extensions,
                  [ required_device_extension ](
                    const auto& available_device_extension)
                  {
                    return strcmp(available_device_extension.extensionName,
                             required_device_extension) == 0;
                  });
              });
          })
        .value_or(false);

    const auto features =
      physical_device.getFeatures2<vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
    const bool supports_required_features =
      features.get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy ==
        vk::Bool32 { requirements.sampler_anisotropy } &&
      features.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering ==
        vk::Bool32 { requirements.dynamic_rendering } &&
      features.get<vk::PhysicalDeviceVulkan13Features>().synchronization2 ==
        vk::Bool32 { requirements.synchronization2 } &&
      features.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>()
          .extendedDynamicState ==
        vk::Bool32 { requirements.extended_dynamic_state };

    return supports_required_device_extensions && supports_required_features;
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
  vma_policy allocator_ {};
  std::uint32_t graphics_qf_index_ { ~0U };
  vk::SampleCountFlagBits msaa_samples_ { vk::SampleCountFlagBits::e1 };
};

}; // namespace vkpp
