module;

#include "contracts_config.hpp"
#include "error/vk_error_config.hpp"

#include <SFML/Window.hpp>
#include <SFML/Window/VideoMode.hpp>
#include <vulkan/vk_platform.h>

#include "imgui_platform.hpp"

#include <backends/imgui_impl_vulkan.h>
#include <imgui.h>
#include <imgui_internal.h>

export module f1st.app;

import std;
import vulkan;
import glm;
import f1st.uniform_buffer;
import vkpp.io.types;
import vkpp.io;
import vkpp.io.mesh;
import vkpp.io.mesh.gltf;
import vkpp.error;
import vkpp.vertex;
import vkpp.memory;
import vkpp.memory.vma;
import vkpp.memory.virtual_block;
import vkpp.image;
import vkpp.buffer;
import vkpp.buffer.upload;
import vkpp.buffer.arena;
import vkpp.buffer.arena.upload;
import vkpp.instance;
import vkpp.device;
import vkpp.swapchain;
import vkpp.command;
import vkpp.frame;
import vkpp.frame_attachments;
import vkpp.texture;
import vkpp.texture.upload;
import vkpp.barrier;
import vkpp.pipeline;
import vkpp.pipeline.compute;
import vkpp.descriptor;
import vkpp.descriptor.indexing;
import vkpp.semaphore;
import vkpp.graph;
import vkpp.query;

namespace f1st
{
using namespace std::string_view_literals;

export constexpr const char* model_path { MODEL_DIRECTORY "911.glb" };

constexpr std::uint32_t window_width { 800 };
constexpr std::uint32_t window_height { 600 };

// During the development i want validation layers (for corectness) in the
// release build
#ifndef NDEBUG
constexpr std::array validation_layers {
  "VK_LAYER_KHRONOS_validation",
  // This one is not checked in the code :(, but should be
  "VK_LAYER_LUNARG_monitor",
};
constexpr bool enable_validation_layers { true };
#else
constexpr std::array<const char*, 0> validation_layers {};
constexpr bool enable_validation_layers { false };
#endif

[[nodiscard]] auto
required_instance_extensions() -> std::vector<const char*>
{
  const auto& sfml = sf::Vulkan::getGraphicsRequiredInstanceExtensions();
  std::vector<const char*> extensions { std::from_range, sfml };
  if constexpr (enable_validation_layers)
  {
    extensions.push_back(vk::EXTDebugUtilsExtensionName);
  }
  return extensions;
}

constexpr std::array required_device_extensions {
  vk::KHRSwapchainExtensionName,
  vk::EXTExtendedDynamicStateExtensionName,
};

constexpr std::size_t max_frames_in_flight { 2UZ };
static_assert(max_frames_in_flight > 0,
  "variable % max_frames_in_flight is used later, so being 0 is UB");

static constexpr vkpp::graphics_pipeline_spec k_pipeline_spec {
  .sample_shading = false,
  .min_sample_shading = 0.2F,
};

static constexpr vkpp::graphics_pipeline_spec k_blend_pipeline_spec {
  .depth_write = false,
  .blend_enable = true,
  .sample_shading = false,
  .min_sample_shading = 0.2F,
};

static constexpr std::array k_set0_bindings {
  vk::DescriptorSetLayoutBinding {
    .binding = 0U,
    .descriptorType = vk::DescriptorType::eUniformBuffer,
    .descriptorCount = 1U,
    .stageFlags =
      vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
    .pImmutableSamplers = nullptr,
  },
  vk::DescriptorSetLayoutBinding {
    .binding = 1U,
    .descriptorType = vk::DescriptorType::eStorageBuffer,
    .descriptorCount = 1U,
    .stageFlags = vk::ShaderStageFlagBits::eFragment,
    .pImmutableSamplers = nullptr,
  },
  vk::DescriptorSetLayoutBinding {
    .binding = 3U,
    .descriptorType = vk::DescriptorType::eStorageBuffer,
    .descriptorCount = 1U,
    .stageFlags =
      vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
    .pImmutableSamplers = nullptr,
  },
};

export class app
{
public:
  void
  run()
  {
    const auto result = init_vulkan().and_then(
      [ this ]() -> std::expected<void, vkpp::error_t> { return main_loop(); });

    (void)device_.device().waitIdle();
    shutdown_imgui();
    if (!result) { std::println(stderr, "{}", vkpp::message(result.error())); }
    swap_chain_.release();
  }

private:
  auto
  init_vulkan() -> std::expected<void, vkpp::error_t>
  {
    return create_instance_context()
      .and_then(std::bind_front(&app::create_surface, this))
      .and_then(std::bind_front(&app::create_device_context, this))
      .and_then(std::bind_front(&app::create_sampler_cache, this))
      .and_then(std::bind_front(&app::create_swap_chain, this))
      .and_then(std::bind_front(&app::create_command_pool, this))
      .and_then(std::bind_front(&app::create_upload_pool, this))
      .and_then(std::bind_front(&app::run_compute_smoke, this))
      .and_then(std::bind_front(&app::create_transfer_upload_pool, this))
      .and_then(std::bind_front(&app::create_frames, this))
      .and_then(std::bind_front(&app::create_frame_timeline, this))
      .and_then(std::bind_front(&app::create_timestamp_ring, this))
      .and_then(std::bind_front(&app::create_descriptor_set_layout, this))
      .and_then(std::bind_front(&app::create_bindless_table, this))
      .and_then(std::bind_front(&app::create_buffers, this))
      .and_then(std::bind_front(&app::create_descriptor_pool, this))
      .and_then(std::bind_front(&app::create_descriptor_sets, this))
      .and_then(std::bind_front(&app::create_graphics_pipelines, this))
      .and_then(std::bind_front(&app::create_imgui_descriptor_pool, this))
      .and_then(std::bind_front(&app::init_imgui, this))
      .and_then(std::bind_front(&app::create_scene_sampler, this));
  }

  auto
  main_loop() -> std::expected<void, vkpp::error_t>
  {
    while (window_.isOpen())
    {
      while (const auto event = window_.pollEvent())
      {
        imgui_process_event(*event);
        if (event->is<sf::Event::Closed>()) { window_.close(); }
        if (event->is<sf::Event::Resized>()) { resized_ = true; }
      }
      if (!window_.isOpen()) { break; }
      if (auto result = draw_frame(); !result) { return result; }
    }
    return UTILS_VK(device_.device().waitIdle(), ^^vk::raii::Device::waitIdle);
  }

private:
  auto
  create_instance_context() -> std::expected<void, vkpp::error_t>
  {
    static constexpr vk::ApplicationInfo app_info {
      .pApplicationName = "f1st",
      .applicationVersion = vk::makeVersion(1, 0, 0),
      .pEngineName = "No Engine",
      .engineVersion = vk::makeVersion(1, 0, 0),
      .apiVersion = vk::ApiVersion14,
    };
    const auto extensions = required_instance_extensions();
    return vkpp::instance_context::create(
      {
        .app_info = app_info,
        .extensions = extensions,
        .layers = validation_layers,
        .enable_validation = enable_validation_layers,
      })
      .transform([ this ](vkpp::instance_context&& context)
        { instance_ = std::move(context); });
  }

  auto
  create_surface() -> std::expected<void, vkpp::error_t>
  {
    VkSurfaceKHR _surface {};
    if (!window_.createVulkanSurface(*instance_.instance(), _surface))
    {
      return std::unexpected {
        vkpp::app_error {
          .kind = vkpp::app_error_kind::surface_creation,
          .detail = "Faild to create window surface"sv,
        },
      };
    }
    instance_.adopt_surface(
      vk::raii::SurfaceKHR(instance_.instance(), _surface));
    return {};
  }

  auto
  create_device_context() -> std::expected<void, vkpp::error_t>
  {
    return vkpp::device_context::create(instance_,
      {
        .extensions = required_device_extensions,
        .min_api_version = vk::ApiVersion13,
        .features = {
          .sampler_anisotropy = true,
          .sample_rate_shading = true,
          .dynamic_rendering = true,
          .synchronization2 = true,
          .extended_dynamic_state = true,
          .timeline_semaphore = true,
          .host_query_reset = true,
          .descriptor_indexing = true,
        },
        .require_present = true,
        .request_dedicated_transfer = true,
      })
      .transform([ this ](vkpp::device_context&& device) -> void
        { device_ = std::move(device); });
  }

  auto
  create_sampler_cache() -> std::expected<void, vkpp::error_t>
  {
    return vkpp::sampler_cache::create(
      device_.device(), device_.physical_device())
      .transform([ this ](vkpp::sampler_cache&& cache) -> void
        { sampler_cache_.emplace(std::move(cache)); });
  }

  auto
  create_swap_chain() -> std::expected<void, vkpp::error_t>
  {
    return vkpp::swapchain::create(device_, instance_.surface(),
      framebuffer_extent_request(),
      [ this ](const vk::SurfaceCapabilitiesKHR& capabilities,
        vk::Extent2D framebuffer)
      { return choose_swap_extent(capabilities, framebuffer); })
      .transform([ this ](vkpp::swapchain&& swap_chain)
        { swap_chain_ = std::move(swap_chain); });
  }

  auto
  framebuffer_extent_request() const -> vkpp::extent_request
  {
    const auto [ width, height ] = window_.getSize();
    return {
      .framebuffer_size = { width, height },
    };
  }

  auto
  choose_swap_extent(const vk::SurfaceCapabilitiesKHR& capabilities,
    vk::Extent2D framebuffer) -> vk::Extent2D
  {
    if (capabilities.currentExtent.width !=
      std::numeric_limits<std::uint32_t>::max())
    {
      return capabilities.currentExtent;
    }
    return {
      .width = std::clamp(framebuffer.width, capabilities.minImageExtent.width,
        capabilities.maxImageExtent.width),
      .height = std::clamp(framebuffer.height,
        capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
    };
  }

  auto
  create_descriptor_set_layout() -> std::expected<void, vkpp::error_t>
  {
    return vkpp::make_descriptor_set_layout(device_.device(), k_set0_bindings)
      .transform([ this ](vk::raii::DescriptorSetLayout&& layout) -> void
        { descriptor_set_layout_ = std::move(layout); });
  }

  auto
  create_bindless_table() -> std::expected<void, vkpp::error_t>
  {
    return vkpp::bindless_table::create(device_.device(),
      { .capacity = 256U, .stages = vk::ShaderStageFlagBits::eFragment })
      .transform([ this ](vkpp::bindless_table&& table) -> void
        { bindless_table_ = std::move(table); });
  }

  auto
  create_graphics_pipelines() -> std::expected<void, vkpp::error_t>
  {
    constexpr std::array vertex_bindings {
      vkpp::vertex::get_binding_description(),
    };
    constexpr auto vertex_attributes =
      vkpp::vertex::get_attribute_descriptions();
    const std::array color_formats { swap_chain_.format() };
    const std::array set_layouts {
      *descriptor_set_layout_,
      *bindless_table_.layout(),
    };

    // Again we nest +1 every time dependency is left up behind (here spirv).
    // Maybe we should create helper about creating more than one
    // graphics_pipeline, so spirv could be shared as one nesting level. Also
    // like I wrote earlier: there should be some utils function that calls some
    // function and the initialises the passed object with resulting value. <- I
    // do not know if this is good idea. It chould be called `do_and_init`, or
    // `dai` if it is too long. What about function with a lot of arguments,
    // this API then could look bad. In perfect world API should look like this:
    // return load_shader()
    //   .and_then(create_first_pipeline)  // <- also initialises the data
    //   member .and_then(create_second_pipeline);

    return vkpp::find_depth_attachment_format(device_.physical_device())
      .and_then(
        [ &, this ](
          vk::Format depth_format) -> std::expected<void, vkpp::error_t>
        {
          const vkpp::graphics_pipeline_runtime_args runtime_args {
            .color_formats = color_formats,
            .depth_format = depth_format,
            .samples = device_.msaa_samples(),
            .set_layouts = set_layouts,
            .vertex_bindings = vertex_bindings,
            .vertex_attributes = vertex_attributes,
            .push_constant_size = static_cast<std::uint32_t>(sizeof(draw_push)),
            .push_constant_stages = vk::ShaderStageFlagBits::eVertex |
              vk::ShaderStageFlagBits::eFragment,
          };
          return vkpp::load_shader_file(SHADER_DIRECTORY "slang.spv")
            .and_then(
              [ &, this ](const std::vector<char>& spirv)
                -> std::expected<void, vkpp::error_t>
              {
                return vkpp::make_graphics_pipeline<k_pipeline_spec>(
                  device_.device(), runtime_args, { .spirv = spirv })
                  .and_then(
                    [ &, this ](vkpp::graphics_pipeline&& opaque)
                      -> std::expected<void, vkpp::error_t>
                    {
                      graphics_pipeline_ = std::move(opaque);
                      return vkpp::make_graphics_pipeline<
                        k_blend_pipeline_spec>(
                        device_.device(), runtime_args, { .spirv = spirv })
                        .transform(
                          [ this ](vkpp::graphics_pipeline&& blend) -> void
                          { blend_pipeline_ = std::move(blend); });
                    });
              });
        });
  }

  auto
  create_command_pool() -> std::expected<void, vkpp::error_t>
  {
    return vkpp::command_pool::create(device_.device(),
      device_.graphics_qf_index(),
      vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
      .transform([ this ](vkpp::command_pool&& pool)
        { command_pool_ = std::move(pool); });
  }

  auto
  create_upload_pool() -> std::expected<void, vkpp::error_t>
  {
    return vkpp::command_pool::create(device_.device(),
      device_.graphics_qf_index(), vk::CommandPoolCreateFlagBits::eTransient)
      .transform([ this ](vkpp::command_pool&& pool) -> void
        { upload_pool_ = std::move(pool); });
  }

  auto
  run_compute_smoke() -> std::expected<void, vkpp::error_t>
  {
    constexpr vk::DeviceSize byte_size { 64UZ * sizeof(std::uint32_t) };
    constexpr std::array compute_bindings {
      vk::DescriptorSetLayoutBinding {
        .binding = 2U,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1U,
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
      },
    };

    auto ssbo = vkpp::make_buffer_resource(device_.allocator(), byte_size,
      vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferSrc,
      vkpp::memory_intent::gpu_only);
    if (!ssbo) { return std::unexpected { std::move(ssbo).error() }; }

    auto staging = vkpp::make_buffer_resource(device_.allocator(), byte_size,
      vk::BufferUsageFlagBits::eTransferDst, vkpp::memory_intent::gpu_to_cpu);
    if (!staging) { return std::unexpected { std::move(staging).error() }; }
    if (staging->mapped() == nullptr)
    {
      return std::unexpected {
        vkpp::app_error {
          .kind = vkpp::app_error_kind::mapping_failed,
          .detail = "compute smoke staging buffer not mapped"sv,
        },
      };
    }

    auto set_layout =
      vkpp::make_descriptor_set_layout(device_.device(), compute_bindings);
    if (!set_layout)
    {
      return std::unexpected { std::move(set_layout).error() };
    }

    auto pool_sizes = vkpp::pool_sizes_for(compute_bindings, 1U);
    auto pool = vkpp::descriptor_pool::create(device_.device(), 1U, pool_sizes);
    if (!pool) { return std::unexpected { std::move(pool).error() }; }

    auto sets = pool->allocate(device_.device(), *set_layout, 1U);
    if (!sets) { return std::unexpected { std::move(sets).error() }; }

    const vk::DescriptorBufferInfo buffer_info {
      .buffer = ssbo->buffer(),
      .offset = 0UZ,
      .range = byte_size,
    };
    const vk::WriteDescriptorSet write {
      .dstSet = (*sets)[ 0 ],
      .dstBinding = 2U,
      .dstArrayElement = 0U,
      .descriptorCount = 1U,
      .descriptorType = vk::DescriptorType::eStorageBuffer,
      .pBufferInfo = &buffer_info,
    };
    vkpp::update_descriptor_sets(device_.device(), std::span { &write, 1UZ });

    auto spirv = vkpp::load_shader_file(SHADER_DIRECTORY "slang.spv");
    if (!spirv) { return std::unexpected { std::move(spirv).error() }; }

    auto pipeline = vkpp::make_compute_pipeline(device_.device(),
      vkpp::compute_pipeline_runtime_args {
        .set_layout = *set_layout,
      },
      vkpp::compute_shader {
        .spirv = *spirv,
      });

    vkpp::single_time_submit submit {
      upload_pool_,
      device_.device(),
      device_.graphics_queue(),
    };
    if (auto begun = submit.begin(); !begun) { return begun; }
    auto& command_buffer = submit.command_buffer();
    command_buffer.bindPipeline(
      vk::PipelineBindPoint::eCompute, *pipeline->pipeline());
    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
      *pipeline->layout(), 0U, (*sets)[ 0 ], nullptr);
    command_buffer.dispatch(1U, 1U, 1U);
    const vkpp::buffer_use_transition after_compute {
      .buffer = ssbo->buffer(),
      .from = vkpp::buffer_use::storage_compute_write,
      .to = vkpp::buffer_use::transfer_src,
    };
    vkpp::record_buffer_use_transitions(
      command_buffer, std::span { &after_compute, 1UZ });
    command_buffer.copyBuffer(
      ssbo->buffer(), staging->buffer(), vk::BufferCopy { .size = byte_size });
    auto submitted = submit.end_and_submit(vkpp::upload::deferred);
    if (!submitted) { return std::unexpected { submitted.error() }; }
    if (auto waited = submitted->wait(); !waited)
    {
      return std::unexpected { waited.error() };
    }
    const auto* words = static_cast<const std::uint32_t*>(staging->mapped());
    for (std::uint32_t i = 0U; i < 64U; ++i)
    {
      if (words[ i ] != i)
      {
        return std::unexpected {
          vkpp::app_error {
            .kind = vkpp::app_error_kind::invalid_argument,
            .detail = "compute smoke SSBO contents mismatch"sv,
          },
        };
      }
    }
    compute_smoke_ok_ = true;
    return {};
  }

  auto
  create_transfer_upload_pool() -> std::expected<void, vkpp::error_t>
  {
    return vkpp::command_pool::create(device_.device(),
      device_.transfer_qf_index(), vk::CommandPoolCreateFlagBits::eTransient)
      .transform([ this ](vkpp::command_pool&& pool) -> void
        { transfer_upload_pool_ = std::move(pool); });
  }

  auto
  create_buffers() -> std::expected<void, vkpp::error_t>
  {
    return vkpp::load_gltf_asset_cpu(model_path,
      {
        .content = vkpp::gltf::content_policy::geometry_and_host_images,
      })
      .and_then(
        [ &, this ](
          vkpp::gltf::asset_cpu&& asset) -> std::expected<void, vkpp::error_t>
        {
          draw_list_ = std::move(asset.draw_list);
          blend_order_.clear();
          blend_order_.reserve(draw_list_.size());

          auto default_sampler = sampler_cache_->get_or_create({});
          if (!default_sampler)
          {
            return std::unexpected {
              std::move(default_sampler).error(),
            };
          }

          textures_.clear();
          textures_.reserve(asset.host_images.size());
          for (auto image_index : std::views::indices(asset.host_images.size()))
          {
            const auto& host = asset.host_images[ image_index ];
            std::expected<vkpp::texture<>, vkpp::error_t> made;
            if (host.decoded.has_value())
            {
              const auto& image = *host.decoded;
              made = vkpp::make_texture({
                .device = device_,
                .pool = upload_pool_,
                .transfer_pool = transfer_upload_pool_,
                .pixels = image.pixels,
                .extent = image.extent,
                .format = image.format,
                .mip_levels = image.mip_levels_present,
                .mip_policy = image.suggested_mip_policy,
                .sampler = {},
                .borrowed_sampler = *default_sampler,
              });
            }
            else if (host.mip_chain.has_value())
            {
              const auto& chain = *host.mip_chain;
              made = vkpp::make_texture({
                .device = device_,
                .pool = upload_pool_,
                .transfer_pool = transfer_upload_pool_,
                .pixels = chain.texels,
                .extent = chain.base_extent,
                .format = chain.format,
                .mip_levels = chain.mip_levels_present,
                .mip_policy = chain.suggested_mip_policy,
                .sampler = {},
                .borrowed_sampler = *default_sampler,
                .level_offsets = chain.level_offsets,
              });
            }
            else
            {
              return std::unexpected {
                vkpp::app_error {
                  .kind = vkpp::app_error_kind::invalid_argument,
                  .detail = "gltf host image empty after realize"sv,
                },
              };
            }
            if (!made) { return std::unexpected { std::move(made).error() }; }
            textures_.push_back(std::move(*made));
          }

          const auto sampler_create_info_for =
            [ & ](const vkpp::gltf::texture_ref_cpu& reference)
            -> std::expected<vkpp::sampler_create_info, vkpp::error_t>
          {
            if (!reference.sampler_index.has_value())
            {
              return vkpp::sampler_create_info {};
            }
            if (*reference.sampler_index >= asset.samplers.size())
            {
              return std::unexpected {
                vkpp::app_error {
                  .kind = vkpp::app_error_kind::invalid_argument,
                  .detail = "gltf sampler index out of range"sv,
                },
              };
            }
            const auto& source = asset.samplers[ *reference.sampler_index ];
            return vkpp::sampler_create_info {
              .mag_filter = source.mag_filter,
              .min_filter = source.min_filter,
              .mipmap_mode = source.mipmap_mode,
              .address_mode_u = source.address_u,
              .address_mode_v = source.address_v,
              .address_mode_w = vk::SamplerAddressMode::eRepeat,
              .anisotropy_enable = true,
              .min_lod = 0.0F,
              .max_lod = source.max_lod,
            };
          };

          std::vector<std::optional<std::uint32_t>> texture_to_slot(
            asset.textures.size(), std::nullopt);
          for (auto texture_index : std::views::indices(asset.textures.size()))
          {
            const auto& reference = asset.textures[ texture_index ];
            const std::optional<std::uint32_t> image_index =
              reference.basisu_image_index.has_value()
              ? reference.basisu_image_index
              : reference.image_index;
            if (!image_index.has_value()) { continue; }
            if (*image_index >= textures_.size())
            {
              return std::unexpected {
                vkpp::app_error {
                  .kind = vkpp::app_error_kind::invalid_argument,
                  .detail = "gltf texture image index out of range"sv,
                },
              };
            }

            auto sampler_create_info = sampler_create_info_for(reference);
            if (!sampler_create_info)
            {
              return std::unexpected { std::move(sampler_create_info).error() };
            }
            auto sampler = sampler_cache_->get_or_create(*sampler_create_info);
            if (!sampler)
            {
              return std::unexpected { std::move(sampler).error() };
            }
            const auto slot = bindless_table_.acquire_index();
            if (!slot)
            {
              return std::unexpected {
                vkpp::app_error {
                  .kind = vkpp::app_error_kind::invalid_argument,
                  .detail = "bindless_table capacity exhausted"sv,
                },
              };
            }
            bindless_table_.write(device_.device(), *slot, *sampler,
              *textures_[ *image_index ].view());
            texture_to_slot[ texture_index ] = *slot;
          }

          const auto same_texture_reference =
            [](const vkpp::gltf::texture_ref_cpu& lhs,
              const vkpp::gltf::texture_ref_cpu& rhs) -> bool
          {
            return lhs.image_index == rhs.image_index &&
              lhs.basisu_image_index == rhs.basisu_image_index &&
              lhs.sampler_index == rhs.sampler_index;
          };

          const auto resolve_slot =
            [ & ](const std::optional<vkpp::gltf::texture_ref_cpu>& ref,
              std::uint32_t bit, std::uint32_t& mask)
            -> std::expected<std::uint32_t, vkpp::error_t>
          {
            if (!ref.has_value()) { return 0U; }
            for (auto texture_index :
              std::views::indices(asset.textures.size()))
            {
              if (!same_texture_reference(
                    *ref, asset.textures[ texture_index ]))
              {
                continue;
              }
              if (!texture_to_slot[ texture_index ].has_value()) { break; }
              mask |= bit;
              return *texture_to_slot[ texture_index ];
            }

            return std::unexpected {
              vkpp::app_error {
                .kind = vkpp::app_error_kind::invalid_argument,
                .detail = "material texture reference was not realized"sv,
              },
            };
          };

          std::vector<material_gpu> materials_gpu {};
          materials_gpu.reserve(asset.materials.size() + 1UZ);
          for (const auto& material : asset.materials)
          {
            std::uint32_t texture_mask { 0U };
            const auto base_color_index =
              resolve_slot(material.base_color_texture, 1U, texture_mask);
            if (!base_color_index)
            {
              return std::unexpected { std::move(base_color_index).error() };
            }
            const auto metallic_roughness_index = resolve_slot(
              material.metallic_roughness_texture, 2U, texture_mask);
            if (!metallic_roughness_index)
            {
              return std::unexpected {
                std::move(metallic_roughness_index).error()
              };
            }
            const auto normal_index =
              resolve_slot(material.normal_texture, 4U, texture_mask);
            if (!normal_index)
            {
              return std::unexpected { std::move(normal_index).error() };
            }
            const auto occlusion_index =
              resolve_slot(material.occlusion_texture, 8U, texture_mask);
            if (!occlusion_index)
            {
              return std::unexpected { std::move(occlusion_index).error() };
            }
            const auto emissive_index =
              resolve_slot(material.emissive_texture, 16U, texture_mask);
            if (!emissive_index)
            {
              return std::unexpected { std::move(emissive_index).error() };
            }

            materials_gpu.push_back({
                .base_color_factor = material.base_color_factor,
                .emissive_factor_and_metallic = {
                  material.emissive_factor[0],
                  material.emissive_factor[1],
                  material.emissive_factor[2],
                  material.metallic_factor,
                },
                .roughness_factor = material.roughness_factor,
                .normal_scale = material.normal_scale,
                .occlusion_strength = material.occlusion_strength,
                .alpha_cutoff = material.alpha_cutoff,
                .base_color_index = *base_color_index,
                .metallic_roughness_index = *metallic_roughness_index,
                .normal_index = *normal_index,
                .occlusion_index = *occlusion_index,
                .emissive_index = *emissive_index,
                .alpha_mode = static_cast<std::uint32_t>(material.alpha_mode),
                .has_texture_mask = texture_mask,
                .transmission_factor = material.transmission_factor,
            });
          }
          materials_gpu.emplace_back();

          auto uploaded_materials = vkpp::upload_device_local_buffer({
            .device = device_,
            .pool = upload_pool_,
            .transfer_pool = transfer_upload_pool_,
            .bytes =
              std::as_bytes(std::span<const material_gpu> { materials_gpu }),
            .gpu_usage = vk::BufferUsageFlagBits::eStorageBuffer,
          });
          if (!uploaded_materials)
          {
            return std::unexpected {
              std::move(uploaded_materials).error(),
            };
          }
          material_buffer_ = std::move(*uploaded_materials);

          std::vector<draw_gpu> draws_gpu {};
          draws_gpu.reserve(draw_list_.size());
          const std::uint32_t default_material_index =
            static_cast<std::uint32_t>(asset.materials.size());
          for (const vkpp::gltf::draw_item_cpu& item : draw_list_)
          {
            if (item.primitive_index >= asset.meshes.primitives.size())
            {
              return std::unexpected {
                vkpp::app_error {
                  .kind = vkpp::app_error_kind::invalid_argument,
                  .detail = "draw item primitive index out of range"sv,
                },
              };
            }
            const vkpp::mesh_streams_cpu& primitive =
              asset.meshes.primitives[ item.primitive_index ];
            const std::uint32_t material_index =
              primitive.material_index.has_value() &&
                *primitive.material_index < asset.materials.size()
              ? *primitive.material_index
              : default_material_index;
            draws_gpu.push_back({
              .world = item.world_transform,
              .normal_matrix_columns = normal_matrix_of(item.world_transform),
              .material_index = material_index,
            });
          }
          auto uploaded_draws = vkpp::upload_device_local_buffer({
            .device = device_,
            .pool = upload_pool_,
            .transfer_pool = transfer_upload_pool_,
            .bytes = std::as_bytes(std::span<const draw_gpu> { draws_gpu }),
            .gpu_usage = vk::BufferUsageFlagBits::eStorageBuffer,
          });
          if (!uploaded_draws)
          {
            return std::unexpected { std::move(uploaded_draws).error() };
          }
          draw_buffer_ = std::move(*uploaded_draws);

          std::vector<vkpp::mesh_interleaved_cpu> packed;
          std::vector<vkpp::gltf::alpha_mode> alpha_modes;
          std::vector<std::uint8_t> double_sided;
          packed.reserve(asset.meshes.primitives.size());
          alpha_modes.reserve(asset.meshes.primitives.size());
          double_sided.reserve(asset.meshes.primitives.size());

          for (const auto& primitive : asset.meshes.primitives)
          {
            packed.push_back(vkpp::pack_interleaved_vertices(primitive));
            vkpp::gltf::alpha_mode mode { vkpp::gltf::alpha_mode::opaque };
            bool two_sided { false };
            if (primitive.material_index.has_value() &&
              *primitive.material_index < asset.materials.size())
            {
              const auto& material =
                asset.materials[ *primitive.material_index ];
              mode = material.alpha_mode;
              if (material.transmission_factor > 0.0F &&
                mode == vkpp::gltf::alpha_mode::opaque)
              {
                mode = vkpp::gltf::alpha_mode::blend;
              }
              two_sided =
                material.double_sided || material.transmission_factor > 0.0F;
            }
            alpha_modes.push_back(mode);
            double_sided.push_back(static_cast<std::uint8_t>(two_sided));
          }

          vk::DeviceSize arena_size { 0UZ };
          for (const auto& pack : packed)
          {
            arena_size += static_cast<vk::DeviceSize>(pack.vertices.size());
            const vk::DeviceSize index_align =
              (pack.index_type == vk::IndexType::eUint32) ? 4UZ : 2UZ;
            arena_size =
              (arena_size + index_align - 1UZ) / index_align * index_align;
            arena_size += static_cast<vk::DeviceSize>(pack.indices.size());
          }

          return vkpp::make_buffer_arena(device_.allocator(), arena_size,
            vk::BufferUsageFlagBits::eVertexBuffer |
              vk::BufferUsageFlagBits::eIndexBuffer,
            vkpp::memory_intent::gpu_only)
            .and_then(
              [ &, this ](vkpp::buffer_arena<>&& arena)
                -> std::expected<void, vkpp::error_t>
              {
                geometry_arena_ = std::move(arena);
                draws_.clear();
                draws_.reserve(packed.size());

                std::vector<vkpp::arena_slice_upload> slices;
                slices.reserve(packed.size() * 2UZ);

                for (auto index : std::views::indices(packed.size()))
                {
                  const auto& pack = packed[ index ];
                  const vk::DeviceSize index_align =
                    (pack.index_type == vk::IndexType::eUint32) ? 4UZ : 2UZ;
                  auto vertex = geometry_arena_.allocate(
                    static_cast<vk::DeviceSize>(pack.vertices.size()), 0UZ);
                  if (!vertex) { return std::unexpected { vertex.error() }; }

                  auto index_slice = geometry_arena_.allocate(
                    static_cast<vk::DeviceSize>(pack.indices.size()),
                    index_align);
                  if (!index_slice)
                  {
                    return std::unexpected { index_slice.error() };
                  }

                  slices.push_back({
                    .bytes = pack.vertices,
                    .dst_offset = vertex->offset,
                  });
                  slices.push_back({
                    .bytes = pack.indices,
                    .dst_offset = index_slice->offset,
                  });
                  draws_.push_back({
                    .vertex_slice = *vertex,
                    .index_slice = *index_slice,
                    .index_count = pack.index_count,
                    .index_type = pack.index_type,
                    .alpha_mode = alpha_modes[ index ],
                    .double_sided = static_cast<bool>(double_sided[ index ]),
                  });
                }

                return vkpp::upload_arena_slices({
                  .device = device_,
                  .pool = upload_pool_,
                  .transfer_pool = transfer_upload_pool_,
                  .arena = geometry_arena_.buffer(),
                  .slices = slices,
                });
              });
        });
  }

  auto
  create_frames() -> std::expected<void, vkpp::error_t>
  {
    return vkpp::create_frames<max_frames_in_flight>(
      {
        .device = device_,
        .pool = command_pool_,
        .ubo_size = sizeof(uniform_buffer_object),
      })
      .transform(
        [ this ](std::array<vkpp::frame, max_frames_in_flight>&& frames) -> void
        { frames_ = std::move(frames); });
  }

  auto
  create_frame_timeline() -> std::expected<void, vkpp::error_t>
  {
    return vkpp::make_timeline_semaphore(device_.device())
      .transform([ this ](vk::raii::Semaphore&& semaphore) -> void
        { frame_timeline_ = std::move(semaphore); });
  }

  auto
  create_timestamp_ring() -> std::expected<void, vkpp::error_t>
  {
    return vkpp::timestamp_ring::create(device_.device(),
      device_.physical_device(),
      static_cast<std::uint32_t>(max_frames_in_flight), 2U)
      .transform([ this ](vkpp::timestamp_ring&& ring) -> void
        { timestamps_ = std::move(ring); });
  }

  void
  update_uniform_buffer(std::uint32_t current_frame)
  {
    static auto start_time = std::chrono::high_resolution_clock::now();
    auto current_time = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration<float>(current_time - start_time).count();
    static constexpr auto degrees { glm::radians(90.0F) };
    static constexpr auto camera_position { glm::vec3 { 2.0F, 2.0F, 2.0F } };
    static constexpr auto target { glm::vec3 { 0.0F, 0.0F, 0.0F } };
    static constexpr auto up { glm::vec3 { 0.0F, 1.0F, 0.0F } };
    static constexpr auto fov_vertical { glm::radians(45.0F) };
    const auto aspect_ratio { //
      static_cast<float>(std::max(1U, scene_extent_.width)) /
      static_cast<float>(std::max(1U, scene_extent_.height))
    };
    static constexpr auto near_plane { 0.1F };
    static constexpr auto far_plane { 10.0F };

    model_matrix_ = glm::gtc::rotate(
      glm::mat4 { 1.0F }, time * degrees, glm::vec3 { 0.0F, 1.0F, 0.0F });
    view_matrix_ = glm::gtc::lookAt(camera_position, target, up);
    uniform_buffer_object ubo {
      .model = model_matrix_,
      .view = view_matrix_,
      .projection = glm::gtc::perspective(
        fov_vertical, aspect_ratio, near_plane, far_plane),
      .light_direction = glm::normalize(glm::vec3 { 0.35F, 0.85F, 0.40F }),
      .light_color = glm::vec3 { 4.0F, 3.85F, 3.6F },
      .camera_position = camera_position,
      .exposure = 1.0F,
    };
    ubo.projection[ 1 ][ 1 ] *= -1;
    frames_[ current_frame ].uniform_buffer.write(&ubo, sizeof(ubo));
  }

  auto
  create_descriptor_pool() -> std::expected<void, vkpp::error_t>
  {
    auto sizes = vkpp::pool_sizes_for(k_set0_bindings, max_frames_in_flight);
    return vkpp::descriptor_pool::create(
      device_.device(), max_frames_in_flight, sizes)
      .transform([ this ](vkpp::descriptor_pool&& pool) -> void
        { descriptor_pool_ = std::move(pool); });
  }

  auto
  create_descriptor_sets() -> std::expected<void, vkpp::error_t>
  {
    return descriptor_pool_
      .allocate(device_.device(), descriptor_set_layout_, max_frames_in_flight)
      .transform(
        [ this ](std::vector<vk::DescriptorSet>&& sets) -> void
        {
          for (auto index : std::views::indices(max_frames_in_flight))
          {
            frames_[ index ].descriptor_set = sets[ index ];
            const vk::DescriptorBufferInfo uniform_buffer_info {
              .buffer = frames_[ index ].uniform_buffer.buffer(),
              .offset = 0U,
              .range = sizeof(uniform_buffer_object),
            };
            const vk::DescriptorBufferInfo material_buffer_info {
              .buffer = material_buffer_.buffer(),
              .offset = 0U,
              .range = material_buffer_.size(),
            };
            const vk::DescriptorBufferInfo draw_buffer_info {
              .buffer = draw_buffer_.buffer(),
              .offset = 0U,
              .range = draw_buffer_.size(),
            };
            const std::array writes {
              vk::WriteDescriptorSet {
                .dstSet = frames_[ index ].descriptor_set,
                .dstBinding = 0U,
                .dstArrayElement = 0U,
                .descriptorCount = 1U,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo = &uniform_buffer_info,
              },
              vk::WriteDescriptorSet {
                .dstSet = frames_[ index ].descriptor_set,
                .dstBinding = 1U,
                .dstArrayElement = 0U,
                .descriptorCount = 1U,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &material_buffer_info,
              },
              vk::WriteDescriptorSet {
                .dstSet = frames_[ index ].descriptor_set,
                .dstBinding = 3U,
                .dstArrayElement = 0U,
                .descriptorCount = 1U,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &draw_buffer_info,
              },
            };
            vkpp::update_descriptor_sets(
              device_.device(), std::span { writes });
          }
        });
  }

  auto
  create_imgui_descriptor_pool() -> std::expected<void, vkpp::error_t>
  {
    constexpr std::array pool_sizes {
      vk::DescriptorPoolSize {
        .type = vk::DescriptorType::eSampledImage,
        .descriptorCount = 16U,
      },
      vk::DescriptorPoolSize {
        .type = vk::DescriptorType::eSampler,
        .descriptorCount = 4U,
      },
    };
    const vk::DescriptorPoolCreateInfo create_info {
      .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      .maxSets = 32U,
      .poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size()),
      .pPoolSizes = pool_sizes.data(),
    };
    return UTILS_VK(device_.device().createDescriptorPool(create_info),
      ^^vk::raii::Device::createDescriptorPool)
      .transform([ this ](vk::raii::DescriptorPool&& pool) -> void
        { imgui_descriptor_pool_ = std::move(pool); });
  }

  auto
  init_imgui_renderer() -> std::expected<void, vkpp::error_t>
  {
    const auto image_count =
      static_cast<std::uint32_t>(swap_chain_.images().size());
    auto color_format = static_cast<VkFormat>(swap_chain_.format());
    VkPipelineRenderingCreateInfo pipeline_rendering_info {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1U,
      .pColorAttachmentFormats = &color_format,
    };

    ImGui_ImplVulkan_InitInfo init_info {
      .ApiVersion = VK_API_VERSION_1_4,
      .Instance = static_cast<VkInstance>(*instance_.instance()),
      .PhysicalDevice =
        static_cast<VkPhysicalDevice>(*device_.physical_device()),
      .Device = static_cast<VkDevice>(*device_.device()),
      .QueueFamily = device_.graphics_qf_index(),
      .Queue = static_cast<VkQueue>(*device_.graphics_queue()),
      .DescriptorPool = static_cast<VkDescriptorPool>(*imgui_descriptor_pool_),
      .MinImageCount = image_count,
      .ImageCount = image_count,
      .PipelineInfoMain = {
        .MSAASamples = VK_SAMPLE_COUNT_1_BIT,
        .PipelineRenderingCreateInfo = pipeline_rendering_info,
      },
      .UseDynamicRendering = true,
      .MinAllocationSize = 1024U * 1024U,
    };

    if (!ImGui_ImplVulkan_Init(&init_info))
    {
      return std::unexpected {
        vkpp::app_error {
          .kind = vkpp::app_error_kind::invalid_argument,
          .detail = "ImGui Vulkan backend initialisation failed"sv,
        },
      };
    }
    imgui_renderer_initialized_ = true;
    return {};
  }

  auto
  init_imgui() -> std::expected<void, vkpp::error_t>
  {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.BackendPlatformName = "f1st_imgui_platform_sfml3";
    return init_imgui_renderer();
  }

  auto
  create_scene_sampler() -> std::expected<void, vkpp::error_t>
  {
    const vk::SamplerCreateInfo create_info {
      .magFilter = vk::Filter::eLinear,
      .minFilter = vk::Filter::eLinear,
      .mipmapMode = vk::SamplerMipmapMode::eNearest,
      .addressModeU = vk::SamplerAddressMode::eClampToEdge,
      .addressModeV = vk::SamplerAddressMode::eClampToEdge,
      .addressModeW = vk::SamplerAddressMode::eClampToEdge,
      .minLod = 0.0F,
      .maxLod = 0.25F,
    };
    return UTILS_VK(device_.device().createSampler(create_info),
      ^^vk::raii::Device::createSampler)
      .transform([ this ](vk::raii::Sampler&& sampler) -> void
        { scene_sampler_ = std::move(sampler); });
  }

  auto
  rebuild_scene_targets(vk::Extent2D extent)
    -> std::expected<void, vkpp::error_t>
  {
    if (scene_texture_id_ != VK_NULL_HANDLE)
    {
      ImGui_ImplVulkan_RemoveTexture(scene_texture_id_);
      scene_texture_id_ = VK_NULL_HANDLE;
    }
    scene_ = {};
    scene_extent_ = vk::Extent2D {};

    return vkpp::frame_attachments::create(device_,
      { .extent = extent,
        .color_format = swap_chain_.format(),
        .samples = device_.msaa_samples(),
        .sink = vkpp::color_sink::sampled })
      .and_then(
        [ this, extent ](
          vkpp::frame_attachments&& scene) -> std::expected<void, vkpp::error_t>
        {
          scene_ = std::move(scene);
          scene_extent_ = extent;
          scene_texture_id_ = ImGui_ImplVulkan_AddTexture(
            static_cast<VkImageView>(*scene_.resolve().view()),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
          if (scene_texture_id_ == VK_NULL_HANDLE)
          {
            return std::unexpected {
              vkpp::app_error {
                .kind = vkpp::app_error_kind::invalid_argument,
                .detail = "ImGui_ImplVulkan_AddTexture returned null"sv,
              },
            };
          }
          return {};
        });
  }

  auto
  ensure_scene_targets() -> std::expected<void, vkpp::error_t>
  {
    vk::Extent2D extent = scene_desired_extent_;
    if (extent.width == 0U) { extent.width = 1U; }
    if (extent.height == 0U) { extent.height = 1U; }
    if (extent.width == scene_extent_.width &&
      extent.height == scene_extent_.height &&
      scene_texture_id_ != VK_NULL_HANDLE)
    {
      return {};
    }
    return UTILS_VK(device_.device().waitIdle(), ^^vk::raii::Device::waitIdle)
      .and_then([ this, extent ]() -> std::expected<void, vkpp::error_t>
        { return rebuild_scene_targets(extent); });
  }

  void
  shutdown_imgui_renderer()
  {
    if (!imgui_renderer_initialized_) { return; }
    if (scene_texture_id_ != VK_NULL_HANDLE)
    {
      ImGui_ImplVulkan_RemoveTexture(scene_texture_id_);
      scene_texture_id_ = {};
    }
    ImGui_ImplVulkan_Shutdown();
    imgui_renderer_initialized_ = false;
  }

  void
  shutdown_imgui()
  {
    shutdown_imgui_renderer();
    if (ImGui::GetCurrentContext() != nullptr)
    {
      ImGui::GetIO().BackendPlatformName = nullptr;
      ImGui::DestroyContext();
    }
    imgui_descriptor_pool_ = nullptr;
  }

  void
  begin_imgui_frame()
  {
    imgui_platform_new_frame(window_, imgui_frame_time_);
    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();
    build_imgui_chrome();
  }

  void
  finish_imgui_frame()
  {
    build_imgui_viewport_image();
    ImGui::Render();
  }

  void
  build_imgui_chrome()
  {
    const auto* viewport = ImGui::GetMainViewport();
    const auto dockspace_id = ImHashStr("f1st DockSpace");

    if (!imgui_layout_built_)
    {
      ImGui::DockBuilderRemoveNode(dockspace_id);
      ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
      ImGui::DockBuilderSetNodeSize(dockspace_id, viewport->Size);

      ImGuiID center = dockspace_id;
      ImGuiID inspector {};
      ImGuiID log {};
      ImGui::DockBuilderSplitNode(
        center, ImGuiDir_Left, 0.22F, &inspector, &center);
      ImGui::DockBuilderSplitNode(center, ImGuiDir_Down, 0.18F, &log, &center);
      ImGui::DockBuilderDockWindow("Inspector", inspector);
      ImGui::DockBuilderDockWindow("Log", log);
      ImGui::DockBuilderDockWindow("Viewport", center);
      ImGui::DockBuilderFinish(dockspace_id);
      imgui_layout_built_ = true;
    }

    ImGui::DockSpaceOverViewport(dockspace_id, viewport);

    ImGui::Begin("Viewport");
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    scene_desired_extent_ = vk::Extent2D {
      .width = static_cast<std::uint32_t>(std::max(1.0F, avail.x)),
      .height = static_cast<std::uint32_t>(std::max(1.0F, avail.y)),
    };
    ImGui::End();

    ImGui::Begin("Inspector");
    ImGui::Text("GPU: %.3f ms", gpu_ms_);
    ImGui::Text("Compute smoke: %s", compute_smoke_ok_ ? "OK" : "not verified");
    ImGui::Text("Skipped BLEND draws: %u", skipped_blend_draws_);
    ImGui::Text("Frames in flight: %zu", max_frames_in_flight);
    ImGui::Text("Swapchain: %u x %u", swap_chain_.extent().width,
      swap_chain_.extent().height);
    ImGui::Text("Viewport: %u x %u", scene_extent_.width, scene_extent_.height);
    ImGui::End();

    ImGui::Begin("Log");
    ImGui::TextUnformatted("Validation and renderer messages remain on stderr");
    ImGui::End();
  }

  void
  build_imgui_viewport_image()
  {
    ImGui::Begin("Viewport");
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    if (scene_texture_id_ != VK_NULL_HANDLE)
    {
      ImGui::Image(scene_texture_id_, avail);
    }
    ImGui::End();
  }

  [[nodiscard]] static auto
  is_mirrored(const std::array<float, 16>& world) -> bool
  {
    const glm::vec3 axis_x { world[ 0 ], world[ 1 ], world[ 2 ] };
    const glm::vec3 axis_y { world[ 4 ], world[ 5 ], world[ 6 ] };
    const glm::vec3 axis_z { world[ 8 ], world[ 9 ], world[ 10 ] };
    return glm::dot(glm::cross(axis_x, axis_y), axis_z) < 0.0F;
  }

  [[nodiscard]] static auto
  normal_matrix_of(const std::array<float, 16>& world) -> std::array<float, 12>
  {
    const glm::mat4 world_matrix {
      glm::vec4 { world[ 0 ], world[ 1 ], world[ 2 ], world[ 3 ] },
      glm::vec4 { world[ 4 ], world[ 5 ], world[ 6 ], world[ 7 ] },
      glm::vec4 { world[ 8 ], world[ 9 ], world[ 10 ], world[ 11 ] },
      glm::vec4 { world[ 12 ], world[ 13 ], world[ 14 ], world[ 15 ] },
    };
    const glm::mat3 normal_matrix =
      glm::transpose(glm::inverse(glm::mat3 { world_matrix }));

    std::array<float, 12> columns {};
    for (auto column : std::views::indices(3UZ))
    {
      for (auto row : std::views::indices(3UZ))
      {
        columns[ column * 4UZ + row ] = normal_matrix[ column ][ row ];
      }
    }
    return columns;
  }

  void
  bind_graphics_pass(vk::raii::CommandBuffer& command_buffer,
    const vkpp::graphics_pipeline& pipeline)
  {
    command_buffer.bindPipeline(
      vk::PipelineBindPoint::eGraphics, *pipeline.pipeline());
    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
      *pipeline.layout(), 0U, frames_[ frame_index_ ].descriptor_set, nullptr);
    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
      *pipeline.layout(), 1U, bindless_table_.set(), nullptr);
  }

  void
  record_draw_item(vk::raii::CommandBuffer& command_buffer,
    std::uint32_t draw_index, const vkpp::graphics_pipeline& pipeline)
  {
    const auto& item = draw_list_[ draw_index ];
    const auto& draw = draws_[ item.primitive_index ];
    command_buffer.bindVertexBuffers(
      0U, geometry_arena_.buffer(), { draw.vertex_slice.offset });
    command_buffer.bindIndexBuffer(
      geometry_arena_.buffer(), draw.index_slice.offset, draw.index_type);
    const draw_push push_constants { .draw_index = draw_index };
    command_buffer.pushConstants(*pipeline.layout(),
      vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0U,
      sizeof(push_constants), &push_constants);
    command_buffer.setCullMode(draw.double_sided //
        ? vk::CullModeFlagBits::eNone
        : vk::CullModeFlagBits::eBack);
    command_buffer.setFrontFace(is_mirrored(item.world_transform)
        ? vk::FrontFace::eClockwise
        : vk::FrontFace::eCounterClockwise);
    command_buffer.drawIndexed(draw.index_count, 1U, 0U, 0U, 0U);
  }

  auto
  record_command_buffer(std::uint32_t image_index)
    -> std::expected<void, vkpp::error_t>
  {
    image_uses_.reset();
    auto& command_buffer = frames_[ frame_index_ ].command_buffer;
    return UTILS_VK(command_buffer.begin({}), ^^vk::raii::CommandBuffer::begin)
      .transform(
        [ this, image_index, &command_buffer ]() -> void
        {
          timestamps_.write(command_buffer, frame_index_, 0U,
            vk::PipelineStageFlagBits2::eNone);

          scene_.record_begin_uses(image_uses_, command_buffer);

          vk::ClearValue clear_color {
            vk::ClearColorValue {
              0.0F,
              0.0F,
              0.0F,
              1.0F,
            },
          };
          vk::ClearValue clear_depth {
            vk::ClearDepthStencilValue {
              .depth = 1.0F,
              .stencil = 0,
            },
          };
          vk::RenderingAttachmentInfo color_attachment_info =
            scene_.color_attachment_info(clear_color);
          vk::RenderingAttachmentInfo depth_attachment_info =
            scene_.depth_attachment_info(clear_depth);

          vk::RenderingInfo rendering_info {
            .renderArea = { .extent = scene_extent_ },
            .layerCount = 1U,
            .colorAttachmentCount = 1U,
            .pColorAttachments = &color_attachment_info,
            .pDepthAttachment = &depth_attachment_info,
          };
          command_buffer.beginRendering(rendering_info);
          command_buffer.setViewport(0U,
            vk::Viewport {
              .x = 0.0F,
              .y = 0.0F,
              .width = static_cast<float>(scene_extent_.width),
              .height = static_cast<float>(scene_extent_.height),
              .minDepth = 0.0F,
              .maxDepth = 1.0F,
            });
          command_buffer.setScissor(0U,
            vk::Rect2D {
              .extent = scene_extent_,
            });
          skipped_blend_draws_ = 0U;
          bind_graphics_pass(command_buffer, graphics_pipeline_);
          for (auto draw_index : std::views::indices(draw_list_.size()))
          {
            const auto& item = draw_list_[ draw_index ];
            if (draws_[ item.primitive_index ].alpha_mode ==
              vkpp::gltf::alpha_mode::blend)
            {
              continue;
            }
            record_draw_item(command_buffer,
              static_cast<std::uint32_t>(draw_index), graphics_pipeline_);
          }

          blend_order_.clear();
          for (auto draw_index : std::views::indices(draw_list_.size()))
          {
            const auto& item = draw_list_[ draw_index ];
            if (draws_[ item.primitive_index ].alpha_mode !=
              vkpp::gltf::alpha_mode::blend)
            {
              continue;
            }
            const glm::vec4 world_origin {
              item.world_transform[ 12 ],
              item.world_transform[ 13 ],
              item.world_transform[ 14 ],
              1.0F,
            };
            const glm::vec4 view_origin =
              view_matrix_ * model_matrix_ * world_origin;
            blend_order_.push_back({
              .draw_index = static_cast<std::uint32_t>(draw_index),
              .view_depth = view_origin.z,
            });
          }
          std::ranges::sort(blend_order_, {}, &blend_entry::view_depth);

          bind_graphics_pass(command_buffer, blend_pipeline_);
          for (const auto& entry : blend_order_)
          {
            record_draw_item(command_buffer, entry.draw_index, blend_pipeline_);
          }

          command_buffer.endRendering();

          scene_.record_after_store(image_uses_, command_buffer);

          image_uses_.transition(command_buffer,
            swap_chain_.images()[ image_index ],
            vkpp::image_use::color_attachment, vk::ImageAspectFlagBits::eColor);

          vk::ClearValue ui_clear { vk::ClearColorValue {
            0.10F,
            0.10F,
            0.12F,
            1.0F,
          } };
          vk::RenderingAttachmentInfo ui_color {
            .imageView = swap_chain_.image_view(image_index),
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = ui_clear,
          };
          vk::RenderingInfo ui_info {
            .renderArea = { .extent = swap_chain_.extent() },
            .layerCount = 1U,
            .colorAttachmentCount = 1U,
            .pColorAttachments = &ui_color,
          };
          command_buffer.beginRendering(ui_info);
          ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(),
            static_cast<VkCommandBuffer>(*command_buffer));
          command_buffer.endRendering();

          timestamps_.write(command_buffer, frame_index_, 1U,
            vk::PipelineStageFlagBits2::eAllCommands);

          image_uses_.transition(command_buffer,
            swap_chain_.images()[ image_index ], vkpp::image_use::present,
            vk::ImageAspectFlagBits::eColor);
        })
      .and_then(
        [ &command_buffer ]() -> std::expected<void, vkpp::error_t>
        {
          return UTILS_VK(command_buffer.end(), ^^vk::raii::CommandBuffer::end);
        });
  }

  auto
  suspend_rendering() -> std::expected<void, vkpp::error_t>
  {
    if (frame_rendering_state_ == frame_rendering_state::suspended &&
      swap_chain_.empty())
    {
      return {};
    }

    frame_rendering_state_ = frame_rendering_state::suspended;
    resized_ = false;

    if (swap_chain_.empty()) { return {}; }

    return UTILS_VK(device_.device().waitIdle(), ^^vk::raii::Device::waitIdle)
      .transform([ this ] { swap_chain_.release(); });
  }

  auto
  reinitialize_imgui_renderer() -> std::expected<void, vkpp::error_t>
  {
    shutdown_imgui_renderer();
    scene_extent_ = vk::Extent2D {};
    return init_imgui_renderer();
  }

  auto
  recreate_swap_chain() -> std::expected<void, vkpp::error_t>
  {
    return swap_chain_
      .recreate(device_, instance_.surface(), framebuffer_extent_request(),
        [ this ](const vk::SurfaceCapabilitiesKHR& capabilities,
          vk::Extent2D framebuffer) -> vk::Extent2D
        { return choose_swap_extent(capabilities, framebuffer); })
      .and_then(std::bind_front(&app::reinitialize_imgui_renderer, this));
  }

  auto
  query_presentability() -> std::expected<vkpp::presentability, vkpp::error_t>
  {
    return vkpp::swapchain::query_presentability(device_, instance_.surface(),
      framebuffer_extent_request(),
      [ this ](const vk::SurfaceCapabilitiesKHR& capabilities,
        vk::Extent2D framebuffer)
      { return choose_swap_extent(capabilities, framebuffer); });
  }

  auto
  recreate_or_suspend() -> std::expected<void, vkpp::error_t>
  {
    return query_presentability().and_then(
      [ this ](vkpp::presentability presentability)
        -> std::expected<void, vkpp::error_t>
      {
        if (!presentability.presentable) { return suspend_rendering(); }
        return recreate_swap_chain();
      });
  }

  auto
  draw_frame_resume() -> std::expected<void, vkpp::error_t>
  {
    return recreate_swap_chain().and_then(
      [ this ] -> std::expected<void, vkpp::error_t>
      {
        frame_rendering_state_ = frame_rendering_state::active;
        return draw_frame_active();
      });
  }

  auto
  draw_frame_active() -> std::expected<void, vkpp::error_t>
  {
    frame_index_ =
      static_cast<std::uint32_t>(frame_counter_ % max_frames_in_flight);
    auto& frame = frames_[ frame_index_ ];

    const std::uint64_t wait_value = frame_counter_ >= max_frames_in_flight
      ? frame_counter_ - max_frames_in_flight + 1ULL
      : 0ULL;
    const vk::Semaphore timeline = *frame_timeline_;
    const vk::SemaphoreWaitInfo timeline_wait_info {
      .semaphoreCount = 1U,
      .pSemaphores = &timeline,
      .pValues = &wait_value,
    };

    if (const auto result = device_.device().waitSemaphores(
          timeline_wait_info, std::numeric_limits<std::uint64_t>::max());
      result != vk::Result::eSuccess)
    {
      return std::unexpected {
        vkpp::vk_error {
          .function = "waitSemaphores",
          .type = "vk::raii::Device",
          .result = result,
        },
      };
    }

    if (frame_counter_ >= max_frames_in_flight)
    {
      if (auto ns = timestamps_.read_and_reset_frame_ns(frame_index_); ns)
      {
        gpu_ms_ = ((*ns)[ 1 ] - (*ns)[ 0 ]) * 1e-6;
        const auto now = std::chrono::steady_clock::now();
        if (now - last_gpu_print_ >= std::chrono::seconds { 1 })
        {
          std::println("GPU frame: {:.3f} ms", gpu_ms_);
          last_gpu_print_ = now;
        }
      }
    }

    auto [ result, image_index ] = swap_chain_.swap_chain().acquireNextImage(
      std::numeric_limits<std::uint64_t>::max(), *frame.present_complete,
      nullptr);
    if (result == vk::Result::eErrorOutOfDateKHR)
    {
      return recreate_or_suspend();
    }
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
    {
      return std::unexpected {
        vkpp::vk_error {
          .function = "acquireNextImage",
          .type = "vk::raii::SwapchainKHR",
          .result = result,
        },
      };
    }

    begin_imgui_frame();
    if (auto result = ensure_scene_targets(); !result) { return result; }
    finish_imgui_frame();
    update_uniform_buffer(frame_index_);

    return UTILS_VK(
      frame.command_buffer.reset(), ^^vk::raii::CommandBuffer::reset)
      .and_then([ this, image_index ] -> std::expected<void, vkpp::error_t>
        { return record_command_buffer(image_index); })
      .and_then(
        [ this, image_index, &frame ] -> std::expected<void, vkpp::error_t>
        {
          const vk::SemaphoreSubmitInfo wait_semaphore_info {
            .semaphore = *frame.present_complete,
            .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
          };
          const vk::CommandBufferSubmitInfo command_buffer_info {
            .commandBuffer = *frame.command_buffer,
          };
          const std::array signal_semaphore_infos {
            vk::SemaphoreSubmitInfo {
              .semaphore = *swap_chain_.render_finished(image_index),
              .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            },
            vk::SemaphoreSubmitInfo {
              .semaphore = *frame_timeline_,
              .value = frame_counter_ + 1ULL,
              .stageMask = vk::PipelineStageFlagBits2::eAllCommands,
            },
          };

          const vk::SubmitInfo2 submit_info {
            .waitSemaphoreInfoCount = 1U,
            .pWaitSemaphoreInfos = &wait_semaphore_info,
            .commandBufferInfoCount = 1U,
            .pCommandBufferInfos = &command_buffer_info,
            .signalSemaphoreInfoCount =
              static_cast<std::uint32_t>(signal_semaphore_infos.size()),
            .pSignalSemaphoreInfos = signal_semaphore_infos.data(),
          };
          return UTILS_VK(
            device_.graphics_queue().submit2(submit_info, nullptr),
            ^^vk::raii::Queue::submit2)
            .transform([ this ] -> void { ++frame_counter_; });
        })
      .and_then(
        [ this, &image_index, &result ] -> std::expected<void, vkpp::error_t>
        {
          const vk::Semaphore wait_semaphore =
            *swap_chain_.render_finished(image_index);
          const vk::SwapchainKHR swapchain = *swap_chain_.swap_chain();

          const vk::PresentInfoKHR present_info {
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &wait_semaphore,
            .swapchainCount = 1,
            .pSwapchains = &swapchain,
            .pImageIndices = &image_index,
            .pResults = nullptr,
          };

          result = device_.graphics_queue().presentKHR(present_info);

          if (result == vk::Result::eSuccess) { return {}; }
          if (result == vk::Result::eSuboptimalKHR ||
            result == vk::Result::eErrorOutOfDateKHR || resized_)
          {
            resized_ = false;
            return recreate_or_suspend();
          }
          return std::unexpected {
            vkpp::vk_error {
              .function = "presentKHR",
              .type = "vk::raii::Queue",
              .result = result,
            },
          };
        });
  }

  auto
  draw_frame() -> std::expected<void, vkpp::error_t>
  {
    return query_presentability().and_then(
      [ this ](vkpp::presentability presentability)
        -> std::expected<void, vkpp::error_t>
      {
        if (!presentability.presentable) { return suspend_rendering(); }
        if (frame_rendering_state_ == frame_rendering_state::suspended)
        {
          return draw_frame_resume();
        }
        return draw_frame_active();
      });
  }

private:
  enum class frame_rendering_state : std::uint8_t
  {
    active,
    suspended,
  };

private:
  sf::WindowBase window_ {
    sf::VideoMode { { window_width, window_height } },
    "Window_title",
  };
  vkpp::instance_context instance_ {};
  vkpp::device_context device_ {};
  vkpp::swapchain swap_chain_ {};

  vk::raii::DescriptorPool imgui_descriptor_pool_ { nullptr };
  std::chrono::steady_clock::time_point imgui_frame_time_ {};
  double gpu_ms_ { 0.0 };
  std::uint32_t skipped_blend_draws_ { 0U };
  bool compute_smoke_ok_ { false };
  bool imgui_renderer_initialized_ { false };
  bool imgui_layout_built_ { false };
  vkpp::frame_attachments scene_ {};
  vk::raii::Sampler scene_sampler_ { nullptr };
  vk::Extent2D scene_extent_ {};
  vk::Extent2D scene_desired_extent_ {};
  VkDescriptorSet scene_texture_id_ { VK_NULL_HANDLE };

  vkpp::image_use_tracker image_uses_ {};

  vkpp::graphics_pipeline graphics_pipeline_;
  vkpp::graphics_pipeline blend_pipeline_;

  struct blend_entry
  {
    std::uint32_t draw_index { 0U };
    float view_depth { 0.0F };
  };
  std::vector<blend_entry> blend_order_ {};
  glm::mat4 model_matrix_ { 1.0F };
  glm::mat4 view_matrix_ { 1.0F };

  vk::raii::DescriptorSetLayout descriptor_set_layout_ { nullptr };
  vkpp::descriptor_pool descriptor_pool_ {};

  vkpp::command_pool command_pool_ {};
  vkpp::command_pool upload_pool_ {};
  vkpp::command_pool transfer_upload_pool_ {};
  std::array<vkpp::frame, max_frames_in_flight> frames_ {};
  std::uint32_t frame_index_ {};
  vk::raii::Semaphore frame_timeline_ { nullptr };
  std::uint64_t frame_counter_ { 0ULL };

  vkpp::timestamp_ring timestamps_ {};
  std::chrono::steady_clock::time_point last_gpu_print_ {};

  struct draw_gpu
  {
    std::array<float, 16> world {};
    std::array<float, 12> normal_matrix_columns {};
    std::uint32_t material_index {};
    std::uint32_t _;
    std::uint32_t _;
    std::uint32_t _;
  };
  static_assert(sizeof(draw_gpu) == 128UZ);

  struct draw_push
  {
    std::uint32_t draw_index { 0U };
  };
  static_assert(sizeof(draw_push) == 4UZ);

  struct material_gpu
  {
    std::array<float, 4> base_color_factor { 1.0F, 1.0F, 1.0F, 1.0F };
    std::array<float, 4> emissive_factor_and_metallic { 0.0F, 0.0F, 0.0F,
      1.0F };
    float roughness_factor { 1.0F };
    float normal_scale { 1.0F };
    float occlusion_strength { 1.0F };
    float alpha_cutoff { 0.5F };
    std::uint32_t base_color_index { 0U };
    std::uint32_t metallic_roughness_index { 0U };
    std::uint32_t normal_index { 0U };
    std::uint32_t occlusion_index { 0U };
    std::uint32_t emissive_index { 0U };
    std::uint32_t alpha_mode { 0U };
    std::uint32_t has_texture_mask { 0U };
    float transmission_factor { 0.0F };
  };
  static_assert(sizeof(material_gpu) == 80UZ);

  std::optional<vkpp::sampler_cache> sampler_cache_ {};
  vkpp::bindless_table bindless_table_ {};
  std::vector<vkpp::texture<>> textures_ {};
  vkpp::buffer_resource<> material_buffer_ {};

  vkpp::buffer_arena<> geometry_arena_ {};
  struct primitive_draw
  {
    vkpp::virtual_slice vertex_slice {};
    vkpp::virtual_slice index_slice {};
    std::uint32_t index_count { 0U };
    vk::IndexType index_type { vk::IndexType::eUint16 };
    vkpp::gltf::alpha_mode alpha_mode { vkpp::gltf::alpha_mode::opaque };
    bool double_sided { false };
  };
  std::vector<primitive_draw> draws_ {};
  std::vector<vkpp::gltf::draw_item_cpu> draw_list_ {};
  vkpp::buffer_resource<> draw_buffer_ {};

  bool resized_ { false };
  frame_rendering_state frame_rendering_state_ {
    frame_rendering_state::active
  };
};
} // namespace f1st
