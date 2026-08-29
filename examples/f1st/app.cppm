module;

#include "contracts_config.hpp"
#include "error/vk_error_config.hpp"

#include <SFML/Window.hpp>
#include <SFML/Window/VideoMode.hpp>
#include <vulkan/vk_platform.h>

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
  .sample_shading = true,
  .min_sample_shading = 0.2F,
};

static constexpr std::array k_set0_bindings { vk::DescriptorSetLayoutBinding {
  .binding = 0U,
  .descriptorType = vk::DescriptorType::eUniformBuffer,
  .descriptorCount = 1U,
  .stageFlags = vk::ShaderStageFlagBits::eVertex,
  .pImmutableSamplers = nullptr,
} };

export class app
{
public:
  void
  run()
  {
    const auto result = init_vulkan().and_then(
      [ this ]() -> std::expected<void, vkpp::error_t> { return main_loop(); });

    (void)device_.device().waitIdle();
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
      .and_then(std::bind_front(&app::create_graphics_pipeline, this))
      .and_then(std::bind_front(&app::create_buffers, this))
      .and_then(std::bind_front(&app::create_descriptor_pool, this))
      .and_then(std::bind_front(&app::create_descriptor_sets, this));
  }

  auto
  main_loop() -> std::expected<void, vkpp::error_t>
  {
    const auto on_close = [ this ](const sf::Event::Closed&) -> void
    { window_.close(); };

    const auto on_resize = [ this ](const sf::Event::Resized&) -> void
    { resized_ = true; };

    while (window_.isOpen())
    {
      window_.handleEvents(on_close, on_resize);
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
  create_graphics_pipeline() -> std::expected<void, vkpp::error_t>
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
    return vkpp::load_shader_file(SHADER_DIRECTORY "slang.spv")
      .and_then(
        [ &, this ](
          const std::vector<char>& spirv) -> std::expected<void, vkpp::error_t>
        {
          return vkpp::make_graphics_pipeline<k_pipeline_spec>(device_.device(),
            vkpp::graphics_pipeline_runtime_args {
              .color_formats = color_formats,
              .depth_format = swap_chain_.depth().format(),
              .samples = device_.msaa_samples(),
              .set_layouts = set_layouts,
              .vertex_bindings = vertex_bindings,
              .vertex_attributes = vertex_attributes,
              .push_constant_size =
                static_cast<std::uint32_t>(sizeof(draw_push)),
              .push_constant_stages = vk::ShaderStageFlagBits::eVertex |
                vk::ShaderStageFlagBits::eFragment,
            },
            { .spirv = spirv })
            .transform([ this ](vkpp::graphics_pipeline&& pipeline) -> void
              { graphics_pipeline_ = std::move(pipeline); });
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
    const vkpp::buffer_barrier after_compute {
      .src_stage = vk::PipelineStageFlagBits2::eComputeShader,
      .src_access = vk::AccessFlagBits2::eShaderStorageWrite,
      .dst_stage = vk::PipelineStageFlagBits2::eCopy,
      .dst_access = vk::AccessFlagBits2::eTransferRead,
      .buffer = ssbo->buffer(),
    };
    vkpp::record_barriers(command_buffer, std::span { &after_compute, 1UZ });
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

          textures_.clear();
          textures_.reserve(asset.host_images.size());
          std::vector<std::uint32_t> image_to_slot(
            asset.host_images.size(), 0U);

          for (auto index : std::views::indices(asset.host_images.size()))
          {
            const auto& host = asset.host_images[ index ];
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
            bindless_table_.write(
              device_.device(), *slot, *made->sampler(), *made->view());
            image_to_slot[ index ] = *slot;
            textures_.push_back(std::move(*made));
          }

          const std::uint32_t fallback_slot =
            image_to_slot.empty() ? 0U : image_to_slot.front();

          auto resolve_base_color =
            [ & ](const vkpp::mesh_streams_cpu& primitive) -> std::uint32_t
          {
            if (!primitive.material_index.has_value()) { return fallback_slot; }
            const auto material_i = *primitive.material_index;
            if (material_i >= asset.materials.size()) { return fallback_slot; }
            const auto& material = asset.materials[ material_i ];
            if (!material.base_color_texture.has_value())
            {
              return fallback_slot;
            }
            const auto& ref = *material.base_color_texture;
            const std::optional<std::uint32_t> image_index =
              ref.basisu_image_index.has_value() ? ref.basisu_image_index
                                                 : ref.image_index;
            if (!image_index.has_value() ||
              *image_index >= image_to_slot.size())
            {
              return fallback_slot;
            }
            return image_to_slot[ *image_index ];
          };

          std::vector<vkpp::mesh_interleaved_cpu> packed;
          std::vector<std::uint32_t> base_color_indices;
          packed.reserve(asset.meshes.primitives.size());
          base_color_indices.reserve(asset.meshes.primitives.size());
          for (const auto& primitive : asset.meshes.primitives)
          {
            base_color_indices.push_back(resolve_base_color(primitive));
            packed.push_back(vkpp::pack_interleaved_vertices(primitive));
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
              [ this, &packed, &base_color_indices ](
                vkpp::buffer_arena<>&& arena)
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
                    .base_color_index = base_color_indices[ index ],
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
    static const auto aspect_ratio { //
      static_cast<float>(swap_chain_.extent().width) /
      static_cast<float>(swap_chain_.extent().height)
    };
    static constexpr auto near_plane { 0.1F };
    static constexpr auto far_plane { 10.0F };

    uniform_buffer_object ubo {
      .model = glm::gtc::rotate(
        glm::mat4 { 1.0F }, time * degrees, glm::vec3 { 0.0F, 1.0F, 0.0F }),
      .view = glm::gtc::lookAt(camera_position, target, up),
      .projection = glm::gtc::perspective(
        fov_vertical, aspect_ratio, near_plane, far_plane),
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
            const vk::DescriptorBufferInfo buffer_info {
              .buffer = frames_[ index ].uniform_buffer.buffer(),
              .offset = 0U,
              .range = sizeof(uniform_buffer_object),
            };
            const vk::WriteDescriptorSet write {
              .dstSet = frames_[ index ].descriptor_set,
              .dstBinding = 0U,
              .dstArrayElement = 0U,
              .descriptorCount = 1U,
              .descriptorType = vk::DescriptorType::eUniformBuffer,
              .pBufferInfo = &buffer_info,
            };
            vkpp::update_descriptor_sets(
              device_.device(), std::span { &write, 1UZ });
          }
        });
  }

  auto
  record_command_buffer(std::uint32_t image_index)
    -> std::expected<void, vkpp::error_t>
  {
    auto& command_buffer = frames_[ frame_index_ ].command_buffer;
    return UTILS_VK(command_buffer.begin({}), ^^vk::raii::CommandBuffer::begin)
      .transform(
        [ this, image_index, &command_buffer ]() -> void
        {
          timestamps_.write(command_buffer, frame_index_, 0U,
            vk::PipelineStageFlagBits2::eNone);
          const std::array pre_pass_transitions {
            vkpp::image_use_transition {
              .image = swap_chain_.images()[ image_index ],
              .from = vkpp::image_use::none,
              .to = vkpp::image_use::color_attachment,
              .aspect = vk::ImageAspectFlagBits::eColor,
            },
            vkpp::image_use_transition {
              .image = swap_chain_.color().image(),
              .from = vkpp::image_use::none,
              .to = vkpp::image_use::color_attachment,
              .aspect = vk::ImageAspectFlagBits::eColor,
            },
            vkpp::image_use_transition {
              .image = swap_chain_.depth().image(),
              .from = vkpp::image_use::none,
              .to = vkpp::image_use::depth_attachment,
              .aspect = vk::ImageAspectFlagBits::eDepth,
            },
          };
          vkpp::record_image_use_transitions(
            command_buffer, std::span { pre_pass_transitions });

          vk::ClearValue clear_color { vk::ClearColorValue {
            0.0F,
            0.0F,
            0.0F,
            1.0F,
          } };
          vk::RenderingAttachmentInfo color_attachment_info {
            .imageView = *swap_chain_.color().view(),
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .resolveMode = vk::ResolveModeFlagBits::eAverage,
            .resolveImageView = swap_chain_.image_view(image_index),
            .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = clear_color,
          };

          vk::ClearValue clear_depth { vk::ClearDepthStencilValue {
            .depth = 1.0F,
            .stencil = 0,
          } };
          vk::RenderingAttachmentInfo depth_attachment_info {
            .imageView = *swap_chain_.depth().view(),
            .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eDontCare,
            .clearValue = clear_depth,
          };

          vk::RenderingInfo rendering_info {
            .renderArea =
              vk::Rect2D {
                .offset = { .x = 0, .y = 0 },
                .extent = swap_chain_.extent(),
              },
            .layerCount = 1U,
            .colorAttachmentCount = 1U,
            .pColorAttachments = &color_attachment_info,
            .pDepthAttachment = &depth_attachment_info,
          };
          command_buffer.beginRendering(rendering_info);
          command_buffer.bindPipeline(
            vk::PipelineBindPoint::eGraphics, *graphics_pipeline_.pipeline());
          command_buffer.setViewport(0U,
            vk::Viewport {
              .x = 0.0F,
              .y = 0.0F,
              .width = static_cast<float>(swap_chain_.extent().width),
              .height = static_cast<float>(swap_chain_.extent().height),
              .minDepth = 0.0F,
              .maxDepth = 1.0F,
            });
          command_buffer.setScissor(0U,
            vk::Rect2D {
              .offset = vk::Offset2D { .x = 0, .y = 0 },
              .extent = swap_chain_.extent(),
            });
          command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
            *graphics_pipeline_.layout(), 0U,
            frames_[ frame_index_ ].descriptor_set, nullptr);
          command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
            *graphics_pipeline_.layout(), 1U, bindless_table_.set(), nullptr);
          for (const auto& item : draw_list_)
          {
            const auto& draw = draws_[ item.primitive_index ];
            command_buffer.bindVertexBuffers(
              0U, geometry_arena_.buffer(), { draw.vertex_slice.offset });
            command_buffer.bindIndexBuffer(geometry_arena_.buffer(),
              draw.index_slice.offset, draw.index_type);

            const draw_push push_constants {
              .world = item.world_transform,
              .texture_index = draw.base_color_index,
            };
            command_buffer.pushConstants(*graphics_pipeline_.layout(),
              vk::ShaderStageFlagBits::eVertex |
                vk::ShaderStageFlagBits::eFragment,
              0U, sizeof(push_constants), &push_constants);
            command_buffer.drawIndexed(draw.index_count, 1U, 0U, 0U, 0U);
          }
          command_buffer.endRendering();
          timestamps_.write(command_buffer, frame_index_, 1U,
            vk::PipelineStageFlagBits2::eAllCommands);

          const vkpp::image_use_transition present_transition {
            .image = swap_chain_.images()[ image_index ],
            .from = vkpp::image_use::color_attachment,
            .to = vkpp::image_use::present,
            .aspect = vk::ImageAspectFlagBits::eColor,
          };
          vkpp::record_image_use_transitions(
            command_buffer, std::span { &present_transition, 1UZ });
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
  recreate_swap_chain() -> std::expected<void, vkpp::error_t>
  {
    return swap_chain_.recreate(device_, instance_.surface(),
      framebuffer_extent_request(),
      [ this ](const vk::SurfaceCapabilitiesKHR& capabilities,
        vk::Extent2D framebuffer)
      { return choose_swap_extent(capabilities, framebuffer); });
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
      // TODO (Konrad): shouldn't we use std::chrono for time? I believe it is
      // one of the best written part of the stdlib
      if (auto ns = timestamps_.read_and_reset_frame_ns(frame_index_); ns)
      {
        const double gpu_ms = ((*ns)[ 1 ] - (*ns)[ 0 ]) * 1e-6;
        const auto now = std::chrono::steady_clock::now();
        if (now - last_gpu_print_ >= std::chrono::seconds { 1 })
        {
          std::println("GPU frame: {:.3f} ms", gpu_ms);
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

  vkpp::graphics_pipeline graphics_pipeline_;

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

  struct draw_push
  {
    std::array<float, 16> world {};
    std::uint32_t texture_index {};
  };

  vkpp::bindless_table bindless_table_ {};
  std::vector<vkpp::texture<>> textures_ {};

  vkpp::buffer_arena<> geometry_arena_ {};
  struct primitive_draw
  {
    vkpp::virtual_slice vertex_slice {};
    vkpp::virtual_slice index_slice {};
    std::uint32_t index_count { 0U };
    vk::IndexType index_type { vk::IndexType::eUint16 };
    std::uint32_t base_color_index { 0U };
  };
  std::vector<primitive_draw> draws_ {};
  std::vector<vkpp::gltf::draw_item_cpu> draw_list_ {};

  bool resized_ { false };
  frame_rendering_state frame_rendering_state_ {
    frame_rendering_state::active
  };
};
} // namespace f1st
