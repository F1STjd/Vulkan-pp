export module vkpp.descriptor;

import std;
import vulkan;

import vkpp.error;
import vkpp.diagnostics;
import vkpp.descriptor.layout_decl;

namespace vkpp
{

export class descriptor_pool
{
public:
  descriptor_pool() = default;

  [[nodiscard]]
  static
  auto
  create(
    const vk::raii::Device&                 device,
    std::uint32_t                           max_sets,
    std::span<const vk::DescriptorPoolSize> pool_sizes,
    vk::DescriptorPoolCreateFlags           flags = {})
    -> std::expected<descriptor_pool, error_t>
  {
    const vk::DescriptorPoolCreateInfo create_info {
      .flags         = flags,
      .maxSets       = max_sets,
      .poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size()),
      .pPoolSizes    = pool_sizes.data(),
    };
    return map_vk_error(device.createDescriptorPool(create_info), std::nullopt)
      .transform([](vk::raii::DescriptorPool&& pool) -> descriptor_pool {
        return descriptor_pool {std::move(pool)};
      });
  }

  [[nodiscard]]
  auto
  allocate(
    const vk::raii::Device& device,
    vk::DescriptorSetLayout layout,
    std::uint32_t           count)
    -> std::expected<std::vector<vk::DescriptorSet>, error_t>
  {
    std::vector layouts(count, layout);

    const vk::DescriptorSetAllocateInfo allocate_info {
      .descriptorPool     = *pool_,
      .descriptorSetCount = count,
      .pSetLayouts        = layouts.data(),
    };

    return map_vk_error(
      device.allocateDescriptorSets(allocate_info), std::nullopt)
      .transform([](std::vector<vk::raii::DescriptorSet>&& owned) {
        std::vector<vk::DescriptorSet> handles;
        handles.reserve(owned.size());
        for (vk::raii::DescriptorSet& set : owned)
        {
          handles.push_back(set.release());
        }
        return handles;
      });
  }

  [[nodiscard]]
  auto
  reset() -> std::expected<void, error_t>
  { return map_vk_error(pool_.reset(), std::nullopt); }

  [[nodiscard]]
  auto
  handle(this auto&& self) -> decltype(auto)
  { return std::forward_like<decltype(self)>(self.pool_); }

private:
  explicit descriptor_pool(vk::raii::DescriptorPool&& pool)
  : pool_ {std::move(pool)}
  {}

  vk::raii::DescriptorPool pool_ {nullptr};
};

export [[nodiscard]]
auto
pool_sizes_for(
  std::span<const vk::DescriptorSetLayoutBinding> bindings,
  std::uint32_t set_count) -> std::vector<vk::DescriptorPoolSize>
{
  struct type_count
  {
    vk::DescriptorType type {};
    std::uint32_t      count {};
  };
  std::vector<type_count> counts_by_type;
  for (const auto& binding : bindings)
  {
    auto existing = std::ranges::find(
      counts_by_type, binding.descriptorType, &type_count::type);
    const std::uint32_t added = binding.descriptorCount * set_count;
    if (existing == counts_by_type.end())
    {
      counts_by_type.push_back({
        .type  = binding.descriptorType,
        .count = added,
      });
    }
    else
    {
      existing->count += added;
    }
  }
  std::vector<vk::DescriptorPoolSize> pool_sizes;
  pool_sizes.reserve(counts_by_type.size());
  for (const auto& element : counts_by_type)
  {
    pool_sizes.push_back({
      .type            = element.type,
      .descriptorCount = element.count,
    });
  }

  return pool_sizes;
}

namespace detail
{

template<declared_resource_layout Layout>
struct set_layout_rows
{
  std::array<vk::DescriptorSetLayoutBinding, Layout::resource_count()>
    bindings {};
  std::array<vk::DescriptorBindingFlags, Layout::resource_count()>
    binding_flags {};

  std::uint32_t                      count {};
  vk::DescriptorSetLayoutCreateFlags layout_flags {};
};

template<declared_resource_layout Layout>
consteval
auto
rows_for_set(std::uint32_t set) -> set_layout_rows<Layout>
{
  set_layout_rows<Layout> rows {};
  for (const binding_assignment& row : Layout::assignment_table())
  {
    if (row.set != set) { continue; }
    rows.bindings[ rows.count ] = {
      .binding            = row.binding,
      .descriptorType     = row.descriptor_type,
      .descriptorCount    = row.count,
      .stageFlags         = row.stage_flags,
      .pImmutableSamplers = nullptr,
    };
    rows.binding_flags[ rows.count ] = row.binding_flags;
    if (row.update_after_bind)
    {
      rows.layout_flags |=
        vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool;
    }
    ++rows.count;
  }
  return rows;
}

template<declared_resource_layout Layout, std::uint32_t Set>
[[nodiscard]]
auto
make_set_layout(const vk::raii::Device& device)
  -> std::expected<vk::raii::DescriptorSetLayout, error_t>
{
  static constexpr set_layout_rows<Layout> rows {rows_for_set<Layout>(Set)};

  const vk::StructureChain chain {
    vk::DescriptorSetLayoutCreateInfo {
      .flags        = rows.layout_flags,
      .bindingCount = rows.count,
      .pBindings    = rows.bindings.data(),
    },
    vk::DescriptorSetLayoutBindingFlagsCreateInfo {
      .bindingCount  = rows.count,
      .pBindingFlags = rows.binding_flags.data(),
    },
  };
  return map_vk_error(
    device.createDescriptorSetLayout(
      chain.get<vk::DescriptorSetLayoutCreateInfo>()),
    std::nullopt);
}

} // namespace detail

export template<declared_resource_layout Layout>
class descriptor_set_layouts
{
  static_assert(
    Layout::set_count == 4U, "descriptor_set_layouts: expects 4 scopes");

public:
  descriptor_set_layouts() = default;

  [[nodiscard]]
  static
  auto
  create(const vk::raii::Device& device)
    -> std::expected<descriptor_set_layouts, error_t>
  {
    return
      [ & ]<std::uint32_t... Sets>(std::integer_sequence<std::uint32_t, Sets...>)
        -> std::expected<descriptor_set_layouts, error_t> {
        std::array<
          std::expected<vk::raii::DescriptorSetLayout, error_t>,
          Layout::set_count
        >
          created {detail::make_set_layout<Layout, Sets>(device)...};

        const auto failed = std::ranges::find_if(
          created,
          [](const std::expected<vk::raii::DescriptorSetLayout, error_t>& layout)
            -> bool { return !layout.has_value(); });
        if (failed != created.end())
        {
          return std::unexpected {failed->error()};
        }

        return descriptor_set_layouts {
          std::array<vk::raii::DescriptorSetLayout, Layout::set_count> {
            std::move(*created[ Sets ])...,
          },
        };
      }(std::make_integer_sequence<std::uint32_t, Layout::set_count> {});
  }

  [[nodiscard]]
  auto
  layout(resource_scope scope) const -> vk::DescriptorSetLayout
  { return *layouts_[ slot_of(scope) ]; }

  [[nodiscard]]
  auto
  handles() const -> std::array<vk::DescriptorSetLayout, Layout::set_count>
  {
    std::array<vk::DescriptorSetLayout, Layout::set_count> handles {};
    std::ranges::transform(
      layouts_,
      handles.begin(),
      [](const vk::raii::DescriptorSetLayout& layout)
        -> vk::DescriptorSetLayout { return *layout; });
    return handles;
  }

private:
  explicit descriptor_set_layouts(
    std::array<vk::raii::DescriptorSetLayout, Layout::set_count>&& layouts)
  : layouts_ {std::move(layouts)}
  {}

  std::array<vk::raii::DescriptorSetLayout, Layout::set_count> layouts_ {
    nullptr, nullptr, nullptr, nullptr
  };
};

export
void
update_descriptor_sets(
  const vk::raii::Device&                 device,
  std::span<const vk::WriteDescriptorSet> writes)
{ device.updateDescriptorSets(writes, {}); }

export struct descriptor_set_arena_create_info
{
  const vk::raii::Device&                         device;
  vk::DescriptorSetLayout                         layout {};
  std::span<const vk::DescriptorSetLayoutBinding> bindings {};
  std::uint32_t                                   set_count {1U};
  vk::DescriptorPoolCreateFlags                   pool_flags {};
  std::span<const vk::DescriptorPoolSize>         pool_sizes_escape {};
};

export class descriptor_set_arena
{
public:
  descriptor_set_arena() = default;

  [[nodiscard]]
  auto
  layout() const -> vk::DescriptorSetLayout
  { return layout_; }

  [[nodiscard]]
  auto
  sets() const -> std::span<const vk::DescriptorSet>
  { return sets_; }

  [[nodiscard]]
  auto
  set(std::uint32_t index) const -> vk::DescriptorSet
  { return sets_[ index ]; }

  [[nodiscard]]
  static
  auto
  create(const descriptor_set_arena_create_info& create_info)
    -> std::expected<descriptor_set_arena, error_t>
  {
    if (create_info.layout == vk::DescriptorSetLayout {})
    {
      return std::unexpected {
        make_app_error(app_error_code::missing_required_argument),
      };
    }
    std::vector<vk::DescriptorPoolSize>     derived {};
    std::span<const vk::DescriptorPoolSize> sizes =
      create_info.pool_sizes_escape;
    if (sizes.empty())
    {
      derived = pool_sizes_for(create_info.bindings, create_info.set_count);
      sizes   = derived;
    }
    return descriptor_pool::create(
      create_info.device, create_info.set_count, sizes, create_info.pool_flags)
      .and_then(
        [ & ](descriptor_pool&& pool)
          -> std::expected<descriptor_set_arena, error_t> {
          return pool
            .allocate(
              create_info.device, create_info.layout, create_info.set_count)
            .transform(
              [ & ](std::vector<vk::DescriptorSet>&& sets) mutable
                -> descriptor_set_arena {
                return descriptor_set_arena {
                  create_info.layout,
                  std::move(pool),
                  std::move(sets),
                };
              });
        });
  }

private:
  descriptor_set_arena(
    vk::DescriptorSetLayout          layout,
    descriptor_pool&&                pool,
    std::vector<vk::DescriptorSet>&& sets)
  : layout_ {layout}, pool_ {std::move(pool)}, sets_ {std::move(sets)}
  {}

  vk::DescriptorSetLayout        layout_ {};
  descriptor_pool                pool_ {};
  std::vector<vk::DescriptorSet> sets_ {};
};

export template<declared_resource_layout Layout, resource_scope Scope>
  requires (Scope != resource_scope::bindless)
class descriptor_scope_arena
{
public:
  static constexpr std::uint32_t set_index {slot_of(Scope)};

  descriptor_scope_arena() = default;

  [[nodiscard]]
  static
  auto
  create(
    const vk::raii::Device&               device,
    const descriptor_set_layouts<Layout>& layouts,
    std::uint32_t                         set_count = 1U)
    -> std::expected<descriptor_scope_arena, error_t>
  {
    static constexpr detail::set_layout_rows<Layout> rows {
      detail::rows_for_set<Layout>(set_index),
    };
    return descriptor_set_arena::create(
      {
        .device    = device,
        .layout    = layouts.layout(Scope),
        .bindings  = std::span {rows.bindings.data(), rows.count},
        .set_count = set_count,
      })
      .transform([](descriptor_set_arena&& arena) -> descriptor_scope_arena {
        return descriptor_scope_arena {std::move(arena)};
      });
  }

  [[nodiscard]]
  auto
  set(std::uint32_t index = 0) const -> vk::DescriptorSet
  { return arena_.set(index); }

  [[nodiscard]]
  auto
  layout() const -> vk::DescriptorSetLayout
  { return arena_.layout(); }

private:
  explicit descriptor_scope_arena(descriptor_set_arena&& arena)
  : arena_ {std::move(arena)}
  {}

  descriptor_set_arena arena_ {};
};

export
template<class Tag, declared_resource_layout Layout, resource_scope Scope>
void
write_descriptor(
  const descriptor_scope_arena<Layout, Scope>& arena,
  const vk::raii::Device&                      device,
  vk::Buffer                                   buffer,
  vk::DeviceSize                               range,
  vk::DeviceSize                               offset    = 0U,
  std::uint32_t                                set_index = 0U)
{
  constexpr binding_assignment row = Layout::template assignment_of<Tag>;
  static_assert(
    row.scope == Scope, "write_descriptor: Tag is declared in another scope");
  static_assert(
    row.descriptor_type == vk::DescriptorType::eStorageBuffer ||
      row.descriptor_type == vk::DescriptorType::eUniformBuffer ||
      row.descriptor_type == vk::DescriptorType::eUniformBufferDynamic,
    "write_descriptor: Tag is not a buffer resource");

  const vk::DescriptorBufferInfo buffer_info {
    .buffer = buffer,
    .offset = offset,
    .range  = range,
  };
  const vk::WriteDescriptorSet write {
    .dstSet          = arena.set(set_index),
    .dstBinding      = row.binding,
    .dstArrayElement = 0U,
    .descriptorCount = 1U,
    .descriptorType  = row.descriptor_type,
    .pBufferInfo     = &buffer_info
  };

  device.updateDescriptorSets(write, nullptr);
}

export
template<class Tag, declared_resource_layout Layout, resource_scope Scope>
void
write_descriptor(
  const descriptor_scope_arena<Layout, Scope>& arena,
  const vk::raii::Device&                      device,
  vk::Sampler                                  sampler,
  vk::ImageView                                view,
  vk::ImageLayout image_layout = vk::ImageLayout::eShaderReadOnlyOptimal,
  std::uint32_t   set_index    = 0U)
{
  constexpr binding_assignment row = Layout::template assignment_of<Tag>;
  static_assert(
    row.scope == Scope, "write_descriptor: Tag is declared in another scope");
  static_assert(
    row.descriptor_type == vk::DescriptorType::eCombinedImageSampler,
    "write_descriptor: Tag is not a combined image sampler");

  const vk::DescriptorImageInfo image_info {
    .sampler     = sampler,
    .imageView   = view,
    .imageLayout = image_layout,
  };
  const vk::WriteDescriptorSet write {
    .dstSet          = arena.set(set_index),
    .dstBinding      = row.binding,
    .dstArrayElement = 0U,
    .descriptorCount = 1U,
    .descriptorType  = row.descriptor_type,
    .pImageInfo      = &image_info
  };

  device.updateDescriptorSets(write, nullptr);
}

} // namespace vkpp
