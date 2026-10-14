export module vkpp.descriptor.spirv_binding_check;

import std;
import vulkan;

import vkpp.error;
import vkpp.diagnostics;
import vkpp.descriptor.layout_decl;

namespace vkpp
{

inline constexpr std::uint32_t spirv_magic_number {0x07'23'02'03U};
inline constexpr std::uint32_t spirv_header_word_count {5UZ};
inline constexpr std::uint32_t spirv_version_1_4 {0x00'01'04'00U};
inline constexpr std::uint32_t spirv_opcode_mask {0xFF'FFU};
inline constexpr std::uint32_t spirv_word_count_shift {16U};

inline constexpr std::uint32_t spirv_op_entry_point {15U};
inline constexpr std::uint32_t spirv_op_type_image {25U};
inline constexpr std::uint32_t spirv_op_type_sampler {26U};
inline constexpr std::uint32_t spirv_op_type_sampled_image {27U};
inline constexpr std::uint32_t spirv_op_type_array {28U};
inline constexpr std::uint32_t spirv_op_type_runtime_array {29U};
inline constexpr std::uint32_t spirv_op_type_struct {30U};
inline constexpr std::uint32_t spirv_op_type_pointer {32U};
inline constexpr std::uint32_t spirv_op_variable {59U};
inline constexpr std::uint32_t spirv_op_decorate {71U};

inline constexpr std::uint32_t spirv_decoration_block {2U};
inline constexpr std::uint32_t spirv_decoration_buffer_block {3U};
inline constexpr std::uint32_t spirv_decoration_binding {33U};
inline constexpr std::uint32_t spirv_decoration_descriptor_set {34U};

inline constexpr std::uint32_t spirv_storage_class_uniform {2U};
inline constexpr std::uint32_t spirv_storage_class_storage_buffer {12U};

inline constexpr std::uint32_t spirv_image_sampled_storage {2U};

export enum class spirv_binding_issue_kind : std::uint8_t
{
  malformed_module,
  unsupported_stage,
  entry_point_missing,
  binding_not_declared,
  descriptor_type_mismatch,
  stage_not_declared,
};

export struct spirv_binding_issue
{
  spirv_binding_issue_kind kind {};
  std::uint32_t            set {};
  std::uint32_t            binding {};
};

enum class shader_resource_class : std::uint8_t
{
  sampled_image,
  storage_image,
  sampler,
  combined_image_sampler,
  uniform_buffer,
  storage_buffer,
  other,
};

struct spirv_type_record
{
  std::uint32_t opcode {};
  std::uint32_t element_id {};
  std::uint32_t storage_class {};
  std::uint32_t image_sampled {};
};

struct spirv_decorations
{
  std::optional<std::uint32_t> set {};
  std::optional<std::uint32_t> binding {};
  bool                         block {};
  bool                         buffer_block {};
};

struct spirv_module_index
{
  std::unordered_map<std::uint32_t, spirv_decorations> decorated {};
  std::unordered_map<std::uint32_t, spirv_type_record> types {};
  std::unordered_map<std::uint32_t, std::uint32_t>     variable_types {};
  std::vector<std::uint32_t>                           checked_variables {};
  bool                                                 stage_checked {};
};

[[nodiscard]]
constexpr
auto
spirv_execution_model_of(vk::ShaderStageFlagBits stage)
  -> std::optional<std::uint32_t>
{
  switch (stage)
  {
  case vk::ShaderStageFlagBits::eVertex                 : return 0U;
  case vk::ShaderStageFlagBits::eTessellationControl    : return 1U;
  case vk::ShaderStageFlagBits::eTessellationEvaluation : return 2U;
  case vk::ShaderStageFlagBits::eGeometry               : return 3U;
  case vk::ShaderStageFlagBits::eFragment               : return 4U;
  case vk::ShaderStageFlagBits::eCompute                : return 5U;
  default                                               : return std::nullopt;
  }
}

[[nodiscard]]
inline
auto
read_literal_string(std::span<const std::uint32_t> words)
  -> std::pair<std::string, std::size_t>
{
  std::string text {};
  text.reserve(words.size() * 4UZ);
  for (const auto [ word_index, word ] : std::views::enumerate(words))
  {
    for (const auto byte_index : std::views::indices(4U))
    {
      const auto character =
        static_cast<char>((word >> (byte_index * 8U)) & 0xFFU);
      if (character == '\0')
      {
        return {std::move(text), static_cast<std::size_t>(word_index) + 1UZ};
      }
      text.push_back(character);
    }
  }
  return {std::move(text), words.size()};
}

[[nodiscard]]
inline
auto
index_spirv_module(
  std::span<const std::uint32_t> words,
  std::string_view               entry,
  vk::ShaderStageFlagBits        stage)
  -> std::expected<spirv_module_index, spirv_binding_issue>
{
  using enum spirv_binding_issue_kind;
  if (words.size() < spirv_header_word_count || words[ 0 ] != spirv_magic_number)
  {
    return std::unexpected {spirv_binding_issue {.kind = malformed_module}};
  }
  const std::optional<std::uint32_t> execution_model =
    spirv_execution_model_of(stage);
  if (execution_model)
  {
    return std::unexpected {spirv_binding_issue {.kind = unsupported_stage}};
  }

  spirv_module_index         index {};
  std::vector<std::uint32_t> interface_ids {};
  bool                       entry_found {};
  std::size_t                word_index {spirv_header_word_count};
  while (word_index < words.size())
  {
    const std::uint32_t word_count =
      words[ word_index ] >> spirv_word_count_shift;
    const std::uint32_t opcode = words[ word_index ] & spirv_opcode_mask;
    if (word_count == 0U || word_index + word_count > words.size())
    {
      return std::unexpected {spirv_binding_issue {.kind = malformed_module}};
    }
    const std::span<const std::uint32_t> instruction =
      words.subspan(word_index, word_count);
    switch (opcode)
    {
    case spirv_op_entry_point :
      if (instruction.size() >= 4UZ && instruction[ 1 ] == *execution_model)
      {
        const auto [ name, name_word_count ] =
          read_literal_string(instruction.subspan(3UZ));
        if (name == entry)
        {
          entry_found = true;
          interface_ids.assign_range(
            instruction.subspan(3UZ + name_word_count));
        }
      }
      break;
    case spirv_op_decorate :
      if (instruction.size() >= 3UZ)
      {
        spirv_decorations&  decorations = index.decorated[ instruction[ 1 ] ];
        const std::uint32_t decoration  = instruction[ 2 ];
        if (decoration == spirv_decoration_block) { decorations.block = true; }
        else if (decoration == spirv_decoration_buffer_block)
        {
          decorations.buffer_block = true;
        }
        else if (
          instruction.size() >= 4UZ &&
          decoration == spirv_decoration_descriptor_set)
        {
          decorations.set = instruction[ 3 ];
        }
        else if (
          instruction.size() >= 4UZ && decoration == spirv_decoration_binding)
        {
          decorations.binding = instruction[ 3 ];
        }
      }
      break;
    case spirv_op_type_pointer :
      if (instruction.size() >= 4UZ)
      {
        index.types[ instruction[ 1 ] ] = spirv_type_record {
          .opcode        = opcode,
          .element_id    = instruction[ 3 ],
          .storage_class = instruction[ 2 ],
        };
      }
      break;
    case spirv_op_type_array :
    case spirv_op_type_runtime_array :
      if (instruction.size() >= 3UZ)
      {
        index.types[ instruction[ 1 ] ] =
          spirv_type_record {.opcode = opcode, .element_id = instruction[ 2 ]};
      }
      break;
    case spirv_op_type_image :
      if (instruction.size() >= 9UZ)
      {
        index.types[ instruction[ 1 ] ] = spirv_type_record {
          .opcode = opcode, .image_sampled = instruction[ 7 ]
        };
      }
      break;
    case spirv_op_type_sampler       :
    case spirv_op_type_sampled_image :
    case spirv_op_type_struct :
      if (instruction.size() >= 2UZ)
      {
        index.types[ instruction[ 1 ] ] = spirv_type_record {.opcode = opcode};
      }
      break;
    case spirv_op_variable :
      if (instruction.size() >= 4UZ)
      {
        index.variable_types[ instruction[ 2 ] ] = instruction[ 1 ];
      }
      break;
    default : break;
    }
    word_index += word_count;
  }
  if (!entry_found)
  {
    return std::unexpected {spirv_binding_issue {.kind = entry_point_missing}};
  }

  const auto has_descriptor_binding = [ & ](std::uint32_t id) -> bool {
    const auto decorated = index.decorated.find(id);
    return index.variable_types.contains(id) &&
           decorated != index.decorated.end() &&
           decorated->second.set.has_value() &&
           decorated->second.binding.has_value();
  };
  index.stage_checked = words[ 1 ] >= spirv_version_1_4;
  if (index.stage_checked)
  {
    index.checked_variables =
      interface_ids |
      std::views::filter(has_descriptor_binding) |
      std::ranges::to<std::vector>();
  }
  else
  {
    index.checked_variables =
      index.variable_types |
      std::views::keys |
      std::views::filter(has_descriptor_binding) |
      std::ranges::to<std::vector>();
  }
  return index;
}

[[nodiscard]]
inline
auto
resource_class_of(const spirv_module_index& index, std::uint32_t variable_id)
  -> shader_resource_class
{
  const auto pointer = index.types.find(index.variable_types.at(variable_id));
  if (
    pointer == index.types.end() ||
    pointer->second.opcode != spirv_op_type_pointer)
  {
    return shader_resource_class::other;
  }
  auto          element = index.types.find(pointer->second.element_id);
  std::uint32_t element_id {pointer->second.element_id};
  while (
    element != index.types.end() &&
    (element->second.opcode == spirv_op_type_array ||
      element->second.opcode == spirv_op_type_runtime_array))
  {
    element_id = element->second.element_id;
    element    = index.types.find(element_id);
  }
  if (element == index.types.end()) { return shader_resource_class::other; }

  switch (element->second.opcode)
  {
  case spirv_op_type_sampler : return shader_resource_class::sampler;
  case spirv_op_type_sampled_image :
    return shader_resource_class::combined_image_sampler;
  case spirv_op_type_image :
    return element->second.image_sampled == spirv_image_sampled_storage
             ? shader_resource_class::storage_image
             : shader_resource_class::sampled_image;
  case spirv_op_type_struct :
  {
    const auto decorated = index.decorated.find(element_id);
    const bool block =
      decorated != index.decorated.end() && decorated->second.block;
    const bool buffer_block =
      decorated != index.decorated.end() && decorated->second.buffer_block;
    const std::uint32_t storage_class = pointer->second.storage_class;
    if (storage_class == spirv_storage_class_storage_buffer && block)
    {
      return shader_resource_class::storage_buffer;
    }
    if (storage_class == spirv_storage_class_uniform && buffer_block)
    {
      return shader_resource_class::storage_buffer;
    }
    if (storage_class == spirv_storage_class_uniform && block)
    {
      return shader_resource_class::uniform_buffer;
    }
    return shader_resource_class::other;
  }
  default : return shader_resource_class::other;
  }
}

[[nodiscard]]
constexpr
auto
descriptor_accepts(
  vk::DescriptorType descriptor_type, shader_resource_class resource) -> bool
{
  switch (descriptor_type)
  {
  case vk::DescriptorType::eCombinedImageSampler :
    return resource == shader_resource_class::combined_image_sampler ||
           resource == shader_resource_class::sampled_image ||
           resource == shader_resource_class::sampler;
  case vk::DescriptorType::eUniformBuffer :
  case vk::DescriptorType::eUniformBufferDynamic :
    return resource == shader_resource_class::uniform_buffer;
  case vk::DescriptorType::eStorageBuffer :
  case vk::DescriptorType::eStorageBufferDynamic :
    return resource == shader_resource_class::storage_buffer;
  default : return true;
  }
}

[[nodiscard]]
constexpr
auto
issue_text(spirv_binding_issue_kind kind) -> std::string_view
{
  switch (kind)
  {
  case spirv_binding_issue_kind::malformed_module :
    return "malformed SPIR-V module";
  case spirv_binding_issue_kind::unsupported_stage :
    return "unsupported shader stage";
  case spirv_binding_issue_kind::entry_point_missing :
    return "entry point not found";
  case spirv_binding_issue_kind::binding_not_declared :
    return "binding not declared in resource_layout";
  case spirv_binding_issue_kind::descriptor_type_mismatch :
    return "descriptor type does not match shader";
  case spirv_binding_issue_kind::stage_not_declared :
    return "stage missing from declared stages";
  }
  return "unknown issue";
}

export
template<declared_resource_layout Layout>
[[nodiscard]]
auto
validate_spirv_bindings(
  std::span<const std::uint32_t> words, std::string_view entry, vk::ShaderStageFlagBits stage)
  -> std::expected<void, spirv_binding_issue>
{
  return index_spirv_module(words, entry, stage)
    .and_then(
      [ stage ](const spirv_module_index& index)
        -> std::expected<void, spirv_binding_issue> {
        using enum spirv_binding_issue_kind;
        constexpr auto table = Layout::assignment_table();
        for (const auto variable_id : index.checked_variables)
        {
          const auto&         decorations = index.decorated.at(variable_id);
          const std::uint32_t set         = *decorations.set;
          const std::uint32_t binding     = *decorations.binding;

          const auto row = std::ranges::find_if(
            table, [ & ](const binding_assignment& assignment) -> bool {
              return assignment.set == set && assignment.binding == binding;
            });
          if (row == table.end())
          {
            return std::unexpected {
              spirv_binding_issue {
                .kind   = binding_not_declared,
                .set    = set,
                .binding = binding,
              },
            };
          }
          if (!descriptor_accepts(
                row->descriptor_type, resource_class_of(index, variable_id)))
          {
            return std::unexpected {
              spirv_binding_issue {
                .kind   = descriptor_type_mismatch,
                .set    = set,
                .binding = binding,
              },
            };
          }
          if (
            index.stage_checked &&
            (row->stage_flags & stage) == vk::ShaderStageFlags {})
          {
            return std::unexpected {
              spirv_binding_issue {
                .kind   = stage_not_declared,
                .set    = set,
                .binding = binding,
              },
            };
          }
        }
      });
}

export
template<class Layout, diagnostics_policy DiagnosticsPolicy>
[[nodiscard]]
auto
check_spirv_bindings(
  std::span<const char>    spirv,
  std::string_view                  entry,
  vk::ShaderStageFlagBits           stage,
  std::optional<diagnostic_buffer&> diagnostics) -> std::expected<void, error_t>
{
  if constexpr (
    std::is_void_v<Layout> || !DiagnosticsPolicy::check_spirv_bindings)
  {
    return {};
  }
  else
  {
    const auto error = make_app_error(app_error_code::shader_binding_mismatch);
    if (spirv.size_bytes() % sizeof(std::uint32_t) != 0UZ)
    {
      return std::unexpected {error};
    }
    const std::size_t word_count = spirv.size_bytes() / sizeof(std::uint32_t);
    const std::span<const std::uint32_t> words {
      std::start_lifetime_as_array<std::uint32_t>(spirv.data(), word_count),
      word_count,
    };
    return validate_spirv_bindings<Layout>(words, entry, stage)
    .or_else(
      [&](const spirv_binding_issue& issue) -> std::expected<void, error_t>
      {
        if (diagnostics)
        {
          diagnostics->report(
            diagnostic_severity::error,
            error,
            std::source_location::current(),
            "SPIR-V binding check, entry '{}': {} (set {}, bindings {})",
            entry,
            issue_text(issue.kind),
            issue.set,
            issue.binding);
        }
        return std::unexpected {error};
      }
    );
  }
}

} // namespace vkpp
