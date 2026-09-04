struct vertex_shader_input {
  [[vk::location(0)]] float3 position;
  [[vk::location(1)]] float3 normal;
  [[vk::location(2)]] float3 color;
  [[vk::location(3)]] float2 texture_coordinates;
};

struct uniform_buffer {
  float4x4 model;
  float4x4 view;
  float4x4 projection;
  float3 light_direction;
  float3 light_color;
  float3 camera_position;
  float exposure;
};
[[vk::binding(0, 0)]]
ConstantBuffer<uniform_buffer> ubo;

struct material_gpu {
  float4 base_color_factor;
  float4 emissive_factor_and_metallic;
  float roughness_factor;
  float normal_scale;
  float occlusion_strength;
  float alpha_cutoff;
  uint base_color_index;
  uint metallic_roughness_index;
  uint normal_index;
  uint occlusion_index;
  uint emissive_index;
  uint alpha_mode;
  uint has_texture_mask;
  float transmission_factor;
};
[[vk::binding(1, 0)]]
StructuredBuffer<material_gpu> materials;

struct draw_gpu {
  float4 world_columns[4];
  float4 normal_matrix_columns[3];
  uint material_index;
  uint _0;
  uint _1;
  uint _2;
};
[[vk::binding(3, 0)]]
StructuredBuffer<draw_gpu> draws;

struct draw_push
{
  uint draw_index;
};
[[vk::push_constant]]
ConstantBuffer<draw_push> draw_pc;

struct vertex_shader_output {
  float4 position : SV_Position;
  [[vk::location(0)]] float3 world_position;
  [[vk::location(1)]] float3 world_normal;
  [[vk::location(2)]] float2 texture_coordinates;
  [[vk::location(3)]] float3 color;
  [[vk::location(4)]] nointerpolation uint material_index;
};

float4 transform_columns(float4 columns[4], float4 value)
{
  return
    columns[0] * value.x + columns[1] * value.y +
    columns[2] * value.z + columns[3] * value.w;
}

float3 transform_normal_columns(float4 columns[3], float3 value)
{
  return
    columns[0].xyz * value.x + columns[1].xyz * value.y +
    columns[2].xyz * value.z;
}

[shader("vertex")]
vertex_shader_output vertex_main(vertex_shader_input input)
{
  vertex_shader_output output;
  const draw_gpu draw = draws[draw_pc.draw_index];
  const float4 node_world =
    transform_columns(draw.world_columns, float4(input.position, 1.0));
  const float4 world = mul(ubo.model, node_world);
  output.world_position = world.xyz;
  output.position = mul(ubo.projection, mul(ubo.view, world));

  const float3 node_normal =
    transform_normal_columns(draw.normal_matrix_columns, input.normal);
  output.world_normal = normalize(mul((float3x3)ubo.model, node_normal));
  output.texture_coordinates = input.texture_coordinates;
  output.color = input.color;
  output.material_index = draw.material_index;
  return output;
}

[[vk::binding(0, 1)]]
Sampler2D textures[];

float3 apply_normal_map(
  float3 geometric_normal, float3 world_position, float2 texture_coordinates,
  float3 encoded_normal, float scale)
{
  const float3 position_dx = ddx(world_position);
  const float3 position_dy = ddy(world_position);
  const float2 uv_dx = ddx(texture_coordinates);
  const float2 uv_dy = ddy(texture_coordinates);
  const float determinant = uv_dx.x * uv_dy.y - uv_dx.y * uv_dy.x;
  if(abs(determinant) < 1e-12)
  {
    return geometric_normal;
  }

  const float inverse_determinant = -1.0 / determinant;
  const float3 raw_tangent =
    (position_dx * uv_dy.y - position_dy * uv_dx.y) * inverse_determinant;
  const float3 raw_bitangent =
    (position_dy * uv_dx.x - position_dx * uv_dy.x) * inverse_determinant;
  const float3 tangent =
    normalize(
      raw_tangent - geometric_normal * dot(geometric_normal, raw_tangent)
    );
  const float handedness =
    dot(cross(geometric_normal, tangent), raw_bitangent) < 0.0
    ? -1.0
    : 1.0;
  const float3 bitangent = handedness * cross(geometric_normal, tangent);

  float3 mapped = encoded_normal * 2.0 - 1.0;
  mapped.xy *= scale;
  mapped = normalize(mapped);
  return normalize(
    tangent * mapped.x + bitangent * mapped.y + geometric_normal * mapped.z
  );
}

static const float pi = 3.14159265359;

float distribution_ggx(float normal_half_cosine, float roughness)
{
  const float alpha = roughness * roughness;
  const float alpha_squared = alpha * alpha;
  const float denominator =
    normal_half_cosine * normal_half_cosine * (alpha_squared - 1.0) + 1.0;
  return alpha_squared / max(pi * denominator * denominator, 1e-7);
}

float geometry_schlick_ggx(float normal_direction_cosine, float roughness)
{
  const float r = roughness + 1.0;
  const float k = r * r / 8.0;
  return
    normal_direction_cosine /
    max(normal_direction_cosine * (1.0 - k) + k, 1e-7);
}

float geometry_smith(
  float normal_view_cosine, float normal_light_cosine, float roughness)
{
  return
    geometry_schlick_ggx(normal_view_cosine, roughness) *
    geometry_schlick_ggx(normal_light_cosine, roughness);
}

float3 fresnel_schlick(float cosine, float3 f0)
{
  return f0 + (1.0 - f0) * pow(saturate(1.0 - cosine), 5.0);
}

float3 tonemap_reinhard_luminance(float3 color)
{
  const float luminance = dot(color, float3(0.2126, 0.7152, 0.0722));
  return color / (1.0 + luminance);
}

[shader("fragment")]
float4 fragment_main(
  vertex_shader_output input, bool front_face : SV_IsFrontFace) : SV_Target
{
  const material_gpu material = materials[input.material_index];
  float4 base_color_sample = float4(1.0, 1.0, 1.0, 1.0);
  if ((material.has_texture_mask & 1U) != 0U)
  {
    base_color_sample = textures[
      NonUniformResourceIndex(material.base_color_index)
    ].Sample(input.texture_coordinates);
  }

  float opacity = base_color_sample.a * material.base_color_factor.a;
  if (material.alpha_mode == 1U && opacity < material.alpha_cutoff)
  {
    discard;
  }
  if (material.transmission_factor > 0.0)
  {
    opacity *= saturate(1.0 - material.transmission_factor);
  }
  const float3 albedo =
    base_color_sample.rgb * material.base_color_factor.rgb * input.color;

  float2 metallic_roughness = float2(
    material.emissive_factor_and_metallic.w,
    material.roughness_factor,
  );
  if ((material.has_texture_mask & 2U) != 0U)
  {
    const float4 sample_value =  textures[
      NonUniformResourceIndex(material.metallic_roughness_index)
    ].Sample(input.texture_coordinates);
    metallic_roughness *= float2(sample_value.b, sample_value.g);
  }
  const float metallic = saturate(metallic_roughness.x);
  const float roughness = clamp(metallic_roughness.y, 0.045, 1.0);

  float occlusion = 1.0;
  if((material.has_texture_mask & 8U) != 0U)
  {
    const float sampled_occlusion = textures[
      NonUniformResourceIndex(material.occlusion_index)
    ].Sample(input.texture_coordinates).r;
    occlusion = lerp(1.0, sampled_occlusion, material.occlusion_strength);
  }

  float3 emissive = material.emissive_factor_and_metallic.rgb;
  if((material.has_texture_mask & 16U) != 0U)
  {
    emissive *= textures[
      NonUniformResourceIndex(material.emissive_index)
    ].Sample(input.texture_coordinates).rgb;
  }

  float3 normal = front_face
    ? normalize(input.world_normal)
    : -normalize(input.world_normal);
  if((material.has_texture_mask & 4U) != 0U)
  {
    const float3 encoded_normal = textures[
      NonUniformResourceIndex(material.normal_index)
    ].Sample(input.texture_coordinates).rgb;
    normal = apply_normal_map(
      normal, input.world_position, input.texture_coordinates, encoded_normal,
      material.normal_scale
    );
  }

  const float3 view_direction = normalize(
    ubo.camera_position - input.world_position
  );
  const float3 light_direction = normalize(ubo.light_direction);
  const float3 half_vector = normalize(light_direction + view_direction);
  const float normal_light_cosine = max(dot(normal, light_direction), 0.0);
  const float normal_view_cosine = max(dot(normal, view_direction), 1e-4);
  const float normal_half_cosine = max(dot(normal, half_vector), 0.0);
  const float view_half_cosine = max(dot(view_direction, half_vector), 0.0);

  const float3 f0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);
  const float3 fresnel = fresnel_schlick(view_half_cosine, f0);
  const float3 specular =
    fresnel *
    distribution_ggx(normal_half_cosine, roughness) *
    geometry_smith(normal_view_cosine, normal_light_cosine, roughness) /
    max(4.0 * normal_view_cosine *  normal_light_cosine, 1e-4);
  const float3 diffuse = (1.0 - fresnel) * (1.0 - metallic) * albedo / pi;
  const float3 direct = (diffuse + specular) * ubo.light_color * normal_light_cosine;
  const float3 ambient = 0.03 * albedo * occlusion;
  const float3 color = direct + ambient + emissive;
  const float3 display_linear = tonemap_reinhard_luminance(color * ubo.exposure);
  return float4(display_linear, opacity);
}

[[vk::binding(2, 0)]]
RWStructuredBuffer<uint> smoke_out;

[shader("compute")]
[numthreads(64, 1, 1)]
void compute_main(uint3 id : SV_DispatchThreadID)
{
  smoke_out[id.x] = id.x;
}