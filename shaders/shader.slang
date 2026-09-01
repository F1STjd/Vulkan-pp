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
  uint reserved;
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
  output.world_normal = normalize(mul((float3x3)ubo.model, input.normal));
  output.texture_coordinates = input.texture_coordinates;
  output.color = input.color;
  output.material_index = draw.material_index;
  return output;
}

[[vk::binding(0, 1)]]
Sampler2D textures[];

[shader("fragment")]
float4 fragment_main(vertex_shader_output input, bool front_face : SV_IsFrontFace) : SV_Target
{
  const material_gpu material = materials[input.material_index];
  float4 sampled = float4(1.0, 1.0, 1.0, 1.0);

  if((material.has_texture_mask & 1U) != 0U)
  {
    sampled = textures [
      NonUniformResourceIndex(material.base_color_index)
    ].Sample(input.texture_coordinates);
  }

  const float opacity = sampled.a * material.base_color_factor.a;
  if (material.alpha_mode == 1U && opacity < material.alpha_cutoff)
  {
    discard;
  }

  const float3 albedo =
    sampled.rgb * material.base_color_factor.rgb * input.color;
  const float3 normal = front_face
    ? normalize(input.world_normal)
    : -normalize(input.world_normal);
  const float3 light_direction = normalize(ubo.light_direction);
  const float normal_light_cosine = max(dot(normal, light_direction), 0.0);
  const float3 ambient = 0.18 * albedo;
  const float3 diffuse = albedo * normal_light_cosine * ubo.light_color;
  return float4(ambient + diffuse, opacity);
}

[[vk::binding(2, 0)]]
RWStructuredBuffer<uint> smoke_out;

[shader("compute")]
[numthreads(64, 1, 1)]
void compute_main(uint3 id : SV_DispatchThreadID)
{
  smoke_out[id.x] = id.x;
}