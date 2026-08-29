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

struct vertex_shader_output {
  float4 position : SV_Position;
  [[vk::location(0)]] float3 world_position;
  [[vk::location(1)]] float3 world_normal;
  [[vk::location(2)]] float2 texture_coordinates;
};

struct draw_push {
  float4x4 world;
  uint texture_index;
  uint alpha_mode;
  float alpha_cutoff;
  float padding;
  float4 base_color_factor;
};

[[vk::push_constant]]
ConstantBuffer<draw_push> draw_pc;

[shader("vertex")]
vertex_shader_output vertex_main(vertex_shader_input input)
{
  vertex_shader_output output;
  const float4 world = mul(ubo.model, mul(draw_pc.world, float4(input.position, 1.0)));
  output.world_position = world.xyz;
  output.position = mul(ubo.projection, mul(ubo.view, world));
  const float3x3 world_n = (float3x3)mul(ubo.model, draw_pc.world);
  output.world_normal = normalize(mul(world_n, input.normal));
  output.texture_coordinates = input.texture_coordinates;
  return output;
}

[[vk::binding(0, 1)]]
Sampler2D textures[];

[shader("fragment")]
float4 fragment_main(vertex_shader_output input, bool front_face : SV_IsFrontFace) : SV_Target
{
  float4 sampled = textures [
    NonUniformResourceIndex(draw_pc.texture_index)
  ].Sample(input.texture_coordinates);

  if (draw_pc.alpha_mode == 1)
  {
    if (sampled.a < draw_pc.alpha_cutoff)
    {
      discard;
    }
  }
  const float3 albedo = sampled.rgb * draw_pc.base_color_factor.rgb;
  const float3 normal =
    front_face
    ? normalize(input.world_normal)
    : -normalize(input.world_normal);
  const float3 light_direction = normalize(ubo.light_direction);
  const float normal_light_cosine = max(dot(normal, light_direction), 0.0);
  const float3 ambient = 0.18 * albedo;
  const float3 diffuse = albedo * normal_light_cosine * ubo.light_color;
  return float4(ambient + diffuse, 1.0);
}

[[vk::binding(2, 0)]]
RWStructuredBuffer<uint> smoke_out;

[shader("compute")]
[numthreads(64, 1, 1)]
void compute_main(uint3 id : SV_DispatchThreadID)
{
  smoke_out[id.x] = id.x;
}