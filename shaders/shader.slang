struct vertex_shader_input {
  float3 position;
  float3 color;
  float2 texture_coordinates;
};

struct uniform_buffer {
  float4x4 model;
  float4x4 view;
  float4x4 projection;
};
[[vk::binding(0, 0)]]
ConstantBuffer<uniform_buffer> ubo;

struct vertex_shader_output {
  float4 position : SV_Position;
  float3 color;
  float2 texture_coordinates;
};

struct draw_push {
  float4x4 world;
  uint texture_index;
};

[[vk::push_constant]]
ConstantBuffer<draw_push> draw_pc;

[shader("vertex")]
vertex_shader_output vertex_main(vertex_shader_input input)
{
  vertex_shader_output output;
  const float4 local = float4(input.position, 1.0);
  output.position = mul(ubo.projection, mul(ubo.view, mul(ubo.model, mul(draw_pc.world, local))));
  output.color = input.color;
  output.texture_coordinates = input.texture_coordinates;
  return output;
}

[[vk::binding(0, 1)]]
Sampler2D textures[];

[shader("fragment")]
float4 fragment_main(vertex_shader_output input) : SV_Target
{
  return textures[NonUniformResourceIndex(draw_pc.texture_index)].Sample(input.texture_coordinates);
}

[[vk::binding(2, 0)]]
RWStructuredBuffer<uint> smoke_out;

[shader("compute")]
[numthreads(64, 1, 1)]
void compute_main(uint3 id : SV_DispatchThreadID)
{
  smoke_out[id.x] = id.x;
}