struct vertex_shader_input {
  float2 position;
  float3 color;
  float2 texture_coordinates;
};

struct uniform_buffer {
  float4x4 model;
  float4x4 view;
  float4x4 projection;
};
ConstantBuffer<uniform_buffer> ubo;

struct vertex_shader_output {
  float4 position : SV_Position;
  float3 color;
  float2 texture_coordinates;
};

[shader("vertex")]
vertex_shader_output vertex_main(vertex_shader_input input)
{
  vertex_shader_output output;
  output.position = mul(ubo.projection, mul(ubo.view, mul(ubo.model, float4(input.position, 0.0, 1.0))));
  output.color = input.color;
  output.texture_coordinates = input.texture_coordinates;
  return output;
}

Sampler2D texture;

[shader("fragment")]
float4 fragment_main(vertex_shader_output input) : SV_Target
{
  return texture.Sample(input.texture_coordinates);
}
