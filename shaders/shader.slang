struct vertex_shader_input {
  float2 position;
  float3 color;
};

struct vertex_shader_output {
  float3 color;
  float4 position : SV_Position;
};

[shader("vertex")]
vertex_shader_output vertex_main(vertex_shader_input input)
{
  vertex_shader_output output;
  output.position = float4(input.position, 0.0, 1.0);
  output.color = input.color;
  return output;
}

[shader("fragment")]
float4 fragment_main(vertex_shader_output input) : SV_Target
{
  return float4(input.color, 1.0);
}
