static float2 positions[3] = float2[]
(
  float2(0.0, -1.0),
  float2(1.0, 1.0),
  float2(-1.0, 1.0),
);

static float3 colors[3] = float3[]
(
  float3(1.0, 0.0, 0.0),
  float3(0.0, 1.0, 0.0),
  float3(0.0, 0.0, 1.0),
);

struct vertex_output {
  float3 color;
  float4 position : SV_Position;
};

[shader("vertex")]
vertex_output vertex_main(uint vertex_id : SV_VulkanVertexID)
{
  vertex_output output;
  output.position = float4(positions[vertex_id], 0.0, 1.0);
  output.color = colors[vertex_id];
  return output;
}

[shader("fragment")]
float4 fragment_main(vertex_output input) : SV_Target
{
  float3 color= input.color;
  return float4(color, 1.0);
}