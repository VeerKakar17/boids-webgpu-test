struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) color: vec3<f32> }
struct BoidVertex { @location(0) position: vec4<f32>, @location(1) velocity: vec4<f32> }
struct Camera { viewProjection: mat4x4<f32> }
@group(0) @binding(0) var<uniform> camera: Camera;

@vertex
fn vertex(@builtin(vertex_index) vertexIndex: u32, boid: BoidVertex) -> VertexOutput {
    let corners = array<vec2<f32>, 6>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0), vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    // The CUDA VBO writes positions in [-1, 1] by multiplying by -1 / scene_scale.
    var clip = camera.viewProjection * vec4<f32>(boid.position.xyz * -0.01, 1.0);
    clip.xy = clip.xy + corners[vertexIndex] * clip.w * vec2<f32>(4.0 / 1280.0, 4.0 / 720.0);
    return VertexOutput(clip, max(abs(boid.velocity.xyz + vec3<f32>(0.3)), vec3<f32>(0.15)));
}

@fragment
fn fragment(input: VertexOutput) -> @location(0) vec4<f32> { return vec4<f32>(input.color, 1.0); }
