struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) color: vec3<f32> }
struct BoidVertex { @location(0) position: vec4<f32>, @location(1) velocity: vec4<f32> }
struct Camera { orbit: vec4<f32> }
@group(0) @binding(0) var<uniform> camera: Camera;

@vertex
fn vertex(@builtin(vertex_index) vertexIndex: u32, boid: BoidVertex) -> VertexOutput {
    let corners = array<vec2<f32>, 6>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0), vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    // Rotate the CUDA-normalized VBO positions with a compact uniform. This
    // avoids matrix-layout differences on mobile WebGPU implementations.
    let position = boid.position.xyz * -0.01;
    let yaw = camera.orbit.x;
    let pitch = camera.orbit.y;
    let yawed = vec3<f32>(
        cos(yaw) * position.x - sin(yaw) * position.z,
        position.y,
        sin(yaw) * position.x + cos(yaw) * position.z
    );
    let pitched = vec3<f32>(
        yawed.x,
        cos(pitch) * yawed.y - sin(pitch) * yawed.z,
        sin(pitch) * yawed.y + cos(pitch) * yawed.z
    );
    let scale = 4.0 / camera.orbit.z;
    var clip = vec4<f32>(pitched.xy * scale, 0.0, 1.0);
    clip.xy = clip.xy + corners[vertexIndex] * vec2<f32>(4.0 / 1280.0, 4.0 / 720.0);
    return VertexOutput(clip, max(abs(boid.velocity.xyz + vec3<f32>(0.3)), vec3<f32>(0.15)));
}

@fragment
fn fragment(input: VertexOutput) -> @location(0) vec4<f32> { return vec4<f32>(input.color, 1.0); }
