const GRID_SIDE: i32 = 22;
const GRID_CELLS: u32 = 10648u;
const SCENE_SCALE: f32 = 100.0;
const GRID_MIN: f32 = -110.0;
const INV_CELL_WIDTH: f32 = 0.1;

struct Boid { position: vec4<f32>, velocity: vec4<f32> }
struct Pair { key: u32, index: u32 }
struct Settings { count: u32, sortSize: u32, _padding: vec2<u32> }
struct RadixBit { bit: u32 }
struct Camera { viewProjection: mat4x4<f32>, viewport: vec2<f32>, _padding: vec2<f32> }

@group(0) @binding(0) var<storage, read_write> current: array<Boid>;
@group(0) @binding(1) var<storage, read_write> sortedBoids: array<Boid>;
@group(0) @binding(2) var<storage, read_write> next: array<Boid>;
@group(0) @binding(3) var<storage, read_write> pairsA: array<Pair>;
@group(0) @binding(4) var<storage, read_write> pairsB: array<Pair>;
@group(0) @binding(5) var<storage, read_write> bits: array<u32>;
@group(0) @binding(6) var<storage, read_write> scan: array<u32>;
@group(0) @binding(7) var<storage, read_write> cellStart: array<u32>;
@group(0) @binding(8) var<storage, read_write> cellEnd: array<u32>;
@group(0) @binding(9) var<storage, read_write> blockSums: array<u32>;
@group(0) @binding(10) var<uniform> settings: Settings;
@group(0) @binding(11) var<uniform> radix: RadixBit;


fn hash(value: u32) -> u32 {
    var a = value;
    a = (a + 0x7ed55d16u) + (a << 12u);
    a = (a ^ 0xc761c23cu) ^ (a >> 19u);
    a = (a + 0x165667b1u) + (a << 5u);
    a = (a + 0xd3a2646cu) ^ (a << 9u);
    a = (a + 0xfd7046c5u) + (a << 3u);
    return (a ^ 0xb55a4f09u) ^ (a >> 16u);
}

fn randomSigned(index: u32, component: u32) -> f32 {
    return f32(hash(index * 3u + component)) / 4294967295.0 * 2.0 - 1.0;
}

fn cellKey(position: vec3<f32>) -> u32 {
    let cell = clamp(vec3<i32>(floor((position - vec3<f32>(GRID_MIN)) * INV_CELL_WIDTH)), vec3<i32>(0), vec3<i32>(GRID_SIDE - 1));
    return u32(cell.x + cell.y * GRID_SIDE + cell.z * GRID_SIDE * GRID_SIDE);
}

@compute @workgroup_size(256)
fn initialize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= settings.sortSize) { return; }
    if (index < settings.count) {
        current[index].position = vec4<f32>(randomSigned(index, 0u), randomSigned(index, 1u), randomSigned(index, 2u), 1.0) * vec4<f32>(SCENE_SCALE, SCENE_SCALE, SCENE_SCALE, 1.0);
        current[index].position.w = 1.0;
        current[index].velocity = vec4<f32>(randomSigned(index, 3u), randomSigned(index, 4u), randomSigned(index, 5u), 0.0);
    } else {
        current[index].position = vec4<f32>(0.0);
        current[index].velocity = vec4<f32>(0.0);
    }
}

@compute @workgroup_size(256)
fn assignPairs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= settings.sortSize) { return; }
    if (index < settings.count) {
        pairsA[index] = Pair(cellKey(current[index].position.xyz), index);
    } else {
        pairsA[index] = Pair(0xffffffffu, 0xffffffffu);
    }
}

fn mapBit(index: u32, value: Pair) {
    let bitSet = (value.key >> radix.bit) & 1u;
    bits[index] = bitSet;
    scan[index] = 1u - bitSet;
}

@compute @workgroup_size(256)
fn radixABMap(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < settings.sortSize) { mapBit(gid.x, pairsA[gid.x]); }
}

@compute @workgroup_size(256)
fn radixBAMap(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < settings.sortSize) { mapBit(gid.x, pairsB[gid.x]); }
}

var<workgroup> sharedScan: array<u32, 512>;

@compute @workgroup_size(256)
fn scanBlocks(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
    let tid = local.x;
    let base = group.x * 512u;
    sharedScan[tid] = scan[base + tid];
    sharedScan[tid + 256u] = scan[base + tid + 256u];
    workgroupBarrier();
    var offset = 1u;
    var d = 256u;
    loop {
        if (tid < d) {
            let ai = offset * (2u * tid + 1u) - 1u;
            let bi = offset * (2u * tid + 2u) - 1u;
            sharedScan[bi] = sharedScan[bi] + sharedScan[ai];
        }
        workgroupBarrier();
        d = d / 2u;
        offset = offset * 2u;
        if (d == 0u) { break; }
    }
    if (tid == 0u) {
        blockSums[group.x] = sharedScan[511u];
        sharedScan[511u] = 0u;
    }
    workgroupBarrier();
    d = 1u;
    loop {
        offset = offset / 2u;
        if (tid < d) {
            let ai = offset * (2u * tid + 1u) - 1u;
            let bi = offset * (2u * tid + 2u) - 1u;
            let temp = sharedScan[ai];
            sharedScan[ai] = sharedScan[bi];
            sharedScan[bi] = sharedScan[bi] + temp;
        }
        workgroupBarrier();
        d = d * 2u;
        if (d == 512u) { break; }
    }
    scan[base + tid] = sharedScan[tid];
    scan[base + tid + 256u] = sharedScan[tid + 256u];
}

@compute @workgroup_size(256)
fn scanBlockSums(@builtin(local_invocation_id) local: vec3<u32>) {
    if (local.x != 0u) { return; }
    var total = 0u;
    for (var i = 0u; i < settings.sortSize / 512u; i++) {
        let value = blockSums[i];
        blockSums[i] = total;
        total = total + value;
    }
}

@compute @workgroup_size(256)
fn addBlockSums(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < settings.sortSize) { scan[index] = scan[index] + blockSums[index / 512u]; }
}

fn scatterPair(index: u32, input: Pair, outputIndex: u32) {
    let totalZeros = scan[settings.sortSize - 1u] + (1u - bits[settings.sortSize - 1u]);
    let destination = select(scan[index], index - scan[index] + totalZeros, bits[index] == 1u);
    if (outputIndex == 0u) { pairsB[destination] = input; } else { pairsA[destination] = input; }
}

@compute @workgroup_size(256)
fn radixABScatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < settings.sortSize) { scatterPair(gid.x, pairsA[gid.x], 0u); }
}

@compute @workgroup_size(256)
fn radixBAScatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < settings.sortSize) { scatterPair(gid.x, pairsB[gid.x], 1u); }
}

@compute @workgroup_size(256)
fn clearCells(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < GRID_CELLS) { cellStart[gid.x] = 0xffffffffu; cellEnd[gid.x] = 0xffffffffu; }
}

@compute @workgroup_size(256)
fn buildCellRanges(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= settings.count) { return; }
    let key = pairsA[index].key;
    if (index == 0u || key != pairsA[index - 1u].key) { cellStart[key] = index; }
    if (index + 1u == settings.count || key != pairsA[index + 1u].key) { cellEnd[key] = index; }
}

@compute @workgroup_size(256)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= settings.count) { return; }
    // A valid radix permutation always has an in-range source index. Keeping
    // the unsorted element as a fallback prevents an invalid key from ever
    // producing an off-screen/undefined vertex.
    let sourceIndex = pairsA[gid.x].index;
    sortedBoids[gid.x] = current[select(gid.x, sourceIndex, sourceIndex < settings.count)];
}

fn addNeighbor(boidIndex: u32, neighbor: u32, position: vec3<f32>, center: ptr<function, vec3<f32>>, count1: ptr<function, u32>, separation: ptr<function, vec3<f32>>, velocity: ptr<function, vec3<f32>>, count3: ptr<function, u32>) {
    if (neighbor == boidIndex) { return; }
    let otherPosition = sortedBoids[neighbor].position.xyz;
    let delta = otherPosition - position;
    let distanceSquared = dot(delta, delta);
    if (distanceSquared < 25.0) { *center = *center + otherPosition; *count1 = *count1 + 1u; }
    if (distanceSquared < 9.0) { *separation = *separation - delta; }
    if (distanceSquared < 25.0) { *velocity = *velocity + sortedBoids[neighbor].velocity.xyz; *count3 = *count3 + 1u; }
}

@compute @workgroup_size(256)
fn update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= settings.count) { return; }
    let boid = sortedBoids[index];
    let position = boid.position.xyz;
    let cell = clamp(vec3<i32>(floor((position - vec3<f32>(GRID_MIN)) * INV_CELL_WIDTH)), vec3<i32>(0), vec3<i32>(GRID_SIDE - 1));
    let localPosition = (position - vec3<f32>(GRID_MIN)) * INV_CELL_WIDTH - vec3<f32>(cell);
    let direction = select(vec3<i32>(-1), vec3<i32>(1), localPosition >= vec3<f32>(0.5));
    var center = vec3<f32>(0.0);
    var centerCount = 0u;
    var separation = vec3<f32>(0.0);
    var averageVelocity = vec3<f32>(0.0);
    var velocityCount = 0u;
    for (var candidate = 0u; candidate < 8u; candidate++) {
        let useX = candidate == 1u || candidate >= 4u;
        let useY = candidate == 2u || candidate == 4u || candidate == 6u || candidate == 7u;
        let useZ = candidate == 3u || candidate == 5u || candidate == 6u || candidate == 7u;
        let neighborCell = cell + vec3<i32>(select(0, direction.x, useX), select(0, direction.y, useY), select(0, direction.z, useZ));
        if (any(neighborCell < vec3<i32>(0)) || any(neighborCell >= vec3<i32>(GRID_SIDE))) { continue; }
        let key = u32(neighborCell.x + neighborCell.y * GRID_SIDE + neighborCell.z * GRID_SIDE * GRID_SIDE);
        let start = cellStart[key];
        let end = cellEnd[key];
        if (start == 0xffffffffu || end == 0xffffffffu) { continue; }
        for (var neighbor = start; neighbor <= end; neighbor++) {
            addNeighbor(index, neighbor, position, &center, &centerCount, &separation, &averageVelocity, &velocityCount);
        }
    }
    var velocity = boid.velocity.xyz + separation * 0.1;
    if (centerCount != 0u) { velocity = velocity + (center / f32(centerCount) - position) * 0.01; }
    if (velocityCount != 0u) { velocity = velocity + averageVelocity / f32(velocityCount) * 0.1; }
    let speed = length(velocity);
    if (speed > 1.0) { velocity = velocity / speed; }
    var newPosition = position + velocity * 0.2;
    newPosition = select(newPosition, vec3<f32>(SCENE_SCALE), newPosition < vec3<f32>(-SCENE_SCALE));
    newPosition = select(newPosition, vec3<f32>(-SCENE_SCALE), newPosition > vec3<f32>(SCENE_SCALE));
    next[index] = Boid(vec4<f32>(newPosition, 1.0), vec4<f32>(velocity, 0.0));
}

