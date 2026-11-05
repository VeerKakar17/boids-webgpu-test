import { GUI } from 'dat.gui';
import computeShader from './shaders/boids.wgsl?raw';
import renderShader from './shaders/boids_render.wgsl?raw';

const WIDTH = 1280;
const HEIGHT = 720;
const DEFAULT_BOIDS = 20;
const MAX_BOIDS = 100000;
const MAX_SORT_SIZE = 131072;
const GRID_SIDE = 22;
const GRID_CELLS = GRID_SIDE * GRID_SIDE * GRID_SIDE;
const WORKGROUP_SIZE = 256;
const SCAN_BLOCK_SIZE = WORKGROUP_SIZE * 2;
const MAX_SCAN_BLOCK_COUNT = MAX_SORT_SIZE / SCAN_BLOCK_SIZE;

const canvas = document.getElementById('mainCanvas') as HTMLCanvasElement;
canvas.width = WIDTH;
canvas.height = HEIGHT;
if (!navigator.gpu) {
    document.body.innerHTML = '<p class="error">This browser does not support WebGPU.</p>';
    throw new Error('WebGPU unavailable');
}
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error('No suitable WebGPU adapter found');
const device = await adapter.requestDevice({
    requiredLimits: { maxStorageBuffersPerShaderStage: 10 }
});
device.addEventListener('uncapturederror', (event) => {
    console.error('WebGPU error:', event.error);
    const message = document.createElement('p');
    message.className = 'error';
    message.textContent = `WebGPU error: ${event.error.message}`;
    document.body.appendChild(message);
});
const context = canvas.getContext('webgpu')!;
const format = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format, alphaMode: 'opaque' });

const fpsCounter = document.getElementById('fpsCounter')!;
const controls = { boidCount: DEFAULT_BOIDS };
const gui = new GUI();

function makeBuffer(label: string, size: number, usage: GPUBufferUsageFlags): GPUBuffer {
    return device.createBuffer({ label, size, usage });
}

function sortSizeFor(count: number): number {
    let size = SCAN_BLOCK_SIZE;
    while (size < count) size *= 2;
    return size;
}

class BoidsApp {
    private readonly computeShader = device.createShaderModule({ label: 'boids compute shaders', code: computeShader });
    private readonly renderShader = device.createShaderModule({ label: 'boids render shaders', code: renderShader });
    private readonly settings = makeBuffer('boid settings', 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    private readonly radixBits = makeBuffer('radix bit uniforms', 32 * 256, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    private readonly boidsA = makeBuffer('boids A', MAX_SORT_SIZE * 32, GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX);
    private readonly boidsB = makeBuffer('boids B', MAX_SORT_SIZE * 32, GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX);
    private readonly reordered = makeBuffer('cell-coherent boids', MAX_SORT_SIZE * 32, GPUBufferUsage.STORAGE);
    private readonly pairsA = makeBuffer('radix pairs A', MAX_SORT_SIZE * 8, GPUBufferUsage.STORAGE);
    private readonly pairsB = makeBuffer('radix pairs B', MAX_SORT_SIZE * 8, GPUBufferUsage.STORAGE);
    private readonly bits = makeBuffer('radix bits', MAX_SORT_SIZE * 4, GPUBufferUsage.STORAGE);
    private readonly scan = makeBuffer('radix scan', MAX_SORT_SIZE * 4, GPUBufferUsage.STORAGE);
    private readonly cellStart = makeBuffer('cell starts', GRID_CELLS * 4, GPUBufferUsage.STORAGE);
    private readonly cellEnd = makeBuffer('cell ends', GRID_CELLS * 4, GPUBufferUsage.STORAGE);
    private readonly blockSums = makeBuffer('scan block sums', MAX_SCAN_BLOCK_COUNT * 4, GPUBufferUsage.STORAGE);
    private readonly computeLayout: GPUBindGroupLayout;
    private readonly computeGroups: [GPUBindGroup, GPUBindGroup];
    private readonly computePipelines: Record<string, GPUComputePipeline>;
    private readonly renderPipeline: GPURenderPipeline;
    private count = DEFAULT_BOIDS;
    private sortSize = sortSizeFor(DEFAULT_BOIDS);
    private activeA = true;

    constructor() {
        const computeEntries: GPUBindGroupLayoutEntry[] = [
            ...[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(binding => ({
                binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' as const }
            })),
            { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 4 } },
        ];
        this.computeLayout = device.createBindGroupLayout({
            label: 'boids compute layout',
            entries: computeEntries
        });
        this.computeGroups = [this.makeComputeGroup(this.boidsA, this.boidsB), this.makeComputeGroup(this.boidsB, this.boidsA)];
        const computeLayout = device.createPipelineLayout({ bindGroupLayouts: [this.computeLayout] });
        this.computePipelines = Object.fromEntries([
            'initialize', 'assignPairs', 'radixABMap', 'radixBAMap', 'scanBlocks', 'scanBlockSums', 'addBlockSums',
            'radixABScatter', 'radixBAScatter', 'clearCells', 'buildCellRanges', 'gather', 'update',
        ].map(entry => [entry, device.createComputePipeline({
            label: entry, layout: computeLayout, compute: { module: this.computeShader, entryPoint: entry }
        })]));
        this.renderPipeline = device.createRenderPipeline({
            label: 'boid render pipeline',
            layout: 'auto',
            vertex: {
                module: this.renderShader,
                entryPoint: 'vertex',
                buffers: [{
                    arrayStride: 32,
                    stepMode: 'instance',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x4' },
                        { shaderLocation: 1, offset: 16, format: 'float32x4' },
                    ]
                }]
            },
            fragment: { module: this.renderShader, entryPoint: 'fragment', targets: [{ format }] },
            primitive: { topology: 'triangle-list' },
        });
        for (let bit = 0; bit < 32; bit++) device.queue.writeBuffer(this.radixBits, bit * 256, new Uint32Array([bit]));
        this.reset(DEFAULT_BOIDS);
    }

    private makeComputeGroup(current: GPUBuffer, next: GPUBuffer): GPUBindGroup {
        const resources = [current, this.reordered, next, this.pairsA, this.pairsB, this.bits, this.scan, this.cellStart, this.cellEnd, this.blockSums, this.settings, this.radixBits];
        return device.createBindGroup({
            label: 'boids compute bindings', layout: this.computeLayout,
            entries: resources.map((value, binding) => ({
                binding,
                resource: binding === 11 ? { buffer: value, size: 16 } : { buffer: value }
            }))
        });
    }

    reset(count: number): void {
        this.count = count;
        this.sortSize = sortSizeFor(count);
        this.activeA = true;
        device.queue.writeBuffer(this.settings, 0, new Uint32Array([count, this.sortSize]));
        const encoder = device.createCommandEncoder({ label: 'initialize boids' });
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.computePipelines.initialize);
        pass.setBindGroup(0, this.computeGroups[0], [0]);
        pass.dispatchWorkgroups(Math.ceil(this.sortSize / WORKGROUP_SIZE));
        pass.end();
        device.queue.submit([encoder.finish()]);
    }

    frame(): void {
        const group = this.computeGroups[this.activeA ? 0 : 1];
        const encoder = device.createCommandEncoder({ label: 'boids frame' });
        const compute = encoder.beginComputePass();
        const run = (name: string, groups: number, bit = 0) => {
            compute.setPipeline(this.computePipelines[name]);
            compute.setBindGroup(0, group, [bit * 256]);
            compute.dispatchWorkgroups(groups);
        };
        run('assignPairs', Math.ceil(this.sortSize / WORKGROUP_SIZE));
        for (let bit = 0; bit < 32; bit++) {
            const direction = bit % 2 === 0 ? 'AB' : 'BA';
            run(`radix${direction}Map`, Math.ceil(this.sortSize / WORKGROUP_SIZE), bit);
            run('scanBlocks', this.sortSize / SCAN_BLOCK_SIZE, bit);
            run('scanBlockSums', 1, bit);
            run('addBlockSums', this.sortSize / SCAN_BLOCK_SIZE, bit);
            run(`radix${direction}Scatter`, Math.ceil(this.sortSize / WORKGROUP_SIZE), bit);
        }
        run('clearCells', Math.ceil(GRID_CELLS / WORKGROUP_SIZE));
        run('buildCellRanges', Math.ceil(this.count / WORKGROUP_SIZE));
        run('gather', Math.ceil(this.count / WORKGROUP_SIZE));
        run('update', Math.ceil(this.count / WORKGROUP_SIZE));
        compute.end();
        const render = encoder.beginRenderPass({
            colorAttachments: [{ view: context.getCurrentTexture().createView(), clearValue: [0, 0, 0, 1], loadOp: 'clear', storeOp: 'store' }]
        });
        render.setPipeline(this.renderPipeline);
        render.setVertexBuffer(0, this.activeA ? this.boidsB : this.boidsA);
        render.draw(6, this.count);
        render.end();
        device.queue.submit([encoder.finish()]);
        this.activeA = !this.activeA;
    }
}

const app = new BoidsApp();
gui.add(controls, 'boidCount', 1, MAX_BOIDS, 1).name('Boid count').onFinishChange((count: number) => app.reset(count));
let measuredFrames = 0;
let measurementStart = performance.now();

async function animate(): Promise<void> {
    app.frame();
    // A frame is counted only after this submission has completed on the GPU.
    // This prevents the mobile counter from reporting queued JavaScript frames.
    await device.queue.onSubmittedWorkDone();
    measuredFrames++;
    const now = performance.now();
    const elapsed = now - measurementStart;
    if (elapsed >= 500) {
        fpsCounter.textContent = `${(measuredFrames * 1000 / elapsed).toFixed(1)} FPS`;
        measuredFrames = 0;
        measurementStart = now;
    }
    requestAnimationFrame(() => { void animate(); });
}
requestAnimationFrame(() => { void animate(); });
