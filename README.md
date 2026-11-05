# WebGPU Boids

A fixed-camera WebGPU port of the coherent-grid CUDA flocking simulation. It uses a GPU radix sort and prefix scan adapted from Project 2 in place of Thrust.

Run `npm install` and `npm run dev`. The upper-right control changes the boid count (1–100,000); the upper-left counter reports FPS. Vite listens on your LAN at port 5173.

Use left-drag to orbit, right-drag to zoom, or the mouse wheel to zoom.
