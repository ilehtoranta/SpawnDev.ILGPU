# SpawnDev.ILGPU Documentation

Comprehensive documentation for SpawnDev.ILGPU — run ILGPU C# kernels on WebGPU, WebGL, and Wasm in Blazor WebAssembly.

> **Your existing ILGPU kernels run in the browser with zero changes to the kernel code.**

## Table of Contents

### Getting Started
- **[Getting Started](getting-started.md)** — Installation, Program.cs setup, and your first GPU kernel in the browser

### Core Concepts
- **[Backends](backends.md)** — WebGPU, WebGL, Wasm, and CPU backends: setup, capabilities, configuration, and auto-selection
- **[Writing Kernels](kernels.md)** — Kernel fundamentals, index types, loading, launching, math functions, and kernel rules
- **[Memory & Buffers](memory-and-buffers.md)** — Allocation, data transfer, async readback, and buffer lifecycle

### Advanced
- **[Advanced Patterns](advanced-patterns.md)** — GPU device sharing, external buffers, rendering pipelines, canvas blitting, and real-time render loops
- **[Limitations & Constraints](limitations.md)** — Blazor WASM restrictions, unsupported features, precision, and browser compatibility

### Reference
- **[API Reference](api-reference.md)** — Public classes, methods, and extension methods organized by namespace

## Quick Links

| Resource | Link |
|----------|------|
| **NuGet Package** | [SpawnDev.ILGPU](https://www.nuget.org/packages/SpawnDev.ILGPU) |
| **GitHub Repository** | [LostBeard/SpawnDev.ILGPU](https://github.com/LostBeard/SpawnDev.ILGPU) |
| **Live Demo** | [lostbeard.github.io/SpawnDev.ILGPU](https://lostbeard.github.io/SpawnDev.ILGPU/) |
| **ILGPU Project** | [m4rs-mt/ILGPU](https://github.com/m4rs-mt/ILGPU) |
| **SpawnDev.BlazorJS** | [LostBeard/SpawnDev.BlazorJS](https://github.com/LostBeard/SpawnDev.BlazorJS) |
