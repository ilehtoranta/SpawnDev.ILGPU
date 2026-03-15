# Forked ILGPU.Algorithms

Modified fork of ILGPU.Algorithms. Same rules as `ILGPU/` — check for `.tt` templates before editing `.cs` files.

## Key Files for Browser Backends
- `ScanExtensions.cs` — `AcceleratorType.Wasm` routes to `CreateSingleGroupScan` (not multi-pass). `ComputeScanTempStorageSize` returns 1 for Wasm.
- `RadixSortExtensions.cs` — `KernelSpecialization` for WebGPU/Wasm. RadixSortKernel1 (line ~673), RadixSortKernel2 (line ~871).
- Algorithm kernel loaders (Histogram, Permutation, Unique, Grid, Optimizer, etc.) — all have WebGPU `KernelSpecialization`.

## Cross-Backend Impact
Same as `ILGPU/` — changes affect all 6 backends.
