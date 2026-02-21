# Writing Kernels

A kernel is a function that runs in parallel on the GPU (or CPU workers). SpawnDev.ILGPU compiles your C# kernel code into GPU shader languages (WGSL, GLSL) or WebAssembly automatically.

## Kernel Basics

A kernel is always a **static void** method. The first parameter is an **index type** that identifies which thread is running. Think of it as the body of a massively parallel `for` loop:

```csharp
// This kernel runs once per element — each thread gets a unique index
static void MyKernel(Index1D index, ArrayView<float> data, float multiplier)
{
    data[index] = data[index] * multiplier;
}
```

When you launch this kernel with 1000 elements, 1000 threads execute simultaneously, each with a different `index` value from 0 to 999.

## Kernel Rules

These rules apply to all kernel code — they come from ILGPU's design and the constraints of GPU execution:

| Rule | Details |
|------|---------|
| Must be `static` | Instance methods are not supported |
| Must return `void` | Kernels don't return values — use output buffers |
| First parameter is the index | `Index1D`, `Index2D`, or `Index3D` |
| Value types only | No classes, no `string`, no reference types |
| No `throw` | The WGSL/GLSL transpiler does not support exception handling |
| No `ref` / `out` | Parameters are passed by value |
| No recursion | GPU hardware doesn't support call stacks |
| No dynamic allocation | No `new` inside kernels (except fixed-size structs) |

## Index Types

The index type determines the dimensionality of the kernel's execution grid:

### Index1D — Linear Processing

```csharp
static void Process1D(Index1D index, ArrayView<float> data, float value)
{
    data[index] = value;
}

// Launch: each element gets one thread
kernel((Index1D)data.Length, data.View, 42.0f);
```

### Index2D — Image/Matrix Processing

```csharp
static void Process2D(
    Index2D index,
    ArrayView2D<uint, Stride2D.DenseX> pixels,
    int width, int height)
{
    int x = index.X;
    int y = index.Y;
    if (x >= width || y >= height) return;

    // Process pixel at (x, y)
    uint r = (uint)(255 * x / width);
    uint g = (uint)(255 * y / height);
    pixels[index] = (0xFFu << 24) | (r << 16) | (g << 8) | 0xFF;
}

// Launch with 2D extent
kernel(buffer.IntExtent, buffer.View, width, height);
```

### Index3D — Volume/Voxel Processing

```csharp
static void Process3D(
    Index3D index,
    ArrayView<float> volume,
    int width, int height, int depth)
{
    int x = index.X, y = index.Y, z = index.Z;
    int i = x + y * width + z * width * height;
    volume[i] = x + y + z;
}
```

## Loading and Launching Kernels

### LoadAutoGroupedStreamKernel (Recommended)

The simplest way to load a kernel. ILGPU automatically determines the optimal workgroup size:

```csharp
// Load once (compile + cache)
var kernel = accelerator.LoadAutoGroupedStreamKernel<
    Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(VectorAddKernel);

// Launch (fire-and-forget — work is queued)
kernel((Index1D)length, bufA.View, bufB.View, bufC.View);

// Wait for completion
await accelerator.SynchronizeAsync();
```

### Kernel Delegate Caching

For render loops and repeated invocations, cache the kernel delegate:

```csharp
// Declare as a field
private Action<Index2D, ArrayView2D<uint, Stride2D.DenseX>, float, float>? _renderKernel;

// Load once
_renderKernel = accelerator.LoadAutoGroupedStreamKernel<
    Index2D, ArrayView2D<uint, Stride2D.DenseX>, float, float>(RenderKernel);

// Invoke repeatedly (no stream argument needed for auto-grouped)
_renderKernel(buffer.IntExtent, buffer.View, time, zoom);
```

> **Note:** The delegate type for `LoadAutoGroupedStreamKernel` does **not** include an `AcceleratorStream` parameter. The index type is the first argument when calling.

### Explicitly Grouped Kernels

For full control over workgroup size (required for shared memory and barriers):

```csharp
static void GroupedKernel(ArrayView<int> data, ArrayView<int> output)
{
    var globalIdx = Grid.GlobalIndex.X;
    var localIdx = Group.IdxX;
    var groupSize = Group.DimX;

    // Use shared memory
    var sharedMem = SharedMemory.Allocate<int>(64);
    sharedMem[localIdx] = data[globalIdx];

    Group.Barrier(); // Wait for all threads in group

    // Process with shared data...
    output[globalIdx] = sharedMem[(localIdx + 1) % groupSize];
}
```

## Parameter Types

### Scalar Parameters

Scalars (`int`, `float`, `double`, etc.) are passed by value:

```csharp
static void ScalarKernel(Index1D index, ArrayView<float> data, float multiplier, int offset)
{
    data[index] = data[index] * multiplier + offset;
}
```

### Struct Parameters

Custom structs work if they are **value types with fixed size**:

```csharp
public struct SimParams
{
    public float DeltaTime;
    public float Gravity;
    public int MaxIterations;
}

static void PhysicsKernel(Index1D index, ArrayView<float> positions, SimParams p)
{
    positions[index] += p.Gravity * p.DeltaTime;
}
```

### ArrayView Parameters

`ArrayView<T>` is the primary way to access GPU memory from kernels:

```csharp
static void CopyKernel(Index1D index, ArrayView<float> source, ArrayView<float> dest)
{
    dest[index] = source[index];
}
```

Multi-dimensional views:

```csharp
static void MatrixKernel(
    Index2D index,
    ArrayView2D<float, Stride2D.DenseX> matrix,
    ArrayView<float> result)
{
    int x = index.X, y = index.Y;
    result[y * matrix.IntExtent.X + x] = matrix[index] * 2.0f;
}
```

## Math Functions

### Supported Functions

ILGPU maps standard .NET math to GPU-native operations:

| C# | GPU Mapping | Notes |
|----|-------------|-------|
| `MathF.Sin(x)` | `sin(x)` | ✅ All backends |
| `MathF.Cos(x)` | `cos(x)` | ✅ All backends |
| `MathF.Tan(x)` | `tan(x)` | ✅ All backends |
| `MathF.Sqrt(x)` | `sqrt(x)` | ✅ All backends |
| `MathF.Pow(x, y)` | `pow(x, y)` | ✅ All backends |
| `MathF.Log(x)` | `log(x)` | ✅ All backends |
| `MathF.Exp(x)` | `exp(x)` | ✅ All backends |
| `MathF.Abs(x)` | `abs(x)` | ✅ All backends |
| `MathF.Floor(x)` | `floor(x)` | ✅ All backends |
| `MathF.Ceiling(x)` | `ceil(x)` | ✅ All backends |
| `Math.Min(a, b)` | `min(a, b)` | ✅ All backends |
| `Math.Max(a, b)` | `max(a, b)` | ✅ All backends |
| `MathF.FusedMultiplyAdd` | `fma(a, b, c)` | ✅ All backends |
| `MathF.Atan2(y, x)` | `atan2(y, x)` | ✅ All backends |

### Unsupported Functions (Contain `throw`)

These .NET methods contain internal `throw` statements and will fail during transpilation:

| C# | Workaround |
|----|------------|
| `Math.Clamp(val, min, max)` | `Math.Min(Math.Max(val, min), max)` |
| `Math.Round(x)` | Avoid — no direct replacement |
| `Math.Truncate(x)` | Avoid — no direct replacement |
| `Math.Sign(x)` | `x > 0 ? 1 : (x < 0 ? -1 : 0)` |

> **Rule of thumb:** If a .NET math method might validate its arguments and throw, it won't work in kernels. Stick to the functions in the "Supported" table above.

## Shared Memory

Shared memory allows threads within a workgroup to share data. It's much faster than global memory but limited in size.

> **Availability:** WebGPU and Wasm backends only. WebGL does not support shared memory.

### Static Shared Memory

```csharp
static void SharedMemKernel(ArrayView<int> data, ArrayView<int> output)
{
    // Allocate shared memory (compile-time size)
    var shared = SharedMemory.Allocate<int>(64);

    var localIdx = Group.IdxX;
    var globalIdx = Grid.GlobalIndex.X;

    // Load data into shared memory
    shared[localIdx] = data[globalIdx];

    // Wait for all threads
    Group.Barrier();

    // Read from neighbor in shared memory
    output[globalIdx] = shared[(localIdx + 1) % Group.DimX];
}
```

### Dynamic Shared Memory

Dynamic shared memory is sized at launch time:

```csharp
static void DynamicSharedKernel(ArrayView<int> data)
{
    var shared = SharedMemory.GetDynamic<int>();
    // Size is determined by the launch configuration
}

// Launch with dynamic shared memory config
var config = SharedMemoryConfig.RequestDynamic<int>(groupSize);
kernel((gridDim, groupDim, config), data.View);
```

## Control Flow

Standard C# control flow works in kernels:

```csharp
static void ControlFlowKernel(Index1D index, ArrayView<float> data, float threshold)
{
    float val = data[index];

    // If/else
    if (val > threshold)
        data[index] = threshold;
    else
        data[index] = val * 2.0f;

    // Loops
    float sum = 0;
    for (int i = 0; i < 10; i++)
        sum += val * i;

    data[index] = sum;
}
```

> **Performance tip:** Avoid divergent branches within a workgroup. When threads in the same workgroup take different paths, performance degrades because the GPU executes both paths sequentially.

## Common Patterns

### Stencil (Neighbor Access)

```csharp
static void Stencil1D(Index1D index, ArrayView<float> input, ArrayView<float> output)
{
    int i = index;
    int len = (int)input.Length;

    float left  = i > 0 ? input[i - 1] : 0;
    float center = input[i];
    float right = i < len - 1 ? input[i + 1] : 0;

    output[i] = (left + center + right) / 3.0f;
}
```

### Bounds Checking

Always guard against out-of-bounds access when the dispatch size may exceed the data size:

```csharp
static void SafeKernel(Index1D index, ArrayView<float> data, int actualLength)
{
    if (index >= actualLength) return;
    data[index] = data[index] * 2.0f;
}
```

### Packed Parameters

When you need many parameters, pack them into a struct or encode multiple values into fewer parameters:

```csharp
// Pack width and height into a single int
int packedSize = width * 65536 + height;

// In kernel:
int width = packedSize / 65536;
int height = packedSize - width * 65536;
```
