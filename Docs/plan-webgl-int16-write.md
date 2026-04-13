# Plan: WebGL Int16 Write via Packed 32-bit Output

**Date:** 2026-04-12
**Status:** Captain approved approach. Planning implementation.
**Approach:** Pack two 16-bit values into one 32-bit Transform Feedback output

---

## Problem

WebGL Transform Feedback outputs 32-bit values only (int/float via GL_TRANSFORM_FEEDBACK_BUFFER). There's no 16-bit output format. Storing a short to `output[idx]` writes 32 bits, wasting space and misaligning subsequent elements.

## Solution

Pack two shorts into one i32. The kernel processes pairs of elements and outputs packed 32-bit values. The readback unpacks them.

## Design

### Buffer layout

A `MemoryBuffer1D<short>(N)` on WebGL:
- GPU buffer: `int[ceil(N/2)]` (half the elements, each holding 2 packed shorts)
- Element 0 of GPU buffer: `(short[1] << 16) | (short[0] & 0xFFFF)`
- Element 1 of GPU buffer: `(short[3] << 16) | (short[2] & 0xFFFF)`
- If N is odd, the last element has only the low 16 bits populated

### Kernel codegen (GLSL Store handler)

When the Store target is a sub-word ArrayView:

**Option A: Modify dispatch to N/2 threads, each writes a pair**

The codegen wraps the user's Store into a packed write:
```glsl
// User wrote: dst[idx] = value;
// Codegen transforms to:
// Thread processes idx = tid * 2
int packed = 0;

// First element (low 16 bits)
int val0 = <user's value expression for idx=tid*2>;
packed = val0 & 0xFFFF;

// Second element (high 16 bits)  
int val1 = <user's value expression for idx=tid*2+1>;
packed = packed | ((val1 & 0xFFFF) << 16);

// Write packed pair to Transform Feedback output
output[tid] = packed;
```

**Challenge:** The codegen needs to evaluate the user's kernel body TWICE per thread (once for each element in the pair). This is complex for arbitrary kernel bodies.

**Option B: Two-pass approach**

Pass 1: Write all even-indexed shorts to low 16 bits of output[idx/2]
Pass 2: Write all odd-indexed shorts to high 16 bits of output[idx/2]

Each pass outputs N/2 values. The second pass does a read-modify-write on the output buffer.

**Challenge:** Requires two dispatches and read-modify-write on the output buffer between passes. WebGL doesn't support reading from a Transform Feedback buffer in the same draw call.

**Option C: Store to a temporary int buffer, pack in a separate kernel**

The user's kernel writes to a full `int[N]` buffer (one int per short, wasting the high 16 bits). A second "packing" kernel reads pairs of ints and packs them into `int[N/2]`. The packed buffer is the actual MemoryBuffer backing.

```glsl
// User's kernel (unchanged):
temp_output[idx] = value;  // writes 32-bit int, only low 16 bits meaningful

// Packing kernel (auto-generated):
packed_output[idx] = (temp_output[idx*2] & 0xFFFF) | ((temp_output[idx*2+1] & 0xFFFF) << 16);
```

**Advantage:** User's kernel is unchanged. Packing is a separate, simple kernel.
**Cost:** Extra temporary buffer + extra dispatch. But the packing kernel is trivial.

### Recommended: Option C (simplest, most correct)

Option C keeps the user's kernel unchanged - no complex codegen transformations. The packing is a standard, testable kernel. The cost is one extra dispatch per sub-word Store operation, which is acceptable since WebGL is already the slowest backend.

### CopyToHostAsync readback

When reading back a `MemoryBuffer1D<short>` on WebGL:
1. MapAsync the packed `int[N/2]` buffer
2. Unpack: for each int, extract low 16 bits as short[i*2], high 16 bits as short[i*2+1]
3. Return `short[N]`

This unpacking happens in CopyToHostAsync, transparent to the user.

### Files to change

1. **WebGL/WebGLAccelerator.cs** - Buffer allocation for sub-word types (allocate temp + packed buffers)
2. **WebGL/Backend/GLSLKernelFunctionGenerator.cs** - Store handler redirects sub-word writes to temp buffer
3. **WebGL/WebGLMemoryBuffer.cs** - CopyToHostAsync unpacking for sub-word buffers
4. **Auto-generated packing kernel** - simple kernel that packs pairs of ints into packed ints

### Test verification

The existing Int16_BufferWrite_Test and Int16_EndToEnd_ReadWrite_Test should pass without modification once the packing is implemented. The tests don't care about the internal packing - they write shorts and read shorts back.

---

## Edge cases

- **Odd element count:** Last packed int only has valid data in low 16 bits. High 16 bits are garbage. Readback ignores the extra short.
- **SubView offsets:** If the user creates a SubView into a short buffer, the sub-view offset needs to account for the packed layout.
- **Concurrent read+write:** The temp buffer approach avoids this - writes go to temp, reads come from packed.

## Performance impact

- One extra kernel dispatch per sub-word Store (the packing kernel)
- One extra buffer allocation (temp int buffer, same size as element count)
- Packing kernel is trivially parallel: N/2 threads, each reads 2 ints and writes 1 packed int
- On WebGL (already the slowest backend), this overhead is negligible
