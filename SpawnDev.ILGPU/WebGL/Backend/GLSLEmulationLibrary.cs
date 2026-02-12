// ---------------------------------------------------------------------------------------
//                                 SpawnDev.ILGPU.WebGL
//                        Copyright (c) 2024 SpawnDev Project
//
// File: GLSLEmulationLibrary.cs
//
// Provides GLSL ES 3.0 helper functions for emu_f64 and emu_i64 emulation.
// emu_f64: Double-float technique using vec2 (high + low)
// emu_i64: Double-word technique using uvec2 (low + high)
//
// Ported from WGSLEmulationLibrary.cs — same algorithms, GLSL syntax.
// ---------------------------------------------------------------------------------------

namespace SpawnDev.ILGPU.WebGL.Backend
{
    /// <summary>
    /// Provides GLSL ES 3.0 code strings for 64-bit type emulation functions.
    /// These functions are prepended to the vertex shader when emulation is used.
    /// Ported from <see cref="WebGPU.Backend.WGSLEmulationLibrary"/>.
    /// </summary>
    public static class GLSLEmulationLibrary
    {
        #region emu_f64 Emulation (Double-Float using vec2)

        /// <summary>
        /// GLSL helper functions for emu_f64 emulation.
        /// Uses the double-float technique where emu_f64 = vec2(high, low).
        /// </summary>
        public const string F64Functions = @"
// ============================================================================
// emu_f64 Emulation Functions (Double-Float: vec2 where x=high, y=low)
// ============================================================================

// --- IEEE 754 double bits to double-float conversion ---
vec2 f64_from_ieee754_bits(uint lo, uint hi) {
    uint sign_bit = (hi >> 31u) & 1u;
    uint exponent = (hi >> 20u) & 0x7FFu;
    uint mantissa_hi = hi & 0xFFFFFu;
    uint mantissa_lo = lo;

    if (exponent == 0u && mantissa_hi == 0u && mantissa_lo == 0u) {
        return vec2(0.0, 0.0);
    }
    if (exponent == 0x7FFu) {
        return vec2(0.0, 0.0);
    }

    int exp_bias = 1023;
    int exp_val = int(exponent) - exp_bias;

    int f32_exp_bias = 127;
    int f32_exp = exp_val + f32_exp_bias;

    if (f32_exp <= 0 || f32_exp >= 255) {
        float val_approx = float(hi) * 0.00000000023283064;
        return vec2(val_approx, 0.0);
    }

    uint f32_bits = (sign_bit << 31u) | (uint(f32_exp) << 23u) | (mantissa_hi << 3u);
    float val = intBitsToFloat(int(f32_bits));

    return vec2(val, 0.0);
}

// Store emu_f64 back to IEEE 754 bits
uvec2 f64_to_ieee754_bits(vec2 v) {
    float val = v.x + v.y;
    if (val == 0.0) {
        return uvec2(0u, 0u);
    }

    uint f32_bits = uint(floatBitsToInt(val));
    uint sign = (f32_bits >> 31u) & 1u;
    uint f32_exp = (f32_bits >> 23u) & 0xFFu;
    uint f32_mantissa = f32_bits & 0x7FFFFFu;

    int f32_bias = 127;
    int f64_bias = 1023;
    int exp_val = int(f32_exp) - f32_bias;
    uint f64_exp = uint(exp_val + f64_bias);

    uint mantissa_hi = f32_mantissa >> 3u;
    uint mantissa_lo = (f32_mantissa & 0x7u) << 29u;

    uint hi = (sign << 31u) | (f64_exp << 20u) | mantissa_hi;
    uint lo = mantissa_lo;

    return uvec2(lo, hi);
}

// Create emu_f64 from float
vec2 f64_from_f32(float v) {
    return vec2(v, 0.0);
}

// Convert emu_f64 to float (loses precision)
float f64_to_f32(vec2 v) {
    return v.x + v.y;
}

// Create emu_f64 from high and low
vec2 f64_new(float hi, float lo) {
    return vec2(hi, lo);
}

// emu_f64 negation
vec2 f64_neg(vec2 a) {
    return vec2(-a.x, -a.y);
}

// emu_f64 addition using Dekker's algorithm
vec2 f64_add(vec2 a, vec2 b) {
    float s = a.x + b.x;
    float v = s - a.x;
    float e = (a.x - (s - v)) + (b.x - v) + a.y + b.y;
    float z_hi = s + e;
    float z_lo = e - (z_hi - s);
    return vec2(z_hi, z_lo);
}

// emu_f64 subtraction
vec2 f64_sub(vec2 a, vec2 b) {
    return f64_add(a, f64_neg(b));
}

// Helper: split float for multiplication
vec2 f64_split(float a) {
    float c = 4097.0 * a;
    float a_hi = c - (c - a);
    float a_lo = a - a_hi;
    return vec2(a_hi, a_lo);
}

// Helper: two-product algorithm
vec2 f64_two_prod(float a, float b) {
    float p = a * b;
    vec2 a_s = f64_split(a);
    vec2 b_s = f64_split(b);
    float e = ((a_s.x * b_s.x - p) + a_s.x * b_s.y + a_s.y * b_s.x) + a_s.y * b_s.y;
    return vec2(p, e);
}

// emu_f64 multiplication
vec2 f64_mul(vec2 a, vec2 b) {
    vec2 p = f64_two_prod(a.x, b.x);
    float e = a.x * b.y + a.y * b.x + p.y;
    float z_hi = p.x + e;
    float z_lo = e - (z_hi - p.x);
    return vec2(z_hi, z_lo);
}

// emu_f64 division
vec2 f64_div(vec2 a, vec2 b) {
    float q = a.x / b.x;
    vec2 r = f64_sub(a, f64_mul(b, f64_from_f32(q)));
    float q2 = r.x / b.x;
    return f64_add(f64_from_f32(q), f64_from_f32(q2));
}

// emu_f64 comparisons
bool f64_lt(vec2 a, vec2 b) {
    return (a.x < b.x) || (a.x == b.x && a.y < b.y);
}

bool f64_le(vec2 a, vec2 b) {
    return (a.x < b.x) || (a.x == b.x && a.y <= b.y);
}

bool f64_gt(vec2 a, vec2 b) {
    return (a.x > b.x) || (a.x == b.x && a.y > b.y);
}

bool f64_ge(vec2 a, vec2 b) {
    return (a.x > b.x) || (a.x == b.x && a.y >= b.y);
}

bool f64_eq(vec2 a, vec2 b) {
    return a.x == b.x && a.y == b.y;
}

bool f64_ne(vec2 a, vec2 b) {
    return a.x != b.x || a.y != b.y;
}

// emu_f64 absolute value
vec2 f64_abs(vec2 a) {
    if (a.x < 0.0 || (a.x == 0.0 && a.y < 0.0)) {
        return f64_neg(a);
    }
    return a;
}

// emu_f64 min/max
vec2 f64_min(vec2 a, vec2 b) {
    if (f64_lt(a, b)) { return a; }
    return b;
}

vec2 f64_max(vec2 a, vec2 b) {
    if (f64_gt(a, b)) { return a; }
    return b;
}
";

        #endregion

        #region emu_i64 Emulation (Double-Word using uvec2)

        /// <summary>
        /// GLSL helper functions for emu_i64/emu_u64 emulation.
        /// Uses double-word technique where emu_i64 = uvec2(low, high).
        /// </summary>
        public const string I64Functions = @"
// ============================================================================
// emu_i64/emu_u64 Emulation Functions (Double-Word: uvec2 where x=low, y=high)
// ============================================================================

// Create emu_i64 from int (sign-extend)
uvec2 i64_from_i32(int v) {
    uint lo = uint(v);
    uint hi = v < 0 ? 0xFFFFFFFFu : 0u;
    return uvec2(lo, hi);
}

// Create emu_u64 from uint
uvec2 u64_from_u32(uint v) {
    return uvec2(v, 0u);
}

// Convert emu_i64 to int (truncate)
int i64_to_i32(uvec2 v) {
    return int(v.x);
}

// Convert emu_u64 to uint (truncate)
uint u64_to_u32(uvec2 v) {
    return v.x;
}

// Create emu_i64 from low and high
uvec2 i64_new(uint lo, uint hi) {
    return uvec2(lo, hi);
}

// emu_i64/emu_u64 addition with carry
uvec2 i64_add(uvec2 a, uvec2 b) {
    uint lo = a.x + b.x;
    uint carry = lo < a.x ? 1u : 0u;
    uint hi = a.y + b.y + carry;
    return uvec2(lo, hi);
}

// emu_i64/emu_u64 subtraction with borrow
uvec2 i64_sub(uvec2 a, uvec2 b) {
    uint borrow = a.x < b.x ? 1u : 0u;
    uint lo = a.x - b.x;
    uint hi = a.y - b.y - borrow;
    return uvec2(lo, hi);
}

// emu_i64 negation (two's complement)
uvec2 i64_neg(uvec2 a) {
    uvec2 inv = uvec2(~a.x, ~a.y);
    return i64_add(inv, uvec2(1u, 0u));
}

// emu_u64 multiplication
uvec2 u64_mul(uvec2 a, uvec2 b) {
    uint a_lo = a.x & 0xFFFFu;
    uint a_hi = a.x >> 16u;
    uint b_lo = b.x & 0xFFFFu;
    uint b_hi = b.x >> 16u;

    uint p0 = a_lo * b_lo;
    uint p1 = a_lo * b_hi;
    uint p2 = a_hi * b_lo;
    uint p3 = a_hi * b_hi;

    uint mid = (p0 >> 16u) + (p1 & 0xFFFFu) + (p2 & 0xFFFFu);
    uint lo = (p0 & 0xFFFFu) | ((mid & 0xFFFFu) << 16u);

    uint hi = p3 + (p1 >> 16u) + (p2 >> 16u) + (mid >> 16u) + a.x * b.y + a.y * b.x;

    return uvec2(lo, hi);
}

// emu_i64 multiplication
uvec2 i64_mul(uvec2 a, uvec2 b) {
    bool neg_a = (a.y & 0x80000000u) != 0u;
    bool neg_b = (b.y & 0x80000000u) != 0u;
    uvec2 abs_a = neg_a ? i64_neg(a) : a;
    uvec2 abs_b = neg_b ? i64_neg(b) : b;
    uvec2 result = u64_mul(abs_a, abs_b);
    if (neg_a != neg_b) { result = i64_neg(result); }
    return result;
}

// Bitwise operations
uvec2 i64_and(uvec2 a, uvec2 b) {
    return uvec2(a.x & b.x, a.y & b.y);
}

uvec2 i64_or(uvec2 a, uvec2 b) {
    return uvec2(a.x | b.x, a.y | b.y);
}

uvec2 i64_xor(uvec2 a, uvec2 b) {
    return uvec2(a.x ^ b.x, a.y ^ b.y);
}

uvec2 i64_not(uvec2 a) {
    return uvec2(~a.x, ~a.y);
}

// Left shift
uvec2 i64_shl(uvec2 a, uint shift) {
    if (shift == 0u) { return a; }
    if (shift >= 64u) { return uvec2(0u, 0u); }
    if (shift >= 32u) {
        return uvec2(0u, a.x << (shift - 32u));
    }
    uint lo = a.x << shift;
    uint hi = (a.y << shift) | (a.x >> (32u - shift));
    return uvec2(lo, hi);
}

// Logical right shift
uvec2 u64_shr(uvec2 a, uint shift) {
    if (shift == 0u) { return a; }
    if (shift >= 64u) { return uvec2(0u, 0u); }
    if (shift >= 32u) {
        return uvec2(a.y >> (shift - 32u), 0u);
    }
    uint lo = (a.x >> shift) | (a.y << (32u - shift));
    uint hi = a.y >> shift;
    return uvec2(lo, hi);
}

// Arithmetic right shift
uvec2 i64_shr(uvec2 a, uint shift) {
    if (shift == 0u) { return a; }
    uint sign = a.y & 0x80000000u;
    if (shift >= 64u) {
        uint fill = sign != 0u ? 0xFFFFFFFFu : 0u;
        return uvec2(fill, fill);
    }
    if (shift >= 32u) {
        int signed_y = int(a.y);
        uint shift_amt = shift - 32u;
        int shifted = signed_y >> int(shift_amt);
        uint lo = uint(shifted);
        uint hi = sign != 0u ? 0xFFFFFFFFu : 0u;
        return uvec2(lo, hi);
    }
    uint lo_r = (a.x >> shift) | (a.y << (32u - shift));
    int signed_y_r = int(a.y);
    int shifted_r = signed_y_r >> int(shift);
    uint hi_r = uint(shifted_r);
    return uvec2(lo_r, hi_r);
}

// Signed comparisons
bool i64_eq(uvec2 a, uvec2 b) {
    return a.x == b.x && a.y == b.y;
}

bool i64_ne(uvec2 a, uvec2 b) {
    return a.x != b.x || a.y != b.y;
}

bool i64_lt(uvec2 a, uvec2 b) {
    bool a_neg = (a.y & 0x80000000u) != 0u;
    bool b_neg = (b.y & 0x80000000u) != 0u;
    if (a_neg && !b_neg) { return true; }
    if (!a_neg && b_neg) { return false; }
    if (a.y != b.y) { return a.y < b.y; }
    return a.x < b.x;
}

bool i64_le(uvec2 a, uvec2 b) {
    return i64_lt(a, b) || i64_eq(a, b);
}

bool i64_gt(uvec2 a, uvec2 b) {
    return i64_lt(b, a);
}

bool i64_ge(uvec2 a, uvec2 b) {
    return !i64_lt(a, b);
}

// Unsigned comparisons
bool u64_lt(uvec2 a, uvec2 b) {
    if (a.y != b.y) { return a.y < b.y; }
    return a.x < b.x;
}

bool u64_le(uvec2 a, uvec2 b) {
    return u64_lt(a, b) || i64_eq(a, b);
}

bool u64_gt(uvec2 a, uvec2 b) {
    return u64_lt(b, a);
}

bool u64_ge(uvec2 a, uvec2 b) {
    return !u64_lt(a, b);
}

// emu_i64 absolute value
uvec2 i64_abs(uvec2 a) {
    if ((a.y & 0x80000000u) != 0u) {
        return i64_neg(a);
    }
    return a;
}
";

        #endregion

        #region Combined Library

        /// <summary>
        /// Gets the full emulation library based on which features are enabled.
        /// </summary>
        public static string GetEmulationLibrary(bool includeF64, bool includeI64)
        {
            var sb = new System.Text.StringBuilder();

            if (includeF64)
            {
                sb.AppendLine(F64Functions);
            }

            if (includeI64)
            {
                sb.AppendLine(I64Functions);
            }

            return sb.ToString();
        }

        #endregion
    }
}
