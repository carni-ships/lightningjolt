#include <metal_stdlib>
using namespace metal;

// BN254 scalar field (Fr) arithmetic in Metal compute shaders.
//
// Each field element is 256 bits stored as 8 × uint32_t (little-endian).
// We use a wide-multiply-then-reduce approach for correctness:
// 1. Compute the full 512-bit product a × b using schoolbook multiplication
// 2. Apply Montgomery reduction to get the result mod p
//
// This is simpler and more debuggable than interleaved CIOS on 32-bit ops.

constant uint32_t MOD[8] = {
    0xf0000001u, 0x43e1f593u,  // limb 0: 0x43e1f593f0000001
    0x79b97091u, 0x2833e848u,  // limb 1: 0x2833e84879b97091
    0x8181585du, 0xb85045b6u,  // limb 2: 0xb85045b68181585d
    0xe131a029u, 0x30644e72u,  // limb 3: 0x30644e72e131a029
};

// INV = -p^{-1} mod 2^32 (we only need the low 32 bits for Montgomery)
// Since full INV mod 2^64 = 0xc2e1f593efffffff,
// INV mod 2^32 = 0xefffffff
constant uint32_t INV32 = 0xefffffffu;

struct Fp256 {
    uint32_t limbs[8];
};

// ---- Wide arithmetic (schoolbook 256×256 → 512 bit) ----

// Add b to a (16-word accumulator), starting at position `offset`
inline void wide_add_at(thread uint32_t *a, uint32_t b, int offset) {
    uint64_t carry = uint64_t(a[offset]) + uint64_t(b);
    a[offset] = uint32_t(carry);
    for (int i = offset + 1; carry >> 32 && i < 16; i++) {
        carry = uint64_t(a[i]) + (carry >> 32);
        a[i] = uint32_t(carry);
    }
}

// Multiply a (8 words) × b (8 words) → product (16 words)
inline void wide_mul(const thread uint32_t *a, const thread uint32_t *b, thread uint32_t *product) {
    for (int i = 0; i < 16; i++) product[i] = 0;

    for (int i = 0; i < 8; i++) {
        uint32_t carry = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t uv = uint64_t(a[i]) * uint64_t(b[j])
                        + uint64_t(product[i + j])
                        + uint64_t(carry);
            product[i + j] = uint32_t(uv);
            carry = uint32_t(uv >> 32);
        }
        product[i + 8] += carry;
        // Propagate any further carry
        for (int k = i + 9; product[i + 8] < carry && k < 16; k++) {
            // This handles rare overflow: if product[i+8] wrapped
            // Actually the += carry above can't wrap product[i+8] past 0
            // unless there was already a max value, so we need proper carry:
            break; // carry from adding `carry` to product[i+8] is at most 1
        }
        // Proper carry propagation for product[i+8] += carry
        if (product[i + 8] < carry) {
            for (int k = i + 9; k < 16; k++) {
                product[k] += 1;
                if (product[k] != 0) break;
            }
        }
    }
}

// Montgomery reduction: given T (16 words = 512 bits), compute T * R^{-1} mod p
// where R = 2^256.
//
// For each of the 8 bottom 32-bit words:
//   m = T[i] * INV32 mod 2^32
//   T += m * MOD << (32*i)
// Then result = T >> 256 (top 8 words), conditionally subtract MOD if >= MOD.
inline Fp256 mont_reduce(thread uint32_t *t) {
    for (int i = 0; i < 8; i++) {
        uint32_t m = t[i] * INV32;
        uint32_t carry = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t uv = uint64_t(m) * uint64_t(MOD[j])
                        + uint64_t(t[i + j])
                        + uint64_t(carry);
            t[i + j] = uint32_t(uv);
            carry = uint32_t(uv >> 32);
        }
        // Propagate carry
        for (int k = i + 8; k < 16; k++) {
            uint64_t s = uint64_t(t[k]) + uint64_t(carry);
            t[k] = uint32_t(s);
            carry = uint32_t(s >> 32);
            if (carry == 0) break;
        }
    }

    // Result is top 8 words: t[8..15]
    // Conditional subtraction: if result >= MOD, subtract MOD
    uint32_t result[8];
    for (int i = 0; i < 8; i++) result[i] = t[i + 8];

    // Check if result >= MOD by trial subtraction
    uint32_t sub[8];
    int borrow = 0;
    for (int i = 0; i < 8; i++) {
        int64_t diff = int64_t(result[i]) - int64_t(MOD[i]) - int64_t(borrow);
        sub[i] = uint32_t(diff);
        borrow = (diff < 0) ? 1 : 0;
    }

    Fp256 out;
    for (int i = 0; i < 8; i++) {
        out.limbs[i] = (borrow != 0) ? result[i] : sub[i];
    }
    return out;
}

// Montgomery multiplication: a * b * R^{-1} mod p
inline Fp256 mont_mul(Fp256 a, Fp256 b) {
    uint32_t product[16];
    wide_mul(a.limbs, b.limbs, product);
    return mont_reduce(product);
}

// Field addition mod p
inline Fp256 fp_add(Fp256 a, Fp256 b) {
    uint32_t carry = 0;
    uint32_t r[8];
    for (int i = 0; i < 8; i++) {
        uint64_t sum = uint64_t(a.limbs[i]) + uint64_t(b.limbs[i]) + uint64_t(carry);
        r[i] = uint32_t(sum);
        carry = uint32_t(sum >> 32);
    }

    // Conditional subtraction
    uint32_t sub[8];
    int borrow = 0;
    for (int i = 0; i < 8; i++) {
        int64_t diff = int64_t(r[i]) - int64_t(MOD[i]) - int64_t(borrow);
        sub[i] = uint32_t(diff);
        borrow = (diff < 0) ? 1 : 0;
    }

    Fp256 result;
    for (int i = 0; i < 8; i++) {
        result.limbs[i] = (borrow != 0) ? r[i] : sub[i];
    }
    return result;
}

// ===== Compute Kernels =====

kernel void batch_field_mul(
    device const Fp256 *a [[buffer(0)]],
    device const Fp256 *b [[buffer(1)]],
    device Fp256 *out      [[buffer(2)]],
    uint tid               [[thread_position_in_grid]]
) {
    out[tid] = mont_mul(a[tid], b[tid]);
}

kernel void batch_field_add(
    device const Fp256 *a [[buffer(0)]],
    device const Fp256 *b [[buffer(1)]],
    device Fp256 *out      [[buffer(2)]],
    uint tid               [[thread_position_in_grid]]
) {
    out[tid] = fp_add(a[tid], b[tid]);
}

kernel void scalar_vec_mul(
    device const Fp256 *scalar [[buffer(0)]],
    device const Fp256 *vec    [[buffer(1)]],
    device Fp256 *out           [[buffer(2)]],
    uint tid                    [[thread_position_in_grid]]
) {
    out[tid] = mont_mul(scalar[0], vec[tid]);
}

kernel void parallel_field_sum(
    device const Fp256 *input        [[buffer(0)]],
    device Fp256 *partial_sums       [[buffer(1)]],
    device const uint *count         [[buffer(2)]],
    uint tid                         [[thread_position_in_grid]],
    uint lid                         [[thread_position_in_threadgroup]],
    uint gid                         [[threadgroup_position_in_grid]],
    uint group_size                  [[threads_per_threadgroup]]
) {
    threadgroup Fp256 shared_data[256];

    uint n = count[0];
    Fp256 val;
    if (tid < n) {
        val = input[tid];
    } else {
        for (int i = 0; i < 8; i++) val.limbs[i] = 0;
    }
    shared_data[lid] = val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_data[lid] = fp_add(shared_data[lid], shared_data[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        partial_sums[gid] = shared_data[0];
    }
}
