#include <metal_stdlib>
using namespace metal;

// BN254 G1 MSM (Multi-Scalar Multiplication) on Metal GPU.
//
// BN254 G1 curve: y^2 = x^3 + 3 over Fq where
//   q = 21888242871839275222246405745257275088696311157297823662689037894645226208583
//
// Points are in affine coordinates (x, y) with a flag for point-at-infinity.
// Intermediate computation uses projective coordinates (X, Y, Z) where
// the affine point is (X/Z^2, Y/Z^3).
//
// Scalars are BN254 Fr (same 256-bit field as in bn254_field.metal).
//
// Strategy: Pippenger's bucket method
// 1. Decompose each scalar into c-bit windows
// 2. For each window, accumulate points into 2^c buckets (GPU kernel)
// 3. Reduce buckets within each window (GPU kernel)
// 4. Combine windows on CPU (sequential, small)

// ---- Base field Fq (NOT Fr! Different modulus) ----

// Fq modulus: 21888242871839275222246405745257275088696311157297823662689037894645226208583
// = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
constant uint32_t FQ_MOD[8] = {
    0xd87cfd47u, 0x3c208c16u,
    0x6871ca8du, 0x97816a91u,
    0x8181585du, 0xb85045b6u,
    0xe131a029u, 0x30644e72u,
};

// -Fq^{-1} mod 2^32
// Fq mod 2^32 = 0xd87cfd47
// We need inv32 such that Fq * inv32 ≡ -1 (mod 2^32)
// inv32 = 0xe4866389 (precomputed; verified: 0xd87cfd47 * 0xe4866389 ≡ 0xffffffff mod 2^32)
constant uint32_t FQ_INV32 = 0xe4866389u;

struct Fq256 {
    uint32_t limbs[8];
};

// ---- Fq field arithmetic (same structure as Fr, different modulus) ----

inline void fq_wide_mul(const thread uint32_t *a, const thread uint32_t *b, thread uint32_t *product) {
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
        uint32_t prev = product[i + 8];
        product[i + 8] = prev + carry;
        if (product[i + 8] < carry) {
            for (int k = i + 9; k < 16; k++) {
                product[k] += 1;
                if (product[k] != 0) break;
            }
        }
    }
}

inline Fq256 fq_mont_reduce(thread uint32_t *t) {
    for (int i = 0; i < 8; i++) {
        uint32_t m = t[i] * FQ_INV32;
        uint32_t carry = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t uv = uint64_t(m) * uint64_t(FQ_MOD[j])
                        + uint64_t(t[i + j])
                        + uint64_t(carry);
            t[i + j] = uint32_t(uv);
            carry = uint32_t(uv >> 32);
        }
        for (int k = i + 8; k < 16; k++) {
            uint64_t s = uint64_t(t[k]) + uint64_t(carry);
            t[k] = uint32_t(s);
            carry = uint32_t(s >> 32);
            if (carry == 0) break;
        }
    }

    uint32_t result[8];
    for (int i = 0; i < 8; i++) result[i] = t[i + 8];

    uint32_t sub[8];
    int borrow = 0;
    for (int i = 0; i < 8; i++) {
        int64_t diff = int64_t(result[i]) - int64_t(FQ_MOD[i]) - int64_t(borrow);
        sub[i] = uint32_t(diff);
        borrow = (diff < 0) ? 1 : 0;
    }

    Fq256 out;
    for (int i = 0; i < 8; i++) {
        out.limbs[i] = (borrow != 0) ? result[i] : sub[i];
    }
    return out;
}

inline Fq256 fq_mul(Fq256 a, Fq256 b) {
    uint32_t product[16];
    fq_wide_mul(a.limbs, b.limbs, product);
    return fq_mont_reduce(product);
}

inline Fq256 fq_add(Fq256 a, Fq256 b) {
    uint32_t carry = 0;
    uint32_t r[8];
    for (int i = 0; i < 8; i++) {
        uint64_t sum = uint64_t(a.limbs[i]) + uint64_t(b.limbs[i]) + uint64_t(carry);
        r[i] = uint32_t(sum);
        carry = uint32_t(sum >> 32);
    }
    uint32_t sub[8];
    int borrow = 0;
    for (int i = 0; i < 8; i++) {
        int64_t diff = int64_t(r[i]) - int64_t(FQ_MOD[i]) - int64_t(borrow);
        sub[i] = uint32_t(diff);
        borrow = (diff < 0) ? 1 : 0;
    }
    Fq256 result;
    for (int i = 0; i < 8; i++) {
        result.limbs[i] = (borrow != 0) ? r[i] : sub[i];
    }
    return result;
}

inline Fq256 fq_sub(Fq256 a, Fq256 b) {
    int borrow = 0;
    uint32_t r[8];
    for (int i = 0; i < 8; i++) {
        int64_t diff = int64_t(a.limbs[i]) - int64_t(b.limbs[i]) - int64_t(borrow);
        r[i] = uint32_t(diff);
        borrow = (diff < 0) ? 1 : 0;
    }
    // If underflow, add modulus
    if (borrow != 0) {
        uint32_t carry = 0;
        for (int i = 0; i < 8; i++) {
            uint64_t sum = uint64_t(r[i]) + uint64_t(FQ_MOD[i]) + uint64_t(carry);
            r[i] = uint32_t(sum);
            carry = uint32_t(sum >> 32);
        }
    }
    Fq256 result;
    for (int i = 0; i < 8; i++) result.limbs[i] = r[i];
    return result;
}

inline Fq256 fq_double(Fq256 a) {
    return fq_add(a, a);
}

inline bool fq_is_zero(Fq256 a) {
    for (int i = 0; i < 8; i++) {
        if (a.limbs[i] != 0) return false;
    }
    return true;
}

// ---- EC Point types ----

// Affine point (x, y) with infinity flag
struct G1Affine {
    Fq256 x;
    Fq256 y;
    uint32_t is_infinity;  // 1 if point at infinity, 0 otherwise
    uint32_t _pad[3];      // Alignment padding
};

// Projective point (X, Y, Z) where affine = (X/Z^2, Y/Z^3)
struct G1Projective {
    Fq256 x;
    Fq256 y;
    Fq256 z;
};

inline G1Projective g1_identity() {
    G1Projective p;
    for (int i = 0; i < 8; i++) {
        p.x.limbs[i] = 0;
        p.y.limbs[i] = 0;
        p.z.limbs[i] = 0;
    }
    // Projective identity: (0, 1, 0) — but we'll use Z==0 as check
    return p;
}

inline bool g1_is_identity(G1Projective p) {
    return fq_is_zero(p.z);
}

// Convert affine to projective
inline G1Projective g1_from_affine(G1Affine a) {
    G1Projective p;
    if (a.is_infinity) {
        return g1_identity();
    }
    p.x = a.x;
    p.y = a.y;
    // Z = 1 in Montgomery form
    // Montgomery form of 1: R mod Fq where R = 2^256
    // R mod Fq = 0x0e0a77c19a07df2f666ea36f7879462c0a78eb28f5c70b3dd35d438dc58f0d9d
    p.z.limbs[0] = 0xc58f0d9du;
    p.z.limbs[1] = 0xd35d438du;
    p.z.limbs[2] = 0xf5c70b3du;
    p.z.limbs[3] = 0x0a78eb28u;
    p.z.limbs[4] = 0x7879462cu;
    p.z.limbs[5] = 0x666ea36fu;
    p.z.limbs[6] = 0x9a07df2fu;
    p.z.limbs[7] = 0x0e0a77c1u;
    return p;
}

// ---- EC point addition (projective + affine → projective) ----
// "mixed addition" — much cheaper than projective + projective
// Uses 7M + 4S formula (Cohen–Miyaji–Ono, 1998)

inline G1Projective g1_add_mixed(G1Projective p, G1Affine q) {
    if (q.is_infinity) return p;
    if (g1_is_identity(p)) return g1_from_affine(q);

    // U1 = P.Y, U2 = Q.Y * P.Z^3
    // S1 = P.X, S2 = Q.X * P.Z^2
    Fq256 z_sq = fq_mul(p.z, p.z);
    Fq256 z_cu = fq_mul(z_sq, p.z);

    Fq256 u2 = fq_mul(q.y, z_cu);
    Fq256 s2 = fq_mul(q.x, z_sq);

    Fq256 h = fq_sub(s2, p.x);      // H = S2 - X1
    Fq256 r = fq_sub(u2, p.y);       // R = U2 - Y1

    // Check for doubling case (H == 0)
    if (fq_is_zero(h)) {
        if (fq_is_zero(r)) {
            // Point doubling
            // Use standard doubling formula
            Fq256 xx = fq_mul(p.x, p.x);
            Fq256 three_xx = fq_add(fq_double(xx), xx);
            // a = 0 for BN254, so M = 3*X^2
            Fq256 m = three_xx;
            Fq256 s = fq_mul(fq_double(fq_double(fq_mul(p.x, fq_mul(p.y, p.y)))),
                             fq_add(fq_mul(p.z, p.z), fq_mul(p.z, p.z)));
            // Simplified doubling:
            Fq256 yy = fq_mul(p.y, p.y);
            Fq256 yyyy = fq_mul(yy, yy);
            Fq256 xy2 = fq_double(fq_mul(p.x, yy));
            s = fq_double(xy2);
            Fq256 m_sq = fq_mul(m, m);
            G1Projective result;
            result.x = fq_sub(m_sq, fq_double(s));
            result.y = fq_sub(fq_mul(m, fq_sub(s, result.x)), fq_double(fq_double(fq_double(yyyy))));
            result.z = fq_mul(fq_double(p.y), p.z);
            return result;
        }
        // Points are negatives of each other
        return g1_identity();
    }

    Fq256 hh = fq_mul(h, h);
    Fq256 hhh = fq_mul(hh, h);
    Fq256 v = fq_mul(p.x, hh);

    G1Projective result;
    Fq256 rr = fq_mul(r, r);
    result.x = fq_sub(fq_sub(rr, hhh), fq_double(v));
    result.y = fq_sub(fq_mul(r, fq_sub(v, result.x)), fq_mul(p.y, hhh));
    result.z = fq_mul(p.z, h);

    return result;
}

// ---- Pippenger bucket accumulation kernel ----
//
// Each thread handles one (scalar, point) pair:
// 1. Extract the c-bit window from the scalar
// 2. Atomically add the point to the corresponding bucket
//
// Since atomic EC point addition isn't possible, we instead write
// (bucket_index, point_index) pairs to a scatter buffer, then
// a separate kernel accumulates each bucket sequentially.

struct ScatterEntry {
    uint32_t bucket_idx;
    uint32_t point_idx;
};

// Phase 1: Scatter — assign each point to a bucket based on scalar window
kernel void pippenger_scatter(
    device const uint32_t *scalars   [[buffer(0)]],  // n × 8 words (Montgomery form scalars)
    device ScatterEntry *scatter     [[buffer(1)]],   // output: n entries
    device const uint32_t *params    [[buffer(2)]],   // [n, window_bits, window_idx]
    uint tid                         [[thread_position_in_grid]]
) {
    uint n = params[0];
    uint window_bits = params[1];
    uint window_idx = params[2];

    if (tid >= n) return;

    // Extract window_bits bits starting at bit position window_idx * window_bits
    uint bit_offset = window_idx * window_bits;
    uint word_idx = bit_offset / 32;
    uint bit_in_word = bit_offset % 32;
    uint mask = (1u << window_bits) - 1u;

    // Read scalar words for this element (8 words per scalar)
    uint32_t lo = scalars[tid * 8 + word_idx];
    uint32_t bucket = (lo >> bit_in_word) & mask;

    // Handle cross-word boundary
    if (bit_in_word + window_bits > 32 && word_idx + 1 < 8) {
        uint32_t hi = scalars[tid * 8 + word_idx + 1];
        bucket |= (hi << (32 - bit_in_word)) & mask;
    }

    scatter[tid].bucket_idx = bucket;
    scatter[tid].point_idx = tid;
}

// Phase 2: Bucket accumulation — each thread processes one bucket
// Accumulates all points assigned to this bucket using mixed addition
kernel void pippenger_accumulate(
    device const G1Affine *points      [[buffer(0)]],  // n affine points
    device const ScatterEntry *scatter [[buffer(1)]],   // n scatter entries (sorted by bucket)
    device const uint32_t *bucket_offsets [[buffer(2)]], // num_buckets+1 offsets into scatter
    device G1Projective *bucket_sums   [[buffer(3)]],   // output: num_buckets projective points
    uint tid                           [[thread_position_in_grid]]
) {
    uint start = bucket_offsets[tid];
    uint end = bucket_offsets[tid + 1];

    G1Projective acc = g1_identity();
    for (uint i = start; i < end; i++) {
        uint pt_idx = scatter[i].point_idx;
        acc = g1_add_mixed(acc, points[pt_idx]);
    }
    bucket_sums[tid] = acc;
}
