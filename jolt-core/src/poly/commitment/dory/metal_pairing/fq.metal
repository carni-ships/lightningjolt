// BN254 Fq field arithmetic in Metal Shading Language
//
// Fq is the base field of BN254 with modulus:
// p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
//
// Represented as 8 × uint32 limbs in little-endian order (limb[0] is least significant).
// All values are in Montgomery form: x_mont = x * R mod p, where R = 2^256.
//
// Montgomery multiplication: given a_mont, b_mont, compute (a * b)_mont = a_mont * b_mont * R^{-1} mod p
// Uses CIOS (Coarsely Integrated Operand Scanning) algorithm for efficiency on GPU.

#include <metal_stdlib>
using namespace metal;

// BN254 Fq modulus p in 8 × u32 limbs (little-endian)
// p = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
constant uint32_t P[8] = {
    0xd87cfd47, 0x3c208c16, 0x6871ca8d, 0x97816a91,
    0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72
};

// Montgomery parameter: p_inv = -p^{-1} mod 2^32
// p mod 2^32 = 0xd87cfd47
// We need inv such that p_low * inv ≡ 0xFFFFFFFF (mod 2^32)
// Verified: 0xd87cfd47 * 0xe4866389 ≡ 0xFFFFFFFF (mod 2^32)
constant uint32_t P_INV = 0xe4866389u;

// R^2 mod p (for converting standard form → Montgomery form)
// R = 2^256
// R^2 mod p = 0x06d89f71cab8351f47ab1eff0a417ff6b5e71911d44501fbf32cfc5b538afa89
constant uint32_t R_SQUARED[8] = {
    0x538afa89, 0xf32cfc5b, 0xd44501fb, 0xb5e71911,
    0x0a417ff6, 0x47ab1eff, 0xcab8351f, 0x06d89f71
};

// Montgomery form of 1: R mod p
// R mod p = 0x0e0a77c19a07df2f666ea36f7879462c0a78eb28f5c70b3dd35d438dc58f0d9d
constant uint32_t R_MOD_P[8] = {
    0xc58f0d9d, 0xd35d438d, 0xf5c70b3d, 0x0a78eb28,
    0x7879462c, 0x666ea36f, 0x9a07df2f, 0x0e0a77c1
};

struct Fq {
    uint32_t limbs[8];
};

// Add with carry: returns (sum, carry)
inline uint2 adc(uint32_t a, uint32_t b, uint32_t carry_in) {
    uint64_t sum = uint64_t(a) + uint64_t(b) + uint64_t(carry_in);
    return uint2(uint32_t(sum), uint32_t(sum >> 32));
}

// Multiply and add: a * b + c + carry_in → (lo, hi)
inline uint2 mac(uint32_t a, uint32_t b, uint32_t c, uint32_t carry_in) {
    uint64_t prod = uint64_t(a) * uint64_t(b) + uint64_t(c) + uint64_t(carry_in);
    return uint2(uint32_t(prod), uint32_t(prod >> 32));
}

// Subtract with borrow: a - b - borrow_in → (result, borrow_out)
inline uint2 sbb(uint32_t a, uint32_t b, uint32_t borrow_in) {
    uint64_t diff = uint64_t(a) - uint64_t(b) - uint64_t(borrow_in);
    // borrow_out is 1 if diff underflowed
    uint32_t borrow_out = (diff >> 63) & 1u;
    return uint2(uint32_t(diff), borrow_out);
}

// Fq addition: result = (a + b) mod p
inline Fq fq_add(Fq a, Fq b) {
    Fq result;
    uint32_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint2 r = adc(a.limbs[i], b.limbs[i], carry);
        result.limbs[i] = r.x;
        carry = r.y;
    }
    // Conditionally subtract p: try result - p, keep if no borrow
    Fq reduced;
    uint32_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint2 r = sbb(result.limbs[i], P[i], borrow);
        reduced.limbs[i] = r.x;
        borrow = r.y;
    }
    // If carry from addition >= borrow from subtraction, the subtraction was valid
    // carry is 0 or 1, borrow is 0 or 1
    // Use reduced when (carry - borrow) >= 0, i.e., carry >= borrow
    bool use_reduced = (carry >= borrow);
    for (int i = 0; i < 8; i++) {
        result.limbs[i] = use_reduced ? reduced.limbs[i] : result.limbs[i];
    }
    return result;
}

// Fq subtraction: result = (a - b) mod p
inline Fq fq_sub(Fq a, Fq b) {
    Fq result;
    uint32_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint2 r = sbb(a.limbs[i], b.limbs[i], borrow);
        result.limbs[i] = r.x;
        borrow = r.y;
    }
    // If borrow, add p back
    if (borrow != 0) {
        uint32_t carry = 0;
        for (int i = 0; i < 8; i++) {
            uint2 r = adc(result.limbs[i], P[i], carry);
            result.limbs[i] = r.x;
            carry = r.y;
        }
    }
    return result;
}

// Montgomery multiplication using CIOS (Coarsely Integrated Operand Scanning)
//
// Given a, b in Montgomery form, computes a * b * R^{-1} mod p (which is (a*b) in Montgomery form).
//
// Algorithm: for each limb i of b:
//   1. Multiply a by b[i] and add to accumulator T
//   2. Compute Montgomery reduction factor m = T[0] * p_inv mod 2^32
//   3. Add m * p to T and shift right by 32 bits
// After all limbs, conditionally subtract p.
inline Fq fq_mul(Fq a, Fq b) {
    // T has 9 limbs (8 + 1 for overflow)
    uint32_t T[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (int i = 0; i < 8; i++) {
        // Step 1: T = T + a * b[i]
        uint32_t carry = 0;
        for (int j = 0; j < 8; j++) {
            uint2 r = mac(a.limbs[j], b.limbs[i], T[j], carry);
            T[j] = r.x;
            carry = r.y;
        }
        // Propagate carry into T[8]
        uint64_t sum = uint64_t(T[8]) + uint64_t(carry);
        T[8] = uint32_t(sum);

        // Step 2: Montgomery reduction
        uint32_t m = T[0] * P_INV;

        // Step 3: T = (T + m * p) >> 32
        carry = 0;
        for (int j = 0; j < 8; j++) {
            uint2 r = mac(m, P[j], T[j], carry);
            if (j > 0) {
                T[j - 1] = r.x;
            }
            // T[0] should be zero after first mac (that's the point of Montgomery reduction)
            carry = r.y;
        }
        sum = uint64_t(T[8]) + uint64_t(carry);
        T[7] = uint32_t(sum);
        T[8] = uint32_t(sum >> 32);
    }

    // Conditional subtraction: if T >= p, subtract p
    Fq result;
    for (int i = 0; i < 8; i++) {
        result.limbs[i] = T[i];
    }

    Fq reduced;
    uint32_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint2 r = sbb(result.limbs[i], P[i], borrow);
        reduced.limbs[i] = r.x;
        borrow = r.y;
    }
    // T[8] is at most 1. If T[8]==0 and borrow==0, result >= p, use reduced.
    // If T[8]==1, definitely >= p (but shouldn't happen with proper Montgomery).
    // Safe check: if no borrow (borrow == 0), use reduced.
    if (borrow == 0) {
        return reduced;
    }
    return result;
}

// Montgomery squaring (same as mul but could be optimized later)
inline Fq fq_sqr(Fq a) {
    return fq_mul(a, a);
}

// Load Fq from constant
inline Fq fq_from_const(constant uint32_t c[8]) {
    Fq r;
    for (int i = 0; i < 8; i++) {
        r.limbs[i] = c[i];
    }
    return r;
}

// Zero
inline Fq fq_zero() {
    Fq r;
    for (int i = 0; i < 8; i++) {
        r.limbs[i] = 0;
    }
    return r;
}

// One (in Montgomery form)
inline Fq fq_one() {
    return fq_from_const(R_MOD_P);
}

// Fq negation: -a mod p
inline Fq fq_neg(Fq a) {
    // Check if a is zero
    bool is_zero = true;
    for (int i = 0; i < 8; i++) {
        if (a.limbs[i] != 0) { is_zero = false; break; }
    }
    if (is_zero) return a;

    Fq result;
    uint32_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint2 r = sbb(P[i], a.limbs[i], borrow);
        result.limbs[i] = r.x;
        borrow = r.y;
    }
    return result;
}

// Fq double: 2a mod p
inline Fq fq_double(Fq a) {
    return fq_add(a, a);
}

// ========== Fq2 = Fq[u]/(u² + 1) ==========
// Element: c0 + c1·u, where u² = -1

struct Fq2 {
    Fq c0;
    Fq c1;
};

inline Fq2 fq2_zero() {
    Fq2 r;
    r.c0 = fq_zero();
    r.c1 = fq_zero();
    return r;
}

inline Fq2 fq2_one() {
    Fq2 r;
    r.c0 = fq_one();
    r.c1 = fq_zero();
    return r;
}

inline Fq2 fq2_add(Fq2 a, Fq2 b) {
    Fq2 r;
    r.c0 = fq_add(a.c0, b.c0);
    r.c1 = fq_add(a.c1, b.c1);
    return r;
}

inline Fq2 fq2_sub(Fq2 a, Fq2 b) {
    Fq2 r;
    r.c0 = fq_sub(a.c0, b.c0);
    r.c1 = fq_sub(a.c1, b.c1);
    return r;
}

inline Fq2 fq2_neg(Fq2 a) {
    Fq2 r;
    r.c0 = fq_neg(a.c0);
    r.c1 = fq_neg(a.c1);
    return r;
}

inline Fq2 fq2_double(Fq2 a) {
    Fq2 r;
    r.c0 = fq_double(a.c0);
    r.c1 = fq_double(a.c1);
    return r;
}

// Fq2 mul: (a0 + a1·u)(b0 + b1·u) = (a0·b0 - a1·b1) + (a0·b1 + a1·b0)·u
// Karatsuba: 3 Fq muls
inline Fq2 fq2_mul(Fq2 a, Fq2 b) {
    Fq v0 = fq_mul(a.c0, b.c0);
    Fq v1 = fq_mul(a.c1, b.c1);
    Fq2 r;
    r.c0 = fq_sub(v0, v1);                                          // a0·b0 - a1·b1
    r.c1 = fq_sub(fq_mul(fq_add(a.c0, a.c1), fq_add(b.c0, b.c1)),
                   fq_add(v0, v1));                                  // (a0+a1)(b0+b1) - v0 - v1
    return r;
}

// Fq2 sqr: optimized for nonresidue = -1
// c0' = (c0-c1)(c0+c1), c1' = 2·c0·c1
inline Fq2 fq2_sqr(Fq2 a) {
    Fq2 r;
    r.c0 = fq_mul(fq_sub(a.c0, a.c1), fq_add(a.c0, a.c1));
    r.c1 = fq_double(fq_mul(a.c0, a.c1));
    return r;
}

// Multiply Fq2 by Fq scalar
inline Fq2 fq2_mul_fq(Fq2 a, Fq s) {
    Fq2 r;
    r.c0 = fq_mul(a.c0, s);
    r.c1 = fq_mul(a.c1, s);
    return r;
}

// Fq2 conjugate: (c0, c1) → (c0, -c1)
inline Fq2 fq2_conjugate(Fq2 a) {
    Fq2 r;
    r.c0 = a.c0;
    r.c1 = fq_neg(a.c1);
    return r;
}

// Fq2 inversion: 1/(c0 + c1·u) = (c0 - c1·u) / (c0² + c1²)
inline Fq2 fq2_inv(Fq2 a) {
    Fq norm = fq_add(fq_mul(a.c0, a.c0), fq_mul(a.c1, a.c1)); // c0² + c1² (since u²=-1)
    // Fq inversion by Fermat: norm^(p-2) mod p
    // For now we won't need Fq2 inversion in the Miller loop hot path
    // (only final exponentiation uses it, and that's done on CPU)
    // Placeholder — will implement if needed
    Fq2 r;
    r.c0 = a.c0;
    r.c1 = a.c1;
    return r;
}

// ========== Fq6 = Fq2[v]/(v³ - ξ), where ξ = 9 + u ==========
// Element: c0 + c1·v + c2·v²

struct Fq6 {
    Fq2 c0;
    Fq2 c1;
    Fq2 c2;
};

// Multiply Fq2 by ξ = 9 + u
// (a + b·u)(9 + u) = (9a - b) + (9b + a)·u
inline Fq2 fq2_mul_by_xi(Fq2 f) {
    // 9x = 8x + x, where 8x = double(double(double(x)))
    Fq a8 = fq_double(fq_double(fq_double(f.c0)));
    Fq b8 = fq_double(fq_double(fq_double(f.c1)));
    Fq2 r;
    r.c0 = fq_sub(fq_add(a8, f.c0), f.c1);  // 9·c0 - c1
    r.c1 = fq_add(fq_add(b8, f.c1), f.c0);  // 9·c1 + c0
    return r;
}

inline Fq6 fq6_zero() {
    Fq6 r;
    r.c0 = fq2_zero();
    r.c1 = fq2_zero();
    r.c2 = fq2_zero();
    return r;
}

inline Fq6 fq6_one() {
    Fq6 r;
    r.c0 = fq2_one();
    r.c1 = fq2_zero();
    r.c2 = fq2_zero();
    return r;
}

inline Fq6 fq6_add(Fq6 a, Fq6 b) {
    Fq6 r;
    r.c0 = fq2_add(a.c0, b.c0);
    r.c1 = fq2_add(a.c1, b.c1);
    r.c2 = fq2_add(a.c2, b.c2);
    return r;
}

inline Fq6 fq6_sub(Fq6 a, Fq6 b) {
    Fq6 r;
    r.c0 = fq2_sub(a.c0, b.c0);
    r.c1 = fq2_sub(a.c1, b.c1);
    r.c2 = fq2_sub(a.c2, b.c2);
    return r;
}

inline Fq6 fq6_neg(Fq6 a) {
    Fq6 r;
    r.c0 = fq2_neg(a.c0);
    r.c1 = fq2_neg(a.c1);
    r.c2 = fq2_neg(a.c2);
    return r;
}

// Fq6 multiplication (Karatsuba — Devegili OhEig Scott Dahab)
// Cost: 6 Fq2 muls + additions
inline Fq6 fq6_mul(Fq6 a, Fq6 b) {
    Fq2 ad = fq2_mul(a.c0, b.c0);
    Fq2 be = fq2_mul(a.c1, b.c1);
    Fq2 cf = fq2_mul(a.c2, b.c2);

    Fq2 x = fq2_sub(fq2_sub(fq2_mul(fq2_add(a.c1, a.c2), fq2_add(b.c1, b.c2)), be), cf);
    Fq2 y = fq2_sub(fq2_sub(fq2_mul(fq2_add(a.c0, a.c1), fq2_add(b.c0, b.c1)), ad), be);
    Fq2 z = fq2_sub(fq2_add(fq2_sub(fq2_mul(fq2_add(a.c0, a.c2), fq2_add(b.c0, b.c2)), ad), be), cf);

    Fq6 r;
    r.c0 = fq2_add(ad, fq2_mul_by_xi(x));  // ad + ξ·x
    r.c1 = fq2_add(y, fq2_mul_by_xi(cf));   // y + ξ·cf
    r.c2 = z;
    return r;
}

// Fq6 squaring (Ch-Sq-3)
inline Fq6 fq6_sqr(Fq6 a) {
    Fq2 s0 = fq2_sqr(a.c0);
    Fq2 ab = fq2_mul(a.c0, a.c1);
    Fq2 s1 = fq2_double(ab);
    Fq2 s2 = fq2_sqr(fq2_add(fq2_sub(a.c0, a.c1), a.c2));
    Fq2 bc = fq2_mul(a.c1, a.c2);
    Fq2 s3 = fq2_double(bc);
    Fq2 s4 = fq2_sqr(a.c2);

    Fq6 r;
    r.c0 = fq2_add(s0, fq2_mul_by_xi(s3));
    r.c1 = fq2_add(s1, fq2_mul_by_xi(s4));
    r.c2 = fq2_add(fq2_sub(fq2_add(s1, s2), s0), fq2_sub(s3, s4));
    return r;
}

// Sparse Fq6 multiply: self * (c0, c1, 0)
// Only positions 0 and 1 of the second operand are non-zero
inline Fq6 fq6_mul_by_01(Fq6 self, Fq2 c0, Fq2 c1) {
    Fq2 a_a = fq2_mul(self.c0, c0);
    Fq2 b_b = fq2_mul(self.c1, c1);
    // t1 = ξ·((self.c1 + self.c2) * c1 - b_b) + a_a
    Fq2 t1 = fq2_add(fq2_mul_by_xi(fq2_sub(fq2_mul(fq2_add(self.c1, self.c2), c1), b_b)), a_a);
    // t2 = (c0 + c1) * (self.c0 + self.c1) - a_a - b_b
    Fq2 t2 = fq2_sub(fq2_sub(fq2_mul(fq2_add(c0, c1), fq2_add(self.c0, self.c1)), a_a), b_b);
    // t3 = c0 * (self.c0 + self.c2) - a_a + b_b = self.c2 * c0 + b_b
    Fq2 t3 = fq2_add(fq2_mul(self.c2, c0), b_b);

    Fq6 r;
    r.c0 = t1;
    r.c1 = t2;
    r.c2 = t3;
    return r;
}

// Multiply Fq6 by nonresidue (0, 1, 0) — cyclic rotation with xi
// (a, b, c) → (ξ·c, a, b)
inline Fq6 fq6_mul_by_nonresidue(Fq6 a) {
    Fq6 r;
    r.c0 = fq2_mul_by_xi(a.c2);
    r.c1 = a.c0;
    r.c2 = a.c1;
    return r;
}

// ========== Fq12 = Fq6[w]/(w² - v) ==========
// Element: c0 + c1·w, where w² = v = (0, 1, 0) in Fq6

struct Fq12 {
    Fq6 c0;
    Fq6 c1;
};

inline Fq12 fq12_one() {
    Fq12 r;
    r.c0 = fq6_one();
    r.c1 = fq6_zero();
    return r;
}

// Fq12 multiplication (Karatsuba over Fq6)
inline Fq12 fq12_mul(Fq12 a, Fq12 b) {
    Fq6 v0 = fq6_mul(a.c0, b.c0);
    Fq6 v1 = fq6_mul(a.c1, b.c1);

    Fq12 r;
    r.c0 = fq6_add(v0, fq6_mul_by_nonresidue(v1));
    r.c1 = fq6_sub(fq6_sub(fq6_mul(fq6_add(a.c0, a.c1), fq6_add(b.c0, b.c1)), v0), v1);
    return r;
}

// Fq12 squaring
inline Fq12 fq12_sqr(Fq12 a) {
    Fq6 v0 = fq6_sub(a.c0, a.c1);
    Fq6 v3 = fq6_sub(a.c0, fq6_mul_by_nonresidue(a.c1));
    Fq6 v2 = fq6_mul(a.c0, a.c1);

    v0 = fq6_mul(v0, v3);

    Fq12 r;
    r.c1 = fq6_add(v2, v2);  // 2·v2
    // result.c0 = v0 + (NONRESIDUE + 1)·v2
    // (NONRESIDUE + 1)·v2 = mul_by_nonresidue(v2) + v2
    r.c0 = fq6_add(v0, fq6_add(fq6_mul_by_nonresidue(v2), v2));
    return r;
}

// Multiply Fq12 by a sparse element (used in Miller loop line functions)
// Sparse Fq12: non-zero at positions 0, 3, 4 in the Fq2 decomposition
// This matches arkworks mul_by_034 for D-type twist
inline Fq12 fq12_mul_by_034(Fq12 f, Fq2 c0, Fq2 c3, Fq2 c4) {
    // a = f.c0 * c0 (scalar broadcast across Fq6)
    Fq6 a;
    a.c0 = fq2_mul(f.c0.c0, c0);
    a.c1 = fq2_mul(f.c0.c1, c0);
    a.c2 = fq2_mul(f.c0.c2, c0);

    // b = f.c1.mul_by_01(c3, c4)
    Fq6 b = fq6_mul_by_01(f.c1, c3, c4);

    // e = (f.c0 + f.c1).mul_by_01(c0 + c3, c4)
    Fq6 e = fq6_mul_by_01(fq6_add(f.c0, f.c1), fq2_add(c0, c3), c4);

    Fq12 r;
    r.c1 = fq6_sub(fq6_sub(e, a), b);
    r.c0 = fq6_add(a, fq6_mul_by_nonresidue(b));
    return r;
}

// ========== G2 Point Structures and Operations ==========

struct G1Affine {
    Fq x;
    Fq y;
};

struct G2Affine {
    Fq2 x;
    Fq2 y;
};

// G2 in homogeneous projective coordinates (for Miller loop accumulation)
struct G2HomProjective {
    Fq2 x;
    Fq2 y;
    Fq2 z;
};

// Line evaluation coefficients (three Fq2 values)
struct EllCoeff {
    Fq2 c0;
    Fq2 c1;
    Fq2 c2;
};

// BN254 G2 twist coefficient b' = 3/(9+u)
// In Montgomery form — these are the raw [u32;8] Montgomery limbs
// b'.c0 = 19485874751759354771024239261021720505790618469301721065564631296452457478373
// b'.c1 = 266929791119991161246907387137283842545076965332900288569378510910307636690
constant uint32_t COEFF_B_C0[8] = {
    0x77b802a8, 0x3bf938e3, 0x3633535d, 0x020b1b27,
    0x49755260, 0x26b7edf0, 0x4384a86d, 0x2514c632
};
constant uint32_t COEFF_B_C1[8] = {
    0xd1dcff67, 0x38e7eccc, 0x93ce0d3e, 0x65f0b37d,
    0x22ac00aa, 0xd749d0dd, 0x4a688d4d, 0x0141b9ce
};

// BN254 twist_mul_by_q coefficients for Frobenius endomorphism
// TWIST_MUL_BY_Q_X
constant uint32_t TWIST_MBQ_X_C0[8] = {
    0x4563ab30, 0xb5773b10, 0xa9aa6454, 0x347f91c8,
    0x242e0991, 0x7a007127, 0x118214ec, 0x1956bcd8
};
constant uint32_t TWIST_MBQ_X_C1[8] = {
    0xa0aa4757, 0x6e849f1e, 0x89f89141, 0xaa1c7b6d,
    0xfae0ca3a, 0xb6e713cd, 0x4e82ebc3, 0x26694fbb
};
// TWIST_MUL_BY_Q_Y
constant uint32_t TWIST_MBQ_Y_C0[8] = {
    0x2936b629, 0xe4bbdd0c, 0xe133bacb, 0xbb30f162,
    0xf9645366, 0x31a9d1b6, 0xa500f8dd, 0x253570be
};
constant uint32_t TWIST_MBQ_Y_C1[8] = {
    0x5ffe77c7, 0xa1d77ce4, 0x7826d1db, 0x07affd11,
    0xbb7edc6b, 0x6d16bd27, 0x85defecc, 0x2c872002
};

// two_inv = (2)^{-1} mod p — in Montgomery form
// = (p + 1) / 2 in standard form, then convert to Montgomery
constant uint32_t TWO_INV[8] = {
    0x4f060572, 0x87bee7d2, 0x2f1c6ae5, 0xd0fd2add,
    0xfcfd4f44, 0x8f5f7492, 0x3d9cbfac, 0x1f37631a
};

// Fq2 multiply by Fq scalar
inline Fq2 fq2_scale(Fq2 a, Fq s) {
    Fq2 r;
    r.c0 = fq_mul(a.c0, s);
    r.c1 = fq_mul(a.c1, s);
    return r;
}

inline Fq fq_load_const(constant uint32_t c[8]) {
    Fq r;
    for (int i = 0; i < 8; i++) r.limbs[i] = c[i];
    return r;
}

// G2 doubling step: doubles R in-place, returns line evaluation coefficients
// Follows arkworks bn/g2.rs double_in_place exactly
inline EllCoeff g2_doubling_step(thread G2HomProjective& r) {
    Fq two_inv = fq_load_const(TWO_INV);
    Fq2 a = fq2_scale(fq2_mul(r.x, r.y), two_inv);
    Fq2 b = fq2_sqr(r.y);
    Fq2 c = fq2_sqr(r.z);

    // e = COEFF_B * (c + c + c) = 3 * COEFF_B * c
    Fq2 coeff_b;
    coeff_b.c0 = fq_load_const(COEFF_B_C0);
    coeff_b.c1 = fq_load_const(COEFF_B_C1);
    Fq2 e = fq2_mul(coeff_b, fq2_add(fq2_double(c), c));

    Fq2 f = fq2_add(fq2_double(e), e);  // 3e
    Fq2 g = fq2_scale(fq2_add(b, f), two_inv);
    Fq2 h = fq2_sub(fq2_sqr(fq2_add(r.y, r.z)), fq2_add(b, c));
    Fq2 i = fq2_sub(e, b);
    Fq2 j = fq2_sqr(r.x);
    Fq2 e_sq = fq2_sqr(e);

    // Update R
    r.x = fq2_mul(a, fq2_sub(b, f));
    r.y = fq2_sub(fq2_sqr(g), fq2_add(fq2_double(e_sq), e_sq));
    r.z = fq2_mul(b, h);

    // Line coefficients for D-type twist
    EllCoeff coeff;
    coeff.c0 = fq2_neg(h);                            // -h
    coeff.c1 = fq2_add(fq2_double(j), j);             // 3j
    coeff.c2 = i;                                       // e - b
    return coeff;
}

// G2 addition step: adds Q (affine) to R (projective), returns line coefficients
// Follows arkworks bn/g2.rs add_in_place exactly
inline EllCoeff g2_addition_step(thread G2HomProjective& r, Fq2 qx, Fq2 qy) {
    Fq2 theta = fq2_sub(r.y, fq2_mul(qy, r.z));
    Fq2 lambda = fq2_sub(r.x, fq2_mul(qx, r.z));
    Fq2 c = fq2_sqr(theta);
    Fq2 d = fq2_sqr(lambda);
    Fq2 e = fq2_mul(lambda, d);
    Fq2 f = fq2_mul(r.z, c);
    Fq2 g = fq2_mul(r.x, d);
    Fq2 h = fq2_sub(fq2_add(e, f), fq2_double(g));

    // Update R
    r.x = fq2_mul(lambda, h);
    r.y = fq2_sub(fq2_mul(theta, fq2_sub(g, h)), fq2_mul(e, r.y));
    r.z = fq2_mul(r.z, e);

    // j = theta * qx - lambda * qy
    Fq2 j = fq2_sub(fq2_mul(theta, qx), fq2_mul(lambda, qy));

    EllCoeff coeff;
    coeff.c0 = lambda;
    coeff.c1 = fq2_neg(theta);
    coeff.c2 = j;
    return coeff;
}

// Apply line evaluation to Fq12 accumulator (D-type twist)
// c0 *= P.y, c1 *= P.x, c2 unchanged, then mul_by_034
inline Fq12 ell(Fq12 f, EllCoeff coeff, Fq py, Fq px) {
    Fq2 c0 = fq2_scale(coeff.c0, py);
    Fq2 c1 = fq2_scale(coeff.c1, px);
    return fq12_mul_by_034(f, c0, c1, coeff.c2);
}

// Frobenius endomorphism on G2: multiply coordinates by twist constants
// Q' = (x^p * TWIST_MUL_BY_Q_X, y^p * TWIST_MUL_BY_Q_Y)
// For Fq2: x^p = conjugate(x) since p ≡ 3 mod 4 for BN254
inline G2Affine g2_mul_by_char(G2Affine q) {
    Fq2 twist_x;
    twist_x.c0 = fq_load_const(TWIST_MBQ_X_C0);
    twist_x.c1 = fq_load_const(TWIST_MBQ_X_C1);
    Fq2 twist_y;
    twist_y.c0 = fq_load_const(TWIST_MBQ_Y_C0);
    twist_y.c1 = fq_load_const(TWIST_MBQ_Y_C1);

    G2Affine result;
    result.x = fq2_mul(fq2_conjugate(q.x), twist_x);
    result.y = fq2_mul(fq2_conjugate(q.y), twist_y);
    return result;
}

// ATE_LOOP_COUNT for BN254 (NAF of 6x+2, stored LSB first)
// From arkworks: same array but in reverse order for forward iteration
constant int8_t ATE_LOOP_COUNT[65] = {
    0, 0, 0, 1, 0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 1, 0,
    0, -1, 0, -1, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, -1, 0,
    0, 1, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, -1,
    0, 1, 0, -1, 0, 0, 0, -1, 0, -1, 0, 0, 0, 1, 0, 1,
    1
};
constant int ATE_LOOP_LEN = 65;

// Single Miller loop for one (G1, G2) pair
// Returns Fq12 element (before final exponentiation)
inline Fq12 miller_loop_single(G1Affine p, G2Affine q) {
    Fq12 f = fq12_one();

    // Initialize R = Q in homogeneous projective
    G2HomProjective r;
    r.x = q.x;
    r.y = q.y;
    r.z = fq2_one();

    G2Affine neg_q;
    neg_q.x = q.x;
    neg_q.y = fq2_neg(q.y);

    // Main loop: iterate from MSB-1 down to bit 0
    // ATE_LOOP_COUNT[64] = 1 (MSB, skip initial)
    for (int i = ATE_LOOP_LEN - 2; i >= 0; i--) {
        if (i != ATE_LOOP_LEN - 2) {
            f = fq12_sqr(f);
        }

        // Doubling step
        EllCoeff dc = g2_doubling_step(r);
        f = ell(f, dc, p.y, p.x);

        int8_t bit = ATE_LOOP_COUNT[i];
        if (bit == 1) {
            EllCoeff ac = g2_addition_step(r, q.x, q.y);
            f = ell(f, ac, p.y, p.x);
        } else if (bit == -1) {
            EllCoeff ac = g2_addition_step(r, neg_q.x, neg_q.y);
            f = ell(f, ac, p.y, p.x);
        }
    }

    // BN254: X_IS_NEGATIVE = false, so no cyclotomic inverse

    // Two final steps: Frobenius endomorphism Q1 and Q2
    G2Affine q1 = g2_mul_by_char(q);
    G2Affine q2 = g2_mul_by_char(q1);
    q2.y = fq2_neg(q2.y);  // negate q2.y

    EllCoeff ac1 = g2_addition_step(r, q1.x, q1.y);
    f = ell(f, ac1, p.y, p.x);

    EllCoeff ac2 = g2_addition_step(r, q2.x, q2.y);
    f = ell(f, ac2, p.y, p.x);

    return f;
}

// ========== Pairing Kernel ==========

// Each thread computes one Miller loop for a (G1, G2) pair
// Results are Fq12 elements that should be multiplied together on CPU
// followed by final exponentiation
kernel void miller_loop_kernel(
    device const G1Affine* g1_points [[buffer(0)]],
    device const G2Affine* g2_points [[buffer(1)]],
    device Fq12* results             [[buffer(2)]],
    uint tid                         [[thread_position_in_grid]]
) {
    results[tid] = miller_loop_single(g1_points[tid], g2_points[tid]);
}

// ========== Benchmark / Test Kernels ==========

// Kernel: batch Fq multiplications
kernel void fq_mul_kernel(
    device const Fq* a       [[buffer(0)]],
    device const Fq* b       [[buffer(1)]],
    device Fq* result        [[buffer(2)]],
    uint tid                 [[thread_position_in_grid]]
) {
    result[tid] = fq_mul(a[tid], b[tid]);
}

// Kernel: batch Fq multiply-accumulate (chain of multiplications)
kernel void fq_mul_chain_kernel(
    device const Fq* a       [[buffer(0)]],
    device const Fq* b       [[buffer(1)]],
    device Fq* result        [[buffer(2)]],
    constant uint32_t& iters [[buffer(3)]],
    uint tid                 [[thread_position_in_grid]]
) {
    Fq val = fq_mul(a[tid], b[tid]);
    for (uint32_t i = 0; i < iters; i++) {
        val = fq_sqr(val);
    }
    result[tid] = val;
}

// Kernel: batch Fq2 multiplications (for testing extension field correctness)
kernel void fq2_mul_kernel(
    device const Fq2* a      [[buffer(0)]],
    device const Fq2* b      [[buffer(1)]],
    device Fq2* result       [[buffer(2)]],
    uint tid                 [[thread_position_in_grid]]
) {
    result[tid] = fq2_mul(a[tid], b[tid]);
}

// Kernel: batch Fq6 multiplications
kernel void fq6_mul_kernel(
    device const Fq6* a      [[buffer(0)]],
    device const Fq6* b      [[buffer(1)]],
    device Fq6* result       [[buffer(2)]],
    uint tid                 [[thread_position_in_grid]]
) {
    result[tid] = fq6_mul(a[tid], b[tid]);
}

// Kernel: batch Fq12 multiplications
kernel void fq12_mul_kernel(
    device const Fq12* a     [[buffer(0)]],
    device const Fq12* b     [[buffer(1)]],
    device Fq12* result      [[buffer(2)]],
    uint tid                 [[thread_position_in_grid]]
) {
    result[tid] = fq12_mul(a[tid], b[tid]);
}

// Kernel: batch Fq12 squarings chain (measures Fq12 throughput)
kernel void fq12_sqr_chain_kernel(
    device const Fq12* input [[buffer(0)]],
    device Fq12* result      [[buffer(1)]],
    constant uint32_t& iters [[buffer(2)]],
    uint tid                 [[thread_position_in_grid]]
) {
    Fq12 val = input[tid];
    for (uint32_t i = 0; i < iters; i++) {
        val = fq12_sqr(val);
    }
    result[tid] = val;
}
