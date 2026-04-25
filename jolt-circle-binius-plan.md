# JoltCircleBinius Implementation Plan - Updated

**Branch**: `jolt-circle-binius`
**Status**: Blocked on binius crate availability
**Target**: Replace Jolt's Dory/KZG backend with CircleBinius native backend

---

## Dependency Status

**BLOCKED**: The binius crate from `https://github.com/irreducibleio/binius` returns 404 (not found).

**Current approach**: Implement what we can without external dependencies:
1. Field type definitions and trait abstractions
2. Tower Trace Adapter (Component 1) - pure data transformation
3. Stub out cryptographic operations with clear interfaces

---

## Phase 1: Dependencies & Field Types (Week 1) - IN PROGRESS

**Status**: External crate unavailable. Implementing stubbed interfaces.

### 1.1 Workspace Dependencies Added (Cargo.toml)
- Added `binius` and `plonky3-circle` as workspace dependencies (commented out until crate is available)

### 1.2 Feature Flags Added (jolt-core/Cargo.toml)
```toml
circle-binius = ["dep:binius", "dep:plonky3-circle"]
```

### 1.3 Binary Tower Field Types (TODO)
Will implement in: `jolt-core/src/poly/commitment/circle_binius/fields.rs`

```rust
/// Binary field F_{2^8} - for opcodes, lookup indices
pub struct BinaryField8b(u8);

/// Binary field F_{2^{32}} - for registers, memory values
pub struct BinaryField32b(u32);

/// Binary field F_{2^{64}} - for addresses
pub struct BinaryField64b(u64);

/// Binary field F_{2^{128}} - for FRI challenges, security parameters
pub struct BinaryField128b(u128);
```

### 1.4 Tower Trace Adapter (Component 1)
Will implement in: `jolt-core/src/poly/commitment/circle_binius/tower_trace.rs`

```rust
/// Maps Jolt's trace columns to binary tower subfield columns
pub struct TowerTrace {
    pub byte_columns: Vec<Column<BinaryField8b>>,      // F_{2^8}
    pub word_columns: Vec<Column<BinaryField32b>>,     // F_{2^{32}}
    pub addr_columns: Vec<Column<BinaryField64b>>,     // F_{2^{64}}
    pub security_columns: Vec<Column<BinaryField128b>>,// F_{2^{128}}
}

impl TowerTrace {
    /// Convert Jolt trace to tower format
    pub fn from_jolt_trace(trace: &JoltTrace) -> Self { ... }

    /// Pack bits for efficient storage
    pub fn pack_bits(&self) -> PackedTrace { ... }
}
```

---

## Phase 2: CommitmentScheme Implementation (Week 2) - BLOCKED ON DEPENDENCY

Will implement in: `jolt-core/src/poly/commitment/circle_binius/mod.rs`

Requires binius crate for:
- `BiniusField` trait implementation
- `BrakedownPCS` struct
- Merkle tree operations with BLAKE3

---

## Component 1: Multi-Tower Trace Adapter

### What it is
A layer that maps Jolt's existing Plonkish trace columns into binary tower subfield columns appropriate for each data type.

### Why
Jolt currently uses one monolithic BLS12-381 field for all trace columns. Binary tower fields allow each column to use the smallest field that fits its data — reducing field element size and enabling bit-packing.

### Column mapping

| Jolt Trace Column | Data Width | Tower Subfield | Bit-Packing |
|---|---|---|---|
| Instruction opcode | 8-bit | F_{2^8} | 16 opcodes per element |
| Register values | 32-bit | F_{2^{32}} | Native |
| Memory addresses | 64-bit | F_{2^{64}} | Native |
| Lookup table indices | 8-bit | F_{2^8} | 16 indices per element |
| FRI challenges | 128-bit | F_{2^{128}} | Native |

### Interface

```rust
// jolt-core/src/poly/commitment/circle_binius/tower_trace.rs

pub struct TowerTrace {
    byte_columns: Vec<DensePolynomial<BinaryField8b>>,
    word_columns: Vec<DensePolynomial<BinaryField32b>>,
    addr_columns: Vec<DensePolynomial<BinaryField64b>>,
    security_columns: Vec<DensePolynomial<BinaryField128b>>,
}

impl TowerTrace {
    /// Create tower trace from Jolt trace columns
    pub fn from_jolt_trace(trace: &JoltTrace) -> Self {
        // Split columns by data width
        // Convert to appropriate binary field representation
        // Pad to power of 2 for FFT compatibility
    }

    /// Get total number of trace rows (padded to power of 2)
    pub fn num_rows(&self) -> usize { ... }

    /// Get trace length in bits
    pub fn trace_length_bits(&self) -> usize { ... }

    /// Convert to evaluation form for FFT
    pub fn to_evaluations(&self) -> Vec<BinaryField32b> { ... }
}
```

### Implementation Notes

1. **JoltField trait compatibility**: Binary fields must implement `JoltField` for compatibility with existing Jolt infrastructure
2. **No cryptographic assumptions**: This is pure data transformation
3. **Round-trip guarantee**: `from_jolt_trace` + `to_evaluations` should reproduce original data

---

## Implementation Steps

### Step 1: Create circle_binius module structure
- [ ] `jolt-core/src/poly/commitment/circle_binius/mod.rs`
- [ ] `jolt-core/src/poly/commitment/circle_binius/fields.rs`
- [ ] `jolt-core/src/poly/commitment/circle_binius/tower_trace.rs`

### Step 2: Implement binary field types
- [ ] `BinaryField8b` with add, mul, negate operations
- [ ] `BinaryField32b` with add, mul, negate operations
- [ ] `BinaryField64b` with add, mul, negate operations
- [ ] `BinaryField128b` with add, mul, negate operations
- [ ] Implement `JoltField` trait for each

### Step 3: Implement TowerTrace
- [ ] `TowerTrace::from_jolt_trace()`
- [ ] `TowerTrace::to_evaluations()`
- [ ] Round-trip test

---

## Next Steps

1. **Resolve binius dependency**: Need to find correct crate URL
   - Check if it's published on crates.io as `binius` or similar
   - Or if it lives under a different organization

2. **Meanwhile**: Implement TowerTrace adapter (Component 1) which is independent

3. **Stub the crypto layer**: Define the `CircleBiniusPCS` struct with trait implementations but panic with "TODO" for the cryptographic operations until dependencies are resolved

---

## Updated Timeline

| Phase | Status | Work |
|-------|--------|------|
| Phase 1 | IN PROGRESS | Dependencies + field types (blocked on crate) |
| Phase 2 | BLOCKED | CommitmentScheme impl |
| Phase 3 | BLOCKED | Circle FFT + FRI |
| Phase 4 | BLOCKED | Brakedown PCS |
| Phase 5 | BLOCKED | GKR batching |
| Phase 6 | BLOCKED | Binary verifier circuit |

**Note**: Component 1 (TowerTrace) is implementable now and doesn't require external dependencies.