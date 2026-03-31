use std::cell::UnsafeCell;

use allocative::Allocative;
use ark_std::Zero;
use num::Integer;

use crate::field::JoltField;
use crate::field::OptimizedMul;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::subprotocols::read_write_matrix::address_major::AddressMajorMatrixEntry;
use crate::subprotocols::read_write_matrix::cycle_major::CycleMajorMatrixEntry;
use crate::subprotocols::read_write_matrix::OneHotCoeffLookupTable;
use crate::subprotocols::read_write_matrix::ReadWriteMatrixAddressMajor;
use crate::subprotocols::read_write_matrix::ReadWriteMatrixCycleMajor;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::zkvm::ram::remap_address;
use common::jolt_device::MemoryLayout;
use rayon::prelude::*;
use tracer::instruction::{Cycle, RAMAccess};

/// Represents a non-zero entry in the ra(k, j) and Val(k, j) polynomials.
/// Conceptually, both ra and Val can be seen as K x T matrices.
///
/// # Memory Optimization: `prev_val`/`next_val` as `u64`
///
/// These fields store raw memory values (not bound coefficients) because:
/// - Phase 1 (cycle binding) always starts from initial memory state
/// - We never switch to cycle-major after binding address variables
/// - They're only converted to `F` when needed in arithmetic
///
/// This saves 48 bytes per entry (~35% reduction).
///
/// # Type Parameters
///
/// - `F`: The field type for coefficients.
#[derive(Allocative, Debug, PartialEq, Clone, Copy, Default)]
pub struct RamCycleMajorEntry<F: JoltField> {
    /// The row index. Before binding, row \in [0, T)
    pub row: usize,
    /// The column index. Before binding, col \in [0, K)
    pub col: usize,
    /// In round i, each ReadWriteEntry represents a coefficient
    ///   Val(k, j', r)
    /// which is some combination of Val(k, j', 00...0), ...
    /// Val(k, j', 11...1).
    /// `prev_val` contains the unbound coefficient before
    /// Val(k, j', 00...0) –– abusing notation, `prev_val` is
    /// Val(k, j'-1, 11...1)
    pub(crate) prev_val: u64,
    /// In round i, each ReadWriteEntry represents a coefficient
    ///   Val(k, j', r)
    /// which is some combination of Val(k, j', 00...0), ...
    /// Val(k, j', 11...1).
    /// `next_val` contains the unbound coefficient after
    /// Val(k, j', 00...0) –– abusing notation, `next_val` is
    /// Val(k, j'+1, 00...0)
    pub(crate) next_val: u64,
    /// The Val coefficient for this matrix entry.
    pub val_coeff: F,
    /// The ra coefficient for this matrix entry. Note that for RAM,
    /// ra and wa are the same polynomial.
    pub ra_coeff: F,
}

impl<F: JoltField> RamCycleMajorEntry<F> {
    fn from_cycle(cycle: &Cycle, cycle_index: usize, memory_layout: &MemoryLayout) -> Option<Self> {
        let ram_op = cycle.ram_access();
        match ram_op {
            RAMAccess::Write(write) => {
                let pre_value = write.pre_value;
                let post_value = write.post_value;
                Some(RamCycleMajorEntry {
                    row: cycle_index,
                    col: remap_address(write.address, memory_layout).unwrap() as usize,
                    ra_coeff: F::one(),
                    val_coeff: F::from_u64(pre_value),
                    prev_val: pre_value,
                    next_val: post_value,
                })
            }
            RAMAccess::Read(read) => {
                let read_value = read.value;
                Some(RamCycleMajorEntry {
                    row: cycle_index,
                    col: remap_address(read.address, memory_layout).unwrap() as usize,
                    ra_coeff: F::one(),
                    val_coeff: F::from_u64(read_value),
                    prev_val: read_value,
                    next_val: read_value,
                })
            }
            _ => None,
        }
    }
}

impl<F: JoltField> ReadWriteMatrixCycleMajor<F, RamCycleMajorEntry<F>> {
    /// Creates a new `ReadWriteMatrixCycleMajor` to represent the ra and Val polynomials
    /// for the RAM read/write checking sumcheck.
    #[tracing::instrument(skip_all, name = "ReadWriteMatrixCycleMajor::new")]
    pub fn new(trace: &[Cycle], val_init: Vec<F>, memory_layout: &MemoryLayout) -> Self {
        let entries: Vec<_> = trace
            .par_iter()
            .enumerate()
            .filter_map(|(j, cycle)| RamCycleMajorEntry::from_cycle(cycle, j, memory_layout))
            .collect();

        ReadWriteMatrixCycleMajor {
            entries,
            ra_lookup_table: None,
            wa_lookup_table: None,
            val_init: val_init.into(),
        }
    }
}

impl<F: JoltField> CycleMajorMatrixEntry<F> for RamCycleMajorEntry<F> {
    type AddressMajor = RamAddressMajorEntry<F>;

    fn row(&self) -> usize {
        self.row
    }

    fn column(&self) -> usize {
        self.col
    }

    fn bind_entries(
        even: Option<&Self>,
        odd: Option<&Self>,
        r: F::Challenge,
        _ra_lookup_table: Option<&OneHotCoeffLookupTable<F>>,
        _wa_lookup_table: Option<&OneHotCoeffLookupTable<F>>,
    ) -> Self {
        match (even, odd) {
            (Some(even), Some(odd)) => {
                debug_assert!(even.row.is_even());
                debug_assert!(odd.row.is_odd());
                debug_assert_eq!(even.col, odd.col);
                RamCycleMajorEntry {
                    row: even.row / 2,
                    col: even.col,
                    ra_coeff: even.ra_coeff + r.mul_0_optimized(odd.ra_coeff - even.ra_coeff),
                    val_coeff: even.val_coeff + r.mul_0_optimized(odd.val_coeff - even.val_coeff),
                    prev_val: even.prev_val,
                    next_val: odd.next_val,
                }
            }
            (Some(even), None) => {
                // For ReadWriteMatrixCycleMajor, the absence of a matrix entry implies
                // that its coeff has not been bound yet.
                // The absence of an odd-row entry in the same column as even
                // means that its implicit Val coeff is even.next_val, and its implicit
                // ra coeff is 0.
                let odd_val_coeff = F::from_u64(even.next_val);
                RamCycleMajorEntry {
                    row: even.row / 2,
                    col: even.col,
                    ra_coeff: (F::one() - r).mul_1_optimized(even.ra_coeff),
                    val_coeff: even.val_coeff + r.mul_0_optimized(odd_val_coeff - even.val_coeff),
                    prev_val: even.prev_val,
                    next_val: even.next_val,
                }
            }
            (None, Some(odd)) => {
                // For ReadWriteMatrixCycleMajor, the absence of a matrix entry implies
                // that its coeff has not been bound yet.
                // The absence of an even-row entry in the same column as odd
                // means that its implicit Val coeff is odd.prev_val, and its implicit
                // ra coeff is 0.
                let even_val_coeff = F::from_u64(odd.prev_val);
                RamCycleMajorEntry {
                    row: odd.row / 2,
                    col: odd.col,
                    ra_coeff: r.mul_1_optimized(odd.ra_coeff),
                    val_coeff: even_val_coeff + r.mul_0_optimized(odd.val_coeff - even_val_coeff),
                    prev_val: odd.prev_val,
                    next_val: odd.next_val,
                }
            }
            (None, None) => panic!("Both entries are None"),
        }
    }

    fn compute_evals(
        even: Option<&Self>,
        odd: Option<&Self>,
        inc_evals: [F; 2],
        gamma: F,
        _ra_lookup_table: Option<&OneHotCoeffLookupTable<F>>,
        _wa_lookup_table: Option<&OneHotCoeffLookupTable<F>>,
    ) -> [F::UnreducedProduct; 2] {
        match (even, odd) {
            (Some(even), Some(odd)) => {
                debug_assert!(even.row.is_even());
                debug_assert!(odd.row.is_odd());
                debug_assert_eq!(even.col, odd.col);
                let ra_evals = [even.ra_coeff, odd.ra_coeff - even.ra_coeff];
                let val_evals = [even.val_coeff, odd.val_coeff - even.val_coeff];
                [
                    ra_evals[0]
                        .mul_to_product(val_evals[0] + gamma * (inc_evals[0] + val_evals[0])),
                    ra_evals[1]
                        .mul_to_product(val_evals[1] + gamma * (inc_evals[1] + val_evals[1])),
                ]
            }
            (Some(even), None) => {
                let odd_val_coeff = F::from_u64(even.next_val);
                let ra_evals = [even.ra_coeff, -even.ra_coeff];
                let val_evals = [even.val_coeff, odd_val_coeff - even.val_coeff];
                [
                    ra_evals[0]
                        .mul_to_product(val_evals[0] + gamma * (inc_evals[0] + val_evals[0])),
                    ra_evals[1]
                        .mul_to_product(val_evals[1] + gamma * (inc_evals[1] + val_evals[1])),
                ]
            }
            (None, Some(odd)) => {
                let even_val_coeff = F::from_u64(odd.prev_val);
                let ra_evals = [F::zero(), odd.ra_coeff];
                let val_evals = [even_val_coeff, odd.val_coeff - even_val_coeff];
                [
                    F::UnreducedProduct::zero(),
                    ra_evals[1]
                        .mul_to_product(val_evals[1] + gamma * (inc_evals[1] + val_evals[1])),
                ]
            }
            (None, None) => panic!("Both entries are None"),
        }
    }

    fn to_address_major(
        self,
        _: Option<&OneHotCoeffLookupTable<F>>,
        _: Option<&OneHotCoeffLookupTable<F>>,
    ) -> Self::AddressMajor {
        RamAddressMajorEntry {
            row: self.row,
            col: self.col,
            prev_val: F::from_u64(self.prev_val),
            next_val: F::from_u64(self.next_val),
            val_coeff: self.val_coeff,
            ra_coeff: self.ra_coeff,
        }
    }
}

impl<F: JoltField> ReadWriteMatrixCycleMajor<F, RamCycleMajorEntry<F>> {
    /// Materializes the ra and Val polynomials represented by this `ReadWriteMatrixCycleMajor`.
    ///
    /// After partial binding of cycle and address variables, there are `K_prime` columns
    /// (remaining address positions) and `T_prime` rows (remaining cycle positions) in the matrix.
    /// This expands the sparse representation to dense polynomials of size `K_prime * T_prime`.
    ///
    /// The output layout is address-major: index(addr k, cycle t) = k * T_prime + t.
    /// This matches the layout expected by `phase3_compute_message`.
    ///
    /// When `T_prime == 1` (all cycle variables bound), this is equivalent to the simple case
    /// where each entry has `row == 0` and entries have distinct `col` values.
    #[tracing::instrument(skip_all, name = "ReadWriteMatrixCycleMajor::materialize")]
    pub fn materialize(
        self,
        K_prime: usize,
        T_prime: usize,
    ) -> (MultilinearPolynomial<F>, MultilinearPolynomial<F>) {
        let len = K_prime * T_prime;

        // Initialize ra to zero
        let mut ra: Vec<F> = unsafe_allocate_zero_vec(len);

        // Initialize val: expand val_init from size K to size K * T_prime
        // Each address k gets val_init[k] replicated across all T_prime cycle positions
        // Layout: address-major, index(k, t) = k * T_prime + t
        let mut val: Vec<F> = unsafe_allocate_zero_vec(len);
        // Extract the coefficient slice from self.val_init (which is a LargeScalars polynomial)
        let val_init_coeffs = match &self.val_init {
            MultilinearPolynomial::LargeScalars(poly) => &poly.Z,
            _ => panic!("val_init must be LargeScalars"),
        };
        val.par_chunks_mut(T_prime)
            .zip(val_init_coeffs.par_iter())
            .for_each(|(chunk, &v)| {
                chunk.fill(v);
            });

        // Update ra and val at positions where we have entries.
        // Index is col * T_prime + row (address-major layout).
        let ra_ptr = ra.as_mut_ptr() as usize;
        let val_ptr = val.as_mut_ptr() as usize;

        self.entries.into_par_iter().for_each(|entry| {
            debug_assert!(
                entry.row() < T_prime,
                "row {} >= T_prime {T_prime}",
                entry.row()
            );
            let idx = entry.column() * T_prime + entry.row();
            // SAFETY: Each entry has a unique (row, col) pair,
            // so writes to ra[idx] and val[idx] are disjoint across parallel iterations.
            unsafe {
                let ra_p = ra_ptr as *mut F;
                let val_p = val_ptr as *mut F;
                *ra_p.add(idx) = entry.ra_coeff;
                *val_p.add(idx) = entry.val_coeff;
            }
        });

        (ra.into(), val.into())
    }
}

/// Represents a non-zero entry in the ra(k, j) and Val(k, j) polynomials.
/// Conceptually, both ra and Val can be seen as K x T matrices.
///
/// # Type Parameters
///
/// - `F`: The field type for coefficients.
#[derive(Allocative, Debug, PartialEq, Clone, Copy, Default)]
pub struct RamAddressMajorEntry<F: JoltField> {
    /// The row index. Before binding, row \in [0, T)
    pub row: usize,
    /// The column index. Before binding, col \in [0, K)
    pub col: usize,
    /// In round i, each ReadWriteEntry represents a coefficient
    ///   Val(k, j', r)
    /// which is some combination of Val(k, j', 00...0), ...
    /// Val(k, j', 11...1).
    /// `prev_val` contains the unbound coefficient before
    /// Val(k, j', 00...0) –– abusing notation, `prev_val` is
    /// Val(k, j'-1, 11...1)
    pub(crate) prev_val: F,
    /// In round i, each ReadWriteEntry represents a coefficient
    ///   Val(k, j', r)
    /// which is some combination of Val(k, j', 00...0), ...
    /// Val(k, j', 11...1).
    /// `next_val` contains the unbound coefficient after
    /// Val(k, j', 00...0) –– abusing notation, `next_val` is
    /// Val(k, j'+1, 00...0)
    pub(crate) next_val: F,
    /// The Val coefficient for this matrix entry.
    pub val_coeff: F,
    /// The ra coefficient for this matrix entry. Note that for RAM,
    /// ra and wa are the same polynomial.
    pub ra_coeff: F,
}

impl<F: JoltField> AddressMajorMatrixEntry<F> for RamAddressMajorEntry<F> {
    fn row(&self) -> usize {
        self.row
    }

    fn column(&self) -> usize {
        self.col
    }

    fn prev_val(&self) -> F {
        self.prev_val
    }

    fn next_val(&self) -> F {
        self.next_val
    }

    fn bind_entries(
        even: Option<&Self>,
        odd: Option<&Self>,
        even_checkpoint: F,
        odd_checkpoint: F,
        r: F::Challenge,
    ) -> Self {
        match (even, odd) {
            (Some(even), Some(odd)) => {
                debug_assert!(even.col.is_even());
                debug_assert!(odd.col.is_odd());
                debug_assert_eq!(even.row, odd.row);
                RamAddressMajorEntry {
                    row: even.row,
                    col: even.col / 2,
                    ra_coeff: even.ra_coeff + r.mul_0_optimized(odd.ra_coeff - even.ra_coeff),
                    val_coeff: even.val_coeff + r.mul_0_optimized(odd.val_coeff - even.val_coeff),
                    prev_val: even.prev_val + r.mul_0_optimized(odd.prev_val - even.prev_val),
                    next_val: even.next_val + r.mul_0_optimized(odd.next_val - even.next_val),
                }
            }
            (Some(even), None) => {
                // For SparseMatrixPolynomial, the absence of a matrix entry implies
                // that its coeff has not been bound yet.
                // The absence of an odd-col entry in the same row as even
                // means that its implicit Val coeff is odd_checkpoint, and its implicit
                // ra coeff is 0.
                RamAddressMajorEntry {
                    row: even.row,
                    col: even.col / 2,
                    ra_coeff: (F::one() - r).mul_1_optimized(even.ra_coeff),
                    val_coeff: even.val_coeff + r.mul_0_optimized(odd_checkpoint - even.val_coeff),
                    prev_val: even.prev_val + r.mul_0_optimized(odd_checkpoint - even.prev_val),
                    next_val: even.next_val + r.mul_0_optimized(odd_checkpoint - even.next_val),
                }
            }
            (None, Some(odd)) => {
                // For SparseMatrixPolynomial, the absence of a matrix entry implies
                // that its coeff has not been bound yet.
                // The absence of an even-col entry in the same row as odd
                // means that its implicit Val coeff is even_checkpoint, and its implicit
                // ra coeff is 0.
                RamAddressMajorEntry {
                    row: odd.row,
                    col: odd.col / 2,
                    ra_coeff: r.mul_1_optimized(odd.ra_coeff),
                    val_coeff: even_checkpoint + r.mul_0_optimized(odd.val_coeff - even_checkpoint),
                    prev_val: even_checkpoint + r.mul_0_optimized(odd.prev_val - even_checkpoint),
                    next_val: even_checkpoint + r.mul_0_optimized(odd.next_val - even_checkpoint),
                }
            }
            (None, None) => panic!("Both entries are None"),
        }
    }

    fn compute_evals(
        even: Option<&Self>,
        odd: Option<&Self>,
        even_checkpoint: F,
        odd_checkpoint: F,
        inc_eval: F,
        eq_eval: F,
        gamma: F,
    ) -> [F::UnreducedProduct; 2] {
        match (even, odd) {
            (Some(even), Some(odd)) => {
                debug_assert!(even.col.is_even());
                debug_assert!(odd.col.is_odd());
                debug_assert_eq!(even.row, odd.row);
                let ra_evals = [even.ra_coeff, odd.ra_coeff + odd.ra_coeff - even.ra_coeff];
                let val_evals = [
                    even.val_coeff,
                    odd.val_coeff + odd.val_coeff - even.val_coeff,
                ];
                [
                    eq_eval.mul_to_product(
                        ra_evals[0] * (val_evals[0] + gamma * (inc_eval + val_evals[0])),
                    ),
                    eq_eval.mul_to_product(
                        ra_evals[1] * (val_evals[1] + gamma * (inc_eval + val_evals[1])),
                    ),
                ]
            }
            (Some(even), None) => {
                let ra_evals = [even.ra_coeff, -even.ra_coeff];
                let val_evals = [
                    even.val_coeff,
                    odd_checkpoint + odd_checkpoint - even.val_coeff,
                ];
                [
                    eq_eval.mul_to_product(
                        ra_evals[0] * (val_evals[0] + gamma * (inc_eval + val_evals[0])),
                    ),
                    eq_eval.mul_to_product(
                        ra_evals[1] * (val_evals[1] + gamma * (inc_eval + val_evals[1])),
                    ),
                ]
            }
            (None, Some(odd)) => {
                let ra_evals = [F::zero(), odd.ra_coeff + odd.ra_coeff];
                let val_evals = [
                    even_checkpoint,
                    odd.val_coeff + odd.val_coeff - even_checkpoint,
                ];
                [
                    F::UnreducedProduct::zero(),
                    eq_eval.mul_to_product(
                        ra_evals[1] * (val_evals[1] + gamma * (inc_eval + val_evals[1])),
                    ),
                ]
            }
            (None, None) => panic!("Both entries are None"),
        }
    }
}

impl<F: JoltField> ReadWriteMatrixAddressMajor<F, RamAddressMajorEntry<F>> {
    /// Materializes the ra and Val polynomials represented by this `ReadWriteMatrixAddressMajor`.
    /// Some number of cycle and address variables have already been bound, so at this point
    /// there are `K_prime` columns and `T_prime` rows left in the matrix.
    #[tracing::instrument(skip_all, name = "ReadWriteMatrixAddressMajor::materialize")]
    pub fn materialize(
        self,
        K_prime: usize,
        T_prime: usize,
    ) -> (MultilinearPolynomial<F>, MultilinearPolynomial<F>) {
        let total = K_prime * T_prime;

        // SAFETY: Each column chunk writes to a disjoint index range
        // `[k * T_prime .. (k+1) * T_prime]`, so no data races occur.
        struct SyncUnsafeSlice<T>(Vec<UnsafeCell<T>>);
        unsafe impl<T: Send> Sync for SyncUnsafeSlice<T> {}
        impl<T: Send> SyncUnsafeSlice<T> {
            fn new(v: Vec<T>) -> Self {
                Self(v.into_iter().map(UnsafeCell::new).collect())
            }
            unsafe fn set(&self, idx: usize, val: T) {
                *self.0[idx].get() = val;
            }
            fn into_vec(self) -> Vec<T> {
                self.0.into_iter().map(|cell| cell.into_inner()).collect()
            }
        }

        let ra = SyncUnsafeSlice::new(vec![F::zero(); total]);
        let val = SyncUnsafeSlice::new(vec![F::zero(); total]);

        self.entries
            .par_chunk_by(|a, b| a.column() == b.column())
            .for_each(|column| {
                let k = column[0].column();
                let mut current_val_coeff = self.val_init.get_bound_coeff(k);
                let mut column_iter = column.iter().peekable();
                for j in 0..T_prime {
                    let idx = k * T_prime + j;
                    if let Some(entry) = column_iter.next_if(|&entry| entry.row() == j) {
                        unsafe {
                            ra.set(idx, entry.ra_coeff);
                            val.set(idx, entry.val_coeff);
                        }
                        current_val_coeff = entry.next_val();
                        continue;
                    }
                    unsafe { val.set(idx, current_val_coeff) };
                }
            });

        (ra.into_vec().into(), val.into_vec().into())
    }
}
