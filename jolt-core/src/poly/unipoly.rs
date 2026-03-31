use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use std::cmp::Ordering;
use std::iter::zip;
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub};

use crate::poly::lagrange_poly::LagrangeHelper;
use allocative::Allocative;
use ark_serialize::*;
use rand_core::{CryptoRng, RngCore};

use super::multilinear_polynomial::MultilinearPolynomial;
use crate::utils::small_scalar::SmallScalar;

// ax^2 + bx + c stored as vec![c,b,a]
// ax^3 + bx^2 + cx + d stored as vec![d,c,b,a]
#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone, PartialEq, Allocative)]
pub struct UniPoly<F: CanonicalSerialize + CanonicalDeserialize> {
    pub coeffs: Vec<F>,
}

// ax^2 + bx + c stored as vec![c,a]
// ax^3 + bx^2 + cx + d stored as vec![d,b,a]
#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct CompressedUniPoly<F: JoltField> {
    pub coeffs_except_linear_term: Vec<F>,
}

impl<F: JoltField> UniPoly<F> {
    pub fn from_coeff(coeffs: Vec<F>) -> Self {
        UniPoly { coeffs }
    }

    /// Interpolate a polynomial from its evaluations at the points 0, 1, 2, ..., n-1.
    pub fn from_evals(evals: &[F]) -> Self {
        match evals.len() {
            0 => UniPoly {
                coeffs: vec![F::zero()],
            },
            1 => UniPoly {
                coeffs: vec![evals[0]],
            },
            2 => {
                let c0 = evals[0];
                let c1 = evals[1] - evals[0];
                UniPoly {
                    coeffs: vec![c0, c1],
                }
            }
            3 => Self::from_evals_deg2(evals),
            4 => Self::from_evals_deg3(evals),
            _ => UniPoly {
                coeffs: Self::newton_interpolation(evals),
            },
        }
    }

    /// Degree-2 interpolation from evaluations at x = 0, 1, 2.
    /// Uses Newton forward differences to avoid Gaussian elimination.
    fn from_evals_deg2(evals: &[F]) -> Self {
        let (e0, e1, e2) = (evals[0], evals[1], evals[2]);
        let d1 = e1 - e0;
        let d2 = e2 - e1 - d1; // e2 - 2*e1 + e0
        let inv2 = F::from_u64(2).inverse().unwrap();
        let c2 = d2 * inv2;
        let c1 = d1 - c2;
        UniPoly {
            coeffs: vec![e0, c1, c2],
        }
    }

    /// Degree-3 interpolation from evaluations at x = 0, 1, 2, 3.
    /// Uses Newton forward differences to avoid Gaussian elimination.
    fn from_evals_deg3(evals: &[F]) -> Self {
        let (e0, e1, e2, e3) = (evals[0], evals[1], evals[2], evals[3]);
        let d1 = e1 - e0;
        let d2 = e2 - e1 - d1; // Δ² = e2 - 2*e1 + e0
        let d12 = e2 - e1;
        let d23 = e3 - e2;
        let d3 = d23 - d12 - (d12 - d1); // Δ³ = e3 - 3*e2 + 3*e1 - e0
        let inv6 = F::from_u64(6).inverse().unwrap();
        let inv2 = inv6 + inv6 + inv6; // 3/6 = 1/2
        let inv3 = inv6 + inv6; // 2/6 = 1/3
        let c3 = d3 * inv6; // Δ³/6
        let c2 = d2 * inv2 - d3 * inv2; // Δ²/2 - Δ³/2
        let c1 = d1 - d2 * inv2 + d3 * inv3; // Δ¹ - Δ²/2 + Δ³/3
        UniPoly {
            coeffs: vec![e0, c1, c2, c3],
        }
    }

    /// Interpolate a polynomial `p(x)` from its evaluations at even points `0, 2, 3, ..., n-1`
    /// and a hint `p(0) + p(1)`.
    pub fn from_evals_and_hint(hint: F, evals: &[F]) -> Self {
        let eval_at_1 = hint - evals[0];
        match evals.len() {
            1 => Self::from_evals(&[evals[0], eval_at_1]),
            2 => Self::from_evals(&[evals[0], eval_at_1, evals[1]]),
            3 => Self::from_evals(&[evals[0], eval_at_1, evals[1], evals[2]]),
            _ => {
                let mut full = Vec::with_capacity(evals.len() + 1);
                full.push(evals[0]);
                full.push(eval_at_1);
                full.extend_from_slice(&evals[1..]);
                Self::from_evals(&full)
            }
        }
    }

    /// Interpolates a polynomial from its evaluations on `[0, 1, ..., degree - 1, inf]`.
    pub fn from_evals_toom(evals: &[F]) -> Self {
        match evals.len() {
            3 => {
                // Degree 2: f(0), f(1), f(∞)
                let (e0, e1, e_inf) = (evals[0], evals[1], evals[2]);
                return UniPoly {
                    coeffs: vec![e0, e1 - e0 - e_inf, e_inf],
                };
            }
            4 => {
                // Degree 3: f(0), f(1), f(2), f(∞)
                let (e0, e1, e2, e_inf) = (evals[0], evals[1], evals[2], evals[3]);
                let inv2 = F::from_u64(2).inverse().unwrap();
                let e_inf_6 = e_inf + e_inf + e_inf + e_inf + e_inf + e_inf;
                let c2 = (e2 + e0 - e1 - e1 - e_inf_6) * inv2;
                let c1 = e1 - e0 - e_inf - c2;
                return UniPoly {
                    coeffs: vec![e0, c1, c2, e_inf],
                };
            }
            _ => {}
        }

        // The leading coefficient equals the evaluation at infinity.
        let n = evals.len();
        let leading = evals[n - 1];

        // Subtract leading * x^{n-1} from the finite evaluations to reduce
        // to a degree-(n-2) interpolation at points 0, 1, ..., n-2.
        let mut reduced = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            let x_pow = Self::pow_u64(i as u64, n - 1);
            reduced.push(evals[i] - leading * x_pow);
        }

        let mut coeffs = Self::newton_interpolation(&reduced);
        coeffs.push(leading);
        UniPoly { coeffs }
    }

    /// O(n²) interpolation via Newton divided differences for equispaced points 0,1,...,n-1.
    /// Replaces O(n³) Gaussian elimination for the general case.
    fn newton_interpolation(evals: &[F]) -> Vec<F> {
        let n = evals.len();
        if n == 0 {
            return vec![F::zero()];
        }
        if n == 1 {
            return vec![evals[0]];
        }

        // Step 1: Compute divided differences for equispaced points.
        // After processing, dd[k] = f[0, 1, ..., k] (k-th divided difference).
        let mut dd: Vec<F> = evals.to_vec();
        for k in 1..n {
            for i in (k..n).rev() {
                dd[i] = dd[i] - dd[i - 1];
            }
            let inv_k = F::from_u64(k as u64).inverse().unwrap();
            for i in k..n {
                dd[i] *= inv_k;
            }
        }

        // Step 2: Convert Newton basis to monomial basis using Horner's method.
        // p(x) = dd[0] + dd[1]*x + dd[2]*x*(x-1) + dd[3]*x*(x-1)*(x-2) + ...
        let mut coeffs = vec![F::zero(); n];
        coeffs[0] = dd[n - 1];
        for k in (0..n - 1).rev() {
            let shift = F::from_u64(k as u64);
            for i in (1..n).rev() {
                coeffs[i] = coeffs[i - 1] - shift * coeffs[i];
            }
            coeffs[0] = dd[k] - shift * coeffs[0];
        }

        coeffs
    }

    /// Compute base^exp as a field element using repeated squaring.
    fn pow_u64(base: u64, exp: usize) -> F {
        if exp == 0 {
            return F::one();
        }
        let mut result = F::one();
        let b = F::from_u64(base);
        let mut current = b;
        let mut e = exp;
        while e > 0 {
            if e & 1 == 1 {
                result *= current;
            }
            current = current * current;
            e >>= 1;
        }
        result
    }

    /// Divide self by another polynomial, and returns the
    /// quotient and remainder.
    #[tracing::instrument(skip_all, name = "UniPoly::divide_with_remainder")]
    pub fn divide_with_remainder(&self, divisor: &Self) -> Option<(Self, Self)> {
        if self.is_zero() {
            Some((Self::zero(), Self::zero()))
        } else if divisor.is_zero() {
            None
        } else if self.degree() < divisor.degree() {
            Some((Self::zero(), self.clone()))
        } else {
            // Now we know that self.degree() >= divisor.degree();
            let mut quotient = vec![F::zero(); self.degree() - divisor.degree() + 1];
            let mut remainder: Self = self.clone();
            // Can unwrap here because we know self is not zero.
            let divisor_leading_inv = divisor.leading_coefficient().unwrap().inverse().unwrap();
            while !remainder.is_zero() && remainder.degree() >= divisor.degree() {
                let cur_q_coeff = *remainder.leading_coefficient().unwrap() * divisor_leading_inv;
                let cur_q_degree = remainder.degree() - divisor.degree();
                quotient[cur_q_degree] = cur_q_coeff;

                for (i, div_coeff) in divisor.coeffs.iter().enumerate() {
                    remainder.coeffs[cur_q_degree + i] -= cur_q_coeff * *div_coeff;
                }

                while let Some(true) = remainder.coeffs.last().map(|c| c == &F::zero()) {
                    remainder.coeffs.pop();
                }
            }
            Some((Self::from_coeff(quotient), remainder))
        }
    }

    fn is_zero(&self) -> bool {
        self.coeffs.is_empty() || self.coeffs.iter().all(|c| c == &F::zero())
    }

    fn leading_coefficient(&self) -> Option<&F> {
        self.coeffs.last()
    }

    pub fn zero() -> Self {
        Self::from_coeff(Vec::new())
    }

    pub fn degree(&self) -> usize {
        self.coeffs.len() - 1
    }

    pub fn as_vec(&self) -> Vec<F> {
        self.coeffs.clone()
    }

    pub fn eval_at_zero(&self) -> F {
        self.coeffs[0]
    }

    pub fn eval_at_one(&self) -> F {
        (0..self.coeffs.len()).map(|i| self.coeffs[i]).sum()
    }

    pub fn evaluate<C>(&self, r: &C) -> F
    where
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        Self::eval_with_coeffs(&self.coeffs, r)
    }

    pub fn eval_with_coeffs<C>(coeffs: &[F], r: &C) -> F
    where
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let mut eval = coeffs[0];
        let mut power = (*r).into();
        for i in 1..coeffs.len() {
            eval += power * coeffs[i];

            #[allow(clippy::assign_op_pattern)]
            {
                power = power * *r;
            }
        }
        eval
    }

    #[tracing::instrument(skip_all, name = "UniPoly::eval_as_univariate")]
    pub fn eval_as_univariate(poly: &MultilinearPolynomial<F>, r: &F) -> F {
        match poly {
            MultilinearPolynomial::LargeScalars(poly) => {
                let mut eval = poly.Z[0];
                let mut power = *r;
                for coeff in poly.evals_ref()[1..].iter() {
                    eval += power * coeff;
                    power *= *r;
                }
                eval
            }
            MultilinearPolynomial::U8Scalars(poly) => {
                let mut eval = F::zero();
                let mut power = F::one();
                for coeff in poly.coeffs.iter() {
                    eval += coeff.field_mul(power);
                    power *= *r;
                }
                eval
            }
            MultilinearPolynomial::U16Scalars(poly) => {
                let mut eval = F::zero();
                let mut power = F::one();
                for coeff in poly.coeffs.iter() {
                    eval += coeff.field_mul(power);
                    power *= *r;
                }
                eval
            }
            MultilinearPolynomial::U32Scalars(poly) => {
                let mut eval = F::zero();
                let mut power = F::one();
                for coeff in poly.coeffs.iter() {
                    eval += coeff.field_mul(power);
                    power *= *r;
                }
                eval
            }
            MultilinearPolynomial::U64Scalars(poly) => {
                let mut eval = F::zero();
                let mut power = F::one();
                for coeff in poly.coeffs.iter() {
                    eval += coeff.field_mul(power);
                    power *= *r;
                }
                eval
            }
            MultilinearPolynomial::I64Scalars(poly) => {
                let mut eval = F::zero();
                let mut power = F::one();
                for coeff in poly.coeffs.iter() {
                    eval += coeff.field_mul(power);
                    power *= *r;
                }
                eval
            }
            _ => unimplemented!("Unsupported MultilinearPolynomial variant"),
        }
    }

    pub fn compress(&self) -> CompressedUniPoly<F> {
        let mut coeffs_except_linear_term = Vec::with_capacity(self.coeffs.len() - 1);
        coeffs_except_linear_term.push(self.coeffs[0]);
        coeffs_except_linear_term.extend_from_slice(&self.coeffs[2..]);
        debug_assert_eq!(coeffs_except_linear_term.len() + 1, self.coeffs.len());
        CompressedUniPoly {
            coeffs_except_linear_term,
        }
    }

    pub fn random<R: RngCore + CryptoRng>(num_vars: usize, mut rng: &mut R) -> Self {
        Self::from_coeff(
            std::iter::from_fn(|| Some(F::random(&mut rng)))
                .take(num_vars)
                .collect(),
        )
    }

    pub fn shift_coefficients(&mut self, rhs: &F) {
        for c in &mut self.coeffs {
            *c += *rhs;
        }
    }

    /// This function computes a cubic polynomial s(X), given the following conditions:
    /// - s(X) = l(X) * t(X), where l(X) is linear and t(X) is quadratic,
    /// - l(X) = a + bX is given by l(0) = a and l(\infty) = b,
    /// - t(X) = c + dX + eX^2 is given by t(0) = c and t(\infty) = e (but d is missing),
    /// - s(0) + s(1) = hint,
    ///
    /// This is used in the optimized sum-check evaluation with split eq polynomial.
    pub fn from_linear_times_quadratic_with_hint(
        linear_coeffs: [F; 2],
        quadratic_coeff_0: F,
        quadratic_coeff_2: F,
        hint: F,
    ) -> Self {
        let linear_eval_one = linear_coeffs[0] + linear_coeffs[1];

        let cubic_coeff_0 = linear_coeffs[0] * quadratic_coeff_0;

        // Compute the linear coefficient of the quadratic polynomial from the hint
        // Given that s(0) + s(1) = hint, we can rewrite this as:
        // a * c + (a + b) * (c + d + e) = hint, which means we can solve for d as:
        // d = (hint - a * c) / (a + b) - c - e
        let quadratic_coeff_1 =
            (hint - cubic_coeff_0) / linear_eval_one - quadratic_coeff_0 - quadratic_coeff_2;

        // Now derive the coefficients of the cubic polynomial from the evaluations
        // We have s(X) = (a + bX) * (c + dX + eX^2) = ac + (ad + bc)X + (ae + bd)X^2 + beX^3
        let coeffs = [
            cubic_coeff_0,
            linear_coeffs[0] * quadratic_coeff_1 + linear_coeffs[1] * quadratic_coeff_0,
            linear_coeffs[0] * quadratic_coeff_2 + linear_coeffs[1] * quadratic_coeff_1,
            linear_coeffs[1] * quadratic_coeff_2,
        ];
        Self::from_coeff(coeffs.to_vec())
    }

    /// Evaluate on a symmetric integer domain of size N and verify a domain-sum constraint.
    ///
    /// Contract/assumptions:
    /// - Domain nodes are consecutive integers centered at 0: t_i = start + i where
    ///   start = -floor((N-1)/2) and i ∈ {0..N-1}.
    /// - N ≤ 16 (univariate-skip setting). For k ≤ N-1, the power sums S_k = Σ_i t_i^k fit in i64,
    ///   and are computed via `LagrangeHelper::power_sums::<N>()`.
    /// - `self` is the full, uncompressed univariate polynomial s(X) with monomial coefficients
    ///   of degree exactly N-1 (asserted in debug builds).
    /// - `claim` is the expected field value of Σ_{t in domain} s(t). In the first outer round of
    ///   Spartan, this `claim` is zero.
    ///
    /// Behavior:
    /// - Computes Σ_{t in domain} s(t) = Σ_j a_j · S_j using i64 power sums and checks equality to `claim`.
    ///
    /// Returns:
    /// - true iff the domain-sum equals `claim`
    pub fn check_sum_evals<const N: usize, const OUT_LEN: usize>(&self, claim: F) -> bool {
        // Relaxed: compute Σ_{t in symmetric N-window} s(t) via i128 power sums up to deg(s)
        debug_assert_eq!(self.degree() + 1, OUT_LEN);
        let power_sums = LagrangeHelper::power_sums::<N, OUT_LEN>();

        // Check domain sum Σ_j a_j * S_j == claim
        let mut sum = F::zero();
        for (j, coeff) in self.coeffs.iter().enumerate() {
            sum += coeff.mul_i128(power_sums[j]);
        }
        sum == claim
    }
}

impl<F: JoltField> AddAssign<&Self> for UniPoly<F> {
    fn add_assign(&mut self, rhs: &Self) {
        let ordering = self.coeffs.len().cmp(&rhs.coeffs.len());
        #[allow(clippy::disallowed_methods)]
        for (lhs, rhs) in self.coeffs.iter_mut().zip(&rhs.coeffs) {
            *lhs += *rhs;
        }
        if matches!(ordering, Ordering::Less) {
            self.coeffs
                .extend(rhs.coeffs[self.coeffs.len()..].iter().cloned());
        }
    }
}

impl<F: JoltField> Add for &UniPoly<F> {
    type Output = UniPoly<F>;

    fn add(self, rhs: Self) -> UniPoly<F> {
        let mut coeffs = vec![F::zero(); self.coeffs.len().max(rhs.coeffs.len())];
        zip(&mut coeffs, &self.coeffs).for_each(|(acc, lhs)| *acc += *lhs);
        zip(&mut coeffs, &rhs.coeffs).for_each(|(acc, rhs)| *acc += *rhs);
        UniPoly { coeffs }
    }
}

impl<F: JoltField> Sub for UniPoly<F> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        let ordering = self.coeffs.len().cmp(&rhs.coeffs.len());
        #[allow(clippy::disallowed_methods)]
        for (lhs, rhs) in self.coeffs.iter_mut().zip(&rhs.coeffs) {
            *lhs -= *rhs;
        }
        if matches!(ordering, Ordering::Less) {
            self.coeffs
                .extend(rhs.coeffs[self.coeffs.len()..].iter().map(|v| v.neg()));
        }
        self
    }
}

impl<F: JoltField> Mul<F> for UniPoly<F> {
    type Output = Self;

    fn mul(mut self, rhs: F) -> Self {
        // UniPoly coefficients are typically degree 2-5; sequential is faster than par_iter
        for c in &mut self.coeffs {
            *c *= rhs;
        }
        self
    }
}

impl<F: JoltField> Mul<&F> for UniPoly<F> {
    type Output = Self;

    fn mul(mut self, rhs: &F) -> Self {
        for c in &mut self.coeffs {
            *c *= *rhs;
        }
        self
    }
}

impl<F: JoltField> Mul<F> for &UniPoly<F> {
    type Output = UniPoly<F>;

    fn mul(self, rhs: F) -> UniPoly<F> {
        UniPoly::from_coeff(self.coeffs.iter().map(|c| *c * rhs).collect::<Vec<_>>())
    }
}

impl<F: JoltField> Index<usize> for UniPoly<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coeffs[index]
    }
}

impl<F: JoltField> IndexMut<usize> for UniPoly<F> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coeffs[index]
    }
}

impl<F: JoltField> MulAssign<&F> for UniPoly<F> {
    fn mul_assign(&mut self, rhs: &F) {
        for c in &mut self.coeffs {
            *c *= *rhs;
        }
    }
}

impl<F: JoltField> CompressedUniPoly<F> {
    pub fn get_compressed_coeffs(&self) -> &[F] {
        &self.coeffs_except_linear_term
    }

    // we require eval(0) + eval(1) = hint, so we can solve for the linear term as:
    // linear_term = hint - 2 * constant_term - deg2 term - deg3 term
    pub fn decompress(&self, hint: &F) -> UniPoly<F> {
        let mut linear_term =
            *hint - self.coeffs_except_linear_term[0] - self.coeffs_except_linear_term[0];
        for i in 1..self.coeffs_except_linear_term.len() {
            linear_term -= self.coeffs_except_linear_term[i];
        }

        let mut coeffs = vec![self.coeffs_except_linear_term[0], linear_term];
        coeffs.extend(&self.coeffs_except_linear_term[1..]);
        assert_eq!(self.coeffs_except_linear_term.len() + 1, coeffs.len());
        UniPoly { coeffs }
    }

    // In the verifier we do not have to check that f(0) + f(1) = hint as we can just
    // recover the linear term assuming the prover did it right, then eval the poly
    pub fn eval_from_hint(&self, hint: &F, x: &F::Challenge) -> F {
        let mut linear_term =
            *hint - self.coeffs_except_linear_term[0] - self.coeffs_except_linear_term[0];
        for i in 1..self.coeffs_except_linear_term.len() {
            linear_term -= self.coeffs_except_linear_term[i];
        }

        let mut running_point: F = (*x).into();
        let mut running_sum = self.coeffs_except_linear_term[0] + *x * linear_term;
        for i in 1..self.coeffs_except_linear_term.len() {
            running_point = running_point * x;
            running_sum += self.coeffs_except_linear_term[i] * running_point;
        }
        running_sum
    }

    pub fn degree(&self) -> usize {
        self.coeffs_except_linear_term.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[test]
    fn test_from_evals_toom() {
        // Our degree 3 polynomial is: 5 + x + 3x^2 + 9x^3.
        let gt_poly = UniPoly::<Fr>::from_coeff(vec![5.into(), 1.into(), 3.into(), 9.into()]);
        let degree = 3;
        let finite_evals = (0..degree)
            .map(|x| gt_poly.evaluate::<Fr>(&x.into()))
            .collect();
        let eval_at_infinity = *gt_poly.coeffs.last().unwrap();
        let toom_evals = [finite_evals, vec![eval_at_infinity]].concat();

        let poly = UniPoly::from_evals_toom(&toom_evals);

        assert_eq!(gt_poly, poly);
    }

    #[test]
    fn test_from_evals_quad() {
        test_from_evals_quad_helper::<Fr>()
    }

    fn test_from_evals_quad_helper<F: JoltField>() {
        // polynomial is 2x^2 + 3x + 1
        let e0 = F::one();
        let e1 = F::from_u64(6u64);
        let e2 = F::from_u64(15u64);
        let evals = vec![e0, e1, e2];
        let poly = UniPoly::from_evals(&evals);

        assert_eq!(poly.eval_at_zero(), e0);
        assert_eq!(poly.eval_at_one(), e1);
        assert_eq!(poly.coeffs.len(), 3);
        assert_eq!(poly.coeffs[0], F::one());
        assert_eq!(poly.coeffs[1], F::from_u64(3u64));
        assert_eq!(poly.coeffs[2], F::from_u64(2u64));

        let hint = e0 + e1;
        let compressed_poly = poly.compress();
        let decompressed_poly = compressed_poly.decompress(&hint);
        for i in 0..decompressed_poly.coeffs.len() {
            assert_eq!(decompressed_poly.coeffs[i], poly.coeffs[i]);
        }

        let e3 = F::from_u64(28u64);
        assert_eq!(poly.evaluate::<F>(&F::from_u64(3u64)), e3);
    }

    #[test]
    fn test_from_evals_cubic() {
        test_from_evals_cubic_helper::<Fr>()
    }
    fn test_from_evals_cubic_helper<F: JoltField>() {
        // polynomial is x^3 + 2x^2 + 3x + 1
        let e0 = F::one();
        let e1 = F::from_u64(7u64);
        let e2 = F::from_u64(23u64);
        let e3 = F::from_u64(55u64);
        let evals = vec![e0, e1, e2, e3];
        let poly = UniPoly::from_evals(&evals);

        assert_eq!(poly.eval_at_zero(), e0);
        assert_eq!(poly.eval_at_one(), e1);
        assert_eq!(poly.coeffs.len(), 4);
        assert_eq!(poly.coeffs[0], F::one());
        assert_eq!(poly.coeffs[1], F::from_u64(3u64));
        assert_eq!(poly.coeffs[2], F::from_u64(2u64));
        assert_eq!(poly.coeffs[3], F::one());

        let hint = e0 + e1;
        let compressed_poly = poly.compress();
        let decompressed_poly = compressed_poly.decompress(&hint);
        for i in 0..decompressed_poly.coeffs.len() {
            assert_eq!(decompressed_poly.coeffs[i], poly.coeffs[i]);
        }

        let e4 = F::from_u64(109u64);
        assert_eq!(poly.evaluate::<F>(&F::from_u64(4u64)), e4);
    }

    pub fn naive_mul<F: JoltField>(ours: &UniPoly<F>, other: &UniPoly<F>) -> UniPoly<F> {
        if ours.is_zero() || other.is_zero() {
            UniPoly::zero()
        } else {
            let mut result = vec![F::zero(); ours.degree() + other.degree() + 1];
            for (i, self_coeff) in ours.coeffs.iter().enumerate() {
                for (j, other_coeff) in other.coeffs.iter().enumerate() {
                    result[i + j] += *self_coeff * *other_coeff;
                }
            }
            UniPoly::from_coeff(result)
        }
    }

    #[test]
    fn test_divide_poly() {
        let rng = &mut ChaCha20Rng::from_seed([0u8; 32]);

        for a_degree in 0..50 {
            for b_degree in 0..50 {
                let dividend = UniPoly::<Fr>::random(a_degree, rng);
                let divisor = UniPoly::<Fr>::random(b_degree, rng);

                if let Some((quotient, remainder)) =
                    UniPoly::divide_with_remainder(&dividend, &divisor)
                {
                    let mut prod = naive_mul(&divisor, &quotient);
                    prod += &remainder;
                    assert_eq!(dividend, prod)
                }
            }
        }
    }

    #[test]
    fn test_newton_interpolation_degree5() {
        // p(x) = 2x^4 + x^3 - 3x^2 + 5x + 7
        let gt_poly = UniPoly::<Fr>::from_coeff(vec![
            7.into(),
            5.into(),
            -Fr::from_u64(3),
            1.into(),
            2.into(),
        ]);
        let evals: Vec<Fr> = (0..5)
            .map(|x| gt_poly.evaluate::<Fr>(&Fr::from_u64(x)))
            .collect();
        let poly = UniPoly::from_evals(&evals);
        assert_eq!(poly.coeffs.len(), gt_poly.coeffs.len());
        for i in 0..poly.coeffs.len() {
            assert_eq!(poly.coeffs[i], gt_poly.coeffs[i]);
        }
    }

    #[test]
    fn test_newton_interpolation_degree8() {
        let rng = &mut ChaCha20Rng::from_seed([42u8; 32]);
        // random(9) => 9 coefficients => degree 8
        let gt_poly = UniPoly::<Fr>::random(9, rng);
        let evals: Vec<Fr> = (0..9)
            .map(|x| gt_poly.evaluate::<Fr>(&Fr::from_u64(x)))
            .collect();
        let poly = UniPoly::from_evals(&evals);
        for i in 0..poly.coeffs.len() {
            assert_eq!(poly.coeffs[i], gt_poly.coeffs[i]);
        }
    }

    #[test]
    fn test_toom_interpolation_degree8() {
        let rng = &mut ChaCha20Rng::from_seed([99u8; 32]);
        // random(9) => 9 coefficients => degree 8
        let gt_poly = UniPoly::<Fr>::random(9, rng);
        let degree = 8;
        let finite_evals: Vec<Fr> = (0..degree)
            .map(|x| gt_poly.evaluate::<Fr>(&Fr::from_u64(x)))
            .collect();
        let eval_at_infinity = *gt_poly.coeffs.last().unwrap();
        let toom_evals: Vec<Fr> = [finite_evals, vec![eval_at_infinity]].concat();
        let poly = UniPoly::from_evals_toom(&toom_evals);
        assert_eq!(poly.coeffs.len(), gt_poly.coeffs.len());
        for i in 0..poly.coeffs.len() {
            assert_eq!(poly.coeffs[i], gt_poly.coeffs[i]);
        }
    }

    #[test]
    fn test_toom_interpolation_degree16() {
        let rng = &mut ChaCha20Rng::from_seed([77u8; 32]);
        // random(17) => 17 coefficients => degree 16
        let gt_poly = UniPoly::<Fr>::random(17, rng);
        let degree = 16;
        let finite_evals: Vec<Fr> = (0..degree)
            .map(|x| gt_poly.evaluate::<Fr>(&Fr::from_u64(x)))
            .collect();
        let eval_at_infinity = *gt_poly.coeffs.last().unwrap();
        let toom_evals: Vec<Fr> = [finite_evals, vec![eval_at_infinity]].concat();
        let poly = UniPoly::from_evals_toom(&toom_evals);
        assert_eq!(poly.coeffs.len(), gt_poly.coeffs.len());
        for i in 0..poly.coeffs.len() {
            assert_eq!(poly.coeffs[i], gt_poly.coeffs[i]);
        }
    }

    #[test]
    fn test_from_linear_times_quadratic_with_hint() {
        // polynomial is s(x) = (x + 1) * (x^2 + 2x + 3) = x^3 + 3x^2 + 5x + 3
        // hint = s(0) + s(1) = 3 + (1 + 3 + 5 + 3) = 15
        let linear_coeffs = [Fr::from_u64(1u64), Fr::from_u64(1u64)];
        let quadratic_coeff_0 = Fr::from_u64(3u64);
        let quadratic_coeff_2 = Fr::from_u64(1u64);
        let true_poly = UniPoly::from_coeff(vec![
            Fr::from_u64(3u64),
            Fr::from_u64(5u64),
            Fr::from_u64(3u64),
            Fr::from_u64(1u64),
        ]);
        let hint = Fr::from_u64(15u64);
        let poly = UniPoly::from_linear_times_quadratic_with_hint(
            linear_coeffs,
            quadratic_coeff_0,
            quadratic_coeff_2,
            hint,
        );
        assert_eq!(poly.coeffs, true_poly.coeffs);
    }
}
