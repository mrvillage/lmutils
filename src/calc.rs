use std::{convert::identity, io::Write, ops::SubAssign};

use faer::{
    diag::Diag,
    get_global_parallelism,
    linalg::solvers::{DenseSolveCore, Solve},
    mat::AsMatRef,
    Col, ColMut, ColRef, Mat, MatMut, MatRef, RowMut, Side,
};
use pulp::{Arch, Simd, WithSimd};
use rand_distr::{Distribution, StandardNormal};
use rayon::{iter::IntoParallelIterator, prelude::*};
use statrs::distribution::{ContinuousCDF, StudentsT};
use tracing::{debug, error, trace, warn};

use crate::{mean, variance};

pub static mut DISABLE_PREDICTED: bool = false;

pub fn should_disable_predicted() -> bool {
    if cfg!(test) {
        return false;
    }
    // this allows us to override this for internal calls
    if unsafe { DISABLE_PREDICTED } {
        return true;
    }
    let enabled = std::env::var("LMUTILS_ENABLE_PREDICTED").is_ok();
    let disabled = std::env::var("LMUTILS_DISABLE_PREDICTED").is_ok();
    // disabled overrides
    if disabled {
        return true;
    }
    // only if enabled is set do we enable predicted
    if enabled {
        return false;
    }
    true
}

#[derive(Debug, Clone)]
pub struct R2 {
    pub(crate) r2: f64,
    pub(crate) adj_r2: f64,
    pub(crate) predicted: Vec<f64>,
    pub(crate) betas: Vec<f64>,
    pub(crate) data: Option<String>,
    pub(crate) outcome: Option<String>,
    pub(crate) n: u32,
    pub(crate) m: u32,
}

impl R2 {
    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn r2(&self) -> f64 {
        self.r2
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn adj_r2(&self) -> f64 {
        self.adj_r2
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn predicted(&self) -> &[f64] {
        &self.predicted
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn betas(&self) -> &[f64] {
        &self.betas
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn data(&self) -> Option<&str> {
        self.data.as_deref()
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn set_data(&mut self, data: String) {
        self.data = Some(data);
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn outcome(&self) -> Option<&str> {
        self.outcome.as_deref()
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn set_outcome(&mut self, outcome: String) {
        self.outcome = Some(outcome);
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn n(&self) -> u32 {
        self.n
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn m(&self) -> u32 {
        self.m
    }
}

#[tracing::instrument(skip(data, outcomes))]
pub fn get_r2s(data: MatRef<f64>, outcomes: MatRef<f64>) -> Vec<R2> {
    debug!("Calculating R2s");
    let n = data.nrows();
    let m = data.ncols();

    let c_all = data.transpose() * outcomes;
    trace!("Computed c_all");
    let c_matrix = data.transpose() * data;
    let betas = match c_matrix.llt(Side::Lower) {
        Ok(chol) => chol.solve(c_all),
        Err(_) => {
            warn!("Using pseudo inverse");
            c_matrix
                .as_mat_ref()
                .thin_svd()
                .expect("could not compute thin SVD for pseudoinverse")
                .pseudoinverse()
                * c_all
        },
    };

    debug!("Calculated betas");

    let r2s = (0..outcomes.ncols())
        .into_par_iter()
        .map(|i| {
            let betas = betas.col(i);
            let mut predicted = (data * betas)
                .try_as_col_major()
                .unwrap()
                .as_slice()
                .to_vec();
            let actual = outcomes.col(i).try_as_col_major().unwrap().as_slice();
            let r2 = R2Simd::new(actual, &predicted).calculate();
            let adj_r2 = calculate_adj_r2(r2, n, m);
            let mut betas = betas.try_as_col_major().unwrap().as_slice().to_vec();
            if should_disable_predicted() {
                predicted = Vec::new();
            }
            R2 {
                r2,
                adj_r2,
                predicted,
                betas,
                outcome: None,
                data: None,
                n: n as u32,
                m: m as u32,
            }
        })
        .collect();
    debug!("Calculated R2s");
    r2s
}

#[derive(Debug, Clone)]
pub struct PValue {
    p_value: f64,
    beta: f64,
    intercept: f64,
    pub(crate) data: Option<String>,
    pub(crate) data_column: Option<u32>,
    pub(crate) outcome: Option<String>,
}

impl PValue {
    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn p_value(&self) -> f64 {
        self.p_value
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn beta(&self) -> f64 {
        self.beta
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn data(&self) -> Option<&str> {
        self.data.as_deref()
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn set_data(&mut self, data: String) {
        self.data = Some(data);
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn data_column(&self) -> Option<u32> {
        self.data_column
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn set_data_column(&mut self, data_column: u32) {
        self.data_column = Some(data_column);
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn outcome(&self) -> Option<&str> {
        self.outcome.as_deref()
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn set_outcome(&mut self, outcome: String) {
        self.outcome = Some(outcome);
    }
}

#[tracing::instrument(skip(xs, ys))]
pub fn p_value(xs: &[f64], ys: &[f64]) -> PValue {
    debug!("Calculating p-values");
    let mut x = Mat::new();
    x.resize_with(
        xs.len(),
        2,
        #[inline(always)]
        |i, j| if j == 0 { xs[i] } else { 1.0 },
    );
    let y: MatRef<'_, f64> = MatRef::from_column_major_slice(ys, ys.len(), 1);
    let c_all = x.transpose() * y;
    let mut c_matrix = faer::Mat::zeros(2, 2);
    faer::linalg::matmul::triangular::matmul(
        c_matrix.as_mut(),
        faer::linalg::matmul::triangular::BlockStructure::TriangularLower,
        faer::Accum::Replace,
        x.transpose(),
        faer::linalg::matmul::triangular::BlockStructure::Rectangular,
        &x,
        faer::linalg::matmul::triangular::BlockStructure::Rectangular,
        1.0,
        get_global_parallelism(),
    );
    let chol = c_matrix.llt(Side::Lower).unwrap();
    let inv_matrix = chol.solve(Mat::<f64>::identity(2, 2));
    let betas = chol.solve(c_all);
    let m = betas.get(0, 0);
    let intercept = betas.get(1, 0);
    let df = xs.len() as f64 - 2.0;
    let residuals = ys
        .iter()
        .zip(xs.iter())
        .map(|(y, x)| (y - (intercept + m * x)))
        .collect::<Vec<_>>();

    let se =
        (inv_matrix[(0, 0)] * ((residuals.iter().map(|x| x.powi(2)).sum::<f64>()) / df)).sqrt();
    let t = m / se;
    let t_distr = StudentsT::new(0.0, 1.0, (xs.len() - 2) as f64).unwrap();
    PValue {
        p_value: 2.0 * (1.0 - t_distr.cdf(t.abs())),
        beta: *m,
        intercept: *intercept,
        data: None,
        data_column: None,
        outcome: None,
    }
}

pub fn standardize_column(mut x: ColMut<f64>) {
    let mut mean = mean(x.as_ref().try_as_col_major().unwrap().as_slice());
    let mut std = variance(x.as_ref().try_as_col_major().unwrap().as_slice(), 1);
    let std = std.sqrt();
    if std == 0.0 {
        x.fill(0.0);
        return;
    }
    let xx = x.as_mut();
    let std_recip = 1.0 / std;
    if let Some(x) = xx.try_as_col_major_mut() {
        let x = x.as_slice_mut();
        Arch::new().dispatch(|| {
            for x in x.iter_mut() {
                *x = (*x - mean) * std_recip;
            }
        });
    } else {
        for x in x.iter_mut() {
            *x = (*x - mean) * std_recip;
        }
    }
}

pub fn standardize_row(mut x: RowMut<f64>) {
    let mut mean = [0.0];
    let mut std = [0.0];
    faer::stats::col_mean(
        ColMut::from_slice_mut(&mut mean),
        x.as_ref().as_mat(),
        faer::stats::NanHandling::Ignore,
    );
    faer::stats::col_varm(
        ColMut::from_slice_mut(&mut std),
        x.as_ref().as_mat(),
        ColRef::from_slice(&mean),
        faer::stats::NanHandling::Ignore,
    );
    let std = std[0].sqrt();
    let mean = mean[0];
    if std == 0.0 {
        x.fill(0.0);
        return;
    }
    let xx = x.as_mut();
    let std_recip = 1.0 / std;
    if let Some(x) = xx.try_as_row_major_mut() {
        let x = x.as_slice_mut();
        Arch::new().dispatch(|| {
            for x in x.iter_mut() {
                *x = (*x - mean) * std_recip;
            }
        });
    } else {
        for x in x.iter_mut() {
            *x = (*x - mean) * std_recip;
        }
    }
}

pub fn calculate_adj_r2(r2: f64, nrows: usize, ncols: usize) -> f64 {
    1.0 - (1.0 - r2) * (nrows as f64 - 1.0) / (nrows as f64 - ncols as f64 - 1.0)
}

#[derive(Debug, Clone)]
pub struct LinearModel {
    slopes: Vec<f64>,
    intercept: f64,
    predicted: Vec<f64>,
    r2: f64,
    adj_r2: f64,
}

impl LinearModel {
    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn slopes(&self) -> &[f64] {
        &self.slopes
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn predicted(&self) -> &[f64] {
        &self.predicted
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn r2(&self) -> f64 {
        self.r2
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn adj_r2(&self) -> f64 {
        self.adj_r2
    }

    pub fn predict(&self, x: &[f64]) -> f64 {
        self.intercept
            + self
                .slopes
                .iter()
                .zip(x.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>()
    }
}

#[tracing::instrument(skip(xs, ys))]
pub fn linear_regression(xs: MatRef<'_, f64>, ys: &[f64]) -> LinearModel {
    let ncols = xs.ncols();
    let mut x = xs.to_owned();
    x.resize_with(
        xs.nrows(),
        xs.ncols() + 1,
        #[inline(always)]
        |_, _| 1.0,
    );
    let y: MatRef<'_, f64> = MatRef::from_column_major_slice(ys, ys.len(), 1);
    let c_all = x.transpose() * y;
    let c_matrix = x.transpose() * &x;
    let betas = match c_matrix.llt(Side::Lower) {
        Ok(chol) => chol.solve(c_all),
        Err(_) => {
            warn!("Using pseudo inverse");
            c_matrix
                .as_mat_ref()
                .thin_svd()
                .expect("could not compute thin SVD for pseudoinverse")
                .pseudoinverse()
                * &c_all
        },
    };
    let betas = betas.col(0).try_as_col_major().unwrap().as_slice();
    let intercept = betas[ncols];
    let mut predicted = (0..ys.len())
        .map(|i| intercept + (0..ncols).map(|j| betas[j] * x[(i, j)]).sum::<f64>())
        .collect::<Vec<_>>();
    let r2 = R2Simd::new(ys, &predicted).calculate();
    let adj_r2 = calculate_adj_r2(r2, ys.len(), ncols);
    if should_disable_predicted() {
        predicted = Vec::new();
    }
    LinearModel {
        slopes: betas[..ncols].to_vec(),
        intercept,
        predicted,
        r2,
        adj_r2,
    }
}

#[derive(Debug, Clone)]
pub struct R2Simd<'a> {
    actual: &'a [f64],
    predicted: &'a [f64],
}

impl<'a> R2Simd<'a> {
    pub fn new(actual: &'a [f64], predicted: &'a [f64]) -> Self {
        assert!(actual.len() == predicted.len());
        Self { actual, predicted }
    }

    pub fn calculate(self) -> f64 {
        let arch = Arch::new();
        arch.dispatch(self)
    }

    pub fn calculate_no_simd(self) -> f64 {
        let mean = self.actual.iter().sum::<f64>() / self.actual.len() as f64;
        let mut rss = 0.0;
        for (actual, predicted) in self.actual.iter().zip(self.predicted.iter()) {
            rss += (actual - predicted).powi(2);
        }
        let mut tss = 0.0;
        for actual in self.actual.iter() {
            tss += (actual - mean).powi(2);
        }
        1.0 - rss / tss
    }
}

impl WithSimd for R2Simd<'_> {
    type Output = f64;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (actual_head, actual_tail) = S::f64s_as_simd(self.actual);
        let (predicted_head, predicted_tail) = S::f64s_as_simd(self.predicted);

        let mean = mean(self.actual);
        let simd_mean = simd.f64s_splat(mean);

        let mut rss0 = simd.f64s_splat(0.0);
        let mut rss1 = simd.f64s_splat(0.0);
        let mut rss2 = simd.f64s_splat(0.0);
        let mut rss3 = simd.f64s_splat(0.0);
        let mut tss0 = simd.f64s_splat(0.0);
        let mut tss1 = simd.f64s_splat(0.0);
        let mut tss2 = simd.f64s_splat(0.0);
        let mut tss3 = simd.f64s_splat(0.0);

        let (actual_head4, actual_head1) = pulp::as_arrays::<4, _>(actual_head);
        let (predicted_head4, predicted_head1) = pulp::as_arrays::<4, _>(predicted_head);

        for (
            [actual0, actual1, actual2, actual3],
            [predicted0, predicted1, predicted2, predicted3],
        ) in actual_head4.iter().zip(predicted_head4.iter())
        {
            let rs0 = simd.f64s_sub(*actual0, *predicted0);
            let rs1 = simd.f64s_sub(*actual1, *predicted1);
            let rs2 = simd.f64s_sub(*actual2, *predicted2);
            let rs3 = simd.f64s_sub(*actual3, *predicted3);
            rss0 = simd.f64s_mul_add(rs0, rs0, rss0);
            rss1 = simd.f64s_mul_add(rs1, rs1, rss1);
            rss2 = simd.f64s_mul_add(rs2, rs2, rss2);
            rss3 = simd.f64s_mul_add(rs3, rs3, rss3);
            let ts0 = simd.f64s_sub(*actual0, simd_mean);
            let ts1 = simd.f64s_sub(*actual1, simd_mean);
            let ts2 = simd.f64s_sub(*actual2, simd_mean);
            let ts3 = simd.f64s_sub(*actual3, simd_mean);
            tss0 = simd.f64s_mul_add(ts0, ts0, tss0);
            tss1 = simd.f64s_mul_add(ts1, ts1, tss1);
            tss2 = simd.f64s_mul_add(ts2, ts2, tss2);
            tss3 = simd.f64s_mul_add(ts3, ts3, tss3);
        }

        tss0 = simd.f64s_add(tss0, tss2);
        tss1 = simd.f64s_add(tss1, tss3);
        rss0 = simd.f64s_add(rss0, rss2);
        rss1 = simd.f64s_add(rss1, rss3);

        let mut tss = simd.f64s_add(tss0, tss1);
        let mut rss = simd.f64s_add(rss0, rss1);

        for (actual, predicted) in actual_head1.iter().zip(predicted_head1.iter()) {
            let rs = simd.f64s_sub(*actual, *predicted);
            rss = simd.f64s_mul_add(rs, rs, rss);
            let ts = simd.f64s_sub(*actual, simd_mean);
            tss = simd.f64s_mul_add(ts, ts, tss);
        }

        let mut tss = simd.f64s_reduce_sum(tss);
        let mut rss = simd.f64s_reduce_sum(rss);

        for (actual, predicted) in actual_tail.iter().zip(predicted_tail.iter()) {
            rss += (*actual - *predicted).powi(2);
            tss += (*actual - mean).powi(2);
        }

        1.0 - rss / tss
    }
}

#[derive(Debug, Clone)]
pub struct LogisticModel {
    slopes: Vec<f64>,
    intercept: f64,
    predicted: Vec<f64>,
    r2: f64,
    adj_r2: f64,
}

impl LogisticModel {
    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn slopes(&self) -> &[f64] {
        &self.slopes
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn predicted(&self) -> &[f64] {
        &self.predicted
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn r2(&self) -> f64 {
        self.r2
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn adj_r2(&self) -> f64 {
        self.adj_r2
    }

    pub fn predict(&self, x: &[f64]) -> f64 {
        self.intercept
            + self
                .slopes
                .iter()
                .zip(x.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>()
    }
}

#[inline(always)]
fn logit(x: f64) -> f64 {
    (x / (1.0 - x)).ln()
}

#[inline(always)]
fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
fn logit_prime(x: f64) -> f64 {
    1.0 / (x * (1.0 - x))
}

#[inline(always)]
fn v(x: f64) -> f64 {
    x * (1.0 - x)
}

#[inline(always)]
fn ll(p: &[f64], y: &[f64]) -> f64 {
    p.iter()
        .zip(y)
        .map(|(p, y)| y * p.ln() + (1.0 - y) * (1.0 - p).ln())
        .sum()
}

#[tracing::instrument(skip(xs, ys))]
pub fn logistic_regression_irls(xs: MatRef<'_, f64>, ys: &[f64]) -> LogisticModel {
    let mut mu = vec![0.5; ys.len()];
    let mut delta = 1.0;
    let mut l = 0.0;
    let mut x = xs.to_owned();
    x.resize_with(
        xs.nrows(),
        xs.ncols() + 1,
        #[inline(always)]
        |_, _| 1.0,
    );
    let mut slopes = vec![0.0; xs.ncols()];
    let mut intercept = 0.0;
    let mut z = vec![0.0; ys.len()];
    // let mut w = Mat::zeros(ys.len(), ys.len());
    let mut w = vec![0.0; ys.len()];
    let xt = x.transpose();
    let mut xtw = Mat::<f64>::zeros(x.ncols(), ys.len());
    let mut xtwx = Mat::zeros(x.ncols(), x.ncols());
    let mut xtwz = Col::zeros(x.ncols());
    while delta > 1e-5 {
        for ((z, mu), y) in z.iter_mut().zip(mu.iter()).zip(ys) {
            *z = logit(*mu) + (y - mu) * logit_prime(*mu);
        }
        for (i, mu) in mu.iter().enumerate() {
            w[i] = 1.0 / (logit_prime(*mu).powi(2) * v(*mu));
        }

        // faer::linalg::matmul::triangular::matmul(
        //     xtw.as_mut(),
        //     faer::linalg::matmul::triangular::BlockStructure::Rectangular,
        //     xt,
        //     faer::linalg::matmul::triangular::BlockStructure::Rectangular,
        //     &w,
        //     faer::linalg::matmul::triangular::BlockStructure::TriangularLower,
        //     None,
        //     1.0,
        //     get_global_parallelism(),
        // );
        xtw.par_col_chunks_mut(1)
            .zip(w.par_iter())
            .enumerate()
            .for_each(|(j, (mut xtw, w))| {
                xtw.col_iter_mut().for_each(|x| {
                    x.iter_mut()
                        .enumerate()
                        .for_each(|(i, x)| *x = xt[(i, j)] * w)
                })
            });
        faer::linalg::matmul::matmul(
            xtwx.as_mut(),
            faer::Accum::Replace,
            &xtw,
            &x,
            1.0,
            get_global_parallelism(),
        );
        faer::linalg::matmul::matmul(
            xtwz.as_mut(),
            faer::Accum::Replace,
            &xtw,
            ColRef::from_slice(z.as_slice()),
            1.0,
            get_global_parallelism(),
        );

        let beta = match xtwx.llt(Side::Lower) {
            Ok(chol) => chol.solve(&xtwz),
            Err(_) => {
                warn!("Using pseudo inverse");
                xtwx.as_mat_ref()
                    .thin_svd()
                    .expect("could not compute thin SVD for pseudoinverse")
                    .pseudoinverse()
                    * &xtwz
            },
        };
        let b = beta.try_as_col_major().unwrap().as_slice();
        slopes.as_mut_slice().copy_from_slice(&b[..xs.ncols()]);
        intercept = b[xs.ncols()];
        let eta = &x * beta;
        let eta = eta.try_as_col_major().unwrap().as_slice();
        for (mu, eta) in mu.iter_mut().zip(eta) {
            *mu = logistic(*eta);
        }
        let old_ll = l;
        l = ll(mu.as_slice(), ys);
        delta = (l - old_ll).abs();
    }

    let r2 = R2Simd::new(ys, &mu).calculate();
    let adj_r2 = calculate_adj_r2(r2, ys.len(), xs.ncols());

    if should_disable_predicted() {
        mu = Vec::new();
    }

    LogisticModel {
        slopes,
        intercept,
        predicted: mu,
        r2,
        adj_r2,
    }
}

#[tracing::instrument(skip(xs, ys))]
pub fn logistic_regression_newton_raphson(xs: MatRef<'_, f64>, ys: &[f64]) -> LogisticModel {
    let mut x = xs.to_owned();
    x.resize_with(
        xs.nrows(),
        xs.ncols() + 1,
        #[inline(always)]
        |_, _| 1.0,
    );
    let mut beta = vec![0.0; x.ncols()];
    let mut mu = vec![0.0; ys.len()];
    // let mut w = Mat::zeros(ys.len(), ys.len());
    let mut w = vec![0.0; ys.len()];
    let mut linear_predictor = Col::zeros(ys.len());
    let mut ys_sub_mu = vec![0.0; ys.len()];
    let xt = x.transpose();
    let mut jacobian = Col::zeros(x.ncols());
    let mut xtw = Mat::<f64>::zeros(x.ncols(), ys.len());
    let mut hessian = Mat::zeros(x.ncols(), x.ncols());
    for _ in 0..100 {
        faer::linalg::matmul::matmul(
            linear_predictor.as_mut(),
            faer::Accum::Replace,
            &x,
            ColRef::from_slice(beta.as_slice()),
            1.0,
            get_global_parallelism(),
        );
        for (mu, l) in mu
            .iter_mut()
            .zip(linear_predictor.try_as_col_major().unwrap().as_slice())
        {
            *mu = logistic(*l);
        }
        for (i, mu) in mu.iter().enumerate() {
            w[i] = mu * (1.0 - mu);
        }
        for (i, (mu, y)) in mu.iter().zip(ys).enumerate() {
            ys_sub_mu[i] = *y - mu;
        }

        faer::linalg::matmul::matmul(
            jacobian.as_mut(),
            faer::Accum::Replace,
            xt,
            ColRef::from_slice(ys_sub_mu.as_slice()),
            1.0,
            get_global_parallelism(),
        );
        // faer::linalg::matmul::triangular::matmul(
        //     xtw.as_mut(),
        //     faer::linalg::matmul::triangular::BlockStructure::Rectangular,
        //     xt,
        //     faer::linalg::matmul::triangular::BlockStructure::Rectangular,
        //     &w,
        //     faer::linalg::matmul::triangular::BlockStructure::TriangularLower,
        //     None,
        //     1.0,
        //     get_global_parallelism(),
        // );
        xtw.par_col_chunks_mut(1)
            .zip(w.par_iter())
            .enumerate()
            .for_each(|(j, (mut xtw, w))| {
                xtw.col_iter_mut().for_each(|x| {
                    x.iter_mut()
                        .enumerate()
                        .for_each(|(i, x)| *x = xt[(i, j)] * w)
                })
            });
        faer::linalg::matmul::matmul(
            hessian.as_mut(),
            faer::Accum::Replace,
            &xtw,
            &x,
            1.0,
            get_global_parallelism(),
        );

        let beta_new = ColRef::from_slice(beta.as_slice())
            + match hessian.llt(Side::Lower) {
                Ok(chol) => chol.solve(&jacobian),
                Err(_) => {
                    warn!("Using pseudo inverse");
                    hessian
                        .as_mat_ref()
                        .thin_svd()
                        .expect("could not compute thin SVD for pseudoinverse")
                        .pseudoinverse()
                        * &jacobian
                },
            };

        if (&beta_new - ColRef::from_slice(beta.as_slice())).norm_l1() < 1e-5 {
            break;
        }
        beta.copy_from_slice(beta_new.try_as_col_major().unwrap().as_slice());
    }
    let r2 = R2Simd::new(ys, &mu).calculate();
    let adj_r2 = calculate_adj_r2(r2, ys.len(), xs.ncols());

    let predicted = if should_disable_predicted() {
        Vec::new()
    } else {
        (&x * ColRef::from_slice(beta.as_slice()))
            .try_as_col_major()
            .unwrap()
            .as_slice()
            .iter()
            .map(|x| logistic(*x))
            .collect()
    };

    LogisticModel {
        predicted,
        intercept: beta[x.ncols() - 1],
        slopes: {
            beta.truncate(x.ncols() - 1);
            beta
        },
        r2,
        adj_r2,
    }
}

// UNFINISHED
fn logistic_regression_glm(xs: MatRef<'_, f64>, ys: &[f64]) -> LogisticModel {
    let mut x = xs.to_owned();
    x.resize_with(
        xs.nrows(),
        xs.ncols() + 1,
        #[inline(always)]
        |_, _| 1.0,
    );
    let weights = vec![1.0; ys.len()];
    let mustart = weights
        .iter()
        .zip(ys)
        .map(|(w, y)| (w * y + 0.5) / (w + 1.0))
        .collect::<Vec<_>>();
    let mut eta = mustart.iter().map(|x| logit(*x)).collect::<Vec<_>>();
    let mut mu = mustart;
    let mut delta = 1.0;
    let mut l = 0.0;
    let mut slopes = vec![0.0; xs.ncols()];
    let mut intercept = 0.0;
    let mut i = 0;
    while delta > 1e-5 {
        fn mu_eta_val(eta: f64) -> f64 {
            let exp = eta.exp();
            exp / (1.0 + exp).powi(2)
        }
        let z = eta
            .iter()
            .zip(ys)
            .zip(mu.iter())
            .map(|((e, y), m)| e + (y - m) / mu_eta_val(*e))
            .collect::<Vec<_>>();
        let w = mu
            .iter()
            .zip(weights.iter())
            .zip(eta.iter())
            .map(|((m, w), e)| (w * mu_eta_val(*e).powi(2)).sqrt() / v(*m))
            .collect::<Vec<_>>();
        let mut xw = x.clone();
        xw.row_iter_mut()
            .zip(w.iter())
            .for_each(|(x, w)| x.iter_mut().for_each(|x| *x *= *w));
        let zw = z.iter().zip(w).map(|(z, w)| z * w).collect::<Vec<_>>();
        std::io::stdout().flush().unwrap();
        let qr = xw.qr();
        // let beta = match xw.cholesky(Side::Lower) {
        //     Ok(chol) => chol.solve([aer::col::from_slice(zw.as_slice())),
        //     Err(_) => {
        //         warn!("Using pseudo inverse");
        //         ThinSvd::new(xw.as_mat_ref()).pseudoinverse() *
        // faer::col::from_slice(zw.as_slice())     },
        // };
        let beta = (qr.thin_R().thin_svd().unwrap().inverse()
            * qr.compute_thin_Q().transpose()
            * ColRef::from_slice(zw.as_slice()));
        let beta_slice = beta.try_as_col_major().unwrap().as_slice();
        slopes.copy_from_slice(&beta_slice[..xs.ncols()]);
        intercept = beta[xs.ncols()];
        let eta = (&x * beta).try_as_col_major().unwrap().as_slice().to_vec();
        for (i, eta) in eta.iter().enumerate() {
            mu[i] = logistic(*eta);
        }
        let old_ll = l;
        l = ll(mu.as_slice(), ys);
        delta = (l - old_ll).abs();
        i += 1;
        if i > 100 {
            error!("Did not converge");
            break;
        }
    }
    let r2 = R2Simd::new(ys, &mu).calculate();
    let adj_r2 = calculate_adj_r2(r2, ys.len(), xs.ncols());

    LogisticModel {
        slopes,
        intercept,
        predicted: mu,
        r2,
        adj_r2,
    }
}

// actual is the actual values (0, 1), predicted is the predicted probabilities
pub fn compute_r2_tjur(actual: &[f64], predicted: &[f64]) -> f64 {
    let mut ones_sum = 0.0;
    let mut zeros_sum = 0.0;
    let mut ones_count = 0.0;
    let mut zeros_count = 0.0;
    for (a, p) in actual.iter().zip(predicted.iter()) {
        if *a == 1.0 {
            ones_sum += p;
            ones_count += 1.0;
        } else {
            zeros_sum += p;
            zeros_count += 1.0;
        }
    }
    ((ones_sum / ones_count) - (zeros_sum / zeros_count))
}

#[cfg(test)]
mod tests {
    use faer::RowRef;
    use rand::SeedableRng;
    use test_log::test;

    use super::*;
    use crate::{IntoMatrix, Matrix, OwnedMatrix};

    macro_rules! assert_float_eq {
        ($a:expr, $b:expr, $tol:expr) => {
            assert!(($a - $b).abs() < $tol, "{:.22} != {:.22}", $a, $b);
        };
    }

    macro_rules! float_eq {
        ($a:expr, $b:expr) => {
            assert_float_eq!($a, $b, 1e-12);
        };
    }

    macro_rules! rough_eq {
        ($a:expr, $b:expr) => {
            assert_float_eq!($a, $b, 1e-3);
        };
    }

    #[test]
    fn test_standardize_column() {
        let mut data = [1.0, 2.0, 3.0];
        standardize_column(ColMut::from_slice_mut(&mut data));
        assert_eq!(data, [-1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_standardize_row() {
        let mut data = [1.0, 2.0, 3.0];
        standardize_row(RowMut::from_slice_mut(&mut data));
        assert_eq!(data, [-1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_get_r2s_normalized() {
        std::env::set_var("LMUTILS_ENABLE_PREDICTED", "1");
        let mut data = OwnedMatrix::new(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0], None)
            .into_matrix();
        data.standardize_columns();
        let mut outcomes =
            OwnedMatrix::new(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 8.0], None)
                .into_matrix();
        outcomes.standardize_columns();
        let r2s = get_r2s(data.as_mat_ref_loaded(), outcomes.as_mat_ref_loaded());
        assert_eq!(r2s.len(), 2);
        float_eq!(r2s[0].r2(), 1.0);
        float_eq!(r2s[0].adj_r2(), 1.0);
        let pred: [f64; 4] = [
            -1.1618950038622262,
            -0.3872983346207394,
            0.3872983346207394,
            1.1618950038622262,
        ];
        assert_eq!(r2s[0].predicted().len(), 4);
        for (a, b) in r2s[0].predicted().iter().copied().zip(pred.iter().copied()) {
            float_eq!(a, b);
        }
        assert_eq!(r2s[0].n(), 4);
        assert_eq!(r2s[0].m(), 2);
        assert!(r2s[0].data().is_none());
        assert!(r2s[0].outcome().is_none());
        float_eq!(r2s[1].r2(), 0.9629629629629629);
        float_eq!(r2s[1].adj_r2(), 0.8888888888888891);
        let pred: [f64; 4] = [
            -0.9999999999999998,
            -0.6666666666666672,
            0.6666666666666672,
            0.9999999999999998,
        ];
        assert_eq!(r2s[1].predicted().len(), 4);
        for (a, b) in r2s[1].predicted().iter().copied().zip(pred.iter().copied()) {
            float_eq!(a, b);
        }
        assert_eq!(r2s[1].n(), 4);
        assert_eq!(r2s[1].m(), 2);
        assert!(r2s[1].data().is_none());
        assert!(r2s[1].outcome().is_none());
    }

    #[test]
    fn test_get_r2s_not_normalized() {
        std::env::set_var("LMUTILS_ENABLE_PREDICTED", "1");
        let data = MatRef::from_column_major_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0], 4, 2);
        let outcomes =
            MatRef::from_column_major_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 8.0], 4, 2);
        let r2s = get_r2s(data.as_mat_ref(), outcomes.as_mat_ref());
        assert_eq!(r2s.len(), 2);
        float_eq!(r2s[0].r2(), 1.0);
        float_eq!(r2s[0].adj_r2(), 1.0);
        let pred: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(r2s[0].predicted().len(), 4);
        for (a, b) in r2s[0].predicted().iter().copied().zip(pred.iter().copied()) {
            float_eq!(a, b);
        }
        assert_eq!(r2s[0].n(), 4);
        assert_eq!(r2s[0].m(), 2);
        assert!(r2s[0].data().is_none());
        assert!(r2s[0].outcome().is_none());
        float_eq!(r2s[1].r2(), 0.9592740150509075);
        float_eq!(r2s[1].adj_r2(), 0.8778220451527224);
        let pred: [f64; 4] = [
            5.235059760956178,
            5.864541832669323,
            7.6454183266932265,
            8.274900398406372,
        ];
        assert_eq!(r2s[1].predicted().len(), 4);
        for (a, b) in r2s[1].predicted().iter().copied().zip(pred.iter().copied()) {
            float_eq!(a, b);
        }
        assert_eq!(r2s[1].n(), 4);
        assert_eq!(r2s[1].m(), 2);
        assert!(r2s[1].data().is_none());
        assert!(r2s[1].outcome().is_none());
    }

    #[test]
    fn test_p_value() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [1.0, 2.0, 3.0, 4.0, 5.0];
        let p_value = p_value(&xs, &ys);
        float_eq!(p_value.p_value(), 0.0);
        assert!(p_value.data().is_none());
        assert!(p_value.outcome().is_none());
    }

    #[test]
    fn test_linear_regression() {
        let xs = MatRef::from_column_major_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], 5, 1);
        let ys = [1.0, 2.0, 3.0, 4.0, 5.0];
        let model = linear_regression(xs.as_mat_ref(), &ys);
        assert_eq!(model.slopes().len(), 1);
        float_eq!(model.slopes()[0], 1.0);
        float_eq!(model.intercept(), 0.0);
        assert_eq!(model.predicted().len(), 5);
        let predicted = [1.0, 2.0, 3.0, 4.0, 5.0];
        for (a, b) in model
            .predicted()
            .iter()
            .copied()
            .zip(predicted.iter().copied())
        {
            float_eq!(a, b);
        }
        float_eq!(model.r2(), 1.0);
        float_eq!(model.adj_r2(), 1.0);
    }

    #[test]
    fn test_linear_regression_predict() {
        let xs = MatRef::from_column_major_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], 5, 1);
        let ys = [1.0, 2.0, 3.0, 4.0, 5.0];
        let model = linear_regression(xs.as_mat_ref(), &ys);
        float_eq!(model.predict(&[6.0]), 6.0);
        let xs = MatRef::from_column_major_slice(&[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], 4, 2);
        let ys = [2.0, 4.0, 6.0, 8.0];
        let model = linear_regression(xs.as_mat_ref(), &ys);
        float_eq!(model.predict(&[1.0, 1.0]), 2.0);
    }

    #[test]
    fn test_r2_simd() {
        for _ in 0..100 {
            let random_actual: Vec<f64> = (0..1000).map(|_| rand::random()).collect();
            let random_predicted: Vec<f64> = (0..1000).map(|_| rand::random()).collect();

            let r2 = R2Simd::new(&random_actual, &random_predicted).calculate();
            let r2_no_simd = R2Simd::new(&random_actual, &random_predicted).calculate_no_simd();
            float_eq!(r2, r2_no_simd);
        }
    }

    #[test]
    fn test_r2() {
        let actual = [1.0, 2.0, 3.0, 4.0, 5.0];
        let predicted = [1.0, 2.0, 3.0, 4.0, 5.0];
        let r2 = R2Simd::new(&actual, &predicted).calculate();
        float_eq!(r2, 1.0);
        let actual = [1.0, 2.0, 3.0, 5.0, 6.0];
        let predicted = [1.0, 2.0, 3.0, 4.0, 5.0];
        let r2 = R2Simd::new(&actual, &predicted).calculate();
        float_eq!(r2, 0.8837209302325582);
    }

    #[test]
    fn test_logistic_regression() {
        let nrows = 50;
        let xs = statrs::distribution::Normal::new(0.0, 1.0).unwrap();
        let ys = statrs::distribution::Bernoulli::new(0.5).unwrap();
        let xs = xs
            .sample_iter(rand::thread_rng())
            .take(nrows)
            .collect::<Vec<_>>();
        let ys = ys
            .sample_iter(rand::thread_rng())
            .take(nrows)
            .collect::<Vec<_>>();
        let xs = MatRef::from_column_major_slice(xs.as_slice(), nrows, 1);
        let m1 = logistic_regression_irls(xs, ys.as_slice());
        let m2 = logistic_regression_newton_raphson(xs, ys.as_slice());
        for (a, b) in m1.slopes.iter().zip(m2.slopes.iter()) {
            rough_eq!(a, b);
        }
        rough_eq!(m1.intercept, m2.intercept);
        for (a, b) in m1.predicted.iter().zip(m2.predicted.iter()) {
            rough_eq!(a, b);
        }
    }
}
