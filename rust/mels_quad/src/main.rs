//! mels_quad -- a mixed-effects location-scale model (Hedeker, Mermelstein and Demirtas 2008) for replicate scores
//! of learners on task conditions, fitted by maximum likelihood with the random effects integrated out exactly
//! (FLAGS.md items 16 and 23: melsvmp's variational fit was degenerate on these data).
//!
//! Level 1 (replicate r of condition c, learner l):
//!   y_rcl = beta_l + beta_c + nu_cl + e_rcl,   e_rcl ~ N(0, exp(tau_l + gamma_c + omega_cl))
//! Level 2 (cell = learner l on condition c):
//!   nu_cl ~ N(0, exp(alpha_l)),   omega_cl ~ N(0, sigma_omega^2),   independent
//! gamma_c (a fixed log-variance effect for each condition, gamma of the first condition = 0) is the addition to
//! melsvmp's model: the within-cell variances span about six orders of magnitude across conditions, and without
//! gamma_c the single random scale effect has to absorb that. With --no-gamma the model is melsvmp's.
//!
//! The likelihood of a cell needs only its replicate count R, mean ybar and within sum of squares SSW. Given omega,
//! nu integrates out in closed form (the cell is multivariate normal with covariance s^2 I + v 11'):
//!   l(omega) = -(R/2) log 2pi - ((R-1)/2) log s^2 - (1/2) log(s^2 + R v) - SSW/(2 s^2) - R (ybar - mu)^2 / (2(s^2 + R v))
//! and omega is integrated numerically on a fixed grid in z = omega / sigma_omega (trapezoid rule, which for these smooth integrands is accurate to far below the optimizer's tolerance). The gradient
//! is analytic (posterior expectations over the grid). Optimization by L-BFGS (argmin); standard errors from the
//! inverse of a numerical Hessian (central differences of the analytic gradient); optionally a bootstrap that
//! resamples conditions (each with all its cells).
//!
//! Input: CSV with header, columns learner (0-based int), cond (0-based int), cell (0-based int), y.
//! Output (stdout): CSV parameter, estimate, se [, boot_se, boot_lo, boot_hi, boot_ok], then a line "#loglik,<value>".
//! Usage: mels_quad <input.csv> [--no-gamma] [--boot B] [--seed S]
//!
//! Eliot R. Weinstein
//! eliotw@uchicago.edu
//! erweinstein.com

use argmin::core::{CostFunction, Error, Executor, Gradient, State};
use argmin::solver::linesearch::{condition::ArmijoCondition, BacktrackingLineSearch};
use argmin::solver::quasinewton::LBFGS;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::f64::consts::PI;

#[derive(Clone)]
struct Cell { learner: usize, cond: usize, r: f64, ybar: f64, ssw: f64 }

/// The between-cell log variance actually used: a smooth floor at ALPHA_FLOOR (softplus), so that an estimate on
/// the boundary (variance 0, alpha -> -infinity) settles near the floor instead of drifting, which stalled the line
/// search. Returns (alpha_eff, d alpha_eff / d alpha).
const ALPHA_FLOOR: f64 = -20.0;
static GEVALS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static EVALS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
fn alpha_eff(a: f64) -> (f64, f64) {
    let x = a - ALPHA_FLOOR;
    let sp = if x > 30.0 { x } else { x.exp().ln_1p() };
    (ALPHA_FLOOR + sp, 1.0 / (1.0 + (-x).exp()))
}

#[derive(Clone)]
struct Model {
    cells: Vec<Cell>,
    nl: usize,
    nc: usize,
    gamma: bool,
    z: Vec<f64>,
    lw: Vec<f64>, // log of (trapezoid weight x standard normal density)
}

// parameter layout: beta_l (nl), beta_c (nc - 1), alpha_l (nl), tau_l (nl), gamma_c (nc - 1, if gamma), log sigma_omega
impl Model {
    fn new(cells: Vec<Cell>, nl: usize, nc: usize, gamma: bool) -> Self {
        let k = 241; let lo = -8.0; let h = 16.0 / (k as f64 - 1.0);
        let z: Vec<f64> = (0..k).map(|i| lo + h * i as f64).collect();
        let lw = z.iter().enumerate().map(|(i, zz)| {
            let tw = if i == 0 || i == k - 1 { 0.5 * h } else { h };
            tw.ln() - 0.5 * zz * zz - 0.5 * (2.0 * PI).ln() }).collect();
        Model { cells, nl, nc, gamma, z, lw }
    }
    fn npar(&self) -> usize { 3 * self.nl + (self.nc - 1) * if self.gamma { 2 } else { 1 } + 1 }
    fn idx_beta_c(&self, c: usize) -> Option<usize> { if c == 0 { None } else { Some(self.nl + c - 1) } }
    fn idx_alpha(&self, l: usize) -> usize { self.nl + self.nc - 1 + l }
    fn idx_tau(&self, l: usize) -> usize { 2 * self.nl + self.nc - 1 + l }
    fn idx_gamma(&self, c: usize) -> Option<usize> { if !self.gamma || c == 0 { None } else { Some(3 * self.nl + self.nc - 1 + c - 1) } }
    fn idx_lsig(&self) -> usize { self.npar() - 1 }

    /// negative log-likelihood and its gradient
    fn nll_grad(&self, p: &[f64], want_grad: bool) -> (f64, Vec<f64>) {
        let np = self.npar();
        let lsig = p[self.idx_lsig()]; let sig = lsig.exp();
        let parts: Vec<(f64, Vec<(usize, f64)>)> = self.cells.par_iter().map(|c| {
            let mu = p[c.learner] + self.idx_beta_c(c.cond).map_or(0.0, |i| p[i]);
            let (al, dal_draw) = alpha_eff(p[self.idx_alpha(c.learner)]); let v = al.exp();
            let s0 = p[self.idx_tau(c.learner)] + self.idx_gamma(c.cond).map_or(0.0, |i| p[i]);
            let dev = c.ybar - mu; let r = c.r;
            let mut lf = vec![0.0; self.z.len()];
            let mut mx = f64::NEG_INFINITY;
            for (q, zz) in self.z.iter().enumerate() {
                let w = s0 + sig * zz; let s2 = w.exp(); let u = s2 + r * v;
                let l = -0.5 * r * (2.0 * PI).ln() - 0.5 * (r - 1.0) * w - 0.5 * u.ln() - c.ssw / (2.0 * s2) - r * dev * dev / (2.0 * u);
                lf[q] = l + self.lw[q]; if lf[q] > mx { mx = lf[q]; }
            }
            let mut sum = 0.0; for x in lf.iter() { sum += (x - mx).exp(); }
            let ll = mx + sum.ln();
            let mut g = Vec::new();
            if want_grad {
                let (mut gmu, mut gs0, mut gal, mut gls) = (0.0, 0.0, 0.0, 0.0);
                for (q, zz) in self.z.iter().enumerate() {
                    let pq = (lf[q] - ll).exp(); if pq < 1e-300 { continue; }
                    let w = s0 + sig * zz; let s2 = w.exp(); let u = s2 + r * v;
                    let dmu = r * dev / u;
                    let dw = -0.5 * (r - 1.0) - 0.5 * s2 / u + c.ssw / (2.0 * s2) + r * dev * dev * s2 / (2.0 * u * u);
                    let dal = -0.5 * r * v / u + r * dev * dev * r * v / (2.0 * u * u);
                    gmu += pq * dmu; gs0 += pq * dw; gal += pq * dal; gls += pq * dw * sig * zz;
                }
                g.push((c.learner, gmu));
                if let Some(i) = self.idx_beta_c(c.cond) { g.push((i, gmu)); }
                g.push((self.idx_alpha(c.learner), gal * dal_draw));
                g.push((self.idx_tau(c.learner), gs0));
                if let Some(i) = self.idx_gamma(c.cond) { g.push((i, gs0)); }
                g.push((self.idx_lsig(), gls));
            }
            (ll, g)
        }).collect();
        let mut nll = 0.0; let mut grad = vec![0.0; np];
        for (ll, g) in parts { nll -= ll; for (i, x) in g { grad[i] -= x; } }
        (nll, grad)
    }

    fn start(&self) -> Vec<f64> {
        let mut p = vec![0.0; self.npar()];
        // learner means, condition offsets, log within variances
        let mut sl = vec![0.0f64; self.nl]; let mut nlc = vec![0.0f64; self.nl];
        for c in &self.cells { sl[c.learner] += c.ybar; nlc[c.learner] += 1.0; }
        for l in 0..self.nl { p[l] = sl[l] / nlc[l].max(1.0); }
        let mut sc = vec![0.0f64; self.nc]; let mut ncc = vec![0.0f64; self.nc];
        for c in &self.cells { sc[c.cond] += c.ybar - p[c.learner]; ncc[c.cond] += 1.0; }
        for k in 1..self.nc { p[self.nl + k - 1] = sc[k] / ncc[k].max(1.0) - sc[0] / ncc[0].max(1.0); }
        let mut lv = vec![0.0f64; self.nl]; let mut lc = vec![0.0f64; self.nc];
        for c in &self.cells { let w = (c.ssw / (c.r - 1.0)).max(1e-12).ln(); lv[c.learner] += w; lc[c.cond] += w; }
        for l in 0..self.nl { p[self.idx_tau(l)] = lv[l] / nlc[l].max(1.0); p[self.idx_alpha(l)] = -3.0; }
        if self.gamma { for k in 1..self.nc { p[self.idx_gamma(k).unwrap()] = lc[k] / ncc[k].max(1.0) - lc[0] / ncc[0].max(1.0); } }
        let il = self.idx_lsig(); p[il] = 0.0;
        p
    }
}

impl CostFunction for Model {
    type Param = Vec<f64>; type Output = f64;
    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let f = self.nll_grad(p, false).0;
        // a step into a region where the likelihood overflows is rejected by the line search
        let f = if f.is_finite() { f } else { f64::INFINITY };
        if std::env::var("MELS_VERBOSE").is_ok() {
            let n = EVALS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n % 500 == 0 { eprintln!("eval {} cost {:.6} lsig {:.3} max|p| {:.2}", n, f, p[self.idx_lsig()], p.iter().fold(0.0f64, |a, x| a.max(x.abs()))); }
        }
        Ok(f) }
}
impl Gradient for Model {
    type Param = Vec<f64>; type Gradient = Vec<f64>;
    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        if std::env::var("MELS_VERBOSE").is_ok() {
            let n = GEVALS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n % 100 == 0 { eprintln!("grad eval {}", n); }
        }
        Ok(self.nll_grad(p, true).1) }
}

fn fit(m: &Model, init: Vec<f64>) -> (Vec<f64>, f64) {
    let mut p = init;
    // L-BFGS restarts until the objective stops improving (the scale parameters can start far from the optimum)
    let mut last = f64::INFINITY;
    // a between-cell variance whose estimate is on the boundary (alpha -> -infinity) is held at alpha = -25
    // (variance 1e-11) after each round, so the optimizer does not chase it
    let rounds: usize = std::env::var("MELS_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    for _ in 0..rounds {
        // backtracking (Armijo) rather than More-Thuente: on these data a first step can overflow the likelihood,
        // and backtracking simply shrinks it, where More-Thuente's interpolation was carried off to infinity
        let ls = BacktrackingLineSearch::new(ArmijoCondition::new(1e-4f64).unwrap()).rho(0.5f64).unwrap();
        let solver = LBFGS::new(ls, 20).with_tolerance_grad(1e-6).unwrap().with_tolerance_cost(1e-10).unwrap();
        let res = Executor::new(m.clone(), solver).configure(|s| s.param(p.clone()).max_iters(400)).run();
        match res {
            Ok(r) => { p = r.state().get_best_param().unwrap().clone(); }
            Err(_) => break,
        }
        for l in 0..m.nl { let i = m.idx_alpha(l); if p[i] < ALPHA_FLOOR - 15.0 { p[i] = ALPHA_FLOOR - 15.0; } }
        let f = m.nll_grad(&p, false).0;
        if std::env::var("MELS_VERBOSE").is_ok() { eprintln!("round objective {:.6}", f); }
        if (last - f).abs() < 1e-7 { last = f; break; }
        last = f;
    }
    (p, last)
}

fn hessian_se(m: &Model, p: &[f64]) -> Vec<f64> {
    let n = p.len(); let mut h = vec![vec![0.0; n]; n];
    for j in 0..n {
        let eps = 1e-5 * (1.0 + p[j].abs());
        let mut a = p.to_vec(); a[j] += eps; let ga = m.nll_grad(&a, true).1;
        let mut b = p.to_vec(); b[j] -= eps; let gb = m.nll_grad(&b, true).1;
        for i in 0..n { h[i][j] = (ga[i] - gb[i]) / (2.0 * eps); }
    }
    for i in 0..n { for j in 0..i { let s = 0.5 * (h[i][j] + h[j][i]); h[i][j] = s; h[j][i] = s; } }
    // invert by Gauss-Jordan with partial pivoting; NaN where singular
    let mut a = h.clone(); let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n { inv[i][i] = 1.0; }
    for c in 0..n {
        let mut piv = c; for r in c..n { if a[r][c].abs() > a[piv][c].abs() { piv = r; } }
        if a[piv][c].abs() < 1e-12 { return vec![f64::NAN; n]; }
        a.swap(c, piv); inv.swap(c, piv);
        let d = a[c][c]; for k in 0..n { a[c][k] /= d; inv[c][k] /= d; }
        for r in 0..n { if r != c { let f = a[r][c]; if f != 0.0 { for k in 0..n { a[r][k] -= f * a[c][k]; inv[r][k] -= f * inv[c][k]; } } } }
    }
    (0..n).map(|i| if inv[i][i] > 0.0 { inv[i][i].sqrt() } else { f64::NAN }).collect()
}

fn names(m: &Model) -> Vec<String> {
    let mut v = Vec::new();
    for l in 0..m.nl { v.push(format!("beta_learner_{}", l)); }
    for k in 1..m.nc { v.push(format!("beta_cond_{}", k)); }
    for l in 0..m.nl { v.push(format!("alpha_learner_{}", l)); }
    for l in 0..m.nl { v.push(format!("tau_learner_{}", l)); }
    if m.gamma { for k in 1..m.nc { v.push(format!("gamma_cond_{}", k)); } }
    v.push("log_sigma_omega".to_string());
    v
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let gamma = !args.iter().any(|a| a == "--no-gamma");
    let boot: usize = args.iter().position(|a| a == "--boot").map(|i| args[i + 1].parse().unwrap()).unwrap_or(0);
    let seed: u64 = args.iter().position(|a| a == "--seed").map(|i| args[i + 1].parse().unwrap()).unwrap_or(1);
    let text = std::fs::read_to_string(path).expect("input");
    // read rows: learner, cond, cell, y
    let mut rows: Vec<(usize, usize, usize, f64)> = Vec::new();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        rows.push((f[0].trim().parse().unwrap(), f[1].trim().parse().unwrap(), f[2].trim().parse().unwrap(), f[3].trim().parse().unwrap()));
    }
    let ncell = rows.iter().map(|r| r.2).max().unwrap() + 1;
    let nl = rows.iter().map(|r| r.0).max().unwrap() + 1;
    let nc = rows.iter().map(|r| r.1).max().unwrap() + 1;
    let mut ys: Vec<Vec<f64>> = vec![Vec::new(); ncell]; let mut lc = vec![(0usize, 0usize); ncell];
    for r in &rows { ys[r.2].push(r.3); lc[r.2] = (r.0, r.1); }
    let cells: Vec<Cell> = (0..ncell).filter(|&i| ys[i].len() >= 2).map(|i| {
        let y = &ys[i]; let r = y.len() as f64; let yb = y.iter().sum::<f64>() / r;
        Cell { learner: lc[i].0, cond: lc[i].1, r, ybar: yb, ssw: y.iter().map(|a| (a - yb).powi(2)).sum() } }).collect();
    let m = Model::new(cells.clone(), nl, nc, gamma);
    if args.iter().any(|a| a == "--check-grad") {
        // analytic gradient against central differences, at the starting values moved a little
        let mut p0 = m.start(); for (i, x) in p0.iter_mut().enumerate() { *x += 0.1 * ((i % 7) as f64 - 3.0) / 3.0; }
        let g = m.nll_grad(&p0, true).1; let mut worst: f64 = 0.0;
        for j in 0..p0.len() { let h = 1e-6 * (1.0 + p0[j].abs());
            let mut a = p0.clone(); a[j] += h; let mut b = p0.clone(); b[j] -= h;
            let fd = (m.nll_grad(&a, false).0 - m.nll_grad(&b, false).0) / (2.0 * h);
            worst = worst.max((fd - g[j]).abs() / (1.0 + fd.abs())); }
        println!("#gradient_check_max_rel_diff,{:e}", worst); return;
    }
    if std::env::var("MELS_VERBOSE").is_ok() { eprintln!("start: {} cells, {} parameters, objective {:.4}", m.cells.len(), m.npar(), m.nll_grad(&m.start(), false).0); }
    let (p, nll) = fit(&m, m.start());
    let se = hessian_se(&m, &p);
    // bootstrap: resample conditions with replacement, each with all its cells (a resampled condition gets a new
    // index, so its beta_c and gamma_c are its own); a replicate counts if the fit returns a finite objective
    let mut bs: Vec<Vec<f64>> = Vec::new();
    if boot > 0 {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let draws: Vec<Vec<usize>> = (0..boot).map(|_| (0..nc).map(|_| rng.gen_range(0..nc)).collect()).collect();
        bs = draws.iter().filter_map(|d| {
            let mut cs = Vec::new();
            for (newk, &k) in d.iter().enumerate() {
                for c in cells.iter().filter(|c| c.cond == k) { let mut c2 = c.clone(); c2.cond = newk; cs.push(c2); }
            }
            // every learner must appear (it does: each condition has all learners that ran on it)
            let mb = Model::new(cs, nl, nc, gamma);
            let (pb, fb) = fit(&mb, p.clone());
            if fb.is_finite() { Some(pb) } else { None }
        }).collect();
    }
    let nm = names(&m);
    // report the between-cell log variances on the scale used (alpha_eff), with delta-method SEs
    let mut p = p; let mut se = se;
    for l in 0..m.nl { let i = m.idx_alpha(l); let (a, da) = alpha_eff(p[i]); p[i] = a; se[i] *= da; }
    for b in bs.iter_mut() { for l in 0..m.nl { let i = m.idx_alpha(l); b[i] = alpha_eff(b[i]).0; } }
    if boot > 0 { println!("parameter,estimate,se,boot_se,boot_lo,boot_hi,boot_ok"); } else { println!("parameter,estimate,se"); }
    for i in 0..p.len() {
        if boot > 0 {
            let mut v: Vec<f64> = bs.iter().map(|b| b[i]).filter(|x| x.is_finite()).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let k = v.len(); let mean = v.iter().sum::<f64>() / k as f64;
            let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (k as f64 - 1.0)).sqrt();
            let q = |a: f64| v[((a * (k as f64 - 1.0)).round() as usize).min(k - 1)];
            println!("{},{},{},{},{},{},{}", nm[i], p[i], se[i], sd, q(0.025), q(0.975), k);
        } else { println!("{},{},{}", nm[i], p[i], se[i]); }
    }
    println!("#loglik,{}", -nll);
}
