#!/usr/bin/env python3
"""Synthetic HDVF validation: Heston recovery and leverage regimes.

This file is intentionally notebook-like and self-contained.  It keeps the
section order of ``Sathvik_heston.ipynb`` while using the more stable multi-seed
simulation and weak-matrix construction from the paper-v2 runner.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


import numpy as np

# Simple run defaults. Outputs are written directly beside this script.
# Main synthetic recovery follows Heston_Main (1).pdf, Phase 1 setup.
N_STEPS = 25_000
DT = 1 / 252
SEEDS = range(1, 31)
LEVERAGE_RHOS = [-0.20, -0.50, -0.65, -0.85]
KERNEL_CENTERS = 30
BANDWIDTH_FACTOR = 2.0
KERNEL_CENTER_GRID = [20, 30, 50, 80]
BANDWIDTH_FACTOR_GRID = [1.0, 1.5, 2.0, 3.0]

# Synthetic OHLC/proxy stress follows Sathvik_heston.ipynb.
NOTEBOOK_T_YEARS = 100
NOTEBOOK_DT_HIGHFREQ = 0.0001
PROXY_DAYS = int(NOTEBOOK_T_YEARS / DT)
INTRADAY_STEPS = int(DT / NOTEBOOK_DT_HIGHFREQ)
EWMA_SPANS = [3, 5, 7, 10, 14, 21, 30, 60]
NONLINEAR_BETAS = [0.0, 2.5, 5.0, 10.0, 15.0]
MODEL_SELECTION_BOOTSTRAPS = 30
EPS = 1e-12


@dataclass(frozen=True)
class HestonParams:
    mu: float = 0.05
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.7
    v0: float = 0.04
    x0: float = 0.0


@dataclass(frozen=True)
class NonlinearSVParams:
    mu: float = 0.05
    kappa: float = 1.5
    theta: float = 0.04
    xi: float = 0.45
    rho: float = -0.5
    beta: float = 0.0
    v0: float = 0.04
    x0: float = 0.0


HESTON_CACHE: dict[tuple[float, int, int], dict[str, np.ndarray]] = {}
OHLC_CACHE: dict[tuple[float, int, int, int], dict[str, np.ndarray]] = {}
NONLINEAR_CACHE: dict[tuple[float, int, int], dict[str, np.ndarray]] = {}


# ==========================================
# 2. Heston Data Generation
# ==========================================


def simulate_heston(params: HestonParams, n_steps: int, dt: float, seed: int) -> dict[str, np.ndarray]:
    """Simulate a synthetic Heston path with true latent variance available."""
    rng = np.random.default_rng(seed)
    x = np.empty(n_steps + 1)
    v = np.empty(n_steps + 1)
    dws = np.empty(n_steps)
    dwv = np.empty(n_steps)
    x[0] = params.x0
    v[0] = params.v0

    chol = np.linalg.cholesky(np.array([[1.0, params.rho], [params.rho, 1.0]]))
    sqrt_dt = math.sqrt(dt)

    for i in range(n_steps):
        z = chol @ rng.normal(size=2)
        dws[i] = z[0]
        dwv[i] = z[1]
        vt = max(v[i], 1e-10)
        x[i + 1] = x[i] + (params.mu - 0.5 * vt) * dt + math.sqrt(vt) * sqrt_dt * z[0]
        v_next = v[i] + params.kappa * (params.theta - vt) * dt + params.xi * math.sqrt(vt) * sqrt_dt * z[1]
        v[i + 1] = max(v_next, 1e-10)

    return {"x": x, "v": v, "dws": dws, "dwv": dwv}


def simulate_heston_cached(params: HestonParams, n_steps: int, seed: int) -> dict[str, np.ndarray]:
    key = (float(params.rho), int(n_steps), int(seed))
    if key not in HESTON_CACHE:
        HESTON_CACHE[key] = simulate_heston(params, n_steps, DT, seed)
    return HESTON_CACHE[key]


def simulate_nonlinear_sv(params: NonlinearSVParams, n_steps: int, dt: float, seed: int) -> dict[str, np.ndarray]:
    """Simulate Heston-like variance with optional quadratic drift curvature."""
    rng = np.random.default_rng(seed)
    x = np.empty(n_steps + 1)
    v = np.empty(n_steps + 1)
    x[0] = params.x0
    v[0] = params.v0
    chol = np.linalg.cholesky(np.array([[1.0, params.rho], [params.rho, 1.0]]))
    sqrt_dt = math.sqrt(dt)

    for i in range(n_steps):
        z = chol @ rng.normal(size=2)
        vt = max(v[i], 1e-10)
        drift = params.kappa * (params.theta - vt) - params.beta * (vt - params.theta) ** 2
        x[i + 1] = x[i] + (params.mu - 0.5 * vt) * dt + math.sqrt(vt) * sqrt_dt * z[0]
        v_next = v[i] + drift * dt + params.xi * math.sqrt(vt) * sqrt_dt * z[1]
        v[i + 1] = max(v_next, 1e-10)

    return {"x": x, "v": v}


def simulate_nonlinear_sv_cached(params: NonlinearSVParams, n_steps: int, seed: int) -> dict[str, np.ndarray]:
    key = (float(params.beta), int(n_steps), int(seed))
    if key not in NONLINEAR_CACHE:
        NONLINEAR_CACHE[key] = simulate_nonlinear_sv(params, n_steps, DT, seed)
    return NONLINEAR_CACHE[key]


def simulate_heston_ohlc(params: HestonParams, n_days: int, intraday_steps: int, seed: int) -> dict[str, np.ndarray]:
    """Simulate intraday synthetic Heston paths and aggregate to daily OHLC bars."""
    rng = np.random.default_rng(seed)
    dt_intraday = DT / intraday_steps
    sqrt_dt = math.sqrt(dt_intraday)
    x = params.x0
    v = params.v0
    chol = np.linalg.cholesky(np.array([[1.0, params.rho], [params.rho, 1.0]]))

    opens = np.empty(n_days)
    highs = np.empty(n_days)
    lows = np.empty(n_days)
    closes = np.empty(n_days)
    true_v = np.empty(n_days)
    log_open = np.empty(n_days)
    log_close = np.empty(n_days)

    for day in range(n_days):
        opens[day] = math.exp(x)
        log_open[day] = x
        true_v[day] = v
        day_prices = []
        for _ in range(intraday_steps):
            z = chol @ rng.normal(size=2)
            vt = max(v, 1e-10)
            x = x + (params.mu - 0.5 * vt) * dt_intraday + math.sqrt(vt) * sqrt_dt * z[0]
            v_next = v + params.kappa * (params.theta - vt) * dt_intraday + params.xi * math.sqrt(vt) * sqrt_dt * z[1]
            v = max(v_next, 1e-10)
            day_prices.append(math.exp(x))
        highs[day] = max(max(day_prices), opens[day])
        lows[day] = min(min(day_prices), opens[day])
        closes[day] = day_prices[-1]
        log_close[day] = x

    return {"open": opens, "high": highs, "low": lows, "close": closes, "x": log_open, "log_close": log_close, "v": true_v}


def simulate_heston_ohlc_cached(params: HestonParams, n_days: int, intraday_steps: int, seed: int) -> dict[str, np.ndarray]:
    key = (float(params.rho), int(n_days), int(intraday_steps), int(seed))
    if key not in OHLC_CACHE:
        OHLC_CACHE[key] = simulate_heston_ohlc(params, n_days, intraday_steps, seed)
    return OHLC_CACHE[key]


def garman_klass_variance(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, dt: float) -> np.ndarray:
    log_hl = np.log(np.maximum(high, EPS) / np.maximum(low, EPS))
    log_co = np.log(np.maximum(close, EPS) / np.maximum(open_, EPS))
    gk = 0.5 * log_hl**2 - (2 * math.log(2) - 1) * log_co**2
    return np.maximum(gk / dt, 1e-10)


def ewma(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for idx in range(1, len(values)):
        out[idx] = alpha * values[idx] + (1 - alpha) * out[idx - 1]
    return np.maximum(out, 1e-10)


def shifted_ewma(values: np.ndarray, span: int) -> np.ndarray:
    smooth = ewma(values, span)
    out = np.empty_like(smooth)
    out[0] = smooth[0]
    out[1:] = smooth[:-1]
    return np.maximum(out, 1e-10)


def proxy_stats(proxy: np.ndarray, true_v: np.ndarray) -> tuple[float, float, int]:
    n = min(len(proxy), len(true_v))
    proxy = proxy[:n]
    true_v = true_v[:n]
    nsr = float(np.std(proxy - true_v) / max(np.std(true_v), EPS))
    corr = float(np.corrcoef(proxy, true_v)[0, 1]) if n > 2 else np.nan
    max_lag = min(20, n // 5)
    best_lag = 0
    best_corr = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = proxy[:lag], true_v[-lag:]
        elif lag > 0:
            a, b = proxy[lag:], true_v[:-lag]
        else:
            a, b = proxy, true_v
        if len(a) > 2:
            candidate = np.corrcoef(a, b)[0, 1]
            if np.isfinite(candidate) and candidate > best_corr:
                best_corr = candidate
                best_lag = lag
    return nsr, corr, best_lag


def true_bv(v: np.ndarray, params: HestonParams) -> np.ndarray:
    return params.kappa * (params.theta - v)


def true_avv(v: np.ndarray, params: HestonParams) -> np.ndarray:
    return params.xi**2 * v


def true_axv(v: np.ndarray, params: HestonParams) -> np.ndarray:
    return params.rho * params.xi * v


# ==========================================
# 3. Weak-Form SINDy Core
# ==========================================


def gaussian_weights(state: np.ndarray, centers: np.ndarray, bandwidth: float) -> np.ndarray:
    weights = np.exp(-0.5 * ((state[None, :] - centers[:, None]) / max(bandwidth, EPS)) ** 2)
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), EPS)


def build_weak_matrices(
    x: np.ndarray,
    v_state: np.ndarray,
    dt: float,
    m: int = KERNEL_CENTERS,
    bandwidth_factor: float = BANDWIDTH_FACTOR,
) -> dict[str, np.ndarray | float]:
    """Build spatial-kernel weak matrices for drift, diffusion, and cross-variation."""
    n = min(len(x), len(v_state)) - 1
    state = np.maximum(np.asarray(v_state[:n]), 1e-10)
    dx = np.diff(np.asarray(x[: n + 1]))
    dv = np.diff(np.asarray(v_state[: n + 1]))

    quantiles = np.linspace(0.02, 0.98, min(m, len(state)))
    centers = np.quantile(state, quantiles)
    spacing = np.diff(np.unique(centers))
    base_bandwidth = float(np.median(spacing)) if len(spacing) else float(np.std(state))
    bandwidth = max(base_bandwidth * bandwidth_factor, 1e-8)

    weights = gaussian_weights(state, centers, bandwidth)
    theta = np.column_stack([np.ones_like(state), state])

    return {
        "state": state,
        "dx": dx,
        "dv": dv,
        "centers": centers,
        "bandwidth": bandwidth,
        "weights": weights,
        "drift_design": weights @ (theta * dt),
        "drift_target": weights @ dv,
        "linear_design": weights @ (state[:, None] * dt),
        "qv_target": weights @ (dv * dv),
        "qx_target": weights @ (dx * dx),
        "qxv_target": weights @ (dx * dv),
    }


def relative_l2(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(estimate - truth) / max(np.linalg.norm(truth), EPS))


def fit_ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(x, y, rcond=None)[0]


def quadratic_design(v: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(v), v, v * v])


def weak_project_direct(v: np.ndarray, y: np.ndarray, x: np.ndarray, m: int, bandwidth_factor: float) -> tuple[np.ndarray, np.ndarray]:
    centers = np.quantile(v, np.linspace(0.02, 0.98, min(m, len(v))))
    spacing = np.diff(np.unique(centers))
    base_bandwidth = float(np.median(spacing)) if len(spacing) else float(np.std(v))
    bandwidth = max(base_bandwidth * bandwidth_factor, 1e-8)
    weights = gaussian_weights(v, centers, bandwidth)
    return weights @ x, weights @ y


def bootstrap_quadratic_support(xw: np.ndarray, yw: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed + 4001)
    n = len(yw)
    chunk = max(n // 8, 4)
    selected = 0
    for _ in range(MODEL_SELECTION_BOOTSTRAPS):
        starts = rng.integers(0, max(n - chunk, 1), size=max(n // chunk, 1))
        idx = np.concatenate([np.arange(start, min(start + chunk, n)) for start in starts])[:n]
        xx = xw[idx]
        yy = yw[idx]
        linear_coef = fit_ols(xx[:, :2], yy)
        quadratic_coef = fit_ols(xx, yy)
        linear_rss = float(np.sum((yy - xx[:, :2] @ linear_coef) ** 2))
        quadratic_rss = float(np.sum((yy - xx @ quadratic_coef) ** 2))
        linear_bic = len(yy) * math.log(max(linear_rss, EPS) / len(yy)) + 2 * math.log(len(yy))
        quadratic_bic = len(yy) * math.log(max(quadratic_rss, EPS) / len(yy)) + 3 * math.log(len(yy))
        selected += int(quadratic_bic < linear_bic)
    return selected / MODEL_SELECTION_BOOTSTRAPS


def model_selection_stats(v: np.ndarray, dv: np.ndarray, seed: int) -> dict[str, float | int]:
    y = dv / DT
    design = quadratic_design(v)
    xw, yw = weak_project_direct(v, y, design, KERNEL_CENTERS, BANDWIDTH_FACTOR)
    cut = max(int(0.70 * len(yw)), 5)

    linear_coef = fit_ols(xw[:, :2], yw)
    quadratic_coef = fit_ols(xw, yw)
    linear_rss = float(np.sum((yw - xw[:, :2] @ linear_coef) ** 2))
    quadratic_rss = float(np.sum((yw - xw @ quadratic_coef) ** 2))
    bic_linear = len(yw) * math.log(max(linear_rss, EPS) / len(yw)) + 2 * math.log(len(yw))
    bic_quadratic = len(yw) * math.log(max(quadratic_rss, EPS) / len(yw)) + 3 * math.log(len(yw))

    linear_train = fit_ols(xw[:cut, :2], yw[:cut])
    quadratic_train = fit_ols(xw[:cut], yw[:cut])
    linear_loss = float(np.mean((yw[cut:] - xw[cut:, :2] @ linear_train) ** 2))
    quadratic_loss = float(np.mean((yw[cut:] - xw[cut:] @ quadratic_train) ** 2))
    validation_improvement = (linear_loss - quadratic_loss) / max(linear_loss, EPS)
    bootstrap_support = bootstrap_quadratic_support(xw, yw, seed)
    delta_bic = bic_quadratic - bic_linear

    bic_pass = int(delta_bic < 0.0)
    validation_pass = int(validation_improvement >= 0.20)
    bootstrap_pass = int(bootstrap_support >= 0.70)
    quadratic_selected = int(bic_pass and validation_pass and bootstrap_pass)
    linear_accepted = int((not bic_pass) and validation_improvement <= 0.05 and bootstrap_support <= 0.50)
    if quadratic_selected:
        decision = "quadratic_detected"
    elif linear_accepted:
        decision = "linear_confirmed"
    else:
        decision = "inconclusive"
    return {
        "bic_linear": bic_linear,
        "bic_quadratic": bic_quadratic,
        "delta_bic_quadratic_minus_linear": delta_bic,
        "validation_loss_linear": linear_loss,
        "validation_loss_quadratic": quadratic_loss,
        "validation_improvement_quadratic": validation_improvement,
        "bootstrap_quadratic_support": bootstrap_support,
        "bic_gate_pass": bic_pass,
        "validation_gate_pass": validation_pass,
        "bootstrap_gate_pass": bootstrap_pass,
        "quadratic_selected": quadratic_selected,
        "linear_accepted": linear_accepted,
        "decision": decision,
    }


# ==========================================
# 4. Sparse Regression & Execution
# ==========================================


def recover_heston_parameters(
    x: np.ndarray,
    v_state: np.ndarray,
    params: HestonParams,
    kernel_centers: int = KERNEL_CENTERS,
    bandwidth_factor: float = BANDWIDTH_FACTOR,
    dt: float = DT,
) -> dict[str, float | np.ndarray]:
    """Recover Heston generator functions and structural parameters."""
    mats = build_weak_matrices(x, v_state, dt, kernel_centers, bandwidth_factor)

    # Variance drift: b_v(v) = kappa*theta - kappa*v.
    drift_coef = fit_ols(mats["drift_design"], mats["drift_target"])
    c0 = float(drift_coef[0])
    c1 = float(drift_coef[1])
    kappa_hat = max(-c1, EPS)
    theta_hat = c0 / kappa_hat

    # Bias-corrected variance quadratic variation, matching the notebook logic.
    state = mats["state"]
    drift_on_state = c0 + c1 * state
    qv_corrected = mats["qv_target"] - mats["weights"] @ (drift_on_state * drift_on_state * dt * dt)

    # Diffusions: a_xx(v)=v, a_vv(v)=xi^2 v, a_xv(v)=rho*xi*v.
    avv_coef = float(fit_ols(mats["linear_design"], qv_corrected).item())
    axx_coef = float(fit_ols(mats["linear_design"], mats["qx_target"]).item())
    axv_coef = float(fit_ols(mats["linear_design"], mats["qxv_target"]).item())
    xi_hat = math.sqrt(max(avv_coef, EPS))
    rho_hat = axv_coef / math.sqrt(max(axx_coef * avv_coef, EPS))
    rho_hat = float(np.clip(rho_hat, -1.5, 1.5))

    grid_min, grid_max = np.quantile(state, [0.02, 0.98])
    grid = np.linspace(max(float(grid_min), 1e-8), max(float(grid_max), 1e-7), 200)
    b_hat = c0 + c1 * grid
    avv_hat = avv_coef * grid
    axv_hat = axv_coef * grid

    return {
        "kappa_hat": kappa_hat,
        "theta_hat": theta_hat,
        "xi_hat": xi_hat,
        "rho_hat": rho_hat,
        "axx_coef": axx_coef,
        "avv_coef": avv_coef,
        "axv_coef": axv_coef,
        "grid": grid,
        "b_hat": b_hat,
        "avv_hat": avv_hat,
        "axv_hat": axv_hat,
        "drift_l2_error": relative_l2(b_hat, true_bv(grid, params)),
        "diffusion_l2_error": relative_l2(avv_hat, true_avv(grid, params)),
        "cross_diffusion_l2_error": relative_l2(axv_hat, true_axv(grid, params)),
    }


def recovery_row(
    experiment: str,
    seed: int,
    params: HestonParams,
    rec: dict[str, float | np.ndarray],
    **extra: float | int | str,
) -> dict[str, float | int | str]:
    rho_hat = float(rec["rho_hat"])
    row: dict[str, float | int | str] = {
        "experiment": experiment,
        "seed": seed,
        "n_steps": N_STEPS,
        "kernel_centers": KERNEL_CENTERS,
        "bandwidth_factor": BANDWIDTH_FACTOR,
        "kappa_true": params.kappa,
        "kappa_hat": float(rec["kappa_hat"]),
        "kappa_rel_error": abs(float(rec["kappa_hat"]) - params.kappa) / max(abs(params.kappa), EPS),
        "theta_true": params.theta,
        "theta_hat": float(rec["theta_hat"]),
        "theta_rel_error": abs(float(rec["theta_hat"]) - params.theta) / max(abs(params.theta), EPS),
        "xi_true": params.xi,
        "xi_hat": float(rec["xi_hat"]),
        "xi_rel_error": abs(float(rec["xi_hat"]) - params.xi) / max(abs(params.xi), EPS),
        "rho_true": params.rho,
        "rho_hat": rho_hat,
        "rho_abs_error": abs(rho_hat - params.rho),
        "rho_sign_match": int(np.sign(rho_hat) == np.sign(params.rho)),
        "drift_l2_error": float(rec["drift_l2_error"]),
        "diffusion_l2_error": float(rec["diffusion_l2_error"]),
        "cross_diffusion_l2_error": float(rec["cross_diffusion_l2_error"]),
    }
    row.update(extra)
    return row


def run_synthetic_ground_truth() -> list[dict[str, float | int | str]]:
    print("Running synthetic ground-truth Heston recovery...")
    params = HestonParams(rho=-0.7)
    rows = []
    for seed in SEEDS:
        sim = simulate_heston_cached(params, N_STEPS, seed)
        rec = recover_heston_parameters(sim["x"], sim["v"], params)
        rows.append(recovery_row("synthetic_ground_truth", seed, params, rec))
        if seed == 1 or seed % 10 == 0:
            print(f"  seed {seed:02d}: kappa={rec['kappa_hat']:.4f}, xi={rec['xi_hat']:.4f}, rho={rec['rho_hat']:.4f}")
    return rows


def run_leverage_regimes() -> list[dict[str, float | int | str]]:
    print("Running multi-seed leverage-regime recovery...")
    rows = []
    total = len(list(SEEDS)) * len(LEVERAGE_RHOS)
    done = 0
    for rho in LEVERAGE_RHOS:
        params = HestonParams(rho=rho)
        for seed in SEEDS:
            sim = simulate_heston_cached(params, N_STEPS, seed)
            rec = recover_heston_parameters(sim["x"], sim["v"], params)
            rows.append(recovery_row("leverage_regime", seed, params, rec))
            done += 1
            if done == 1 or done % 20 == 0 or done == total:
                print(f"  {done:03d}/{total}: rho={rho:+.2f}, seed={seed:02d}, rho_hat={rec['rho_hat']:.4f}")
    return rows


def run_kernel_bandwidth_robustness() -> list[dict[str, float | int | str]]:
    print("Running kernel/bandwidth robustness grid...")
    params = HestonParams(rho=-0.7)
    jobs = [(seed, centers, bandwidth) for seed in SEEDS for centers in KERNEL_CENTER_GRID for bandwidth in BANDWIDTH_FACTOR_GRID]
    rows = []
    for done, (seed, centers, bandwidth) in enumerate(jobs, start=1):
        sim = simulate_heston_cached(params, N_STEPS, seed)
        rec = recover_heston_parameters(sim["x"], sim["v"], params, centers, bandwidth)
        rows.append(
            recovery_row(
                "kernel_bandwidth_robustness",
                seed,
                params,
                rec,
                kernel_centers=centers,
                bandwidth_factor=bandwidth,
            )
        )
        if done == 1 or done % 80 == 0 or done == len(jobs):
            print(f"  {done:03d}/{len(jobs)}: centers={centers}, bandwidth={bandwidth:g}, seed={seed:02d}")
    return rows


def proxy_variants(raw_gk: np.ndarray, true_v: np.ndarray) -> list[tuple[str, int, np.ndarray]]:
    variants = [("true_v", -1, true_v), ("raw_gk", -1, raw_gk)]
    for span in EWMA_SPANS:
        variants.append((f"shifted_ewma_{span}", span, shifted_ewma(raw_gk, span)))
    return variants


def run_proxy_stress_tests() -> list[dict[str, float | int | str]]:
    print("Running synthetic proxy stress tests...")
    params = HestonParams(rho=-0.7)
    rows = []
    total = len(list(SEEDS)) * (2 + len(EWMA_SPANS))
    done = 0
    for seed in SEEDS:
        data = simulate_heston_ohlc_cached(params, PROXY_DAYS, INTRADAY_STEPS, seed)
        raw_gk = garman_klass_variance(data["open"], data["high"], data["low"], data["close"], DT)
        for proxy_label, span, proxy in proxy_variants(raw_gk, data["v"]):
            n = min(len(data["x"]), len(proxy))
            rec = recover_heston_parameters(data["x"][:n], proxy[:n], params)
            nsr, corr, lag = proxy_stats(proxy[:n], data["v"][:n])
            rows.append(
                recovery_row(
                    "proxy_stress_test",
                    seed,
                    params,
                    rec,
                    n_steps=PROXY_DAYS,
                    proxy_type=proxy_label,
                    ewma_span=span,
                    proxy_nsr=nsr,
                    proxy_corr=corr,
                    proxy_lag=lag,
                )
            )
            done += 1
            if done == 1 or done % 45 == 0 or done == total:
                print(f"  {done:03d}/{total}: {proxy_label}, seed={seed:02d}, nsr={nsr:.3f}")
    return rows


def run_nonlinear_model_selection() -> list[dict[str, float | int | str]]:
    print("Running nonlinear model-selection failure-boundary sweep...")
    rows: list[dict[str, float | int | str]] = []
    jobs = [(seed, beta) for seed in SEEDS for beta in NONLINEAR_BETAS]
    for done, (seed, beta) in enumerate(jobs, start=1):
        params = NonlinearSVParams(beta=beta)
        sim = simulate_nonlinear_sv_cached(params, N_STEPS, seed)
        state = sim["v"][:-1]
        dv = np.diff(sim["v"])
        stats = model_selection_stats(state, dv, seed + int(beta * 1000))
        true_quadratic = int(beta > 0)
        quadratic_selected = int(stats["quadratic_selected"])
        linear_accepted = int(stats["linear_accepted"])
        false_quadratic = int(true_quadratic == 0 and quadratic_selected == 1)
        missed_quadratic = int(true_quadratic == 1 and quadratic_selected == 0)
        rows.append(
            {
                "experiment": "nonlinear_model_selection",
                "seed": seed,
                "n_steps": N_STEPS,
                "beta_true": beta,
                "true_quadratic": true_quadratic,
                "quadratic_selected": quadratic_selected,
                "linear_accepted": linear_accepted,
                "decision": str(stats["decision"]),
                "false_quadratic": false_quadratic,
                "missed_quadratic": missed_quadratic,
                "bic_gate_pass": int(stats["bic_gate_pass"]),
                "validation_gate_pass": int(stats["validation_gate_pass"]),
                "bootstrap_gate_pass": int(stats["bootstrap_gate_pass"]),
                "bic_linear": float(stats["bic_linear"]),
                "bic_quadratic": float(stats["bic_quadratic"]),
                "delta_bic_quadratic_minus_linear": float(stats["delta_bic_quadratic_minus_linear"]),
                "validation_loss_linear": float(stats["validation_loss_linear"]),
                "validation_loss_quadratic": float(stats["validation_loss_quadratic"]),
                "validation_improvement_quadratic": float(stats["validation_improvement_quadratic"]),
                "bootstrap_quadratic_support": float(stats["bootstrap_quadratic_support"]),
            }
        )
        if done == 1 or done % 45 == 0 or done == len(jobs):
            print(
                f"  {done:03d}/{len(jobs)}: beta={beta:g}, seed={seed:02d}, "
                f"decision={stats['decision']}, support={stats['bootstrap_quadratic_support']:.2f}"
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_parameters_used(path: Path) -> None:
    rows: list[dict[str, str | float | int]] = [
        {"parameter": "N_STEPS", "value": N_STEPS, "source": "Heston_Main Phase 1 synthetic baseline"},
        {"parameter": "DT", "value": DT, "source": "Heston_Main and Sathvik_heston daily macro step"},
        {"parameter": "mu", "value": HestonParams().mu, "source": "Heston_Main and Sathvik_heston Heston parameters"},
        {"parameter": "kappa", "value": HestonParams().kappa, "source": "Heston_Main and Sathvik_heston Heston parameters"},
        {"parameter": "theta", "value": HestonParams().theta, "source": "Heston_Main and Sathvik_heston Heston parameters"},
        {"parameter": "xi", "value": HestonParams().xi, "source": "Heston_Main and Sathvik_heston Heston parameters"},
        {"parameter": "rho", "value": HestonParams().rho, "source": "Heston_Main and Sathvik_heston Heston parameters"},
        {"parameter": "v0", "value": HestonParams().v0, "source": "Heston_Main and Sathvik_heston initial variance"},
        {"parameter": "KERNEL_CENTERS", "value": KERNEL_CENTERS, "source": "Heston_Main Phase 1 J=30"},
        {"parameter": "BANDWIDTH_FACTOR", "value": BANDWIDTH_FACTOR, "source": "Heston_Main bandwidth set to twice inter-centre spacing"},
        {"parameter": "NOTEBOOK_T_YEARS", "value": NOTEBOOK_T_YEARS, "source": "Sathvik_heston simulate_heston_and_ohlc(T=100)"},
        {"parameter": "NOTEBOOK_DT_HIGHFREQ", "value": NOTEBOOK_DT_HIGHFREQ, "source": "Sathvik_heston dt_highfreq=0.0001"},
        {"parameter": "PROXY_DAYS", "value": PROXY_DAYS, "source": "int(T / (1/252)) from Sathvik_heston"},
        {"parameter": "INTRADAY_STEPS", "value": INTRADAY_STEPS, "source": "int((1/252) / 0.0001) from Sathvik_heston"},
        {"parameter": "EWMA_PRIMARY_SPAN", "value": 14, "source": "Sathvik_heston and Heston_Main EWMA span"},
        {"parameter": "EWMA_SPANS", "value": ",".join(str(v) for v in EWMA_SPANS), "source": "Proxy stress around notebook/paper span discussion"},
        {"parameter": "SEEDS", "value": "1-30", "source": "HDVF multi-seed extension"},
    ]
    write_csv(path, rows)


def quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def summarize_parameter_recovery(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | str]]:
    specs = [
        ("kappa", "kappa_true", "kappa_hat", "kappa_rel_error"),
        ("theta", "theta_true", "theta_hat", "theta_rel_error"),
        ("xi", "xi_true", "xi_hat", "xi_rel_error"),
        ("rho", "rho_true", "rho_hat", "rho_abs_error"),
    ]
    summary = []
    for label, true_col, hat_col, error_col in specs:
        estimates = np.array([float(row[hat_col]) for row in rows], dtype=float)
        errors = np.array([float(row[error_col]) for row in rows], dtype=float)
        summary.append(
            {
                "parameter": label,
                "true": float(rows[0][true_col]),
                "mean_estimate": float(np.mean(estimates)),
                "std": float(np.std(estimates, ddof=1)),
                "median": float(np.median(estimates)),
                "p25": quantile(estimates, 0.25),
                "p75": quantile(estimates, 0.75),
                "median_error": float(np.median(errors)),
            }
        )
    return summary


def summarize_leverage_regimes(rows: list[dict[str, float | int | str]]) -> list[dict[str, float]]:
    summary = []
    for rho in LEVERAGE_RHOS:
        group = [row for row in rows if abs(float(row["rho_true"]) - rho) < 1e-12]
        rho_hat = np.array([float(row["rho_hat"]) for row in group], dtype=float)
        rho_error = np.array([float(row["rho_abs_error"]) for row in group], dtype=float)
        sign_match = np.array([float(row["rho_sign_match"]) for row in group], dtype=float)
        cross_error = np.array([float(row["cross_diffusion_l2_error"]) for row in group], dtype=float)
        summary.append(
            {
                "rho_true": rho,
                "rho_hat_mean": float(np.mean(rho_hat)),
                "rho_hat_std": float(np.std(rho_hat, ddof=1)),
                "rho_hat_median": float(np.median(rho_hat)),
                "rho_hat_p25": quantile(rho_hat, 0.25),
                "rho_hat_p75": quantile(rho_hat, 0.75),
                "rho_abs_error_mean": float(np.mean(rho_error)),
                "rho_abs_error_median": float(np.median(rho_error)),
                "rho_sign_match_rate": float(np.mean(sign_match)),
                "cross_diffusion_l2_error_median": float(np.median(cross_error)),
            }
        )
    return summary


def summarize_kernel_robustness(rows: list[dict[str, float | int | str]]) -> list[dict[str, float]]:
    summary = []
    for centers in KERNEL_CENTER_GRID:
        for bandwidth in BANDWIDTH_FACTOR_GRID:
            group = [
                row
                for row in rows
                if int(row["kernel_centers"]) == centers and abs(float(row["bandwidth_factor"]) - bandwidth) < 1e-12
            ]
            drift = np.array([float(row["drift_l2_error"]) for row in group], dtype=float)
            diffusion = np.array([float(row["diffusion_l2_error"]) for row in group], dtype=float)
            cross = np.array([float(row["cross_diffusion_l2_error"]) for row in group], dtype=float)
            rho_error = np.array([float(row["rho_abs_error"]) for row in group], dtype=float)
            summary.append(
                {
                    "kernel_centers": centers,
                    "bandwidth_factor": bandwidth,
                    "drift_l2_error_median": float(np.median(drift)),
                    "drift_l2_error_p75": quantile(drift, 0.75),
                    "diffusion_l2_error_median": float(np.median(diffusion)),
                    "diffusion_l2_error_p75": quantile(diffusion, 0.75),
                    "cross_diffusion_l2_error_median": float(np.median(cross)),
                    "cross_diffusion_l2_error_p75": quantile(cross, 0.75),
                    "rho_abs_error_median": float(np.median(rho_error)),
                }
            )
    return summary


def proxy_sort_key(label: str) -> tuple[int, int]:
    if label == "true_v":
        return (0, 0)
    if label == "raw_gk":
        return (1, 0)
    if label.startswith("shifted_ewma_"):
        return (2, int(label.rsplit("_", 1)[-1]))
    return (9, 0)


def summarize_proxy_stress(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | str]]:
    labels = sorted({str(row["proxy_type"]) for row in rows}, key=proxy_sort_key)
    summary = []
    for label in labels:
        group = [row for row in rows if str(row["proxy_type"]) == label]
        drift = np.array([float(row["drift_l2_error"]) for row in group], dtype=float)
        diffusion = np.array([float(row["diffusion_l2_error"]) for row in group], dtype=float)
        cross = np.array([float(row["cross_diffusion_l2_error"]) for row in group], dtype=float)
        nsr = np.array([float(row["proxy_nsr"]) for row in group], dtype=float)
        corr = np.array([float(row["proxy_corr"]) for row in group], dtype=float)
        rho_error = np.array([float(row["rho_abs_error"]) for row in group], dtype=float)
        summary.append(
            {
                "proxy_type": label,
                "ewma_span": float(group[0]["ewma_span"]),
                "proxy_nsr_median": float(np.median(nsr)),
                "proxy_corr_median": float(np.median(corr)),
                "drift_l2_error_median": float(np.median(drift)),
                "diffusion_l2_error_median": float(np.median(diffusion)),
                "cross_diffusion_l2_error_median": float(np.median(cross)),
                "rho_abs_error_median": float(np.median(rho_error)),
            }
        )
    return summary


def summarize_nonlinear_model_selection(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    for beta in NONLINEAR_BETAS:
        group = [row for row in rows if abs(float(row["beta_true"]) - beta) < 1e-12]
        selected = np.array([float(row["quadratic_selected"]) for row in group], dtype=float)
        linear = np.array([float(row["linear_accepted"]) for row in group], dtype=float)
        false_quad = np.array([float(row["false_quadratic"]) for row in group], dtype=float)
        missed_quad = np.array([float(row["missed_quadratic"]) for row in group], dtype=float)
        bootstrap = np.array([float(row["bootstrap_quadratic_support"]) for row in group], dtype=float)
        improvement = np.array([float(row["validation_improvement_quadratic"]) for row in group], dtype=float)
        delta_bic = np.array([float(row["delta_bic_quadratic_minus_linear"]) for row in group], dtype=float)
        detection_rate = float(np.mean(selected))
        linear_rate = float(np.mean(linear))
        inconclusive_rate = max(0.0, 1.0 - detection_rate - linear_rate)
        bic_pass = np.array([float(row["bic_gate_pass"]) for row in group], dtype=float)
        validation_pass = np.array([float(row["validation_gate_pass"]) for row in group], dtype=float)
        bootstrap_pass = np.array([float(row["bootstrap_gate_pass"]) for row in group], dtype=float)
        summary.append(
            {
                "beta_true": beta,
                "true_model_class": "linear" if beta == 0 else "quadratic_nonlinear",
                "linear_confirmed_rate": linear_rate,
                "inconclusive_rate": inconclusive_rate,
                "quadratic_selected_rate": detection_rate,
                "false_quadratic_rate": float(np.mean(false_quad)),
                "missed_quadratic_rate": float(np.mean(missed_quad)),
                "bic_prefers_quadratic_rate": float(np.mean(bic_pass)),
                "validation_gate_pass_rate": float(np.mean(validation_pass)),
                "bootstrap_gate_pass_rate": float(np.mean(bootstrap_pass)),
                "bootstrap_support_median": float(np.median(bootstrap)),
                "validation_improvement_median": float(np.median(improvement)),
                "delta_bic_median": float(np.median(delta_bic)),
                "passes_detection_gate": int(detection_rate >= 0.80) if beta > 0 else int(detection_rate <= 0.10),
            }
        )
    return summary


# ==========================================
# 5. Publication-Ready Plotting
# ==========================================


def set_matplotlib_rc():
    try:
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman"],
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42
        })
        return True
    except ImportError:
        return False

def plot_synthetic_recovery(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not set_matplotlib_rc(): return
    import matplotlib.pyplot as plt

    param_specs = [
        ("κ", "kappa_hat", "kappa_true"),
        ("θ", "theta_hat", "theta_true"),
        ("ξ", "xi_hat", "xi_true"),
        ("ρ", "rho_hat", "rho_true"),
    ]
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Synthetic Ground-Truth Recovery Across 30 Seeds", fontsize=16, fontweight="bold")
    
    param_values = []
    for _, hat_col, true_col in param_specs:
        vals = np.array([float(row[hat_col]) / float(row[true_col]) for row in rows], dtype=float)
        param_values.append(vals)
        
    axs[0].boxplot(param_values, positions=np.arange(1, len(param_specs) + 1), patch_artist=True,
                   boxprops=dict(facecolor="#dde9fa", color="#2357a5", linewidth=2),
                   medianprops=dict(color="#c44137", linewidth=2),
                   showmeans=True, meanprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black', markersize=5))
                   
    rng = np.random.default_rng(42)
    for idx, vals in enumerate(param_values, start=1):
        jitter = rng.uniform(-0.1, 0.1, size=len(vals))
        axs[0].scatter(idx + jitter, vals, color='#2357a5', s=10, alpha=0.5)
        
    axs[0].axhline(1.0, color='black', linewidth=2, linestyle='-')
    axs[0].text(len(param_specs) + 0.3, 1.0, 'true value = 1', va='center')
    axs[0].set_xticks(np.arange(1, len(param_specs) + 1))
    axs[0].set_xticklabels([label for label, _, _ in param_specs])
    axs[0].set_ylim(0.55, 1.40)
    axs[0].set_title("A. Normalized Parameter Recovery", fontweight="bold")
    axs[0].set_xlabel("Parameter")
    axs[0].set_ylabel("Estimate / true (unitless ratio)")
    axs[0].grid(True, linestyle="--", alpha=0.7)
    
    error_specs = [
        ("drift", "drift_l2_error"),
        ("diffusion", "diffusion_l2_error"),
        ("cross-diffusion", "cross_diffusion_l2_error"),
    ]
    error_values = [np.array([float(row[col]) for row in rows], dtype=float) for _, col in error_specs]
    
    axs[1].boxplot(error_values, positions=np.arange(1, len(error_specs) + 1), patch_artist=True,
                   boxprops=dict(facecolor="#dde9fa", color="#2357a5", linewidth=2),
                   medianprops=dict(color="#c44137", linewidth=2),
                   showmeans=True, meanprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black', markersize=5))
                   
    axs[1].set_xticks(np.arange(1, len(error_specs) + 1))
    axs[1].set_xticklabels([label for label, _ in error_specs])
    axs[1].set_title("B. Generator Error Distributions", fontweight="bold")
    axs[1].set_xlabel("Generator component")
    axs[1].set_ylabel("Relative L2 error (unitless)")
    axs[1].grid(True, linestyle="--", alpha=0.7)
    
    fig.text(0.5, 0.01, "Boxes summarize independent simulations; lower generator error means better recovered dynamics.", ha="center", fontsize=10, color="gray")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_rho_true_vs_hat(rows: list[dict[str, float | int | str]], path: Path) -> None:
    if not set_matplotlib_rc(): return
    import matplotlib.pyplot as plt

    rho_true = np.array([float(row["rho_true"]) for row in rows], dtype=float)
    rho_hat = np.array([float(row["rho_hat"]) for row in rows], dtype=float)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle("Multi-Seed Leverage Recovery", fontsize=16, fontweight="bold")
    
    ax.scatter(rho_true, rho_hat, color='#2357a5', alpha=0.7, label='one seed')
    ax.plot([-0.9, -0.15], [-0.9, -0.15], 'k--', linewidth=2, label='perfect recovery')
    
    ax.set_title("Recovered vs True Correlation", fontweight="bold")
    ax.set_xlabel("True ρ")
    ax.set_ylabel("Recovered ρ_hat (unitless)")
    ax.set_xlim(-0.9, -0.15)
    ax.set_ylim(-0.9, -0.15)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc='upper left')
    
    fig.text(0.5, 0.01, "Each dot is one seed in one leverage regime; dashed line is exact recovery.", ha="center", fontsize=10, color="gray")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_leverage_error_by_rho(rows: list[dict[str, float | int | str]], path: Path) -> None:
    if not set_matplotlib_rc(): return
    import matplotlib.pyplot as plt

    grouped_errors = []
    labels = []
    for rho in LEVERAGE_RHOS:
        errors = [float(row["rho_abs_error"]) for row in rows if abs(float(row["rho_true"]) - rho) < 1e-12]
        grouped_errors.append(errors)
        labels.append(f"{rho:.2f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Leverage Error by Correlation Regime", fontsize=16, fontweight="bold")
    
    ax.boxplot(grouped_errors, positions=np.arange(1, len(grouped_errors) + 1), patch_artist=True,
               boxprops=dict(facecolor="#dde9fa", color="#2357a5", linewidth=2),
               medianprops=dict(color="#c44137", linewidth=2),
               showmeans=True, meanprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black', markersize=5))
               
    ax.set_xticks(np.arange(1, len(grouped_errors) + 1))
    ax.set_xticklabels(labels)
    ax.set_title("Distribution Across 30 Seeds", fontweight="bold")
    ax.set_xlabel("True ρ")
    ax.set_ylabel("|ρ_hat - ρ_true| (correlation points)")
    ax.grid(True, linestyle="--", alpha=0.7)
    
    fig.text(0.5, 0.01, "Absolute rho error stays small across weak and strong negative leverage regimes.", ha="center", fontsize=10, color="gray")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_kernel_robustness(summary_rows: list[dict[str, float | int | str]], path: Path) -> None:
    if not set_matplotlib_rc(): return
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    metrics = [
        ("drift_l2_error_median", "Variance Drift Error"),
        ("diffusion_l2_error_median", "Variance Diffusion Error"),
        ("cross_diffusion_l2_error_median", "Leverage / Cross-Diffusion Error"),
    ]
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Kernel and Bandwidth Robustness", fontsize=16, fontweight="bold")
    
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["#f6f8fc", "#466996"])
    
    for idx, (metric, title) in enumerate(metrics):
        values = np.array([float(row[metric]) for row in summary_rows], dtype=float)
        vmin, vmax = float(np.min(values)), float(np.max(values))
        
        data = np.zeros((len(KERNEL_CENTER_GRID), len(BANDWIDTH_FACTOR_GRID)))
        lookup = {
            (int(row["kernel_centers"]), float(row["bandwidth_factor"])): float(row[metric])
            for row in summary_rows
        }
        
        for y_idx, centers in enumerate(KERNEL_CENTER_GRID):
            for x_idx, bandwidth in enumerate(BANDWIDTH_FACTOR_GRID):
                data[y_idx, x_idx] = lookup[(centers, bandwidth)]
                
        im = axs[idx].imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        
        axs[idx].set_xticks(np.arange(len(BANDWIDTH_FACTOR_GRID)))
        axs[idx].set_yticks(np.arange(len(KERNEL_CENTER_GRID)))
        axs[idx].set_xticklabels([f"{v:g}" for v in BANDWIDTH_FACTOR_GRID])
        axs[idx].set_yticklabels([str(v) for v in KERNEL_CENTER_GRID])
        axs[idx].set_title(title, fontweight="bold")
        axs[idx].set_xlabel("Bandwidth factor")
        if idx == 0:
            axs[idx].set_ylabel("Centers")
            
        for i in range(len(KERNEL_CENTER_GRID)):
            for j in range(len(BANDWIDTH_FACTOR_GRID)):
                val = data[i, j]
                color = "white" if (val - vmin) / max(1e-12, vmax - vmin) > 0.6 else "black"
                axs[idx].text(j, i, f"{val:.3g}", ha="center", va="center", color=color)
                
                # Mark default
                if KERNEL_CENTER_GRID[i] == KERNEL_CENTERS and abs(BANDWIDTH_FACTOR_GRID[j] - BANDWIDTH_FACTOR) < 1e-12:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='red', linewidth=3)
                    axs[idx].add_patch(rect)
                    
        cbar = fig.colorbar(im, ax=axs[idx], orientation='horizontal', fraction=0.046, pad=0.15)
        cbar.set_label("median relative L2 error", color="gray", size=10)
        
    fig.text(0.5, 0.01, "Rows are kernel centers; columns are bandwidth factors; cells show median relative L2 error; red box marks default.", ha="center", fontsize=10, color="gray")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def short_proxy_label(label: str) -> str:
    if label == "true_v": return "true"
    if label == "raw_gk": return "raw"
    if label.startswith("shifted_ewma_"): return "s" + label.rsplit("_", 1)[-1]
    return label


def plot_proxy_stress(rows: list[dict[str, float | int | str]], summary_rows: list[dict[str, float | str]], path: Path) -> None:
    if not set_matplotlib_rc(): return
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    labels = [str(row["proxy_type"]) for row in summary_rows]
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1.5]})
    fig.suptitle("Synthetic Proxy Stress Tests", fontsize=16, fontweight="bold")
    
    # Panel A
    nsr_groups = [
        np.array([float(row["proxy_nsr"]) for row in rows if str(row["proxy_type"]) == label], dtype=float)
        for label in labels
    ]
    
    axs[0].boxplot(nsr_groups, positions=np.arange(1, len(labels) + 1), patch_artist=True,
                   boxprops=dict(facecolor="#dde9fa", color="#2357a5", linewidth=2),
                   medianprops=dict(color="#c44137", linewidth=2),
                   showmeans=True, meanprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black', markersize=5))
                   
    axs[0].axhline(0.32, color='red', linewidth=2, linestyle='-')
    axs[0].text(len(labels) + 0.3, 0.32, 'stress gate = 0.32', color='red', va='bottom', ha='right')
    
    axs[0].set_xticks(np.arange(1, len(labels) + 1))
    axs[0].set_xticklabels([short_proxy_label(l) for l in labels])
    axs[0].set_ylim(0, 1.0)
    axs[0].set_title("A. Proxy Noise-to-Signal", fontweight="bold")
    axs[0].set_xlabel("Proxy choice")
    axs[0].set_ylabel("Proxy noise / signal (unitless ratio)")
    axs[0].grid(True, linestyle="--", alpha=0.7)
    
    # Panel B
    selected = [row for row in summary_rows if str(row["proxy_type"]) in {"true_v", "raw_gk", "shifted_ewma_14"}]
    metric_specs = [
        ("drift_l2_error_median", "Drift"),
        ("diffusion_l2_error_median", "Diffusion"),
        ("cross_diffusion_l2_error_median", "Leverage"),
        ("rho_abs_error_median", "Rho abs"),
    ]
    
    data = np.zeros((len(selected), len(metric_specs)))
    for r_idx, row in enumerate(selected):
        for c_idx, (metric, _) in enumerate(metric_specs):
            data[r_idx, c_idx] = float(row[metric])
            
    outlier_threshold = 10.0
    valid_data = data[data <= outlier_threshold]
    vmin, vmax = (np.min(valid_data), np.max(valid_data)) if len(valid_data) > 0 else (0, 1)
    
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["#f6f8fc", "#466996"])
    cmap.set_over('#965650') # Red for outliers
    
    im = axs[1].imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    
    axs[1].set_xticks(np.arange(len(metric_specs)))
    axs[1].set_yticks(np.arange(len(selected)))
    axs[1].set_xticklabels([label for _, label in metric_specs])
    axs[1].set_yticklabels([short_proxy_label(str(row["proxy_type"])) for row in selected])
    axs[1].set_title("B. Proxy Reliability Scorecard", fontweight="bold")
    
    for i in range(len(selected)):
        for j in range(len(metric_specs)):
            val = data[i, j]
            if val > outlier_threshold:
                text = f"{val:.1f}*"
                color = "white"
            else:
                text = f"{val:.2g}"
                color = "white" if (val - vmin) / max(1e-12, vmax - vmin) > 0.6 else "black"
            axs[1].text(j, i, text, ha="center", va="center", color=color)
            
    cbar = fig.colorbar(im, ax=axs[1], orientation='horizontal', fraction=0.046, pad=0.15, extend='max')
    cbar.set_label("median error scale", color="gray", size=10)
    axs[1].text(0, -0.25, "* off-scale raw GK drift/diffusion errors > 10", color='gray', fontsize=10, transform=axs[1].transAxes)
    
    fig.text(0.5, 0.01, "Left: proxy noise across seeds. Right: which proxies preserve recovered generator components.", ha="center", fontsize=10, color="gray")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_nonlinear_model_selection(summary_rows: list[dict[str, float | str]], path: Path) -> None:
    if not set_matplotlib_rc(): return
    import matplotlib.pyplot as plt

    betas = np.array([float(row["beta_true"]) for row in summary_rows], dtype=float)
    linear = np.array([float(row["linear_confirmed_rate"]) for row in summary_rows], dtype=float)
    inconclusive = np.array([float(row["inconclusive_rate"]) for row in summary_rows], dtype=float)
    selected = np.array([float(row["quadratic_selected_rate"]) for row in summary_rows], dtype=float)
    bic_pass = np.array([float(row["bic_prefers_quadratic_rate"]) for row in summary_rows], dtype=float)
    validation_pass = np.array([float(row["validation_gate_pass_rate"]) for row in summary_rows], dtype=float)
    bootstrap_pass = np.array([float(row["bootstrap_gate_pass_rate"]) for row in summary_rows], dtype=float)
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Nonlinear Drift Model-Selection: What Is Tested and What Fails", fontsize=16, fontweight="bold")
    
    # Panel A
    params = NonlinearSVParams()
    v_grid = np.linspace(0.005, 0.13, 200)
    
    axs[0].plot(v_grid, params.kappa * (params.theta - v_grid) - 0.0 * (v_grid - params.theta) ** 2, 'k-', linewidth=2, label="beta=0 linear")
    axs[0].plot(v_grid, params.kappa * (params.theta - v_grid) - 5.0 * (v_grid - params.theta) ** 2, color='#2357a5', linewidth=2, label="beta=5 nonlinear")
    axs[0].plot(v_grid, params.kappa * (params.theta - v_grid) - 15.0 * (v_grid - params.theta) ** 2, color='#c44137', linewidth=2, label="beta=15 nonlinear")
    
    axs[0].set_title("A. Ground-Truth Drift Shapes", fontweight="bold")
    axs[0].set_xlabel("Variance, v (annualized 1/year)")
    axs[0].set_ylabel("$b_v(v)$, annualized variance/year")
    axs[0].grid(True, linestyle="--", alpha=0.7)
    axs[0].legend(loc="lower left")
    
    # Panel B
    width = 0.6
    axs[1].bar(np.arange(len(betas)), linear, width, label='linear confirmed', color='#767676')
    axs[1].bar(np.arange(len(betas)), inconclusive, width, bottom=linear, label='inconclusive', color='#dde9fa')
    axs[1].bar(np.arange(len(betas)), selected, width, bottom=linear+inconclusive, label='quadratic detected', color='#2357a5')
    
    axs[1].set_xticks(np.arange(len(betas)))
    axs[1].set_xticklabels([f"{b:g}" for b in betas])
    axs[1].set_ylim(0, 1.0)
    axs[1].set_title("B. Final Decision Across 30 Seeds", fontweight="bold")
    axs[1].set_xlabel("Quadratic strength beta")
    axs[1].set_ylabel("Fraction of seeds (0-1)")
    axs[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    
    # Panel C
    axs[2].plot(betas, bic_pass, 'k-o', linewidth=2, label='BIC prefers quadratic')
    axs[2].plot(betas, validation_pass, color='#c44137', marker='o', linestyle='-', linewidth=2, label='validation improves')
    axs[2].plot(betas, bootstrap_pass, color='#2357a5', marker='o', linestyle='-', linewidth=2, label='bootstrap supports term')
    
    axs[2].axhline(0.70, color='gray', linestyle='--', linewidth=1)
    axs[2].text(np.max(betas), 0.72, '0.70 support gate', color='gray', ha='right')
    
    axs[2].set_xticks([0, 5, 10, 15])
    axs[2].set_ylim(-0.05, 1.05)
    axs[2].set_title("C. Evidence Gate Pass Rates", fontweight="bold")
    axs[2].set_xlabel("Quadratic strength beta")
    axs[2].set_ylabel("Fraction of seeds (0-1)")
    axs[2].grid(True, linestyle="--", alpha=0.7)
    axs[2].legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    
    fig.text(0.5, 0.01, "The selector is conservative: nonlinear claims require BIC, validation, and bootstrap support to agree.", ha="center", fontsize=10, color="gray")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    synthetic_rows = run_synthetic_ground_truth()
    leverage_rows = run_leverage_regimes()
    kernel_rows = run_kernel_bandwidth_robustness()
    proxy_rows = run_proxy_stress_tests()
    nonlinear_rows = run_nonlinear_model_selection()

    synthetic_summary = summarize_parameter_recovery(synthetic_rows)
    leverage_summary = summarize_leverage_regimes(leverage_rows)
    kernel_summary = summarize_kernel_robustness(kernel_rows)
    proxy_summary = summarize_proxy_stress(proxy_rows)
    nonlinear_summary = summarize_nonlinear_model_selection(nonlinear_rows)

    (ROOT / "csvs").mkdir(exist_ok=True)
    (ROOT / "pdfs").mkdir(exist_ok=True)

    write_parameters_used(ROOT / "csvs" / "run_parameters_used.csv")
    write_csv(ROOT / "csvs" / "synthetic_ground_truth_recovery.csv", synthetic_rows)
    write_csv(ROOT / "csvs" / "synthetic_ground_truth_summary.csv", synthetic_summary)
    write_csv(ROOT / "csvs" / "leverage_regime_recovery.csv", leverage_rows)
    write_csv(ROOT / "csvs" / "leverage_regime_summary.csv", leverage_summary)
    write_csv(ROOT / "csvs" / "kernel_bandwidth_robustness.csv", kernel_rows)
    write_csv(ROOT / "csvs" / "kernel_bandwidth_summary.csv", kernel_summary)
    write_csv(ROOT / "csvs" / "proxy_stress_recovery.csv", proxy_rows)
    write_csv(ROOT / "csvs" / "proxy_stress_summary.csv", proxy_summary)
    write_csv(ROOT / "csvs" / "nonlinear_model_selection.csv", nonlinear_rows)
    write_csv(ROOT / "csvs" / "nonlinear_model_selection_summary.csv", nonlinear_summary)

    plot_synthetic_recovery(ROOT / "pdfs" / "synthetic_heston_recovery.pdf", synthetic_rows)
    plot_rho_true_vs_hat(leverage_rows, ROOT / "pdfs" / "rho_true_vs_hat.pdf")
    plot_leverage_error_by_rho(leverage_rows, ROOT / "pdfs" / "leverage_error_by_rho.pdf")
    plot_kernel_robustness(kernel_summary, ROOT / "pdfs" / "kernel_bandwidth_robustness.pdf")
    plot_proxy_stress(proxy_rows, proxy_summary, ROOT / "pdfs" / "proxy_stress_tests.pdf")
    plot_nonlinear_model_selection(nonlinear_summary, ROOT / "pdfs" / "nonlinear_model_selection_boundaries.pdf")

    print("\nDone. Wrote outputs directly to:")
    for name in [
        "csvs/synthetic_ground_truth_recovery.csv",
        "csvs/run_parameters_used.csv",
        "csvs/synthetic_ground_truth_summary.csv",
        "csvs/leverage_regime_recovery.csv",
        "csvs/leverage_regime_summary.csv",
        "csvs/kernel_bandwidth_robustness.csv",
        "csvs/kernel_bandwidth_summary.csv",
        "csvs/proxy_stress_recovery.csv",
        "csvs/proxy_stress_summary.csv",
        "csvs/nonlinear_model_selection.csv",
        "csvs/nonlinear_model_selection_summary.csv",
        "pdfs/synthetic_heston_recovery.pdf",
        "pdfs/rho_true_vs_hat.pdf",
        "pdfs/leverage_error_by_rho.pdf",
        "pdfs/kernel_bandwidth_robustness.pdf",
        "pdfs/proxy_stress_tests.pdf",
        "pdfs/nonlinear_model_selection_boundaries.pdf",
    ]:
        print(f"  {ROOT / name}")


if __name__ == "__main__":
    main()

