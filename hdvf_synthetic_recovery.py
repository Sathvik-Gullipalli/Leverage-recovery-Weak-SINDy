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
from PIL import Image, ImageDraw, ImageFont, ImageChops

def save_tight_pdf(image: Image.Image, path: Path) -> None:
    bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = ImageChops.difference(image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        pad = 20
        bbox = (
            max(0, bbox[0] - pad),
            max(0, bbox[1] - pad),
            min(image.size[0], bbox[2] + pad),
            min(image.size[1], bbox[3] + pad),
        )
        image = image.crop(bbox)
    image.save(path, format="PDF", resolution=100.0)


# ==========================================
# 1. Publication-Ready Plot Settings
# ==========================================

WHITE = (255, 255, 255)
BLACK = (35, 35, 35)
GRID = (224, 224, 224)
BLUE = (35, 87, 165)
RED = (196, 65, 55)
GREY = (110, 110, 110)
LIGHT_BLUE = (221, 233, 250)


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


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


FONT_TINY = load_font(18)
FONT_SMALL = load_font(22)
FONT = load_font(26)
FONT_BOLD = load_font(30, bold=True)
FONT_TITLE = load_font(36, bold=True)


def nice_range(values: list[np.ndarray] | np.ndarray) -> tuple[float, float]:
    arr = np.concatenate([np.asarray(v, dtype=float).ravel() for v in values]) if isinstance(values, list) else np.asarray(values, dtype=float).ravel()
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not math.isfinite(lo) or not math.isfinite(hi):
        return 0.0, 1.0
    if abs(hi - lo) < EPS:
        pad = max(abs(hi) * 0.1, 1e-3)
    else:
        pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def map_point(x: float, y: float, xlim: tuple[float, float], ylim: tuple[float, float], box: tuple[int, int, int, int]) -> tuple[int, int]:
    left, top, right, bottom = box
    px = left + (x - xlim[0]) / max(xlim[1] - xlim[0], EPS) * (right - left)
    py = bottom - (y - ylim[0]) / max(ylim[1] - ylim[0], EPS) * (bottom - top)
    return int(round(px)), int(round(py))


def nice_tick_values(lo: float, hi: float, count: int = 5) -> list[float]:
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return [lo, hi]
    raw_step = (hi - lo) / max(count - 1, 1)
    exponent = math.floor(math.log10(raw_step))
    base = raw_step / (10 ** exponent)
    if base <= 1:
        nice_base = 1
    elif base <= 2:
        nice_base = 2
    elif base <= 5:
        nice_base = 5
    else:
        nice_base = 10
    step = nice_base * (10 ** exponent)
    start = math.ceil(lo / step) * step
    ticks = []
    value = start
    while value <= hi + step * 0.5:
        if lo - step * 0.25 <= value <= hi + step * 0.25:
            ticks.append(0.0 if abs(value) < EPS else value)
        value += step
    if not ticks:
        ticks = [lo, hi]
    return ticks


def tick_label(value: float) -> str:
    if abs(value) >= 10:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.2g}"
    if abs(value) >= 0.01:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3g}"


def draw_polyline(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], color: tuple[int, int, int], width: int = 4, dashed: bool = False) -> None:
    if len(points) < 2:
        return
    if not dashed:
        draw.line(points, fill=color, width=width, joint="curve")
        return
    for idx in range(len(points) - 1):
        if idx % 2 == 0:
            draw.line([points[idx], points[idx + 1]], fill=color, width=width)


def draw_axes(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    title: str,
    xlabel: str,
    ylabel: str,
    show_x_ticks: bool = True,
    show_y_ticks: bool = True,
    x_ticks: list[float] | None = None,
    y_ticks: list[float] | None = None,
    x_tick_labels: list[str] | None = None,
    y_tick_labels: list[str] | None = None,
) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline=BLACK, width=2)
    x_ticks = nice_tick_values(*xlim) if x_ticks is None else x_ticks
    y_ticks = nice_tick_values(*ylim) if y_ticks is None else y_ticks
    x_tick_labels = [tick_label(v) for v in x_ticks] if x_tick_labels is None else x_tick_labels
    y_tick_labels = [tick_label(v) for v in y_ticks] if y_tick_labels is None else y_tick_labels
    for x_val in x_ticks:
        if xlim[0] < x_val < xlim[1]:
            x, _ = map_point(x_val, ylim[0], xlim, ylim, box)
            draw.line([(x, top), (x, bottom)], fill=GRID, width=1)
    for y_val in y_ticks:
        if ylim[0] < y_val < ylim[1]:
            _, y = map_point(xlim[0], y_val, xlim, ylim, box)
            draw.line([(left, y), (right, y)], fill=GRID, width=1)
    draw.text(((left + right) // 2, top - 42), title, fill=BLACK, font=FONT_BOLD, anchor="mm")
    draw.text(((left + right) // 2, bottom + 42), xlabel, fill=BLACK, font=FONT, anchor="mm")
    draw_y_axis_label(draw, (left - 68, (top + bottom) // 2), ylabel)
    for x_val, label in zip(x_ticks, x_tick_labels):
        if show_x_ticks:
            x_px, _ = map_point(x_val, ylim[0], xlim, ylim, box)
            draw.text((x_px, bottom + 10), label, fill=GREY, font=FONT_SMALL, anchor="mt")
    for y_val, label in zip(y_ticks, y_tick_labels):
        if show_y_ticks:
            _, y_px = map_point(xlim[0], y_val, xlim, ylim, box)
            draw.text((left - 10, y_px), label, fill=GREY, font=FONT_SMALL, anchor="rm")


def draw_y_axis_label(draw: ImageDraw.ImageDraw, center: tuple[int, int], text: str) -> None:
    bbox = draw.textbbox((0, 0), text, font=FONT_SMALL)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    label = Image.new("RGBA", (w + 16, h + 16), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((8, 8), text, fill=BLACK, font=FONT_SMALL)
    rotated = label.rotate(90, expand=True)
    x = int(center[0] - rotated.size[0] / 2)
    y = int(center[1] - rotated.size[1] / 2)
    draw._image.paste(rotated, (x, y), rotated)


def draw_footer(draw: ImageDraw.ImageDraw, width: int, height: int, text: str) -> None:
    draw.text((width // 2, height - 30), text, fill=GREY, font=FONT_SMALL, anchor="mm")


def draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int, labels: list[tuple[str, tuple[int, int, int], bool]]) -> None:
    for idx, (label, color, dashed) in enumerate(labels):
        yy = y + idx * 32
        draw_polyline(draw, [(x, yy), (x + 42, yy)], color, width=4, dashed=dashed)
        draw.text((x + 54, yy), label, fill=BLACK, font=FONT_SMALL, anchor="lm")


def draw_dot_legend(draw: ImageDraw.ImageDraw, x: int, y: int, labels: list[tuple[str, tuple[int, int, int], str]]) -> None:
    for idx, (label, color, style) in enumerate(labels):
        yy = y + idx * 32
        if style == "ring":
            draw.ellipse((x - 11, yy - 11, x + 11, yy + 11), outline=color, width=4)
        else:
            draw.ellipse((x - 8, yy - 8, x + 8, yy + 8), fill=color)
        draw.text((x + 26, yy), label, fill=BLACK, font=FONT_SMALL, anchor="lm")


def draw_swatch_legend(draw: ImageDraw.ImageDraw, x: int, y: int, labels: list[tuple[str, tuple[int, int, int]]], step: int = 32) -> None:
    for idx, (label, color) in enumerate(labels):
        yy = y + idx * step
        draw.rectangle((x, yy - 10, x + 28, yy + 10), fill=color, outline=GREY, width=1)
        draw.text((x + 40, yy), label, fill=BLACK, font=FONT_SMALL, anchor="lm")


def draw_boxplot_key(draw: ImageDraw.ImageDraw, x: int, y: int, include_seed_dots: bool = False) -> None:
    draw.rectangle((x, y - 12, x + 34, y + 12), outline=BLUE, fill=LIGHT_BLUE, width=3)
    draw.text((x + 46, y), "box = middle 50% of seeds", fill=BLACK, font=FONT_SMALL, anchor="lm")
    draw.line([(x, y + 34), (x + 34, y + 34)], fill=RED, width=4)
    draw.text((x + 46, y + 34), "red line = median", fill=BLACK, font=FONT_SMALL, anchor="lm")
    draw.ellipse((x + 11, y + 60, x + 23, y + 72), fill=BLACK)
    draw.text((x + 46, y + 66), "black dot = mean", fill=BLACK, font=FONT_SMALL, anchor="lm")
    if include_seed_dots:
        draw.ellipse((x + 10, y + 90, x + 24, y + 104), fill=BLUE)
        draw.text((x + 46, y + 97), "blue dots = individual seeds", fill=BLACK, font=FONT_SMALL, anchor="lm")


def draw_box_distribution(
    draw: ImageDraw.ImageDraw,
    values: np.ndarray,
    x_pos: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
    fill: tuple[int, int, int],
    box_width: float = 0.25,
) -> None:
    q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    vmin, vmax = float(np.min(values)), float(np.max(values))
    mean = float(np.mean(values))
    cx = map_point(x_pos, med, xlim, ylim, box)[0]
    x_left = map_point(x_pos - box_width, med, xlim, ylim, box)[0]
    x_right = map_point(x_pos + box_width, med, xlim, ylim, box)[0]
    y_q1 = map_point(x_pos, float(q1), xlim, ylim, box)[1]
    y_q3 = map_point(x_pos, float(q3), xlim, ylim, box)[1]
    y_med = map_point(x_pos, float(med), xlim, ylim, box)[1]
    y_min = map_point(x_pos, vmin, xlim, ylim, box)[1]
    y_max = map_point(x_pos, vmax, xlim, ylim, box)[1]
    y_mean = map_point(x_pos, mean, xlim, ylim, box)[1]
    draw.line([(cx, y_min), (cx, y_max)], fill=BLACK, width=2)
    draw.rectangle((x_left, y_q3, x_right, y_q1), outline=color, fill=fill, width=3)
    draw.line([(x_left, y_med), (x_right, y_med)], fill=RED, width=3)
    draw.ellipse((cx - 5, y_mean - 5, cx + 5, y_mean + 5), fill=BLACK)


def plot_synthetic_recovery(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    """Plot HDVF-style repeated-trial recovery, not one generator curve."""
    img = Image.new("RGB", (1750, 890), WHITE)
    draw = ImageDraw.Draw(img)
    draw.text((875, 38), "Synthetic Ground-Truth Recovery Across 30 Seeds", fill=BLACK, font=FONT_TITLE, anchor="mm")
    draw_footer(draw, 1750, 890, "Boxes summarize independent simulations; lower generator error means better recovered dynamics.")

    # Panel A: recovered structural parameters normalized by the true value.
    param_specs = [
        ("κ", "kappa_hat", "kappa_true"),
        ("θ", "theta_hat", "theta_true"),
        ("ξ", "xi_hat", "xi_true"),
        ("ρ", "rho_hat", "rho_true"),
    ]
    param_values = []
    for _, hat_col, true_col in param_specs:
        vals = np.array([float(row[hat_col]) / float(row[true_col]) for row in rows], dtype=float)
        param_values.append(vals)

    box_a = (110, 160, 825, 650)
    ylim_a = (0.55, 1.40)
    xlim_a = (0.5, len(param_specs) + 0.5)
    draw_axes(
        draw,
        box_a,
        xlim_a,
        ylim_a,
        "A. Normalized Parameter Recovery",
        "Parameter",
        "Estimate / true (unitless ratio)",
        show_x_ticks=False,
        y_ticks=[0.60, 0.80, 1.00, 1.20, 1.40],
        y_tick_labels=["0.60", "0.80", "1.00", "1.20", "1.40"],
    )
    y_true = map_point(0.5, 1.0, xlim_a, ylim_a, box_a)[1]
    draw.line([(box_a[0], y_true), (box_a[2], y_true)], fill=BLACK, width=3)
    draw.text((box_a[2] - 12, y_true - 10), "true value = 1", fill=BLACK, font=FONT_SMALL, anchor="rs")

    for idx, ((label, _, _), vals) in enumerate(zip(param_specs, param_values), start=1):
        rng = np.random.default_rng(idx)
        for val in vals:
            jitter = float(rng.uniform(-0.10, 0.10))
            px, py = map_point(idx + jitter, float(val), xlim_a, ylim_a, box_a)
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=BLUE)
        draw_box_distribution(draw, vals, idx, xlim_a, ylim_a, box_a, BLUE, LIGHT_BLUE, box_width=0.22)
        draw.text((map_point(idx, ylim_a[0], xlim_a, ylim_a, box_a)[0], box_a[3] + 12), label, fill=GREY, font=FONT_SMALL, anchor="mt")

    # Panel B: reliability of the recovered generator components across seeds.
    error_specs = [
        ("drift", "drift_l2_error"),
        ("diffusion", "diffusion_l2_error"),
        ("cross-diffusion", "cross_diffusion_l2_error"),
    ]
    error_values = [np.array([float(row[col]) for row in rows], dtype=float) for _, col in error_specs]

    box_b = (980, 160, 1660, 650)
    max_error = float(max(np.max(vals) for vals in error_values))
    yhi_b = max(0.25, math.ceil(max_error / 0.05) * 0.05)
    ylim_b = (0.0, yhi_b)
    xlim_b = (0.5, len(error_specs) + 0.5)
    draw_axes(
        draw,
        box_b,
        xlim_b,
        ylim_b,
        "B. Generator Error Distributions",
        "Generator component",
        "Relative L2 error (unitless)",
        show_x_ticks=False,
        y_ticks=[round(v, 2) for v in np.linspace(0.0, yhi_b, 4)],
    )
    for idx, ((label, _), vals) in enumerate(zip(error_specs, error_values), start=1):
        draw_box_distribution(draw, vals, idx, xlim_b, ylim_b, box_b, BLUE, LIGHT_BLUE, box_width=0.22)
        draw.text((map_point(idx, ylim_b[0], xlim_b, ylim_b, box_b)[0], box_b[3] + 12), label, fill=GREY, font=FONT_SMALL, anchor="mt")

    draw_boxplot_key(draw, 1210, 730, include_seed_dots=True)
    save_tight_pdf(img, path)


def plot_rho_true_vs_hat(rows: list[dict[str, float | int | str]], path: Path) -> None:
    rho_true = np.array([float(row["rho_true"]) for row in rows], dtype=float)
    rho_hat = np.array([float(row["rho_hat"]) for row in rows], dtype=float)
    lo, hi = -0.90, -0.15
    xlim = (lo, hi)
    ylim = (lo, hi)
    box = (125, 170, 810, 705)

    img = Image.new("RGB", (940, 830), WHITE)
    draw = ImageDraw.Draw(img)
    draw.text((470, 38), "Multi-Seed Leverage Recovery", fill=BLACK, font=FONT_TITLE, anchor="mm")
    draw_footer(draw, 940, 830, "Each dot is one seed in one leverage regime; dashed line is exact recovery.")
    rho_ticks = [-0.85, -0.65, -0.50, -0.20]
    rho_tick_labels = ["-0.85", "-0.65", "-0.50", "-0.20"]
    draw_axes(
        draw,
        box,
        xlim,
        ylim,
        "Recovered vs True Correlation",
        "True ρ",
        "Recovered ρ_hat (unitless)",
        x_ticks=rho_ticks,
        y_ticks=rho_ticks,
        x_tick_labels=rho_tick_labels,
        y_tick_labels=rho_tick_labels,
    )
    diag = [map_point(lo, lo, xlim, ylim, box), map_point(hi, hi, xlim, ylim, box)]
    draw_polyline(draw, diag, BLACK, width=3, dashed=True)
    for x, y in zip(rho_true, rho_hat):
        px, py = map_point(float(x), float(y), xlim, ylim, box)
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=BLUE)
    draw_legend(draw, 150, 200, [("dashed = perfect recovery", BLACK, True)])
    draw_dot_legend(draw, 150, 245, [("blue dot = one seed", BLUE, "dot")])
    save_tight_pdf(img, path)


def plot_leverage_error_by_rho(rows: list[dict[str, float | int | str]], path: Path) -> None:
    grouped_errors = []
    labels = []
    for rho in LEVERAGE_RHOS:
        errors = [float(row["rho_abs_error"]) for row in rows if abs(float(row["rho_true"]) - rho) < 1e-12]
        grouped_errors.append(errors)
        labels.append(f"{rho:.2f}")

    all_errors = np.array([err for group in grouped_errors for err in group], dtype=float)
    xlim = (0.5, len(grouped_errors) + 0.5)
    yhi = max(0.025, math.ceil(float(np.max(all_errors)) / 0.005) * 0.005)
    ylim = (0.0, yhi)
    box = (130, 165, 795, 645)

    img = Image.new("RGB", (930, 900), WHITE)
    draw = ImageDraw.Draw(img)
    draw.text((465, 38), "Leverage Error by Correlation Regime", fill=BLACK, font=FONT_TITLE, anchor="mm")
    draw_footer(draw, 930, 900, "Absolute rho error stays small across weak and strong negative leverage regimes.")
    draw_axes(
        draw,
        box,
        xlim,
        ylim,
        "Distribution Across 30 Seeds",
        "True ρ",
        "|ρ_hat - ρ_true| (correlation points)",
        show_x_ticks=False,
        y_ticks=[round(v, 3) for v in np.linspace(0.0, yhi, 5)],
        y_tick_labels=[f"{v:.3f}".rstrip("0").rstrip(".") for v in np.linspace(0.0, yhi, 5)],
    )
    left, top, right, bottom = box
    width = (right - left) / len(grouped_errors)
    for idx, errors in enumerate(grouped_errors, start=1):
        vals = np.array(errors, dtype=float)
        q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        mean = float(np.mean(vals))
        cx = left + (idx - 0.5) * width
        box_w = width * 0.34
        y_q1 = map_point(idx, float(q1), xlim, ylim, box)[1]
        y_q3 = map_point(idx, float(q3), xlim, ylim, box)[1]
        y_med = map_point(idx, float(med), xlim, ylim, box)[1]
        y_min = map_point(idx, vmin, xlim, ylim, box)[1]
        y_max = map_point(idx, vmax, xlim, ylim, box)[1]
        y_mean = map_point(idx, mean, xlim, ylim, box)[1]
        draw.line([(cx, y_min), (cx, y_max)], fill=BLACK, width=2)
        draw.rectangle((cx - box_w, y_q3, cx + box_w, y_q1), outline=BLUE, fill=LIGHT_BLUE, width=3)
        draw.line([(cx - box_w, y_med), (cx + box_w, y_med)], fill=RED, width=3)
        draw.ellipse((cx - 5, y_mean - 5, cx + 5, y_mean + 5), fill=BLACK)
        draw.text((cx, bottom + 10), labels[idx - 1], fill=GREY, font=FONT_SMALL, anchor="mt")
    draw_boxplot_key(draw, 600, 720, include_seed_dots=False)
    save_tight_pdf(img, path)


def plot_kernel_robustness(summary_rows: list[dict[str, float | int | str]], path: Path) -> None:
    """Show kernel robustness as heatmaps for drift, diffusion, and leverage."""
    def heat_color(value: float, lo: float, hi: float) -> tuple[int, int, int]:
        t = 0.0 if hi <= lo else (value - lo) / (hi - lo)
        t = max(0.0, min(1.0, t))
        start = np.array([246, 248, 252], dtype=float)
        end = np.array([70, 105, 150], dtype=float)
        rgb = start * (1.0 - t) + end * t
        return tuple(int(x) for x in rgb)

    def draw_color_scale(
        box: tuple[int, int, int, int],
        value_range: tuple[float, float],
    ) -> None:
        left, top, right, bottom = box
        lo, hi = value_range
        bar_left = left + 38
        bar_right = right - 38
        bar_top = bottom + 76
        bar_bottom = bar_top + 16
        for x in range(bar_left, bar_right):
            t = (x - bar_left) / max(1, bar_right - bar_left - 1)
            value = lo + t * (hi - lo)
            draw.line([(x, bar_top), (x, bar_bottom)], fill=heat_color(value, lo, hi))
        draw.rectangle((bar_left, bar_top, bar_right, bar_bottom), outline=GREY, width=1)
        draw.text((bar_left, bar_bottom + 8), f"{lo:.3g}", fill=GREY, font=FONT_TINY, anchor="lt")
        draw.text((bar_right, bar_bottom + 8), f"{hi:.3g}", fill=GREY, font=FONT_TINY, anchor="rt")
        draw.text(((bar_left + bar_right) // 2, bar_bottom + 30), "lower median L2 error  ->  higher", fill=GREY, font=FONT_TINY, anchor="mt")

    def draw_heatmap_panel(
        box: tuple[int, int, int, int],
        metric: str,
        title: str,
        value_range: tuple[float, float],
    ) -> None:
        left, top, right, bottom = box
        draw.text(((left + right) // 2, top - 36), title, fill=BLACK, font=FONT_BOLD, anchor="mm")
        cell_w = (right - left) / len(BANDWIDTH_FACTOR_GRID)
        cell_h = (bottom - top) / len(KERNEL_CENTER_GRID)
        lookup = {
            (int(row["kernel_centers"]), float(row["bandwidth_factor"])): float(row[metric])
            for row in summary_rows
        }
        for y_idx, centers in enumerate(KERNEL_CENTER_GRID):
            for x_idx, bandwidth in enumerate(BANDWIDTH_FACTOR_GRID):
                x0 = int(left + x_idx * cell_w)
                x1 = int(left + (x_idx + 1) * cell_w)
                y0 = int(top + y_idx * cell_h)
                y1 = int(top + (y_idx + 1) * cell_h)
                value = lookup[(centers, bandwidth)]
                draw.rectangle((x0, y0, x1, y1), fill=heat_color(value, *value_range), outline=WHITE, width=2)
                draw.text(((x0 + x1) // 2, (y0 + y1) // 2), f"{value:.3g}", fill=BLACK, font=FONT_SMALL, anchor="mm")
                if centers == KERNEL_CENTERS and abs(bandwidth - BANDWIDTH_FACTOR) < 1e-12:
                    draw.rectangle((x0 + 4, y0 + 4, x1 - 4, y1 - 4), outline=RED, width=4)
        draw.rectangle(box, outline=BLACK, width=2)
        for x_idx, bandwidth in enumerate(BANDWIDTH_FACTOR_GRID):
            x = left + (x_idx + 0.5) * cell_w
            draw.text((x, bottom + 12), f"{bandwidth:g}", fill=GREY, font=FONT_SMALL, anchor="mt")
        for y_idx, centers in enumerate(KERNEL_CENTER_GRID):
            y = top + (y_idx + 0.5) * cell_h
            draw.text((left - 12, y), str(centers), fill=GREY, font=FONT_SMALL, anchor="rm")
        draw.text(((left + right) // 2, bottom + 45), "Bandwidth factor", fill=BLACK, font=FONT_SMALL, anchor="mt")
        draw.text((left - 4, top - 8), "Centers", fill=BLACK, font=FONT_SMALL, anchor="rs")
        draw_color_scale(box, value_range)

    metrics = [
        ("drift_l2_error_median", "Variance Drift Error"),
        ("diffusion_l2_error_median", "Variance Diffusion Error"),
        ("cross_diffusion_l2_error_median", "Leverage / Cross-Diffusion Error"),
    ]
    img = Image.new("RGB", (1850, 735), WHITE)
    draw = ImageDraw.Draw(img)
    draw.text((925, 42), "Kernel and Bandwidth Robustness", fill=BLACK, font=FONT_TITLE, anchor="mm")
    draw_footer(draw, 1850, 735, "Rows are kernel centers; columns are bandwidth factors; cells show median relative L2 error; red box marks default.")
    boxes = [(105, 160, 530, 520), (710, 160, 1135, 520), (1315, 160, 1740, 520)]
    for box, (metric, title) in zip(boxes, metrics):
        values = np.array([float(row[metric]) for row in summary_rows], dtype=float)
        draw_heatmap_panel(box, metric, title, (float(np.min(values)), float(np.max(values))))
    save_tight_pdf(img, path)


def short_proxy_label(label: str) -> str:
    if label == "true_v":
        return "true"
    if label == "raw_gk":
        return "raw"
    if label.startswith("shifted_ewma_"):
        return "s" + label.rsplit("_", 1)[-1]
    return label


def plot_proxy_stress(rows: list[dict[str, float | int | str]], summary_rows: list[dict[str, float | str]], path: Path) -> None:
    labels = [str(row["proxy_type"]) for row in summary_rows]
    box_a = (125, 165, 785, 655)
    box_b = (930, 165, 1690, 655)
    img = Image.new("RGB", (1760, 820), WHITE)
    draw = ImageDraw.Draw(img)
    draw.text((880, 38), "Synthetic Proxy Stress Tests", fill=BLACK, font=FONT_TITLE, anchor="mm")
    draw_footer(draw, 1760, 820, "Left: proxy noise across seeds. Right: which proxies preserve recovered generator components.")

    nsr_groups = [
        np.array([float(row["proxy_nsr"]) for row in rows if str(row["proxy_type"]) == label], dtype=float)
        for label in labels
    ]
    xlim = (0.5, len(labels) + 0.5)
    ylim = (0.0, 1.0)
    draw_axes(
        draw,
        box_a,
        xlim,
        ylim,
        "A. Proxy Noise-to-Signal",
        "Proxy choice",
        "Proxy noise / signal (unitless ratio)",
        show_x_ticks=False,
        y_ticks=[0.00, 0.25, 0.50, 0.75, 1.00],
        y_tick_labels=["0.00", "0.25", "0.50", "0.75", "1.00"],
    )
    y_gate = map_point(0.5, 0.32, xlim, ylim, box_a)[1]
    if box_a[1] <= y_gate <= box_a[3]:
        draw.line([(box_a[0], y_gate), (box_a[2], y_gate)], fill=RED, width=3)
        draw.text((box_a[2] - 12, y_gate - 8), "stress gate = 0.32", fill=RED, font=FONT_SMALL, anchor="rs")
    for idx, vals in enumerate(nsr_groups, start=1):
        draw_box_distribution(draw, vals, idx, xlim, ylim, box_a, BLUE, LIGHT_BLUE, box_width=0.20)
        draw.text((map_point(idx, ylim[0], xlim, ylim, box_a)[0], box_a[3] + 12), short_proxy_label(labels[idx - 1]), fill=GREY, font=FONT_SMALL, anchor="mt")

    selected = [row for row in summary_rows if str(row["proxy_type"]) in {"true_v", "raw_gk", "shifted_ewma_14"}]
    metric_specs = [
        ("drift_l2_error_median", "Drift"),
        ("diffusion_l2_error_median", "Diffusion"),
        ("cross_diffusion_l2_error_median", "Leverage"),
        ("rho_abs_error_median", "Rho abs"),
    ]
    left, top, right, bottom = box_b
    draw.text(((left + right) // 2, top - 42), "B. Proxy Reliability Scorecard", fill=BLACK, font=FONT_BOLD, anchor="mm")
    draw.text(((left + right) // 2, top - 12), "Cells show median error; darker blue means larger error", fill=GREY, font=FONT_TINY, anchor="mm")

    table_width = 620
    table_left = int((left + right - table_width) / 2)
    table_top = top + 72
    table_right = table_left + table_width
    table_bottom = bottom - 95
    label_w = 110
    cell_w = (table_right - table_left - label_w) / len(metric_specs)
    cell_h = (table_bottom - table_top) / len(selected)

    outlier_threshold = 10.0
    selected_values = [float(row[metric]) for row in selected for metric, _ in metric_specs]
    gradient_values = [v for v in selected_values if v <= outlier_threshold]
    color_lo = min(gradient_values) if gradient_values else 0.0
    color_hi = max(gradient_values) if gradient_values else 1.0

    def reliability_color(value: float) -> tuple[int, int, int]:
        if value > outlier_threshold:
            return (150, 86, 80)
        t = 0.0 if color_hi <= color_lo else (float(value) - color_lo) / (color_hi - color_lo)
        t = max(0.0, min(1.0, t))
        start = np.array([246, 248, 252], dtype=float)
        end = np.array([70, 105, 150], dtype=float)
        rgb = start * (1.0 - t) + end * t
        return tuple(int(x) for x in rgb)

    for col_idx, (_, label) in enumerate(metric_specs):
        x = table_left + label_w + (col_idx + 0.5) * cell_w
        draw.text((x, table_top - 18), label, fill=BLACK, font=FONT_SMALL, anchor="mb")
    for row_idx, row in enumerate(selected):
        y0 = int(table_top + row_idx * cell_h)
        y1 = int(table_top + (row_idx + 1) * cell_h)
        proxy_label = short_proxy_label(str(row["proxy_type"]))
        draw.text((table_left + label_w - 14, (y0 + y1) // 2), proxy_label, fill=BLACK, font=FONT_SMALL, anchor="rm")
        for col_idx, (metric, _) in enumerate(metric_specs):
            x0 = int(table_left + label_w + col_idx * cell_w)
            x1 = int(table_left + label_w + (col_idx + 1) * cell_w)
            value = float(row[metric])
            draw.rectangle((x0, y0, x1, y1), fill=reliability_color(value), outline=WHITE, width=3)
            text = f"{value:.2g}" if value <= outlier_threshold else f"{value:.1f}*"
            draw.text(((x0 + x1) // 2, (y0 + y1) // 2), text, fill=BLACK, font=FONT_SMALL, anchor="mm")
    draw.rectangle((table_left + label_w, table_top, table_right, table_bottom), outline=BLACK, width=2)

    bar_left = table_left + label_w
    bar_right = table_right
    bar_top = bottom - 56
    bar_bottom = bar_top + 16
    for x in range(bar_left, bar_right):
        frac = (x - bar_left) / max(1, bar_right - bar_left - 1)
        value = color_lo + frac * (color_hi - color_lo)
        draw.line([(x, bar_top), (x, bar_bottom)], fill=reliability_color(value))
    draw.rectangle((bar_left, bar_top, bar_right, bar_bottom), outline=GREY, width=1)
    draw.text((bar_left, bar_bottom + 8), f"{color_lo:.2g}", fill=GREY, font=FONT_TINY, anchor="lt")
    draw.text((bar_right, bar_bottom + 8), f"{color_hi:.2g}", fill=GREY, font=FONT_TINY, anchor="rt")
    draw.text(((bar_left + bar_right) // 2, bar_bottom + 28), "median error scale", fill=GREY, font=FONT_TINY, anchor="mt")
    draw.rectangle((bar_left, bar_bottom + 42, bar_left + 20, bar_bottom + 58), fill=(150, 86, 80), outline=GREY, width=1)
    draw.text((bar_left + 28, bar_bottom + 50), "* off-scale raw GK drift/diffusion errors > 10", fill=GREY, font=FONT_TINY, anchor="lm")
    save_tight_pdf(img, path)


def draw_metric_line(
    draw: ImageDraw.ImageDraw,
    x_values: np.ndarray,
    y_values: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
    label_points: bool = False,
) -> None:
    points = [map_point(float(x), float(y), xlim, ylim, box) for x, y in zip(x_values, y_values)]
    draw_polyline(draw, points, color, width=4)
    for point, y in zip(points, y_values):
        px, py = point
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=color)
        if label_points:
            draw.text((px, py - 14), f"{float(y):.2g}", fill=color, font=FONT_SMALL, anchor="mb")


def plot_nonlinear_model_selection(summary_rows: list[dict[str, float | str]], path: Path) -> None:
    betas = np.array([float(row["beta_true"]) for row in summary_rows], dtype=float)
    linear = np.array([float(row["linear_confirmed_rate"]) for row in summary_rows], dtype=float)
    inconclusive = np.array([float(row["inconclusive_rate"]) for row in summary_rows], dtype=float)
    selected = np.array([float(row["quadratic_selected_rate"]) for row in summary_rows], dtype=float)
    bic_pass = np.array([float(row["bic_prefers_quadratic_rate"]) for row in summary_rows], dtype=float)
    validation_pass = np.array([float(row["validation_gate_pass_rate"]) for row in summary_rows], dtype=float)
    bootstrap_pass = np.array([float(row["bootstrap_gate_pass_rate"]) for row in summary_rows], dtype=float)

    img = Image.new("RGB", (2050, 900), WHITE)
    draw = ImageDraw.Draw(img)
    draw.text((1025, 38), "Nonlinear Drift Model-Selection: What Is Tested and What Fails", fill=BLACK, font=FONT_TITLE, anchor="mm")
    draw_footer(draw, 2050, 900, "The selector is conservative: nonlinear claims require BIC, validation, and bootstrap support to agree.")

    # Panel A: show the actual ground-truth drift shapes.
    params = NonlinearSVParams()
    v_grid = np.linspace(0.005, 0.13, 200)
    shape_betas = [0.0, 5.0, 15.0]
    curves = [
        params.kappa * (params.theta - v_grid) - beta * (v_grid - params.theta) ** 2
        for beta in shape_betas
    ]
    box_a = (90, 165, 605, 650)
    xlim_a = (float(np.min(v_grid)), float(np.max(v_grid)))
    ylim_a = nice_range(curves)
    draw_axes(
        draw,
        box_a,
        xlim_a,
        ylim_a,
        "A. Ground-Truth Drift Shapes",
        "Variance, v (annualized 1/year)",
        "b_v(v), annualized variance/year",
        x_ticks=[0.02, 0.05, 0.08, 0.11],
        x_tick_labels=["0.02", "0.05", "0.08", "0.11"],
    )
    for beta, curve, color, dashed in [
        (0.0, curves[0], BLACK, False),
        (5.0, curves[1], BLUE, False),
        (15.0, curves[2], RED, False),
    ]:
        points = [map_point(float(x), float(y), xlim_a, ylim_a, box_a) for x, y in zip(v_grid, curve)]
        draw_polyline(draw, points, color, width=4, dashed=dashed)
    draw_legend(draw, 155, 730, [("beta=0 linear", BLACK, False), ("beta=5 nonlinear", BLUE, False), ("beta=15 nonlinear", RED, False)])

    # Panel B: final selector outcomes are easier to read as stacked bars.
    box_b = (775, 165, 1290, 650)
    xlim_b = (0.5, len(betas) + 0.5)
    ylim_b = (0.0, 1.0)
    draw_axes(
        draw,
        box_b,
        xlim_b,
        ylim_b,
        "B. Final Decision Across 30 Seeds",
        "Quadratic strength beta",
        "Fraction of seeds (0-1)",
        show_x_ticks=False,
        y_ticks=[0.00, 0.25, 0.50, 0.75, 1.00],
        y_tick_labels=["0", "0.25", "0.50", "0.75", "1.00"],
    )
    left, top, right, bottom = box_b
    bar_w = (right - left) / len(betas) * 0.48
    for idx, beta in enumerate(betas, start=1):
        cx = map_point(idx, 0, xlim_b, ylim_b, box_b)[0]
        y0 = bottom
        segments = [
            (linear[idx - 1], GREY, "linear"),
            (inconclusive[idx - 1], LIGHT_BLUE, "inconclusive"),
            (selected[idx - 1], BLUE, "quadratic"),
        ]
        cumulative = 0.0
        for rate, color, _ in segments:
            cumulative += float(rate)
            y1 = map_point(idx, cumulative, xlim_b, ylim_b, box_b)[1]
            draw.rectangle((cx - bar_w / 2, y1, cx + bar_w / 2, y0), fill=color, outline=WHITE)
            y0 = y1
        draw.text((cx, bottom + 12), f"{beta:g}", fill=GREY, font=FONT_SMALL, anchor="mt")
    draw_swatch_legend(draw, 990, 735, [("linear confirmed", GREY), ("inconclusive", LIGHT_BLUE), ("quadratic detected", BLUE)], step=28)

    # Panel C: show which required evidence gates pass before final selection.
    box_c = (1460, 165, 1975, 650)
    xlim_c = (min(betas) - 0.5, max(betas) + 0.5)
    ylim_c = (-0.05, 1.05)
    draw_axes(
        draw,
        box_c,
        xlim_c,
        ylim_c,
        "C. Evidence Gate Pass Rates",
        "Quadratic strength beta",
        "Fraction of seeds (0-1)",
        x_ticks=[0, 5, 10, 15],
        y_ticks=[0.00, 0.25, 0.50, 0.75, 1.00],
        y_tick_labels=["0", "0.25", "0.50", "0.75", "1.00"],
    )
    y_gate = map_point(xlim_c[0], 0.70, xlim_c, ylim_c, box_c)[1]
    draw.line([(box_c[0], y_gate), (box_c[2], y_gate)], fill=GREY, width=2)
    draw.text((box_c[2] - 10, y_gate - 8), "0.70 support gate", fill=GREY, font=FONT_TINY, anchor="rs")
    draw_metric_line(draw, betas, bic_pass, xlim_c, ylim_c, box_c, BLACK)
    draw_metric_line(draw, betas, validation_pass, xlim_c, ylim_c, box_c, RED)
    draw_metric_line(draw, betas, bootstrap_pass, xlim_c, ylim_c, box_c, BLUE)
    draw_legend(draw, 1575, 730, [("BIC prefers quadratic", BLACK, False), ("validation improves", RED, False), ("bootstrap supports term", BLUE, False)])
    save_tight_pdf(img, path)


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

    write_parameters_used(ROOT / "run_parameters_used.csv")
    write_csv(ROOT / "synthetic_ground_truth_recovery.csv", synthetic_rows)
    write_csv(ROOT / "synthetic_ground_truth_summary.csv", synthetic_summary)
    write_csv(ROOT / "leverage_regime_recovery.csv", leverage_rows)
    write_csv(ROOT / "leverage_regime_summary.csv", leverage_summary)
    write_csv(ROOT / "kernel_bandwidth_robustness.csv", kernel_rows)
    write_csv(ROOT / "kernel_bandwidth_summary.csv", kernel_summary)
    write_csv(ROOT / "proxy_stress_recovery.csv", proxy_rows)
    write_csv(ROOT / "proxy_stress_summary.csv", proxy_summary)
    write_csv(ROOT / "nonlinear_model_selection.csv", nonlinear_rows)
    write_csv(ROOT / "nonlinear_model_selection_summary.csv", nonlinear_summary)

    plot_synthetic_recovery(ROOT / "synthetic_heston_recovery.pdf", synthetic_rows)
    plot_rho_true_vs_hat(leverage_rows, ROOT / "rho_true_vs_hat.pdf")
    plot_leverage_error_by_rho(leverage_rows, ROOT / "leverage_error_by_rho.pdf")
    plot_kernel_robustness(kernel_summary, ROOT / "kernel_bandwidth_robustness.pdf")
    plot_proxy_stress(proxy_rows, proxy_summary, ROOT / "proxy_stress_tests.pdf")
    plot_nonlinear_model_selection(nonlinear_summary, ROOT / "nonlinear_model_selection_boundaries.pdf")

    print("\nDone. Wrote outputs directly to:")
    for name in [
        "synthetic_ground_truth_recovery.csv",
        "run_parameters_used.csv",
        "synthetic_ground_truth_summary.csv",
        "leverage_regime_recovery.csv",
        "leverage_regime_summary.csv",
        "kernel_bandwidth_robustness.csv",
        "kernel_bandwidth_summary.csv",
        "proxy_stress_recovery.csv",
        "proxy_stress_summary.csv",
        "nonlinear_model_selection.csv",
        "nonlinear_model_selection_summary.csv",
        "synthetic_heston_recovery.pdf",
        "rho_true_vs_hat.pdf",
        "leverage_error_by_rho.pdf",
        "kernel_bandwidth_robustness.pdf",
        "proxy_stress_tests.pdf",
        "nonlinear_model_selection_boundaries.pdf",
    ]:
        print(f"  {ROOT / name}")


if __name__ == "__main__":
    main()
