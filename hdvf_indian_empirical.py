#!/usr/bin/env python3
"""Indian 1-minute empirical HDVF extension.

This script is intentionally self-contained.  It reads Kite BSE 1-minute
Parquet candles, builds daily variance proxies, runs a two-term weak-form
Heston-style recovery, and writes paper-facing CSV/PNG/report artifacts beside
the script.
"""

from __future__ import annotations

import argparse
import csv
import math
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
LOCAL_DATA_ROOT = ROOT / "indian_market_data"
FALLBACK_SOURCE_ROOT = ROOT / "indian_market_data"
SOURCE_ROOT = LOCAL_DATA_ROOT if LOCAL_DATA_ROOT.exists() else FALLBACK_SOURCE_ROOT
DATA_ROOT = SOURCE_ROOT
STATE_DB = ROOT / "state.sqlite3"

START_DATE = "2020-02-01"
END_DATE = "2020-06-30"
OUTPUT_PREFIX = "indian_balanced50_covid_crash"
TRADING_DAYS = 252.0
DT = 1.0 / TRADING_DAYS
DEFAULT_MAX_COMPANIES = 50
DEFAULT_KERNELS = 50
DEFAULT_BANDWIDTH = 1.5
EWMA_SPAN = 14
YZ_WINDOW = 14
MIN_DAILY_ROWS = 200
MIN_VALID_DAYS = 500
EPS = 1e-12

PROXIES = ["realized_var_1m", "gk_raw", "ewma_gk14", "parkinson", "yang_zhang14"]

DEFAULT_ARTIFACTS = {
    "company_selection_csv": "indian_empirical_company_selection.csv",
    "daily_proxy_panel_csv": "indian_empirical_daily_proxy_panel.csv",
    "recovery_csv": "indian_empirical_recovery.csv",
    "proxy_summary_csv": "indian_empirical_proxy_summary.csv",
    "recovery_pdf": "indian_empirical_recovery.pdf",
    "proxy_reliability_pdf": "indian_proxy_reliability.pdf",
    "rho_distribution_pdf": "indian_rho_distribution.pdf",
    "company_selection_pdf": "indian_company_selection.pdf",
}

ARTIFACT_SUFFIXES = {
    "company_selection_csv": "company_selection.csv",
    "daily_proxy_panel_csv": "daily_proxy_panel.csv",
    "recovery_csv": "recovery.csv",
    "proxy_summary_csv": "proxy_summary.csv",
    "recovery_pdf": "recovery.pdf",
    "proxy_reliability_pdf": "proxy_reliability.pdf",
    "rho_distribution_pdf": "rho_distribution.pdf",
    "company_selection_pdf": "company_selection.pdf",
}

SECTOR_ORDER = [
    "Banks",
    "Non-bank Financials and Insurance",
    "Information Technology",
    "Energy and Utilities",
    "Consumer Staples",
    "Automobiles",
    "Healthcare",
    "Materials Metals and Cement",
    "Industrials Capital Goods and Construction",
    "Consumer Discretionary Services and Telecom",
    "Financials",
    "Industrials",
    "Materials",
    "Energy & Utilities",
    "Consumer",
    "Other",
]

CURATED_SECTORS = {
    "AAVAS": "Financials",
    "ABCAPITAL": "Financials",
    "AKCAPIT": "Financials",
    "ALANKIT": "Financials",
    "BIRLAMONEY": "Financials",
    "ABB": "Industrials",
    "ACE": "Industrials",
    "AIAENG": "Industrials",
    "AHLUCONT": "Industrials",
    "AGI": "Industrials",
    "ACC": "Materials",
    "AARTIIND": "Materials",
    "ALKYLAMINE": "Materials",
    "AKZOINDIA": "Materials",
    "ALUFLUOR": "Materials",
    "ADANIPOWER": "Energy & Utilities",
    "ADANIGREEN": "Energy & Utilities",
    "ADANIENSOL": "Energy & Utilities",
    "ATGL": "Energy & Utilities",
    "AEGISLOG": "Logistics",
    "ADANIPORTS": "Logistics",
    "ALLCARGO": "Logistics",
    "ABFRL": "Consumer",
    "ADFFOODS": "Consumer",
    "ARE&M": "Consumer",
    "AJANTSOY": "Consumer",
    "ABBOTINDIA": "Healthcare",
    "AJANTPHARM": "Healthcare",
    "ALKEM": "Healthcare",
    "APLLTD": "Healthcare",
    "AARTIDRUGS": "Healthcare",
    "ALEMBICLTD": "Healthcare",
    "ADVENZYMES": "Healthcare",
    "ACCELYA": "Information Technology",
    "ALLDIGI": "Information Technology",
    "ADSL": "Information Technology",
    "ABREL": "Real Estate",
    "AJMERA": "Real Estate",
    "ADANIENT": "Diversified",
}


@dataclass
class CompanyMeta:
    folder: str
    name: str
    symbol: str
    token: str
    state_rows: int
    state_status: str
    state_earliest: str
    state_latest: str
    day_files_2019_2025: int


@dataclass
class DailyRow:
    company_id: str
    symbol: str
    name: str
    date: str
    rows: int
    valid_rows: int
    open: float
    high: float
    low: float
    close: float
    volume: int
    traded_value: float
    log_open: float
    log_close: float
    realized_var_1m: float
    gk_raw: float
    ewma_gk14: float
    parkinson: float
    yang_zhang14: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Indian 1-minute empirical HDVF recovery.")
    parser.add_argument("--data-root", default=str(DATA_ROOT), help="Root containing company day files or a Kite partitioned lake.")
    parser.add_argument("--start-date", default=START_DATE, help="Inclusive empirical window start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", default=END_DATE, help="Inclusive empirical window end date, YYYY-MM-DD.")
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX, help="Prefix for all generated artifacts.")
    parser.add_argument("--covid-crash", action="store_true", help="Use a focused COVID crash window and artifact prefix.")
    return parser.parse_args()


def artifact_path(key: str) -> Path:
    suffix = ARTIFACT_SUFFIXES[key]
    if suffix.endswith(".pdf"):
        folder = ROOT / "pdfs"
    else:
        folder = ROOT / "csvs"
    folder.mkdir(exist_ok=True)
    return folder / f"{OUTPUT_PREFIX}_{suffix}"


def estimate_min_valid_days(start_date: str, end_date: str) -> int:
    from datetime import date

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    calendar_days = max((end - start).days + 1, 1)
    approx_trading_days = int(calendar_days * 5.0 / 7.0)
    return min(500, max(30, int(approx_trading_days * 0.65)))


def clean_text(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def split_company_folder(folder: str) -> tuple[str, str, str]:
    parts = folder.rsplit("__", 2)
    if len(parts) == 3:
        return clean_text(parts[0].replace("_", " ")), parts[1], parts[2]
    return clean_text(folder.replace("_", " ")), folder, ""


def infer_sector(symbol: str, name: str) -> str:
    symbol = symbol.upper()
    name_u = name.upper()
    if symbol in CURATED_SECTORS:
        return CURATED_SECTORS[symbol]
    if any(k in name_u for k in ["BANK", "FINANCE", "CAPITAL", "SECURITIES", "HOUSING", "INVEST", "CREDIT"]):
        return "Financials"
    if any(k in name_u for k in ["PHARMA", "DRUG", "LIFE", "HEALTH", "HOSPITAL", "MEDIC", "LABORATOR"]):
        return "Healthcare"
    if any(k in name_u for k in ["POWER", "ENERGY", "GAS", "GREEN", "TRANSMISSION", "SOLAR", "ELECTRIC"]):
        return "Energy & Utilities"
    if any(k in name_u for k in ["CHEM", "CEMENT", "METAL", "STEEL", "ALUMIN", "FLUOR", "AMINES", "MATERIAL"]):
        return "Materials"
    if any(k in name_u for k in ["LOGISTICS", "PORT", "TRANSPORT", "SHIPPING", "CARGO"]):
        return "Logistics"
    if any(k in name_u for k in ["TECH", "SOFTWARE", "DIGITAL", "INFO", "COMPUTER", "SYSTEM"]):
        return "Information Technology"
    if any(k in name_u for k in ["REALTY", "REAL ESTATE", "HOUSING", "INFRA", "DEVELOPMENT", "CONSTRUCTION"]):
        return "Real Estate"
    if any(k in name_u for k in ["FOODS", "FASHION", "RETAIL", "TEXTILE", "SPINNING", "AUTO", "MOBILITY", "PAINT"]):
        return "Consumer"
    if any(k in name_u for k in ["ENGINEERING", "INDUSTRIES", "ELECTRICAL", "TOOLS", "EQUIPMENT"]):
        return "Industrials"
    if any(k in name_u for k in ["ENTERPRISES", "HOLDINGS"]):
        return "Diversified"
    return "Other"


def date_from_path(path: Path) -> str:
    return f"{path.parts[-3]}-{path.parts[-2]}-{path.stem}"


def resolve_data_root(path: Path) -> Path:
    lake_root = path / "lake" / "ohlcv" / "kite" / "universe=paper2_v4_balanced50" / "interval=minute" / "exchange_preference=nse_then_bse"
    if lake_root.exists():
        return lake_root
    if path.exists() and any(p.is_dir() and p.name.startswith("company=") for p in path.iterdir()):
        return path
    return path


def load_universe_metadata() -> dict[str, dict[str, str]]:
    candidates = []
    if SOURCE_ROOT.exists():
        candidates.extend((SOURCE_ROOT / "metadata").glob("*.csv"))
    candidates.append(ROOT / "indian_market_balanced_universe_50.csv")
    out = {}
    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    symbol = clean_text(row.get("tradingsymbol", "")).upper()
                    if symbol:
                        out[symbol] = row
        except Exception:
            continue
    return out


def in_window(date: str) -> bool:
    return START_DATE <= date <= END_DATE


def median(values: Iterable[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return statistics.median(vals) if vals else default


def mean(values: Iterable[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return statistics.fmean(vals) if vals else default


def pctl(values: list[float], q: float, default: float = 0.0) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return default
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def safe_log_ratio(a: float, b: float) -> float | None:
    if a > 0 and b > 0:
        return math.log(a / b)
    return None


def load_state_rows() -> dict[str, dict[str, str | int]]:
    if not STATE_DB.exists():
        return {}
    con = sqlite3.connect(STATE_DB)
    con.row_factory = sqlite3.Row
    out = {}
    for row in con.execute(
        "select tradingsymbol, name, instrument_token, rows_written, status, "
        "coalesce(earliest_1m_timestamp,''), coalesce(latest_1m_timestamp,'') from instruments"
    ):
        token = str(row["instrument_token"])
        out[token] = {
            "symbol": row["tradingsymbol"],
            "name": row["name"],
            "rows_written": int(row["rows_written"] or 0),
            "status": row["status"],
            "earliest": row[5],
            "latest": row[6],
        }
    con.close()
    return out


def discover_companies() -> list[CompanyMeta]:
    state = load_state_rows()
    universe_meta = load_universe_metadata()
    companies = []
    for folder_path in sorted(DATA_ROOT.iterdir()):
        if not folder_path.is_dir() or folder_path.name == "_staging":
            continue
        if folder_path.name.startswith("company="):
            raw = folder_path.name.split("=", 1)[1]
            parts = raw.rsplit("__", 1)
            symbol = parts[0].replace("_", "-")
            token = parts[1] if len(parts) == 2 else ""
            symbol_from_file, name_from_file = read_folder_identity(folder_path)
            symbol = symbol_from_file or symbol
            meta_row = universe_meta.get(symbol.upper(), {})
            name = clean_text(meta_row.get("company_name", name_from_file or symbol))
            day_files = len({str(d) for d in monthly_file_dates(folder_path) if in_window(str(d))})
        else:
            name, symbol, token = split_company_folder(folder_path.name)
            day_files = sum(1 for p in folder_path.rglob("*.parquet") if in_window(date_from_path(p)))
        s = state.get(token, {})
        companies.append(
            CompanyMeta(
                folder=folder_path.name,
                name=clean_text(str(s.get("name", name))),
                symbol=str(s.get("symbol", symbol)),
                token=token,
                state_rows=int(s.get("rows_written", 0) or 0),
                state_status=str(s.get("status", "")),
                state_earliest=str(s.get("earliest", "")),
                state_latest=str(s.get("latest", "")),
                day_files_2019_2025=day_files,
            )
        )
    return companies


def read_folder_identity(folder: Path) -> tuple[str, str]:
    try:
        import pyarrow.parquet as pq

        path = next(folder.rglob("*.parquet"))
        table = pq.read_table(path, columns=["tradingsymbol", "company_name"])
        data = table.slice(0, 1).to_pydict()
        symbol = clean_text(data.get("tradingsymbol", [""])[0])
        name = clean_text(data.get("company_name", [""])[0])
        return symbol, name
    except Exception:
        return "", ""


def read_day_file(path: Path, meta: CompanyMeta) -> DailyRow | None:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["timestamp", "open", "high", "low", "close", "volume"])
    data = table.to_pydict()
    rows = []
    for idx, ts in enumerate(data["timestamp"]):
        o = float(data["open"][idx])
        h = float(data["high"][idx])
        l = float(data["low"][idx])
        c = float(data["close"][idx])
        v = int(data["volume"][idx] or 0)
        if o > 0 and h > 0 and l > 0 and c > 0 and h >= l:
            rows.append((str(ts), o, h, l, c, v))
    return daily_from_candles(rows, meta, date_from_path(path), table.num_rows)


def daily_from_candles(rows: list[tuple[str, float, float, float, float, int]], meta: CompanyMeta, date: str, raw_rows: int) -> DailyRow | None:
    if len(rows) < 2:
        return None
    rows.sort(key=lambda r: r[0])
    daily_open = rows[0][1]
    daily_close = rows[-1][4]
    daily_high = max(r[2] for r in rows)
    daily_low = min(r[3] for r in rows)
    volume = sum(r[5] for r in rows)
    closes = [r[4] for r in rows]
    rv = 0.0
    for prev, cur in zip(closes[:-1], closes[1:]):
        lr = safe_log_ratio(cur, prev)
        if lr is not None:
            rv += lr * lr
    rv *= TRADING_DAYS

    hl = safe_log_ratio(daily_high, daily_low)
    co = safe_log_ratio(daily_close, daily_open)
    if hl is None or co is None:
        return None
    gk = max((0.5 * hl * hl - (2.0 * math.log(2.0) - 1.0) * co * co) * TRADING_DAYS, EPS)
    parkinson = max((hl * hl / (4.0 * math.log(2.0))) * TRADING_DAYS, EPS)
    return DailyRow(
        company_id=meta.folder,
        symbol=meta.symbol,
        name=meta.name,
        date=date,
        rows=raw_rows,
        valid_rows=len(rows),
        open=daily_open,
        high=daily_high,
        low=daily_low,
        close=daily_close,
        volume=volume,
        traded_value=daily_close * volume,
        log_open=math.log(daily_open),
        log_close=math.log(daily_close),
        realized_var_1m=max(rv, EPS),
        gk_raw=gk,
        ewma_gk14=0.0,
        parkinson=parkinson,
        yang_zhang14=None,
    )


def monthly_file_dates(folder: Path) -> set[str]:
    dates = set()
    for path in folder.rglob("*.parquet"):
        try:
            year = path.parent.parent.name.split("=", 1)[1]
            month = path.parent.name.split("=", 1)[1]
            if not (START_DATE[:7] <= f"{year}-{int(month):02d}" <= END_DATE[:7]):
                continue
            import pyarrow.parquet as pq

            table = pq.read_table(path, columns=["date"])
            dates.update(str(d) for d in table.column("date").to_pylist())
        except Exception:
            continue
    return dates


def read_month_file(path: Path, meta: CompanyMeta) -> list[DailyRow]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["timestamp", "date", "open", "high", "low", "close", "volume"])
    data = table.to_pydict()
    grouped: dict[str, list[tuple[str, float, float, float, float, int]]] = defaultdict(list)
    raw_counts: dict[str, int] = defaultdict(int)
    for idx, ts in enumerate(data["timestamp"]):
        date = str(data["date"][idx])
        if not in_window(date):
            continue
        raw_counts[date] += 1
        o = float(data["open"][idx])
        h = float(data["high"][idx])
        l = float(data["low"][idx])
        c = float(data["close"][idx])
        v = int(data["volume"][idx] or 0)
        if o > 0 and h > 0 and l > 0 and c > 0 and h >= l:
            grouped[date].append((str(ts), o, h, l, c, v))
    rows = []
    for date, candles in grouped.items():
        row = daily_from_candles(candles, meta, date, raw_counts[date])
        if row is not None:
            rows.append(row)
    return rows


def complete_proxy_columns(rows: list[DailyRow]) -> None:
    rows.sort(key=lambda r: r.date)
    alpha = 2.0 / (EWMA_SPAN + 1.0)
    ewma = None
    for row in rows:
        ewma = row.gk_raw if ewma is None else alpha * row.gk_raw + (1.0 - alpha) * ewma
        row.ewma_gk14 = max(ewma, EPS)

    for idx, row in enumerate(rows):
        if idx == 0:
            row.yang_zhang14 = None
            continue
        start = max(1, idx - YZ_WINDOW + 1)
        window = rows[start : idx + 1]
        overnight = []
        open_close = []
        rs_terms = []
        for j, day in enumerate(window, start=start):
            prev_close = rows[j - 1].close
            ro = safe_log_ratio(day.open, prev_close)
            rc = safe_log_ratio(day.close, day.open)
            ho = safe_log_ratio(day.high, day.open)
            hc = safe_log_ratio(day.high, day.close)
            lo = safe_log_ratio(day.low, day.open)
            lc = safe_log_ratio(day.low, day.close)
            if None in (ro, rc, ho, hc, lo, lc):
                continue
            overnight.append(ro)
            open_close.append(rc)
            rs_terms.append(ho * hc + lo * lc)
        n = len(open_close)
        if n < 3:
            row.yang_zhang14 = None
            continue
        k = 0.34 / (1.34 + (n + 1.0) / max(n - 1.0, 1.0))
        yz_daily = sample_variance(overnight) + k * sample_variance(open_close) + (1.0 - k) * mean(rs_terms)
        row.yang_zhang14 = max(yz_daily * TRADING_DAYS, EPS)


def sample_variance(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if len(vals) < 2:
        return 0.0
    m = statistics.fmean(vals)
    return sum((v - m) ** 2 for v in vals) / (len(vals) - 1)


def read_company_daily(meta: CompanyMeta) -> list[DailyRow]:
    folder = DATA_ROOT / meta.folder
    monthly_layout = meta.folder.startswith("company=")
    paths = sorted(folder.rglob("*.parquet"))
    rows = []
    for path in paths:
        try:
            if monthly_layout:
                rows.extend(read_month_file(path, meta))
                continue
            if not in_window(date_from_path(path)):
                continue
            row = read_day_file(path, meta)
        except Exception:
            row = None
        if row is not None:
            rows.append(row)
    complete_proxy_columns(rows)
    return rows


def company_metrics(meta: CompanyMeta, rows: list[DailyRow]) -> dict[str, float | str | int]:
    rows_kept = [r for r in rows if r.valid_rows >= MIN_DAILY_ROWS]
    valid_days = len(rows_kept)
    median_rows = median([r.valid_rows for r in rows_kept])
    median_value = median([r.traded_value for r in rows_kept])
    total_value = sum(r.traded_value for r in rows_kept)
    first_date = rows_kept[0].date if rows_kept else ""
    last_date = rows_kept[-1].date if rows_kept else ""
    coverage_score = min(valid_days / 1700.0, 1.0)
    row_score = min(median_rows / 375.0, 1.0)
    liquidity_score = math.log10(max(median_value, 1.0))
    score = 100.0 * coverage_score + 20.0 * row_score + liquidity_score
    return {
        "company_id": meta.folder,
        "symbol": meta.symbol,
        "name": meta.name,
        "sector": company_sector(meta.symbol, meta.name),
        "token": meta.token,
        "state_status": meta.state_status,
        "state_rows": meta.state_rows,
        "day_files_2019_2025": meta.day_files_2019_2025,
        "valid_days": valid_days,
        "first_date": first_date,
        "last_date": last_date,
        "median_intraday_rows": median_rows,
        "median_traded_value": median_value,
        "total_traded_value": total_value,
        "selection_score": score,
    }


def company_sector(symbol: str, name: str) -> str:
    meta = load_universe_metadata().get(symbol.upper(), {})
    return clean_text(meta.get("segment", "")) or infer_sector(symbol, name)


def solve_2x2(features: list[tuple[float, float]], target: list[float], ridge: float = 1e-10) -> tuple[float, float]:
    s00 = ridge
    s01 = 0.0
    s11 = ridge
    t0 = 0.0
    t1 = 0.0
    for (a0, a1), y in zip(features, target):
        s00 += a0 * a0
        s01 += a0 * a1
        s11 += a1 * a1
        t0 += a0 * y
        t1 += a1 * y
    det = s00 * s11 - s01 * s01
    if abs(det) < EPS:
        return 0.0, 0.0
    return (t0 * s11 - t1 * s01) / det, (s00 * t1 - s01 * t0) / det


def build_weak_matrices(x: list[float], v: list[float], m: int, bandwidth_factor: float) -> dict[str, list[float]]:
    state = v[:-1]
    dx = [x[i + 1] - x[i] for i in range(len(x) - 1)]
    dv = [v[i + 1] - v[i] for i in range(len(v) - 1)]
    lo = pctl(state, 0.05)
    hi = pctl(state, 0.95)
    if hi <= lo:
        hi = lo + max(abs(lo) * 0.1, 1e-4)
    centers = [lo + (hi - lo) * j / max(m - 1, 1) for j in range(m)]
    std = math.sqrt(sample_variance(state))
    h = max(bandwidth_factor * std / math.sqrt(max(m, 1)), (hi - lo) / max(m, 1), 1e-6)
    a0 = []
    a1 = []
    bv = []
    qv = []
    qxv = []
    for center in centers:
        wa0 = wa1 = wb = wqv = wqxv = 0.0
        for s, dxi, dvi in zip(state, dx, dv):
            w = math.exp(-0.5 * ((s - center) / h) ** 2)
            wa0 += w * DT
            wa1 += w * s * DT
            wb += w * dvi
            wqv += w * dvi * dvi
            wqxv += w * dxi * dvi
        a0.append(wa0)
        a1.append(wa1)
        bv.append(wb)
        qv.append(wqv)
        qxv.append(wqxv)
    return {"a0": a0, "a1": a1, "bv": bv, "qv": qv, "qxv": qxv, "centers": centers, "h": [h]}


def recover_company_proxy(company_rows: list[DailyRow], proxy: str, kernels: int, bandwidth: float) -> dict[str, float | str | int]:
    usable = [r for r in company_rows if r.valid_rows >= MIN_DAILY_ROWS and getattr(r, proxy) is not None]
    x = [r.log_open for r in usable]
    v = [float(getattr(r, proxy)) for r in usable]
    if len(x) < MIN_VALID_DAYS or len(set(round(val, 10) for val in v)) < 5:
        return {"status": "insufficient_data", "n_days": len(x)}
    mats = build_weak_matrices(x, v, kernels, bandwidth)
    features = list(zip(mats["a0"], mats["a1"]))
    c0, c1 = solve_2x2(features, mats["bv"])
    state = v[:-1]
    drift_state = [c0 + c1 * s for s in state]
    corr_qv = []
    for center, q in zip(mats["centers"], mats["qv"]):
        h = mats["h"][0]
        correction = 0.0
        for s, b in zip(state, drift_state):
            w = math.exp(-0.5 * ((s - center) / h) ** 2)
            correction += w * b * b
        corr_qv.append(q - correction * DT * DT)
    d0, d1 = solve_2x2(features, corr_qv)
    x0, x1 = solve_2x2(features, mats["qxv"])

    kappa = -c1
    theta = c0 / kappa if abs(kappa) > EPS else float("nan")
    diffusion_slope_positive = d1 > 0
    xi = math.sqrt(max(d1, 0.0))
    rho = x1 / xi if xi > EPS else float("nan")
    warnings = []
    if kappa <= 0:
        warnings.append("kappa_nonpositive")
    if not math.isfinite(theta) or theta <= 0:
        warnings.append("theta_nonpositive")
    if not diffusion_slope_positive:
        warnings.append("diffusion_slope_nonpositive")
    if not math.isfinite(rho) or rho < -1.0 or rho > 1.0:
        warnings.append("rho_outside_unit_interval")
    if math.isfinite(rho) and rho >= 0:
        warnings.append("rho_nonnegative")
    heston_like = not warnings
    return {
        "status": "ok",
        "n_days": len(x),
        "v_min": min(v),
        "v_median": median(v),
        "v_max": max(v),
        "c0": c0,
        "c1": c1,
        "d0": d0,
        "d1": d1,
        "xv0": x0,
        "xv1": x1,
        "kappa": kappa,
        "theta": theta,
        "xi": xi,
        "rho": rho,
        "rho_negative": int(math.isfinite(rho) and rho < 0),
        "rho_in_unit_interval": int(math.isfinite(rho) and -1 <= rho <= 1),
        "heston_like_gate": int(heston_like),
        "warning_flags": ";".join(warnings) if warnings else "none",
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if not fieldnames:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def proxy_value(row: DailyRow, proxy: str) -> float | None:
    return getattr(row, proxy)


def create_daily_panel(selected_data: dict[str, list[DailyRow]]) -> list[dict[str, object]]:
    rows = []
    for company_id, daily in selected_data.items():
        for r in daily:
            if r.valid_rows < MIN_DAILY_ROWS:
                continue
            rows.append(
                {
                    "company_id": company_id,
                    "symbol": r.symbol,
                    "name": r.name,
                    "date": r.date,
                    "intraday_rows": r.rows,
                    "valid_intraday_rows": r.valid_rows,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                    "traded_value": r.traded_value,
                    "log_open": r.log_open,
                    "realized_var_1m": r.realized_var_1m,
                    "gk_raw": r.gk_raw,
                    "ewma_gk14": r.ewma_gk14,
                    "parkinson": r.parkinson,
                    "yang_zhang14": "" if r.yang_zhang14 is None else r.yang_zhang14,
                }
            )
    return rows


def summarize_by_proxy(recovery_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for proxy in PROXIES:
        rows = [r for r in recovery_rows if r["proxy"] == proxy and r["status"] == "ok"]
        if not rows:
            continue
        n = len(rows)
        out.append(
            {
                "proxy": proxy,
                "n_companies": n,
                "median_kappa": median([float(r["kappa"]) for r in rows]),
                "median_theta": median([float(r["theta"]) for r in rows if math.isfinite(float(r["theta"]))]),
                "median_implied_vol": math.sqrt(max(median([float(r["theta"]) for r in rows if math.isfinite(float(r["theta"]))]), 0.0)),
                "median_xi": median([float(r["xi"]) for r in rows]),
                "median_rho": median([float(r["rho"]) for r in rows if math.isfinite(float(r["rho"]))]),
                "kappa_positive_rate": mean([1.0 if float(r["kappa"]) > 0 else 0.0 for r in rows]),
                "theta_positive_rate": mean([1.0 if math.isfinite(float(r["theta"])) and float(r["theta"]) > 0 else 0.0 for r in rows]),
                "xi_positive_rate": mean([1.0 if float(r["xi"]) > 0 else 0.0 for r in rows]),
                "rho_negative_rate": mean([float(r["rho_negative"]) for r in rows]),
                "rho_unit_rate": mean([float(r["rho_in_unit_interval"]) for r in rows]),
                "heston_like_rate": mean([float(r["heston_like_gate"]) for r in rows]),
            }
        )
    return out


def choose_balanced_sample(eligible: list[dict[str, object]], max_companies: int) -> list[dict[str, object]]:
    sector_cap = max(4, math.ceil(max_companies / 7))
    grouped: dict[str, list[dict[str, object]]] = {sector: [] for sector in SECTOR_ORDER}
    for row in eligible:
        grouped.setdefault(str(row["sector"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda m: (float(m["selection_score"]), float(m["median_traded_value"]), int(m["valid_days"])), reverse=True)

    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    sector_counts: dict[str, int] = defaultdict(int)
    while len(selected) < max_companies:
        added = False
        for sector in SECTOR_ORDER:
            if sector_counts[sector] >= sector_cap:
                continue
            bucket = grouped.get(sector, [])
            while bucket and str(bucket[0]["company_id"]) in selected_ids:
                bucket.pop(0)
            if not bucket:
                continue
            row = bucket.pop(0)
            selected.append(row)
            selected_ids.add(str(row["company_id"]))
            sector_counts[sector] += 1
            added = True
            if len(selected) >= max_companies:
                break
        if not added:
            break

    if len(selected) < max_companies:
        remainder = [row for row in eligible if str(row["company_id"]) not in selected_ids]
        remainder.sort(key=lambda m: (float(m["selection_score"]), float(m["median_traded_value"]), int(m["valid_days"])), reverse=True)
        for row in remainder:
            selected.append(row)
            selected_ids.add(str(row["company_id"]))
            if len(selected) >= max_companies:
                break
    return selected


def select_companies(max_companies: int) -> tuple[list[dict[str, object]], dict[str, list[DailyRow]]]:
    metas = discover_companies()
    if max_companies >= DEFAULT_MAX_COMPANIES:
        min_candidates = len(metas)
    else:
        min_candidates = max(max_companies * 2, max_companies)
    candidates = sorted(metas, key=lambda m: (m.day_files_2019_2025, m.state_rows), reverse=True)[:min_candidates]

    metrics = []
    data = {}
    for idx, meta in enumerate(candidates, start=1):
        print(f"Reading candidate {idx:03d}/{len(candidates):03d}: {meta.symbol} ({meta.day_files_2019_2025} day files)")
        daily = read_company_daily(meta)
        m = company_metrics(meta, daily)
        metrics.append(m)
        data[meta.folder] = daily

    eligible = [m for m in metrics if int(m["valid_days"]) >= MIN_VALID_DAYS and float(m["median_intraday_rows"]) >= MIN_DAILY_ROWS]
    eligible.sort(key=lambda m: (float(m["selection_score"]), float(m["median_traded_value"]), int(m["valid_days"])), reverse=True)
    selected_balanced = choose_balanced_sample(eligible, max_companies)
    selected_ids = {str(m["company_id"]) for m in selected_balanced}
    selected_rank = {str(m["company_id"]): idx for idx, m in enumerate(selected_balanced, start=1)}
    selection_rows = []
    for m in sorted(metrics, key=lambda x: float(x["selection_score"]), reverse=True):
        selected = str(m["company_id"]) in selected_ids
        if selected:
            reason = f"selected: balanced best-available sample rank {selected_rank[str(m['company_id'])]} ({m['sector']})"
        elif int(m["valid_days"]) < MIN_VALID_DAYS:
            reason = f"rejected: valid_days<{MIN_VALID_DAYS}"
        elif float(m["median_intraday_rows"]) < MIN_DAILY_ROWS:
            reason = f"rejected: median_intraday_rows<{MIN_DAILY_ROWS}"
        else:
            reason = "rejected: below balanced-sector score cutoff"
        row = dict(m)
        row["selected"] = int(selected)
        row["selection_reason"] = reason
        selection_rows.append(row)

    # Add unprocessed companies to document why they were outside the candidate pool.
    processed_ids = {str(m["company_id"]) for m in metrics}
    for meta in sorted(metas, key=lambda m: (m.day_files_2019_2025, m.state_rows), reverse=True):
        if meta.folder in processed_ids:
            continue
        selection_rows.append(
            {
                "company_id": meta.folder,
                "symbol": meta.symbol,
                "name": meta.name,
                "sector": infer_sector(meta.symbol, meta.name),
                "token": meta.token,
                "state_status": meta.state_status,
                "state_rows": meta.state_rows,
                "day_files_2019_2025": meta.day_files_2019_2025,
                "valid_days": "",
                "first_date": "",
                "last_date": "",
                "median_intraday_rows": "",
                "median_traded_value": "",
                "total_traded_value": "",
                "selection_score": "",
                "selected": 0,
                "selection_reason": "rejected: not in analyzed candidate pool",
            }
        )

    selected_data = {cid: data[cid] for cid in selected_ids}
    return selection_rows, selected_data




# ==========================================
# Rendering helpers.
# ==========================================


def render_all_plots(
    selection: list[dict[str, object]],
    recovery: list[dict[str, object]],
    summary: list[dict[str, object]],
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import matplotlib.colors as mcolors
        import numpy as np
    except ImportError as exc:
        print(f"Skipping plots: {exc}")
        return

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman"],
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42, # Embed fonts in PDFs for text selection
        "ps.fonttype": 42
    })

    def proxy_short(label: str) -> str:
        return {
            "realized_var_1m": "RV 1m",
            "gk_raw": "GK",
            "ewma_gk14": "EWMA GK",
            "parkinson": "Parkinson",
            "yang_zhang14": "YZ",
        }.get(label, label)

    def to_float(value: str, default: float = float("nan")) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def median_key(rows: list[dict[str, object]], key: str, default: float = 0.0) -> float:
        vals = []
        for row in rows:
            value = to_float(row.get(key, ""))
            if math.isfinite(value):
                vals.append(value)
        return statistics.median(vals) if vals else default

    # 1. Primary Indian generator recovery figure
    primary_rows = [r for r in recovery if r.get("proxy") == "realized_var_1m" and r.get("status") == "ok"]
    primary_summary = next((r for r in summary if r.get("proxy") == "realized_var_1m"), {})
    
    if primary_rows:
        c0 = median_key(primary_rows, "c0")
        c1 = median_key(primary_rows, "c1")
        d0 = median_key(primary_rows, "d0")
        d1 = median_key(primary_rows, "d1")
        xv0 = median_key(primary_rows, "xv0")
        xv1 = median_key(primary_rows, "xv1")
        rho = to_float(primary_summary.get("median_rho", median_key(primary_rows, "rho")))
        kappa = to_float(primary_summary.get("median_kappa", median_key(primary_rows, "kappa")))

        if "covid_crash" in OUTPUT_PREFIX:
            xlim = (0.22, 0.38)
        else:
            centers = [to_float(r.get("v_median", "")) for r in primary_rows]
            finite_centers = [v for v in centers if math.isfinite(v)]
            center = statistics.median(finite_centers) if finite_centers else 0.20
            half_width = max(0.08, 0.25 * center)
            xlim = (max(EPS, center - half_width), center + half_width)

        x_vals = np.linspace(xlim[0], xlim[1], 240)
        b_v = c0 + c1 * x_vals
        a_22 = d0 + d1 * x_vals
        a_12 = xv0 + xv1 * x_vals

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Indian 1-Minute Recovered Generators ({START_DATE} to {END_DATE}) | ρ={rho:.2f}, κ={kappa:.2f}", fontsize=16, fontweight="bold")
        
        # Panel 1
        axs[0].plot(x_vals, b_v, color="#c44137", linewidth=2.5, label="estimated")
        axs[0].set_title("Empirical Variance Drift $b_v(v)$", fontweight="bold")
        axs[0].set_xlabel("Annualized variance, v (1/year)")
        axs[0].set_ylabel("$b_v(v)$, annualized variance/year")
        
        # Panel 2
        axs[1].plot(x_vals, a_22, color="#c44137", linewidth=2.5, label="estimated")
        axs[1].set_title("Empirical Variance Diffusion $a_{22}(v)$", fontweight="bold")
        axs[1].set_xlabel("Annualized variance, v (1/year)")
        axs[1].set_ylabel("$a_{22}(v)$, variance$^2$/year")
        
        # Panel 3
        axs[2].plot(x_vals, a_12, color="#c44137", linewidth=2.5, label="estimated")
        axs[2].set_title("Empirical Leverage Effect $a_{12}(v)$", fontweight="bold")
        axs[2].set_xlabel("Annualized variance, v (1/year)")
        axs[2].set_ylabel("$a_{12}(v)$, covariance rate/year")
        
        for ax in axs:
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.legend(loc="lower right")
            
        fig.text(0.5, -0.05, "Balanced Kite 1-minute sample; primary variance state is annualized 1-minute realized variance.", ha="center", fontsize=10, color="gray")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(artifact_path("recovery_pdf"), format="pdf", bbox_inches="tight")
        plt.close(fig)

    # 2. Company selection plot
    selected = [r for r in selection if str(r.get("selected")) == "1"]
    sector_counts = defaultdict(int)
    selected_by_sector: dict[str, list[str]] = defaultdict(list)
    for row in selected:
        sector = str(row.get("sector", "Other"))
        sector_counts[sector] += 1
        selected_by_sector[sector].append(str(row.get("symbol", "")))
        
    sector_items = [(sector, sector_counts[sector]) for sector in SECTOR_ORDER if sector_counts[sector]]
    if "Other" not in {sector for sector, _ in sector_items} and sector_counts["Other"]:
        sector_items.append(("Other", sector_counts["Other"]))
        
    sector_labels = {
        "Banks": "Banks",
        "Non-bank Financials and Insurance": "NBFC/Ins.",
        "Information Technology": "IT",
        "Energy and Utilities": "Energy",
        "Consumer Staples": "Staples",
        "Automobiles": "Autos",
        "Healthcare": "Health",
        "Materials Metals and Cement": "Materials",
        "Industrials Capital Goods and Construction": "Ind.",
        "Consumer Discretionary Services and Telecom": "Disc./Tel.",
        "Financials": "Financials",
        "Industrials": "Industrials",
        "Healthcare": "Healthcare",
        "Materials": "Materials",
        "Energy & Utilities": "Energy",
        "Consumer": "Consumer",
        "Information Technology": "IT",
        "Real Estate": "Real Est.",
        "Logistics": "Logistics",
        "Diversified": "Diversified",
        "Other": "Other",
    }
    
    palette = [
        "#4e6991", "#5a7a65", "#965650", "#706084", "#9c7548",
        "#537f8f", "#767676", "#916574", "#62805a", "#846c52", "#787891"
    ]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    fig.suptitle("Balanced Company Selection", fontsize=16, fontweight="bold")
    ax.set_title("Selected Companies by Segment", fontsize=14, fontweight="bold", pad=20)
    
    card_w = 0.16
    card_h = 0.25
    gap_x = 0.04
    gap_y = 0.10
    start_x = 0.05
    start_y = 0.65
    
    for idx, (sector, count) in enumerate(sector_items):
        col = idx % 5
        row = idx // 5
        x0 = start_x + col * (card_w + gap_x)
        y0 = start_y - row * (card_h + gap_y)
        
        color = palette[(idx - 1) % len(palette)]
        
        rect = patches.Rectangle((x0, y0), card_w, card_h, linewidth=1, edgecolor='#cccccc', facecolor='white', transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        
        header = patches.Rectangle((x0, y0 + card_h - 0.05), card_w, 0.05, linewidth=0, facecolor=color, transform=ax.transAxes, clip_on=False)
        ax.add_patch(header)
        
        label = sector_labels.get(sector, sector)
        ax.text(x0 + 0.01, y0 + card_h - 0.025, label, color='white', fontsize=9, va='center', transform=ax.transAxes)
        ax.text(x0 + card_w - 0.01, y0 + card_h - 0.025, f"{count}", color='white', fontsize=9, ha='right', va='center', transform=ax.transAxes)
        
        for s_idx, symbol in enumerate(selected_by_sector[sector][:5]):
            ax.text(x0 + 0.01, y0 + card_h - 0.08 - s_idx * 0.04, symbol, color='black', fontsize=9, va='center', transform=ax.transAxes)
            
    fig.text(0.5, 0.01, "Balanced Kite 1-minute panel: five companies per segment; not index weighted.", ha="center", fontsize=10, color="gray")
    plt.savefig(artifact_path("company_selection_pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)

    # 3. Proxy reliability scorecard
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Indian Empirical Proxy Reliability", fontsize=16, fontweight="bold")
    
    metrics = [
        ("kappa_positive_rate", "kappa>0"),
        ("theta_positive_rate", "theta>0"),
        ("xi_positive_rate", "xi>0"),
        ("rho_negative_rate", "rho<0"),
        ("rho_unit_rate", "|rho|<=1"),
        ("heston_like_rate", "all gates"),
    ]
    
    y_labels = [proxy_short(row["proxy"]) for row in summary]
    x_labels = [label for _, label in metrics]
    
    data = np.zeros((len(summary), len(metrics)))
    for r_idx, row in enumerate(summary):
        for c_idx, (key, _) in enumerate(metrics):
            data[r_idx, c_idx] = to_float(row[key], 0.0)
            
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["#f6f8fc", "#466996"])
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            val = data[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color)
            
    ax.set_xticks(np.arange(-.5, len(x_labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(y_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.15)
    cbar.set_label("gate pass-rate fraction", color="gray", size=10)
    cbar.ax.tick_params(colors="gray")
    
    fig.text(0.5, 0.01, "Cells show pass-rate fractions from 0 to 1 across selected companies; darker cells indicate stronger gate support.", ha="center", fontsize=10, color="gray")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(artifact_path("proxy_reliability_pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)

    # 4. Rho distribution
    proxy_order = [r["proxy"] for r in summary]
    rho_groups_all = [[to_float(r["rho"]) for r in recovery if r["proxy"] == proxy and r["status"] == "ok"] for proxy in proxy_order]
    rho_groups = [[v for v in group if math.isfinite(v) and -1.0 <= v <= 1.0] for group in rho_groups_all]
    out_of_range = [sum(1 for v in group if math.isfinite(v) and not (-1.0 <= v <= 1.0)) for group in rho_groups_all]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Recovered Leverage by Proxy", fontsize=16, fontweight="bold")
    
    ax.boxplot(rho_groups, positions=np.arange(1, len(proxy_order) + 1), patch_artist=True, 
                       boxprops=dict(facecolor="#dde9fa", color="#2357a5", linewidth=2),
                       medianprops=dict(color="#c44137", linewidth=2),
                       whiskerprops=dict(color="black", linewidth=1.5),
                       capprops=dict(color="black", linewidth=1.5),
                       flierprops=dict(marker='o', markerfacecolor='black', markersize=3, alpha=0.5))
    
    ax.axhline(0, color="#c44137", linewidth=2)
    
    ax.set_xticks(np.arange(1, len(proxy_order) + 1))
    ax.set_xticklabels([proxy_short(p) for p in proxy_order])
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Recovered ρ (unitless correlation)")
    ax.set_xlabel("Variance proxy")
    ax.grid(True, linestyle="--", alpha=0.7)
    
    for idx, n_bad in enumerate(out_of_range, start=1):
        if n_bad > 0:
            ax.text(idx, -1.02, f"{n_bad} out", color="#c44137", ha='center', va='top', fontsize=9)
            
    fig.text(0.5, 0.01, "Boxes show valid Heston rho values inside [-1, 1]; out-of-range fits are counted below.", ha="center", fontsize=10, color="gray")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(artifact_path("rho_distribution_pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    global START_DATE, END_DATE, MIN_VALID_DAYS, OUTPUT_PREFIX, SOURCE_ROOT, DATA_ROOT
    args = parse_args()
    if args.covid_crash:
        args.start_date = "2020-02-01"
        args.end_date = "2020-06-30"
        if args.output_prefix == "indian_empirical":
            args.output_prefix = "indian_balanced50_covid_crash"
    START_DATE = args.start_date
    END_DATE = args.end_date
    OUTPUT_PREFIX = args.output_prefix
    SOURCE_ROOT = Path(args.data_root).expanduser()
    DATA_ROOT = resolve_data_root(SOURCE_ROOT)
    MIN_VALID_DAYS = estimate_min_valid_days(START_DATE, END_DATE)
    if not DATA_ROOT.exists():
        raise SystemExit(f"Data path does not exist: {DATA_ROOT}")

    max_companies = DEFAULT_MAX_COMPANIES
    print(f"Running Indian empirical HDVF for top {max_companies} companies.")
    print(f"Window: {START_DATE} to {END_DATE} | min_valid_days={MIN_VALID_DAYS} | prefix={OUTPUT_PREFIX}")
    print(f"Source: {DATA_ROOT}")

    selection_rows, selected_data = select_companies(max_companies)
    selected_ids = {str(r["company_id"]) for r in selection_rows if str(r.get("selected")) == "1"}
    selected_symbols = [str(r["symbol"]) for r in selection_rows if str(r.get("selected")) == "1"]
    print(f"Selected {len(selected_ids)} companies: {', '.join(selected_symbols[:12])}{'...' if len(selected_symbols) > 12 else ''}")

    daily_panel = create_daily_panel({cid: rows for cid, rows in selected_data.items() if cid in selected_ids})
    recovery_rows = []
    for idx, cid in enumerate(sorted(selected_ids), start=1):
        rows = selected_data[cid]
        meta = next(r for r in selection_rows if str(r["company_id"]) == cid)
        print(f"Recovering {idx:03d}/{len(selected_ids):03d}: {meta['symbol']}")
        for proxy in PROXIES:
            result = recover_company_proxy(rows, proxy, DEFAULT_KERNELS, DEFAULT_BANDWIDTH)
            result.update(
                {
                    "company_id": cid,
                    "symbol": meta["symbol"],
                    "name": meta["name"],
                    "proxy": proxy,
                    "kernel_centers": DEFAULT_KERNELS,
                    "bandwidth_factor": DEFAULT_BANDWIDTH,
                }
            )
            recovery_rows.append(result)

    summary_rows = summarize_by_proxy(recovery_rows)

    write_csv(artifact_path("company_selection_csv"), selection_rows)
    write_csv(artifact_path("daily_proxy_panel_csv"), daily_panel)
    write_csv(artifact_path("recovery_csv"), recovery_rows)
    write_csv(artifact_path("proxy_summary_csv"), summary_rows)
    render_all_plots(selection_rows, recovery_rows, summary_rows)

    print("\nDone. Wrote outputs directly to:")
    for key in [
        "company_selection_csv",
        "daily_proxy_panel_csv",
        "recovery_csv",
        "proxy_summary_csv",
        "recovery_pdf",
        "proxy_reliability_pdf",
        "rho_distribution_pdf",
        "company_selection_pdf",
    ]:
        print(f"  {artifact_path(key)}")


if __name__ == "__main__":
    main()
