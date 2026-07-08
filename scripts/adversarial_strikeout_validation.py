from __future__ import annotations

from math import erf, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "adversarial_validation"
EDGE_FILE = ROOT / "data" / "processed_noopp_wf2025_ext" / "bt_noopp_oos_edges.csv"


def american_to_decimal(odds: pd.Series) -> pd.Series:
    odds = pd.to_numeric(odds, errors="coerce")
    return pd.Series(
        np.where(odds > 0, 1 + odds / 100.0, 1 + 100.0 / odds.abs()),
        index=odds.index,
    )


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def add_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["game_date"] = pd.to_datetime(out["game_date"])
    out["season"] = out["game_date"].dt.year
    out["month"] = out["game_date"].dt.to_period("M").astype(str)
    out["calendar_month"] = out["game_date"].dt.month
    out["line"] = pd.to_numeric(out["line"], errors="coerce")
    out["strikeouts"] = pd.to_numeric(out["strikeouts"], errors="coerce")
    out["edge_pct"] = pd.to_numeric(out["edge_pct"], errors="coerce")
    out["strikeouts_projection"] = pd.to_numeric(out["strikeouts_projection"], errors="coerce")
    out["signed_gap"] = out["strikeouts_projection"] - out["line"]
    out["abs_gap"] = out["signed_gap"].abs()
    out["norm_gap"] = out["abs_gap"] / out["line"].clip(lower=0.5)
    out["edge_gap_product"] = out["edge_pct"] * out["abs_gap"]
    out["side"] = out["best_side"].astype(str).str.lower()
    bet_odds = np.where(
        out["side"].eq("over"),
        pd.to_numeric(out["over_odds"], errors="coerce"),
        pd.to_numeric(out["under_odds"], errors="coerce"),
    )
    out["bet_odds"] = bet_odds
    out["decimal_odds"] = american_to_decimal(pd.Series(bet_odds, index=out.index))
    out["won"] = np.where(
        out["side"].eq("over"),
        out["strikeouts"] > out["line"],
        out["strikeouts"] < out["line"],
    )
    out["push"] = out["strikeouts"] == out["line"]
    out["profit_unit"] = np.where(
        out["push"],
        0.0,
        np.where(out["won"], out["decimal_odds"] - 1.0, -1.0),
    )
    out["direction_correct"] = np.where(
        out["signed_gap"] >= 0,
        out["strikeouts"] > out["line"],
        out["strikeouts"] < out["line"],
    )
    return out


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["game_date", "pitcher_id", "market", "line", "side"]
    return (
        df.sort_values(["edge_pct", "abs_gap"], ascending=[False, False])
        .drop_duplicates(keys, keep="first")
        .reset_index(drop=True)
    )


def bootstrap_ci(values: np.ndarray, n_iter: int = 5000, seed: int = 29) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_iter, len(values)), replace=True).mean(axis=1)
    lo, mid, hi = np.quantile(samples, [0.025, 0.5, 0.975])
    return float(lo), float(mid), float(hi)


def permutation_p_value(parent: pd.DataFrame, selected_mask: pd.Series, n_iter: int = 5000, seed: int = 31) -> float:
    parent_profit = parent["profit_unit"].to_numpy(dtype=float)
    k = int(selected_mask.sum())
    if k == 0:
        return np.nan
    observed = float(parent.loc[selected_mask, "profit_unit"].mean())
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_iter):
        idx = rng.choice(len(parent_profit), size=k, replace=False)
        if float(parent_profit[idx].mean()) >= observed:
            count += 1
    return float((count + 1) / (n_iter + 1))


def summarize(df: pd.DataFrame, parent: pd.DataFrame | None = None, label: str = "") -> dict[str, float | int | str]:
    d = df.copy()
    profits = d["profit_unit"].to_numpy(dtype=float)
    n = len(d)
    wins = int((d["won"] & ~d["push"]).sum())
    losses = int((~d["won"] & ~d["push"]).sum())
    pushes = int(d["push"].sum())
    roi = float(np.nanmean(profits)) if n else np.nan
    std = float(np.nanstd(profits, ddof=1)) if n > 1 else np.nan
    se = std / sqrt(n) if n > 1 else np.nan
    ci_lo, ci_mid, ci_hi = bootstrap_ci(profits) if n else (np.nan, np.nan, np.nan)
    z = roi / se if se and np.isfinite(se) and se > 0 else np.nan
    p_norm = 1 - norm_cdf(z) if np.isfinite(z) else np.nan
    p_perm = np.nan
    if parent is not None and n:
        selected = parent.index.isin(d.index)
        p_perm = permutation_p_value(parent, pd.Series(selected, index=parent.index))
    return {
        "strategy": label,
        "bets": n,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": float(wins / (wins + losses)) if wins + losses else np.nan,
        "roi": roi,
        "profit_units": float(np.nansum(profits)) if n else 0.0,
        "avg_edge_pct": float(d["edge_pct"].mean()) if n else np.nan,
        "avg_abs_gap": float(d["abs_gap"].mean()) if n else np.nan,
        "avg_odds": float(pd.to_numeric(d["bet_odds"], errors="coerce").mean()) if n else np.nan,
        "overs": int(d["side"].eq("over").sum()) if n else 0,
        "unders": int(d["side"].eq("under").sum()) if n else 0,
        "bootstrap_roi_lo": ci_lo,
        "bootstrap_roi_mid": ci_mid,
        "bootstrap_roi_hi": ci_hi,
        "normal_p_roi_gt_0": p_norm,
        "permutation_p_vs_parent": p_perm,
        "brier": brier_score(d),
        "log_loss": log_loss_score(d),
        "ece": expected_calibration_error(d),
        "mce": maximum_calibration_error(d),
    }


def brier_score(df: pd.DataFrame) -> float:
    if df.empty or "over_probability" not in df:
        return np.nan
    y = (df["strikeouts"] > df["line"]).astype(float)
    p = pd.to_numeric(df["over_probability"], errors="coerce").clip(1e-6, 1 - 1e-6)
    valid = p.notna() & y.notna()
    return float(((p[valid] - y[valid]) ** 2).mean()) if valid.any() else np.nan


def log_loss_score(df: pd.DataFrame) -> float:
    if df.empty or "over_probability" not in df:
        return np.nan
    y = (df["strikeouts"] > df["line"]).astype(float)
    p = pd.to_numeric(df["over_probability"], errors="coerce").clip(1e-6, 1 - 1e-6)
    valid = p.notna() & y.notna()
    if not valid.any():
        return np.nan
    return float((-(y[valid] * np.log(p[valid]) + (1 - y[valid]) * np.log(1 - p[valid]))).mean())


def expected_calibration_error(df: pd.DataFrame, bins: int = 10) -> float:
    if df.empty or "over_probability" not in df:
        return np.nan
    y = (df["strikeouts"] > df["line"]).astype(float)
    p = pd.to_numeric(df["over_probability"], errors="coerce")
    valid = p.notna() & y.notna()
    if not valid.any():
        return np.nan
    p = p[valid]
    y = y[valid]
    cuts = pd.cut(p, np.linspace(0, 1, bins + 1), include_lowest=True)
    total = len(p)
    ece = 0.0
    for _, idx in p.groupby(cuts, observed=True).groups.items():
        if len(idx) == 0:
            continue
        ece += len(idx) / total * abs(float(p.loc[idx].mean()) - float(y.loc[idx].mean()))
    return float(ece)


def maximum_calibration_error(df: pd.DataFrame, bins: int = 10) -> float:
    if df.empty or "over_probability" not in df:
        return np.nan
    y = (df["strikeouts"] > df["line"]).astype(float)
    p = pd.to_numeric(df["over_probability"], errors="coerce")
    valid = p.notna() & y.notna()
    if not valid.any():
        return np.nan
    p = p[valid]
    y = y[valid]
    cuts = pd.cut(p, np.linspace(0, 1, bins + 1), include_lowest=True)
    errors = []
    for _, idx in p.groupby(cuts, observed=True).groups.items():
        if len(idx):
            errors.append(abs(float(p.loc[idx].mean()) - float(y.loc[idx].mean())))
    return float(max(errors)) if errors else np.nan


def apply_strategy(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    q = df.copy()
    if "edge_min" in spec:
        q = q[q["edge_pct"] >= spec["edge_min"]]
    if "eg_min" in spec:
        q = q[q["edge_gap_product"] >= spec["eg_min"]]
    if "gap_min" in spec:
        q = q[q["abs_gap"] >= spec["gap_min"]]
    if "norm_gap_min" in spec:
        q = q[q["norm_gap"] >= spec["norm_gap_min"]]
    if spec.get("skip_june"):
        q = q[q["calendar_month"] != 6]
    if spec.get("overs_only"):
        q = q[q["side"] == "over"]
    if spec.get("unders_only"):
        q = q[q["side"] == "under"]
    if "line_min" in spec:
        q = q[q["line"] >= spec["line_min"]]
    if "line_max" in spec:
        q = q[q["line"] <= spec["line_max"]]
    if "odds_min" in spec:
        q = q[q["bet_odds"] >= spec["odds_min"]]
    if "odds_max" in spec:
        q = q[q["bet_odds"] <= spec["odds_max"]]
    return dedupe(q)


def reliability_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    d = df.copy()
    d["over_outcome"] = (d["strikeouts"] > d["line"]).astype(float)
    d["prob_bin"] = pd.cut(
        pd.to_numeric(d["over_probability"], errors="coerce"),
        np.linspace(0, 1, 11),
        include_lowest=True,
    )
    rows = []
    for b, g in d.groupby("prob_bin", observed=True):
        rows.append(
            {
                "strategy": label,
                "bin": str(b),
                "n": len(g),
                "mean_over_prob": g["over_probability"].mean(),
                "actual_over_rate": g["over_outcome"].mean(),
                "abs_error": abs(g["over_probability"].mean() - g["over_outcome"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(EDGE_FILE)
    df = dedupe(add_outcomes(raw))
    df.to_csv(OUT_DIR / "deduped_noopp_oos_edges.csv", index=False)

    y2025 = df[df["season"] == 2025].copy()
    y2026 = df[df["season"] == 2026].copy()

    specs = {
        "all_edges": {},
        "edge>=12": {"edge_min": 12},
        "edge>=20": {"edge_min": 20},
        "edge>=25": {"edge_min": 25},
        "eg>=3": {"eg_min": 3},
        "eg>=5": {"eg_min": 5},
        "eg>=6": {"eg_min": 6},
        "eg>=8": {"eg_min": 8},
        "eg>=10": {"eg_min": 10},
        "eg>=12": {"eg_min": 12},
        "eg>=15": {"eg_min": 15},
        "eg>=12_skip_june": {"eg_min": 12, "skip_june": True},
        "eg>=6_skip_june": {"eg_min": 6, "skip_june": True},
        "eg>=8_skip_june": {"eg_min": 8, "skip_june": True},
        "eg>=12_overs_skip_june": {"eg_min": 12, "skip_june": True, "overs_only": True},
        "eg>=12_unders_skip_june": {"eg_min": 12, "skip_june": True, "unders_only": True},
        "edge>=12_gap>=0.5": {"edge_min": 12, "gap_min": 0.5},
        "edge>=12_normgap>=0.10": {"edge_min": 12, "norm_gap_min": 0.10},
    }

    rows = []
    monthly_rows = []
    side_rows = []
    reliability = []
    for name, spec in specs.items():
        for year, parent in [(2025, y2025), (2026, y2026)]:
            selected = apply_strategy(parent, spec)
            row = summarize(selected, parent=parent, label=name)
            row["year"] = year
            rows.append(row)
            if not selected.empty:
                month_records = []
                for month_key, group in selected.groupby("month"):
                    record = summarize(group)
                    record["month"] = month_key
                    month_records.append(record)
                month = pd.DataFrame(month_records)
                month["strategy"] = name
                month["year"] = year
                monthly_rows.append(month)
                side_records = []
                for side_key, group in selected.groupby("side"):
                    record = summarize(group)
                    record["side"] = side_key
                    side_records.append(record)
                side = pd.DataFrame(side_records)
                side["strategy"] = name
                side["year"] = year
                side_rows.append(side)
                reliability.append(reliability_table(selected, f"{name}_{year}"))

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "strategy_summary.csv", index=False)
    if monthly_rows:
        pd.concat(monthly_rows, ignore_index=True).to_csv(OUT_DIR / "monthly_summary.csv", index=False)
    if side_rows:
        pd.concat(side_rows, ignore_index=True).to_csv(OUT_DIR / "side_summary.csv", index=False)
    if reliability:
        pd.concat(reliability, ignore_index=True).to_csv(OUT_DIR / "reliability_tables.csv", index=False)

    # Threshold grid selected on 2025, evaluated on 2026. Keep simple, pregame-only filters.
    grid_rows = []
    for eg in [0, 3, 5, 6, 8, 10, 12, 15, 18, 20]:
        for skip_june in [False, True]:
            for side_filter in ["all", "over", "under"]:
                spec = {"eg_min": eg}
                if skip_june:
                    spec["skip_june"] = True
                if side_filter == "over":
                    spec["overs_only"] = True
                elif side_filter == "under":
                    spec["unders_only"] = True
                s2025 = apply_strategy(y2025, spec)
                s2026 = apply_strategy(y2026, spec)
                if len(s2025) < 100 or len(s2026) < 25:
                    continue
                r2025 = summarize(s2025, label="2025")
                r2026 = summarize(s2026, label="2026")
                grid_rows.append(
                    {
                        "eg_min": eg,
                        "skip_june": skip_june,
                        "side_filter": side_filter,
                        "bets_2025": r2025["bets"],
                        "roi_2025": r2025["roi"],
                        "lo_2025": r2025["bootstrap_roi_lo"],
                        "hi_2025": r2025["bootstrap_roi_hi"],
                        "p_2025": r2025["normal_p_roi_gt_0"],
                        "bets_2026": r2026["bets"],
                        "roi_2026": r2026["roi"],
                        "lo_2026": r2026["bootstrap_roi_lo"],
                        "hi_2026": r2026["bootstrap_roi_hi"],
                        "p_2026": r2026["normal_p_roi_gt_0"],
                        "profit_2025": r2025["profit_units"],
                        "profit_2026": r2026["profit_units"],
                    }
                )
    grid = pd.DataFrame(grid_rows).sort_values(["roi_2026", "bets_2026"], ascending=[False, False])
    grid.to_csv(OUT_DIR / "threshold_grid_train2025_test2026.csv", index=False)

    # Drift metrics for major numeric features.
    feature_candidates = [
        "strikeouts_projection",
        "line",
        "edge_pct",
        "abs_gap",
        "edge_gap_product",
        "over_probability",
        "league_k",
        "opp_batting_k_rate_roll10",
        "sc_swinging_strike_rate_roll10",
        "adv_whiff_rate_roll10",
        "pitcher_starts_ytd",
        "days_into_season",
    ]
    drift_rows = []
    for col in feature_candidates:
        if col not in df.columns:
            continue
        a = pd.to_numeric(y2025[col], errors="coerce").dropna()
        b = pd.to_numeric(y2026[col], errors="coerce").dropna()
        if a.empty or b.empty:
            continue
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        drift_rows.append(
            {
                "feature": col,
                "mean_2025": a.mean(),
                "mean_2026": b.mean(),
                "std_mean_diff": (b.mean() - a.mean()) / pooled if pooled > 0 else np.nan,
                "median_2025": a.median(),
                "median_2026": b.median(),
            }
        )
    pd.DataFrame(drift_rows).to_csv(OUT_DIR / "feature_prediction_drift.csv", index=False)

    print(f"Wrote adversarial validation outputs to {OUT_DIR}")
    print(summary[summary['strategy'].isin(['eg>=6_skip_june', 'eg>=8_skip_june', 'eg>=12_skip_june', 'eg>=12_overs_skip_june'])].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
