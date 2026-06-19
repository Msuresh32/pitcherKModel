from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BOOKS = ["draftkings", "betonlineag", "fanduel", "betrivers", "pinnacle"]
BOOK_LABELS = {
    "draftkings": "DraftKings",
    "betonlineag": "BetOnline",
    "fanduel": "FanDuel",
    "betrivers": "BetRivers",
    "pinnacle": "Pinnacle",
}

BET_FILES = [
    ("2025_threshold_selection", Path("data/processed_2024/thresh_sel_2025_dk_edges.csv")),
    ("2026_mar_apr_frozen", Path("data/processed/wf2026_p1_mar_apr_edges.csv")),
    ("2026_may_frozen", Path("data/processed_apr2026/wf2026_p2_may_edges.csv")),
    ("2026_jun_frozen", Path("data/processed/wf2026_p3_jun_edges.csv")),
]

LOCAL_CLOSE_FILES = [
    Path("data/odds/historical_pitcher_props_2025.csv"),
    Path("data/odds/full_2026_odds.csv"),
]
PINNACLE_CLOSE_FILE = Path("data/odds/pinnacle_close_cache.csv")

EDGE_BANDS = [
    (0, 5, "0-5%"),
    (5, 10, "5-10%"),
    (10, 15, "10-15%"),
    (15, 20, "15-20%"),
    (20, 25, "20-25%"),
    (25, np.inf, "25%+"),
]

PROB_BINS = [
    (0.50, 0.52, "50-52%"),
    (0.52, 0.54, "52-54%"),
    (0.54, 0.56, "54-56%"),
    (0.56, 0.58, "56-58%"),
    (0.58, 0.60, "58-60%"),
    (0.60, 1.01, "60%+"),
]


def american_to_prob(odds: float) -> float:
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    return 100 / (100 + odds) if odds > 0 else abs(odds) / (abs(odds) + 100)


def american_to_decimal(odds: float) -> float:
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    return 1 + odds / 100 if odds > 0 else 1 + 100 / abs(odds)


def devig_pair(over_odds: float, under_odds: float) -> tuple[float, float]:
    if pd.isna(over_odds) or pd.isna(under_odds):
        return np.nan, np.nan
    over_prob = american_to_prob(over_odds)
    under_prob = american_to_prob(under_odds)
    denom = over_prob + under_prob
    if denom <= 0:
        return np.nan, np.nan
    return over_prob / denom, under_prob / denom


def t_stat(values: pd.Series) -> tuple[int, float, float, float, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    n = len(values)
    if n == 0:
        return 0, np.nan, np.nan, np.nan, np.nan
    mean = float(values.mean())
    if n < 2:
        return n, mean, np.nan, np.nan, np.nan
    se = float(values.std(ddof=1) / math.sqrt(n))
    t = mean / se if se > 0 else np.nan
    return n, mean, t, mean - 1.96 * se, mean + 1.96 * se


def load_bets(edge_min: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces = []
    audit_rows = []
    for label, path in BET_FILES:
        if not path.exists():
            audit_rows.append({"slice": label, "path": str(path), "status": "missing"})
            continue
        df = pd.read_csv(path)
        df = df[df["market"].eq("strikeouts")].copy()
        raw_rows = len(df)
        df["source_slice"] = label
        pieces.append(df)
        dedup_rows = len(df.drop_duplicates(["game_date", "pitcher_name", "line", "best_side"]))
        audit_rows.append(
            {
                "slice": label,
                "path": str(path),
                "status": "ok",
                "raw_rows": raw_rows,
                "dedup_rows": dedup_rows,
                "duplicate_rows": raw_rows - dedup_rows,
                "duplicate_pct": (raw_rows - dedup_rows) / raw_rows if raw_rows else 0.0,
            }
        )

    if not pieces:
        raise FileNotFoundError("No bet edge files found.")

    bets = pd.concat(pieces, ignore_index=True, sort=False)
    bets["game_date"] = pd.to_datetime(bets["game_date"])
    bets = (
        bets.sort_values("edge_pct", ascending=False)
        .drop_duplicates(["game_date", "pitcher_name", "line", "best_side"])
        .reset_index(drop=True)
    )
    bets = bets[pd.to_numeric(bets["edge_pct"], errors="coerce") >= edge_min].copy()
    bets["year"] = bets["game_date"].dt.year
    bets["month"] = bets["game_date"].dt.to_period("M").astype(str)

    bets[["nv_entry_over", "nv_entry_under"]] = bets.apply(
        lambda row: pd.Series(devig_pair(row["over_odds"], row["under_odds"])),
        axis=1,
    )
    bets["nv_entry_side"] = np.where(
        bets["best_side"].eq("over"), bets["nv_entry_over"], bets["nv_entry_under"]
    )
    bets["entry_odds"] = np.where(
        bets["best_side"].eq("over"), bets["over_odds"], bets["under_odds"]
    )
    bets["won"] = np.where(
        bets["best_side"].eq("over"),
        bets["strikeouts"] > bets["line"],
        bets["strikeouts"] < bets["line"],
    )
    bets["profit"] = np.where(
        bets["won"], bets["entry_odds"].map(american_to_decimal) - 1, -1.0
    )
    bets["projection_gap_abs"] = (bets["strikeouts_projection"] - bets["line"]).abs()
    bets["bet_probability"] = np.where(
        bets["best_side"].eq("over"), bets.get("over_probability"), bets.get("under_probability")
    )
    return bets, pd.DataFrame(audit_rows)


def _local_close_file(path: Path, books: Iterable[str]) -> pd.DataFrame:
    odds = pd.read_csv(path)
    odds = odds[odds["snapshot_type"].eq("close") & odds["bookmaker"].isin(books)].copy()
    odds["game_date"] = pd.to_datetime(odds["game_date"])
    odds["fetched_at"] = pd.to_datetime(odds["fetched_at"], errors="coerce")

    # De-vigged CLV requires a real two-sided market row. Filter before deduping so
    # one-sided alternates cannot mix with main lines or blank out a valid close.
    odds = odds[odds["over_odds"].notna() & odds["under_odds"].notna()].copy()
    odds = (
        odds.sort_values("fetched_at")
        .groupby(["game_date", "bookmaker", "player_name", "line"], as_index=False)
        .tail(1)
    )
    odds[["nv_over", "nv_under"]] = odds.apply(
        lambda row: pd.Series(devig_pair(row["over_odds"], row["under_odds"])),
        axis=1,
    )
    return odds


def load_close_index() -> pd.DataFrame:
    pieces = []
    for path in LOCAL_CLOSE_FILES:
        if path.exists():
            pieces.append(_local_close_file(path, ["draftkings", "betonlineag", "fanduel", "betrivers"]))

    if PINNACLE_CLOSE_FILE.exists():
        pin = pd.read_csv(PINNACLE_CLOSE_FILE)
        pin["game_date"] = pd.to_datetime(pin["game_date"])
        pin["bookmaker"] = "pinnacle"
        pin = pin[pin["over_odds"].notna() & pin["under_odds"].notna()].copy()
        pin[["nv_over", "nv_under"]] = pin.apply(
            lambda row: pd.Series(devig_pair(row["over_odds"], row["under_odds"])),
            axis=1,
        )
        pieces.append(pin)

    if not pieces:
        raise FileNotFoundError("No closing odds files found.")
    return pd.concat(pieces, ignore_index=True, sort=False)


def match_clv(bets: pd.DataFrame, close: pd.DataFrame, book: str) -> pd.DataFrame:
    sub = close[close["bookmaker"].eq(book)][
        ["game_date", "player_name", "line", "nv_over", "nv_under"]
    ].copy()
    matched = bets.merge(
        sub,
        left_on=["game_date", "pitcher_name", "line"],
        right_on=["game_date", "player_name", "line"],
        how="left",
    )
    matched = matched.dropna(subset=["nv_over", "nv_under", "nv_entry_side"]).copy()
    matched["nv_close_side"] = np.where(
        matched["best_side"].eq("over"), matched["nv_over"], matched["nv_under"]
    )
    matched["clv_pp"] = (matched["nv_close_side"] - matched["nv_entry_side"]) * 100
    matched["book"] = book
    return matched


def summarize_books(bets: pd.DataFrame, close: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for book in BOOKS:
        matched = match_clv(bets, close, book)
        n, mean, t, lo, hi = t_stat(matched["clv_pp"])
        daily = matched.groupby(matched["game_date"].dt.date)["clv_pp"].mean()
        dn, dmean, dt, dlo, dhi = t_stat(daily)
        rows.append(
            {
                "subset": label,
                "book": BOOK_LABELS[book],
                "bets": len(bets),
                "matched": n,
                "match_pct": n / len(bets) if len(bets) else np.nan,
                "clv_pp": mean,
                "t_stat": t,
                "ci_low": lo,
                "ci_high": hi,
                "positive_pct": float((matched["clv_pp"] > 0).mean()) if n else np.nan,
                "win_pct": float(matched["won"].mean()) if n else np.nan,
                "date_clusters": dn,
                "date_cluster_clv_pp": dmean,
                "date_cluster_t": dt,
                "date_cluster_ci_low": dlo,
                "date_cluster_ci_high": dhi,
            }
        )
    return pd.DataFrame(rows)


def summarize_group(
    bets: pd.DataFrame,
    close: pd.DataFrame,
    group_cols: list[str],
    label: str,
    book: str = "draftkings",
) -> pd.DataFrame:
    matched = match_clv(bets, close, book)
    rows = []
    for keys, group in matched.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        n, mean, t, lo, hi = t_stat(group["clv_pp"])
        row.update(
            {
                "subset": label,
                "book": BOOK_LABELS[book],
                "n": n,
                "clv_pp": mean,
                "t_stat": t,
                "ci_low": lo,
                "ci_high": hi,
                "positive_pct": float((group["clv_pp"] > 0).mean()) if n else np.nan,
                "win_pct": float(group["won"].mean()) if n else np.nan,
                "roi": float(group["profit"].mean()) if n else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def banded_summary(
    bets: pd.DataFrame,
    close: pd.DataFrame,
    bands: list[tuple[float, float, str]],
    value_col: str,
    label: str,
    book: str = "draftkings",
) -> pd.DataFrame:
    matched = match_clv(bets, close, book)
    rows = []
    for lo, hi, band_label in bands:
        group = matched[(matched[value_col] >= lo) & (matched[value_col] < hi)].copy()
        if group.empty:
            continue
        n, mean, t, cil, cih = t_stat(group["clv_pp"])
        rows.append(
            {
                "subset": label,
                "book": BOOK_LABELS[book],
                "band": band_label,
                "n": n,
                "clv_pp": mean,
                "t_stat": t,
                "ci_low": cil,
                "ci_high": cih,
                "positive_pct": float((group["clv_pp"] > 0).mean()),
                "win_pct": float(group["won"].mean()),
                "roi": float(group["profit"].mean()),
            }
        )
    return pd.DataFrame(rows)


def probability_calibration(bets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    df = bets.dropna(subset=["bet_probability"]).copy()
    for lo, hi, label in PROB_BINS:
        sub = df[(df["bet_probability"] >= lo) & (df["bet_probability"] < hi)]
        if sub.empty:
            continue
        rows.append(
            {
                "probability_bin": label,
                "n": len(sub),
                "mean_model_probability": float(sub["bet_probability"].mean()),
                "actual_win_pct": float(sub["won"].mean()),
                "calibration_gap": float(sub["bet_probability"].mean() - sub["won"].mean()),
                "roi": float(sub["profit"].mean()),
            }
        )
    return pd.DataFrame(rows)


def temporal_audit() -> pd.DataFrame:
    rows = []
    config_paths = [
        Path("config/config_2024only.yaml"),
        Path("config/config_apr2026.yaml"),
        Path("config/config.yaml"),
    ]
    try:
        import yaml
    except ImportError:
        return pd.DataFrame([{"check": "config_temporal_integrity", "status": "skipped_no_yaml"}])

    for path in config_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        training = cfg.get("training", {})
        train_end = training.get("train_end")
        backtest_start = training.get("backtest_start")
        ok = pd.to_datetime(backtest_start) > pd.to_datetime(train_end)
        rows.append(
            {
                "check": "config_temporal_integrity",
                "file": str(path),
                "train_end": train_end,
                "backtest_start": backtest_start,
                "backtest_end": training.get("backtest_end"),
                "status": "pass" if ok else "fail",
            }
        )
    return pd.DataFrame(rows)


def rolling_feature_audit(bets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if {"pitcher_id", "game_date", "strikeouts", "p_strikeouts_roll3"}.issubset(bets.columns):
        corrs = []
        for _, group in bets.sort_values("game_date").groupby("pitcher_id"):
            if len(group) >= 5 and group["p_strikeouts_roll3"].notna().sum() >= 3:
                corrs.append(group["strikeouts"].corr(group["p_strikeouts_roll3"]))
        rows.append(
            {
                "check": "same_game_roll3_correlation",
                "n_pitchers": len(corrs),
                "mean_corr": float(np.nanmean(corrs)) if corrs else np.nan,
                "status": "review" if corrs and np.nanmean(corrs) > 0.75 else "pass",
                "note": "High same-game correlation can indicate leakage; moderate autocorrelation is expected.",
            }
        )

    direct_feature_names = {"strikeouts_projection", "line", "edge_pct", "over_probability", "under_probability"}
    suspicious = []
    numeric_cols = [
        col
        for col in bets.columns
        if col not in direct_feature_names and pd.api.types.is_numeric_dtype(bets[col])
    ]
    for col in numeric_cols:
        if col == "strikeouts":
            continue
        corr = bets["strikeouts"].corr(bets[col])
        if pd.notna(corr) and abs(corr) > 0.75:
            suspicious.append((col, corr))
    rows.append(
        {
            "check": "high_target_correlation_scan",
            "n_features_over_abs_0_75": len(suspicious),
            "top_features": json.dumps(
                sorted(suspicious, key=lambda x: abs(x[1]), reverse=True)[:10]
            ),
            "status": "review" if suspicious else "pass",
        }
    )
    return pd.DataFrame(rows)


def write_report(
    out_dir: Path,
    bets: pd.DataFrame,
    close: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    edge_min: float,
) -> None:
    lines = []
    lines.append("# Pitcher Model Validation Report")
    lines.append("")
    lines.append(f"Generated with `scripts/validate_model_edge.py`, edge_min={edge_min:g}.")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Deduped qualifying bets: {len(bets):,}")
    lines.append(f"- Date range: {bets['game_date'].min().date()} to {bets['game_date'].max().date()}")
    lines.append(f"- Closing rows by book: {close['bookmaker'].value_counts().to_dict()}")
    lines.append("")
    lines.append("## Headline Corrected CLV")
    lines.append("")
    lines.append(tables["book_summary"].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## DK Edge Bands")
    lines.append("")
    lines.append(tables["dk_edge_bands"].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Probability Calibration")
    lines.append("")
    lines.append(tables["probability_calibration"].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Audits")
    lines.append("")
    lines.append(tables["input_audit"].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append(tables["temporal_audit"].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append(tables["rolling_audit"].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "This report uses corrected two-sided de-vigged closing prices. "
        "It intentionally avoids `groupby.last()` on odds rows because that can mix "
        "non-null over/under prices from different rows."
    )
    lines.append(
        "The 2025 slice is out-of-sample for the projection model when produced from "
        "`config/config_2024only.yaml`; threshold selection on 2025 should still be "
        "treated as model-selection evidence rather than final blind proof."
    )
    (out_dir / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-min", type=float, default=0.0)
    parser.add_argument("--out-dir", default="data/processed/validation")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bets, input_audit = load_bets(args.edge_min)
    close = load_close_index()

    subsets = {
        "combined_edge0": bets,
        "2025_edge0": bets[bets["year"].eq(2025)].copy(),
        "2026_edge0": bets[bets["year"].eq(2026)].copy(),
        "combined_edge15": bets[bets["edge_pct"].ge(15)].copy(),
        "2026_edge15": bets[bets["year"].eq(2026) & bets["edge_pct"].ge(15)].copy(),
    }

    book_summary = pd.concat(
        [summarize_books(sub, close, label) for label, sub in subsets.items() if not sub.empty],
        ignore_index=True,
    )
    dk_edge_bands = pd.concat(
        [
            banded_summary(sub, close, EDGE_BANDS, "edge_pct", label, "draftkings")
            for label, sub in subsets.items()
            if not sub.empty
        ],
        ignore_index=True,
    )
    pinnacle_edge_bands = pd.concat(
        [
            banded_summary(sub, close, EDGE_BANDS, "edge_pct", label, "pinnacle")
            for label, sub in subsets.items()
            if not sub.empty
        ],
        ignore_index=True,
    )
    side_summary = pd.concat(
        [
            summarize_group(sub, close, ["best_side"], label, "draftkings")
            for label, sub in subsets.items()
            if not sub.empty
        ],
        ignore_index=True,
    )
    month_summary = summarize_group(bets, close, ["month"], "combined_edge0", "draftkings")
    gap_bands = [
        (0, 0.25, "0-0.25"),
        (0.25, 0.5, "0.25-0.50"),
        (0.5, 0.75, "0.50-0.75"),
        (0.75, 1.0, "0.75-1.00"),
        (1.0, np.inf, "1.00+"),
    ]
    gap_summary = banded_summary(bets, close, gap_bands, "projection_gap_abs", "combined_edge0", "draftkings")
    prob_cal = probability_calibration(bets)
    temporal = temporal_audit()
    rolling = rolling_feature_audit(bets)

    tables = {
        "book_summary": book_summary,
        "dk_edge_bands": dk_edge_bands,
        "pinnacle_edge_bands": pinnacle_edge_bands,
        "side_summary": side_summary,
        "month_summary": month_summary,
        "gap_summary": gap_summary,
        "probability_calibration": prob_cal,
        "input_audit": input_audit,
        "temporal_audit": temporal,
        "rolling_audit": rolling,
    }

    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)

    write_report(out_dir, bets, close, tables, args.edge_min)

    print(f"Validation complete. Wrote report and CSVs to {out_dir}")
    print("")
    print("Headline corrected CLV:")
    print(book_summary[book_summary["subset"].eq("combined_edge0")].to_string(index=False))
    print("")
    print("DK edge bands:")
    print(dk_edge_bands[dk_edge_bands["subset"].eq("combined_edge0")].to_string(index=False))


if __name__ == "__main__":
    main()
