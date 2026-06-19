from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.calibration import (
    apply_probability_calibrator,
    build_probability_calibration,
    load_calibration,
    save_calibration,
)
from src.odds.pricing import expected_value, fair_american_odds, kelly_fraction


def _american_to_decimal(odds: float) -> float:
    odds = float(odds)
    return 1 + odds / 100 if odds > 0 else 1 + 100 / abs(odds)


def _prepare_source(path: Path, market: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["market"].eq(market)].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = (
        df.sort_values("edge_pct", ascending=False)
        .drop_duplicates(["game_date", "pitcher_name", "line"])
        .reset_index(drop=True)
    )
    return df


def _score_calibration(df: pd.DataFrame, probability_col: str) -> pd.DataFrame:
    rows = []
    bins = [
        (0.00, 0.40, "<40%"),
        (0.40, 0.45, "40-45%"),
        (0.45, 0.50, "45-50%"),
        (0.50, 0.55, "50-55%"),
        (0.55, 0.60, "55-60%"),
        (0.60, 1.01, "60%+"),
    ]
    outcome = (df["strikeouts"] > df["line"]).astype(float)
    for lo, hi, label in bins:
        sub = df[(df[probability_col] >= lo) & (df[probability_col] < hi)].copy()
        if sub.empty:
            continue
        sub_outcome = outcome.loc[sub.index]
        rows.append(
            {
                "probability_col": probability_col,
                "bin": label,
                "n": len(sub),
                "mean_probability": float(sub[probability_col].mean()),
                "actual_over_rate": float(sub_outcome.mean()),
                "gap": float(sub[probability_col].mean() - sub_outcome.mean()),
                "brier": float(((sub[probability_col] - sub_outcome) ** 2).mean()),
            }
        )
    rows.append(
        {
            "probability_col": probability_col,
            "bin": "ALL",
            "n": len(df),
            "mean_probability": float(df[probability_col].mean()),
            "actual_over_rate": float(outcome.mean()),
            "gap": float(df[probability_col].mean() - outcome.mean()),
            "brier": float(((df[probability_col] - outcome) ** 2).mean()),
        }
    )
    return pd.DataFrame(rows)


def _rescore_edges(df: pd.DataFrame, calibrator: dict, bankroll: float, max_kelly: float) -> pd.DataFrame:
    out = df.copy()
    out["over_probability_calibrated"] = out["over_probability"].map(
        lambda p: apply_probability_calibrator(p, calibrator)
    )
    out["under_probability_calibrated"] = 1 - out["over_probability_calibrated"]
    out["fair_over_odds_calibrated"] = out["over_probability_calibrated"].map(fair_american_odds)
    out["fair_under_odds_calibrated"] = out["under_probability_calibrated"].map(fair_american_odds)
    out["over_ev_calibrated"] = out.apply(
        lambda r: expected_value(r["over_probability_calibrated"], r["over_odds"]),
        axis=1,
    )
    out["under_ev_calibrated"] = out.apply(
        lambda r: expected_value(r["under_probability_calibrated"], r["under_odds"]),
        axis=1,
    )
    out["best_side_calibrated"] = np.where(
        out["over_ev_calibrated"] >= out["under_ev_calibrated"], "over", "under"
    )
    out["edge_pct_calibrated"] = np.where(
        out["best_side_calibrated"].eq("over"),
        out["over_ev_calibrated"],
        out["under_ev_calibrated"],
    ) * 100
    out["bet_probability_calibrated"] = np.where(
        out["best_side_calibrated"].eq("over"),
        out["over_probability_calibrated"],
        out["under_probability_calibrated"],
    )
    out["entry_odds_calibrated"] = np.where(
        out["best_side_calibrated"].eq("over"), out["over_odds"], out["under_odds"]
    )
    out["won_calibrated"] = np.where(
        out["best_side_calibrated"].eq("over"),
        out["strikeouts"] > out["line"],
        out["strikeouts"] < out["line"],
    )
    out["profit_calibrated"] = np.where(
        out["won_calibrated"], out["entry_odds_calibrated"].map(_american_to_decimal) - 1, -1
    )
    out["kelly_fraction_calibrated"] = out.apply(
        lambda r: kelly_fraction(
            r["bet_probability_calibrated"],
            r["entry_odds_calibrated"],
            max_kelly,
        ),
        axis=1,
    )
    out["stake_calibrated"] = bankroll * out["kelly_fraction_calibrated"]
    out["profit_staked_calibrated"] = out["stake_calibrated"] * np.where(
        out["won_calibrated"],
        out["entry_odds_calibrated"].map(_american_to_decimal) - 1,
        -1,
    )
    return out


def _threshold_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for edge_col, side_col, won_col, profit_col, label in [
        ("edge_pct", "best_side", "won_original", "profit_original", "original"),
        (
            "edge_pct_calibrated",
            "best_side_calibrated",
            "won_calibrated",
            "profit_calibrated",
            "calibrated",
        ),
    ]:
        tmp = df.copy()
        if label == "original":
            tmp["won_original"] = np.where(
                tmp["best_side"].eq("over"), tmp["strikeouts"] > tmp["line"], tmp["strikeouts"] < tmp["line"]
            )
            tmp["entry_odds_original"] = np.where(tmp["best_side"].eq("over"), tmp["over_odds"], tmp["under_odds"])
            tmp["profit_original"] = np.where(
                tmp["won_original"], tmp["entry_odds_original"].map(_american_to_decimal) - 1, -1
            )
        for threshold in [0, 5, 10, 12, 15, 18, 20, 25, 30]:
            sub = tmp[tmp[edge_col] >= threshold]
            if len(sub) < 30:
                continue
            rows.append(
                {
                    "version": label,
                    "threshold": threshold,
                    "bets": len(sub),
                    "win_pct": float(sub[won_col].mean()),
                    "roi": float(sub[profit_col].mean()),
                    "avg_edge_pct": float(sub[edge_col].mean()),
                    "over_pct": float(sub[side_col].eq("over").mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/processed_2024/thresh_sel_2025_dk_edges.csv")
    parser.add_argument("--target-calibration", default="data/processed/calibration.json")
    parser.add_argument("--market", default="strikeouts")
    parser.add_argument("--out-dir", default="data/processed/validation")
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--max-kelly", type=float, default=0.05)
    parser.add_argument("--method", choices=["logit", "isotonic"], default="logit")
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_path = Path(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = _prepare_source(source_path, args.market)
    fitted = build_probability_calibration(
        source,
        min_rows=200,
        method=args.method,
        regularization_c=args.regularization_c,
    )
    if args.market not in fitted:
        raise RuntimeError(f"Could not fit probability calibration for {args.market}")

    calibrator = fitted[args.market]
    source["over_probability_calibrated"] = source["over_probability"].map(
        lambda p: apply_probability_calibrator(p, calibrator)
    )
    calibration_report = pd.concat(
        [
            _score_calibration(source, "over_probability"),
            _score_calibration(source, "over_probability_calibrated"),
        ],
        ignore_index=True,
    )
    calibration_report.to_csv(out_dir / "probability_calibration_fit.csv", index=False)

    rescored = _rescore_edges(source, calibrator, args.bankroll, args.max_kelly)
    rescored.to_csv(out_dir / "probability_calibrated_2025_edges.csv", index=False)
    threshold_summary = _threshold_summary(rescored)
    threshold_summary.to_csv(out_dir / "probability_calibrated_threshold_summary.csv", index=False)

    if not args.dry_run:
        target_path = Path(args.target_calibration)
        calibration = load_calibration(target_path)
        calibration.setdefault("markets", {}).setdefault(args.market, {})
        calibration["markets"][args.market]["probability_calibration"] = calibrator
        save_calibration(calibration, target_path)
        write_status = f"Updated {target_path}"
    else:
        write_status = "Dry run; calibration.json not modified"

    print(write_status)
    print("")
    print("Calibration fit:")
    print(
        calibration_report[calibration_report["bin"].eq("ALL")].to_string(
            index=False,
            formatters={
                "mean_probability": "{:.4f}".format,
                "actual_over_rate": "{:.4f}".format,
                "gap": "{:+.4f}".format,
                "brier": "{:.5f}".format,
            },
        )
    )
    print("")
    print("Threshold comparison:")
    print(
        threshold_summary.to_string(
            index=False,
            formatters={
                "win_pct": "{:.1%}".format,
                "roi": "{:+.1%}".format,
                "avg_edge_pct": "{:.1f}".format,
                "over_pct": "{:.1%}".format,
            },
        )
    )


if __name__ == "__main__":
    main()
