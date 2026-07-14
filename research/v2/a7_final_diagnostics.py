"""Generate final, reproducible diagnostics from frozen V2 predictions.

This script does not fit or select a model. It only summarizes the already
frozen 2025 validation and honest 2026 walk-forward predictions, preserving
the holdout protocol.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
S = importlib.import_module("30_strategy")

OUT = ROOT / "research" / "v2"
REPORTS = ROOT / "reports" / "final_model_diagnostics"


def calibration_table(df: pd.DataFrame, probability: str, season: str) -> pd.DataFrame:
    d = df.dropna(subset=[probability, "outcome_over"]).copy()
    d = d[d["outcome_push"] == 0]
    d["bin"] = pd.qcut(d[probability], 10, labels=False, duplicates="drop") + 1
    out = d.groupby("bin", observed=True).agg(
        n=("outcome_over", "size"),
        mean_prediction=(probability, "mean"),
        observed_rate=("outcome_over", "mean"),
    ).reset_index()
    out["calibration_gap"] = out["mean_prediction"] - out["observed_rate"]
    out.insert(0, "season", season)
    return out


def bet_breakdown(bets: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return (bets.groupby(keys, observed=True)
            .agg(n=("pnl", "size"), roi=("pnl", "mean"),
                 units=("pnl", "sum"), win_rate=("won", "mean"),
                 clv=("clv", "mean"))
            .reset_index())


def fmt_metric(m: dict, key: str, pct: bool = False) -> str:
    value = m[key] * (100 if pct else 1)
    return f"{value:+.2f}%" if pct else f"{value:.2f}"


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    p25 = pd.read_parquet(OUT / "preds_2025.parquet")
    p26 = pd.read_parquet(OUT / "adaptive_preds_2026.parquet")
    for d in (p25, p26):
        d["game_date"] = pd.to_datetime(d["game_date"])

    b25 = S.make_bets(p25[p25.outcome_push == 0], "p_mean_count", min_edge=0.08)
    b26 = S.make_bets(p26[p26.outcome_push == 0], "p_h1", min_edge=0.08)
    m25, m26 = S.bet_metrics(b25), S.bet_metrics(b26)

    # Calendar stability.
    monthly = []
    for season, bets in (("2025 validation", b25), ("2026 walk-forward", b26)):
        d = bets.copy()
        d["month"] = d.game_date.dt.to_period("M").astype(str)
        x = bet_breakdown(d, ["month"])
        x.insert(0, "sample", season)
        monthly.append(x)
    monthly = pd.concat(monthly, ignore_index=True)
    monthly.to_csv(REPORTS / "monthly.csv", index=False)

    # A pregame pitcher archetype available in both stored prediction sets.
    archetypes = []
    labels = ["low-K (<8 K/9)", "mid-K (8-10 K/9)", "power (>=10 K/9)"]
    for season, bets in (("2025 validation", b25), ("2026 walk-forward", b26)):
        d = bets.copy()
        d["archetype"] = pd.cut(
            d["p_k9_career"], [-np.inf, 8.0, 10.0, np.inf], labels=labels,
            right=False,
        )
        x = bet_breakdown(d, ["archetype"])
        x.insert(0, "sample", season)
        archetypes.append(x)
    archetypes = pd.concat(archetypes, ignore_index=True)
    archetypes.to_csv(REPORTS / "pitcher_archetypes.csv", index=False)

    # Sportsbook and odds-range stability.
    grouped = []
    for season, bets in (("2025 validation", b25), ("2026 walk-forward", b26)):
        d = bets.copy()
        d["odds_range"] = pd.cut(
            d.odds, [-np.inf, -160, -120, -101, 120, 160, np.inf]
        ).astype(str)
        for dimension in ("book", "odds_range", "bet_side", "line"):
            x = bet_breakdown(d, [dimension]).rename(columns={dimension: "bucket"})
            x.insert(0, "dimension", dimension)
            x.insert(0, "sample", season)
            grouped.append(x)
    pd.concat(grouped, ignore_index=True).to_csv(
        REPORTS / "grouped_stability.csv", index=False
    )

    # Reliability tables and plot use all eligible prop rows, not selected bets.
    calibration = pd.concat([
        calibration_table(p25, "p_mean_count", "2025 validation"),
        calibration_table(p26, "p_h1", "2026 walk-forward"),
    ], ignore_index=True)
    calibration.to_csv(REPORTS / "calibration.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for ax, season in zip(axes, ["2025 validation", "2026 walk-forward"]):
        d = calibration[calibration.season == season]
        ax.plot([0, 1], [0, 1], "--", color="#777777", linewidth=1)
        ax.plot(d.mean_prediction, d.observed_rate, "o-", color="#145da0")
        for row in d.itertuples():
            ax.annotate(str(int(row.n)), (row.mean_prediction, row.observed_rate),
                        xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.set_title(season)
        ax.set_xlabel("Mean predicted P(over)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Observed over rate")
    fig.suptitle("Final MLB pitcher-K model reliability (decile counts labeled)")
    fig.tight_layout()
    fig.savefig(REPORTS / "calibration.png", dpi=180)
    plt.close(fig)

    # Preserve the full candidate screen and fold metrics beside final diagnostics.
    screen = pd.read_csv(OUT / "screen_2025.csv")
    screen.to_csv(REPORTS / "candidate_2025_screen.csv", index=False)
    folds = pd.read_csv(OUT / "fold_metrics.csv")
    folds.to_csv(REPORTS / "candidate_walk_forward_metrics.csv", index=False)

    # The post-final workload experiment must be rejected if 2026 degrades.
    w25 = pd.read_parquet(OUT / "k_workload_preds_2025.parquet")
    w26 = pd.read_parquet(OUT / "k_workload_preds_2026.parquet")
    workload_rows = []
    for sample, d in (("2025 validation", w25), ("2026 walk-forward", w26)):
        for model in ("p0", "p_adj"):
            metrics = S.bet_metrics(S.make_bets(d[d.outcome_push == 0], model, min_edge=0.08))
            workload_rows.append({"sample": sample, "model": model, **metrics})
    workload = pd.DataFrame(workload_rows)
    workload.to_csv(REPORTS / "workload_adjustment.csv", index=False)

    max_gap25 = calibration.loc[
        calibration.season == "2025 validation", "calibration_gap"
    ].abs().max()
    max_gap26 = calibration.loc[
        calibration.season == "2026 walk-forward", "calibration_gap"
    ].abs().max()
    workload_25 = workload[(workload["sample"] == "2025 validation") &
                           (workload["model"] == "p_adj")].iloc[0]
    workload_26 = workload[(workload["sample"] == "2026 walk-forward") &
                           (workload["model"] == "p_adj")].iloc[0]
    report = f"""# Final MLB Pitcher-K Model Audit

Generated from frozen prediction artifacts. No model fitting, threshold tuning, or
2026 candidate selection occurs in this diagnostic pass.

## Verdict

The strongest researched system is the H1 adaptive five-count-model ensemble with
monthly expanding retrains, trailing-90-day Platt recalibration, an 8 percentage-point
minimum model/market edge, and one bet per pitcher-game. **It is not deployable for
real-money betting.** The completed 2026 point estimate is essentially flat, its
confidence interval spans substantial losses, and CLV is modest.

| Sample | Bets | ROI | 90% date-block CI | CLV | CLV positive | Sharpe | Max DD | Profit factor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025 validation | {m25['n']:,} | {fmt_metric(m25, 'roi', True)} | [{m25['roi_lo90']*100:+.1f}%, {m25['roi_hi90']*100:+.1f}%] | {m25['clv_mean']*100:+.2f}pp | {m25['clv_pos_pct']*100:.1f}% | {m25['sharpe_ann']:.2f} | {m25['max_dd']:.1f}u | {m25['profit_factor']:.2f} |
| 2026 walk-forward | {m26['n']:,} | {fmt_metric(m26, 'roi', True)} | [{m26['roi_lo90']*100:+.1f}%, {m26['roi_hi90']*100:+.1f}%] | {m26['clv_mean']*100:+.2f}pp | {m26['clv_pos_pct']*100:.1f}% | {m26['sharpe_ann']:.2f} | {m26['max_dd']:.1f}u | {m26['profit_factor']:.2f} |

The stricter one-shot final gate (June 1-July 10, 2026) is -6.35% ROI on 144
bets with +0.12pp CLV. It fails the pre-registered deployment rule.

## Validation integrity

- Model development: 2022-2024 expanding chronological folds.
- Strategy selection: 2025 only. This means 2025 is out-of-sample for prediction,
  but in-sample for the final edge threshold.
- Honest deployment evidence: 2026 walk-forward through July 10; each prediction
  uses only previously settled games.
- Training ROI is intentionally unavailable because real historical prop odds begin
  in 2025. Inventing 2022-2024 prices would make ROI meaningless.
- Candidate 2026 ROI is intentionally not reported for every candidate: repeatedly
  ranking candidates on the protected period would turn it into another validation set.

## Robustness and stability

- 2025 date-block bootstrap CI: [{m25['roi_lo90']*100:+.1f}%, {m25['roi_hi90']*100:+.1f}%].
- Probability-within-date shuffle test: p < 0.005 (200 trials).
- Correct median-book execution: +11.81% 2025 ROI; DK/FD-only: +7.13%;
  best price excluding BetRivers: +9.90%.
- 2025 edge-to-CLV is monotone; 2026 edge-to-CLV is weaker, which is a principal
  reason real-money stakes are disabled.
- Maximum absolute decile calibration gap: {max_gap25*100:.2f}pp in 2025 and
  {max_gap26*100:.2f}pp in 2026. See `calibration.png` and `calibration.csv`.
- Monthly, sportsbook, odds-range, side, line, and pregame K-archetype breakdowns
  are emitted as CSV files in this directory.

## Model/feature decision

The final probability is the equal-weight mean of isotonic-calibrated Poisson GLM,
XGBoost Poisson, LightGBM Poisson, histogram gradient-boosted Poisson, and CatBoost
Poisson count models, with a Negative-Binomial tail conversion. Classifier families,
random forests, stacking, and single-model variants were rejected for worse or less
stable chronological log loss/calibration. The complete comparison is in
`candidate_walk_forward_metrics.csv` and `candidate_2025_screen.csv`.

Stable leading features include prior career/rolling strikeouts, opponent lineup and
team strikeout rates, recent velocity, workload, home status, and contact rate.
Umpire/weather/park effects are marginal; catcher framing is negligible.

The market-implied workload layer is rejected: it changes ROI from +13.46% to
{workload_25['roi']*100:+.2f}% in 2025 but from +0.26% to
{workload_26['roi']*100:+.2f}% in honest 2026 walk-forward data.

## Deployment

Do not enable real stakes. `research/v2/92_today.py` may be used for paper tracking,
with the CLV, ROI, projection-bias, and edge-to-CLV kill switches retained. A future
candidate needs a newly protected forward period after July 10; further tuning on
this exposed sample cannot create trustworthy evidence. The known Statcast/framing
append-duplication failure is now guarded in `src/data/loaders.py` before rolling
features are calculated.
"""
    (REPORTS / "README.md").write_text(report, encoding="utf-8")
    print(f"Wrote final diagnostics to {REPORTS}")


if __name__ == "__main__":
    main()
