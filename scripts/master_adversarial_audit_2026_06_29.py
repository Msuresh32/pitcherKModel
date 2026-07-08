from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from no_leak_2025_to_2026_model_search import (  # noqa: E402
    EDGE_FILE,
    add_betting_frame,
    filter_strategy,
    summarize,
)
from walk_forward_final_model_compare import (  # noqa: E402
    OUT_DIR as WF_OUT_DIR,
    attach_clv,
    calibration_metrics,
    edge_monotonicity,
    load_clv_index,
    prediction_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "adversarial_master_2026_06_29"
CURRENT_BETS_0618 = ROOT / "reports" / "no_leak_2025_to_2026" / "bets_2026_poisson_top120_a2_e7_eg0_under.csv"
BACKFILL_BETS = ROOT / "reports" / "june_19_30_backfill" / "final_model_bets_2026-06-19_2026-06-30.csv"
WF_SUMMARY = WF_OUT_DIR / "walk_forward_summary.csv"
WF_MONTHLY = WF_OUT_DIR / "walk_forward_monthly.csv"
WF_EDGE = WF_OUT_DIR / "walk_forward_edge_buckets.csv"
WF_BETS = WF_OUT_DIR / "walk_forward_bets.csv"
WF_SCORED = WF_OUT_DIR / "walk_forward_scored_opportunities.csv"


def american_to_decimal_value(odds) -> float:
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    return 1.0 + odds / 100.0 if odds > 0 else 1.0 + 100.0 / abs(odds)


def breakeven(odds) -> float:
    dec = american_to_decimal_value(odds)
    return 1.0 / dec if pd.notna(dec) and dec > 0 else np.nan


def normalize_bets(df: pd.DataFrame, model_label: str, validity: str) -> pd.DataFrame:
    out = df.copy()
    out["game_date"] = pd.to_datetime(out["game_date"]).dt.date.astype(str)
    if "side" not in out and "best_side" in out:
        out["side"] = out["best_side"]
    if "bet_odds" not in out:
        out["bet_odds"] = np.where(out["side"].eq("over"), out["over_odds"], out["under_odds"])
    if "decimal_odds" not in out:
        out["decimal_odds"] = out["bet_odds"].map(american_to_decimal_value)
    if "profit_unit" not in out and "profit_units" in out:
        out["profit_unit"] = out["profit_units"]
    if "profit_units" not in out:
        out["profit_units"] = out["profit_unit"]
    if "won" not in out and "result" in out:
        out["won"] = out["result"].astype(str).str.lower().eq("win")
    if "push" not in out and "result" in out:
        out["push"] = out["result"].astype(str).str.lower().eq("push")
    if "result" not in out:
        out["result"] = np.where(out["push"], "Push", np.where(out["won"], "Win", "Loss"))
    out["model_label"] = model_label
    out["validity"] = validity
    return out


def classify_artifact(path: Path) -> tuple[str, str]:
    p = str(path).replace("\\", "/").lower()
    name = path.name.lower()
    status = "experimental"
    reason = "general repo artifact"
    if "no_leak_2025_to_2026" in p or "walk_forward_comparison" in p or "adversarial_master" in p:
        status, reason = "canonical", "created for no-leak audit/comparison"
    elif "june_19_30_backfill" in p:
        status, reason = "suspicious", "useful extension but full feature cache was incomplete"
    elif "processed_noopp_wf2025_ext/bt_noopp_oos_edges" in p:
        status, reason = "canonical", "primary no-opportunity matrix, but raw league_k column must be excluded"
    elif "processed_poisson" in p or "processed_ensemble" in p or "processed_oos2025" in p:
        status, reason = "deprecated", "saved artifact family affected by prior league_k leakage risk"
    elif name.startswith("v") and name.endswith("_research_report.md"):
        status, reason = "stale", "superseded by leakage audit"
    elif "picks_log" in p or "2026_backtest_extended" in p:
        status, reason = "suspicious", "mixed/live export; not canonical for no-leak proof"
    elif ".env" in p:
        status, reason = "suspicious", "secret/config file; not model evidence"
    elif "raw/pitcher_game_logs" in p:
        status, reason = "suspicious", "currently partial local raw cache; not full historical source"
    elif "odds/snapshots" in p or "pitcher_props.csv" in p:
        status, reason = "experimental", "live snapshot only unless tied to audited bet file"
    elif name.endswith(".joblib"):
        status, reason = "deprecated", "saved model not trusted unless retrained after leakage fix"
    return status, reason


def inventory_artifacts() -> pd.DataFrame:
    patterns = ["*.csv", "*.xlsx", "*.joblib", "*.json", "*.yaml", "*.md", "*.py", "*.ps1"]
    rows = []
    for pattern in patterns:
        for path in ROOT.rglob(pattern):
            if ".git" in path.parts or "__pycache__" in path.parts:
                continue
            try:
                rel = path.relative_to(ROOT)
            except ValueError:
                rel = path
            status, reason = classify_artifact(path)
            rows.append(
                {
                    "path": str(rel).replace("\\", "/"),
                    "kind": path.suffix.lower().lstrip("."),
                    "bytes": path.stat().st_size,
                    "modified": pd.Timestamp(path.stat().st_mtime, unit="s").isoformat(),
                    "classification": status,
                    "reason": reason,
                }
            )
    return pd.DataFrame(rows).sort_values(["classification", "path"])


def classify_feature(feature: str) -> tuple[str, str]:
    f = feature.lower()
    if f == "league_k":
        return "LEAKAGE", "same-day league strikeout average"
    if any(x in f for x in ["strikeouts_projection", "over_probability", "under_probability", "edge", "ev", "odds"]):
        return "LEAKAGE", "betting/model output feature"
    if f.startswith("p_") and ("roll" in f or "career" in f):
        return "SAFE", "pitcher prior rolling/career feature if shifted before game"
    if f.startswith("opp_") and ("prior" in f or "roll" in f):
        return "SAFE", "opponent prior rolling feature if shifted before game"
    if f.startswith("league_k_"):
        return "SAFE", "shifted league environment feature after leakage fix"
    if f.startswith("adv_") or f.startswith("sc_") or f.startswith("lineup_bat_"):
        return "SUSPICIOUS", "requires source timestamp audit for point-in-time availability"
    if f.startswith("opp_lineup_"):
        return "SUSPICIOUS", "lineup feature may depend on confirmation timing"
    if f.startswith("matchup_"):
        return "UNKNOWN", "derived matchup feature; needs construction audit"
    if f in {"venue_id", "temperature", "wind_speed_mph", "home_plate_umpire_id"}:
        return "SUSPICIOUS", "pregame availability/timestamp must be verified"
    if f.startswith("pvt_"):
        return "SAFE", "prior pitcher-vs-team aggregate if shifted"
    return "UNKNOWN", "not proven point-in-time safe by name alone"


def feature_audit() -> pd.DataFrame:
    # Reuse the last WF scored opportunity columns to infer selected features from the
    # dedicated comparison script summary is not enough; recompute top-120 via existing script output.
    import importlib

    mod = importlib.import_module("no_leak_2025_to_2026_model_search")
    raw = pd.read_csv(EDGE_FILE)
    raw["game_date"] = pd.to_datetime(raw["game_date"])
    raw = raw[raw["market"].eq("strikeouts")]
    numeric_cols = set(raw.select_dtypes(include=[np.number]).columns)
    all_features = [col for col in raw.columns if mod.is_feature_col(col, numeric_cols)]
    train_games = raw[raw["game_date"].dt.year.eq(2025)].sort_values("game_date").drop_duplicates(["game_date", "pitcher_id"])
    train_inner = train_games[train_games["game_date"] < "2025-07-01"].copy()
    top120 = mod.top_corr_features(train_inner, all_features, 120)
    rows = []
    for rank, feat in enumerate(top120, start=1):
        status, reason = classify_feature(feat)
        rows.append({"rank": rank, "feature": feat, "classification": status, "reason": reason})
    return pd.DataFrame(rows)


def max_drawdown(units: pd.Series) -> float:
    curve = units.cumsum()
    peak = curve.cummax()
    return float((curve - peak).min()) if len(curve) else 0.0


def longest_losing_streak(results: pd.Series) -> int:
    best = cur = 0
    for r in results:
        if str(r).lower() == "loss":
            cur += 1
            best = max(best, cur)
        elif str(r).lower() != "push":
            cur = 0
    return best


def bootstrap_ci(values: pd.Series, n_iter: int = 5000, seed: int = 11) -> tuple[float, float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_iter, len(x)), replace=True).mean(axis=1)
    return tuple(float(v) for v in np.quantile(means, [0.025, 0.5, 0.975]))


def stats_row(df: pd.DataFrame, label: str, validity: str) -> dict:
    d = df.copy()
    n = len(d)
    wins = int(d["won"].astype(bool).sum()) if n else 0
    pushes = int(d["push"].astype(bool).sum()) if n else 0
    losses = n - wins - pushes
    profit = pd.to_numeric(d["profit_unit"], errors="coerce")
    odds = pd.to_numeric(d["bet_odds"], errors="coerce")
    be = odds.map(breakeven)
    decided = wins + losses
    avg_be = float(be.mean()) if len(be) else np.nan
    ci = bootstrap_ci(profit)
    se = float(profit.std(ddof=1) / math.sqrt(n)) if n > 1 else np.nan
    t = float(profit.mean() / se) if se and np.isfinite(se) and se > 0 else np.nan
    binom_p = float(binomtest(wins, decided, avg_be, alternative="greater").pvalue) if decided and pd.notna(avg_be) and 0 < avg_be < 1 else np.nan
    rng = np.random.default_rng(7)
    pr_le_zero = float((rng.choice(profit.dropna().to_numpy(float), size=(5000, n), replace=True).mean(axis=1) <= 0).mean()) if n else np.nan
    clv = pd.to_numeric(d.get("clv_pct", pd.Series(dtype=float)), errors="coerce")
    return {
        "model": label,
        "validity": validity,
        "start": d["game_date"].min() if n and "game_date" in d else "",
        "end": d["game_date"].max() if n and "game_date" in d else "",
        "bets": n,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": wins / decided if decided else np.nan,
        "avg_odds": float(odds.mean()) if len(odds) else np.nan,
        "breakeven_win_rate": avg_be,
        "roi": float(profit.mean()) if n else np.nan,
        "units": float(profit.sum()) if n else 0.0,
        "binomial_p_vs_breakeven": binom_p,
        "bootstrap_roi_ci_lo": ci[0],
        "bootstrap_roi_ci_mid": ci[1],
        "bootstrap_roi_ci_hi": ci[2],
        "t_stat_units_per_bet": t,
        "max_drawdown_units": max_drawdown(profit),
        "longest_losing_streak": longest_losing_streak(d["result"]) if n else 0,
        "prob_true_roi_lte_0": pr_le_zero,
        "clv_bets": int(clv.notna().sum()),
        "avg_clv_pct": float(clv.mean()) if clv.notna().any() else np.nan,
        "median_clv_pct": float(clv.median()) if clv.notna().any() else np.nan,
        "positive_clv_rate": float((clv.dropna() > 0).mean()) if clv.notna().any() else np.nan,
    }


def load_current_frozen_through_0629() -> pd.DataFrame:
    clv_index = load_clv_index()
    bets_0618 = pd.read_csv(CURRENT_BETS_0618)
    bets_0618 = normalize_bets(attach_clv(bets_0618, clv_index), "frozen_2025_current", "canonical_through_2026_06_18")
    if BACKFILL_BETS.exists():
        back = pd.read_csv(BACKFILL_BETS)
        back = back[pd.to_datetime(back["game_date"]) <= pd.Timestamp("2026-06-29")].copy()
        back = normalize_bets(back, "frozen_2025_current", "feature_cache_limited_backfill_2026_06_19_29")
        return pd.concat([bets_0618, back], ignore_index=True, sort=False)
    return bets_0618


def make_valid_model_comparison(current: pd.DataFrame) -> pd.DataFrame:
    rows = [stats_row(current, "frozen_2025_current_through_2026_06_29", "mixed_canonical_plus_limited_backfill")]
    if WF_SUMMARY.exists():
        wf = pd.read_csv(WF_SUMMARY)
        for _, row in wf.iterrows():
            rows.append(
                {
                    "model": row["model"],
                    "validity": "canonical_complete_matrix_through_2026_06_18",
                    "start": "2026-03-26",
                    "end": "2026-06-18",
                    "bets": row["bets"],
                    "wins": row["wins"],
                    "losses": row["losses"],
                    "pushes": row["pushes"],
                    "win_rate": row["win_rate"],
                    "avg_odds": row["avg_odds"],
                    "breakeven_win_rate": np.nan,
                    "roi": row["roi"],
                    "units": row["profit_units"],
                    "binomial_p_vs_breakeven": np.nan,
                    "bootstrap_roi_ci_lo": row["roi_ci_lo"],
                    "bootstrap_roi_ci_mid": row["roi_ci_mid"],
                    "bootstrap_roi_ci_hi": row["roi_ci_hi"],
                    "t_stat_units_per_bet": np.nan,
                    "max_drawdown_units": np.nan,
                    "longest_losing_streak": np.nan,
                    "prob_true_roi_lte_0": np.nan,
                    "clv_bets": row["clv_bets"],
                    "avg_clv_pct": row["avg_clv_pct"],
                    "median_clv_pct": row["median_clv_pct"],
                    "positive_clv_rate": row["positive_clv_rate"],
                    "mae": row["mae"],
                    "rmse": row["rmse"],
                    "brier": row["brier"],
                    "log_loss": row["log_loss"],
                    "ece": row["ece"],
                    "edge_spearman": row["edge_spearman"],
                }
            )
    return pd.DataFrame(rows)


def clv_audit(current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = current.copy()
    d["month"] = pd.to_datetime(d["game_date"]).dt.to_period("M").astype(str)
    d["edge_bucket"] = pd.cut(pd.to_numeric(d["edge_pct"], errors="coerce"), [7, 10, 15, 20, 30, 50, np.inf], labels=["7-10", "10-15", "15-20", "20-30", "30-50", "50+"], right=False)
    d["line_bucket"] = d["line"].astype(str)
    d["book"] = np.where(d["side"].eq("over"), d.get("over_bookmaker", ""), d.get("under_bookmaker", ""))
    groupings = {"overall": [], "month": ["month"], "line": ["line_bucket"], "edge_bucket": ["edge_bucket"], "book": ["book"]}
    for group_name, cols in groupings.items():
        iterator = [(("overall",), d)] if not cols else d.groupby(cols, dropna=False, observed=True)
        for key, g in iterator:
            if not isinstance(key, tuple):
                key = (key,)
            clv = pd.to_numeric(g["clv_pct"], errors="coerce")
            valid = g[clv.notna()].copy()
            clv_valid = clv.dropna()
            se = clv_valid.std(ddof=1) / math.sqrt(len(clv_valid)) if len(clv_valid) > 1 else np.nan
            rows.append(
                {
                    "group_type": group_name,
                    "group": "|".join(str(x) for x in key),
                    "bets": len(g),
                    "clv_matched": int(clv_valid.notna().sum()),
                    "avg_clv_pct": float(clv_valid.mean()) if len(clv_valid) else np.nan,
                    "median_clv_pct": float(clv_valid.median()) if len(clv_valid) else np.nan,
                    "positive_clv_rate": float((clv_valid > 0).mean()) if len(clv_valid) else np.nan,
                    "clv_t_stat": float(clv_valid.mean() / se) if se and np.isfinite(se) and se > 0 else np.nan,
                    "roi_positive_clv": float(valid[valid["clv_pct"] > 0]["profit_unit"].mean()) if len(valid[valid["clv_pct"] > 0]) else np.nan,
                    "roi_nonpositive_clv": float(valid[valid["clv_pct"] <= 0]["profit_unit"].mean()) if len(valid[valid["clv_pct"] <= 0]) else np.nan,
                    "roi_missing_clv": float(g[clv.isna()]["profit_unit"].mean()) if len(g[clv.isna()]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def adjust_american(odds, cents: int) -> float:
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds >= 100:
        return odds + cents
    return odds + cents  # negative odds: +5 is better, -5 is worse


def execution_sensitivity(current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cents in [-15, -10, -5, 0, 5, 10, 15]:
        # Define positive cents as bettor improvement. For + odds, add; for - odds, add toward zero.
        adj = current.copy()
        adj["adjusted_odds"] = adj["bet_odds"].map(lambda x: adjust_american(x, cents))
        adj["adjusted_decimal"] = adj["adjusted_odds"].map(american_to_decimal_value)
        adj["adjusted_profit_unit"] = np.where(adj["push"], 0.0, np.where(adj["won"], adj["adjusted_decimal"] - 1.0, -1.0))
        rows.append(
            {
                "execution_adjustment_cents": cents,
                "description": "positive = bettor improvement; negative = worse fill",
                "bets": len(adj),
                "roi": float(adj["adjusted_profit_unit"].mean()),
                "units": float(adj["adjusted_profit_unit"].sum()),
            }
        )
    return pd.DataFrame(rows)


def robustness_grid(scored_path: Path, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if scored_path.exists():
        scored = pd.read_csv(scored_path)
        scored = scored[scored["eval_model"].eq("frozen_2025")].copy()
        for edge in [5, 7, 10, 12, 15]:
            for eg in [0, 5, 10, 12, 15]:
                q = filter_strategy(scored, edge, eg, "under")
                row = stats_row(normalize_bets(q, f"edge{edge}_eg{eg}", "canonical_grid_through_2026_06_18"), f"edge>={edge}_eg>={eg}", "canonical_grid_through_2026_06_18")
                row["edge_min"] = edge
                row["edge_gap_min"] = eg
                rows.append(row)
    # Perturb current selected bet list for robustness tests through 06/29.
    base = current.copy().sort_values("game_date").reset_index(drop=True)
    tests = {
        "base_selected_through_0629": base,
        "remove_top5_wins": base.drop(base[base["profit_unit"] > 0].nlargest(5, "profit_unit").index),
        "remove_top10_wins": base.drop(base[base["profit_unit"] > 0].nlargest(10, "profit_unit").index),
        "remove_worst5": base.drop(base.nsmallest(5, "profit_unit").index),
        "first_half": base.iloc[: len(base) // 2],
        "second_half": base.iloc[len(base) // 2 :],
        "favorites": base[pd.to_numeric(base["bet_odds"], errors="coerce") < 0],
        "underdogs": base[pd.to_numeric(base["bet_odds"], errors="coerce") > 0],
        "low_line_<=5.5": base[pd.to_numeric(base["line"], errors="coerce") <= 5.5],
        "high_line_>=6.5": base[pd.to_numeric(base["line"], errors="coerce") >= 6.5],
        "juice_cap_not_worse_than_-140": base[pd.to_numeric(base["bet_odds"], errors="coerce") >= -140],
    }
    for name, df in tests.items():
        row = stats_row(df, name, "selected_bet_robustness_through_2026_06_29")
        row["edge_min"] = np.nan
        row["edge_gap_min"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def grouped_performance(current: pd.DataFrame) -> pd.DataFrame:
    d = current.copy()
    d["month"] = pd.to_datetime(d["game_date"]).dt.to_period("M").astype(str)
    d["odds_bucket"] = pd.cut(pd.to_numeric(d["bet_odds"], errors="coerce"), [-1000, -150, -120, -100, 120, 150, 1000], labels=["<=-150", "-149/-120", "-119/-101", "+100/+120", "+121/+150", ">=+151"])
    d["edge_bucket"] = pd.cut(pd.to_numeric(d["edge_pct"], errors="coerce"), [7, 10, 15, 20, 30, 50, np.inf], labels=["7-10", "10-15", "15-20", "20-30", "30-50", "50+"])
    d["abs_gap_bucket"] = pd.cut(pd.to_numeric(d["abs_gap"], errors="coerce"), [0, .25, .5, .75, 1, 1.5, 99], labels=["0-.25", ".25-.5", ".5-.75", ".75-1", "1-1.5", "1.5+"])
    rows = []
    for group_name, col in [("month", "month"), ("line", "line"), ("odds_bucket", "odds_bucket"), ("edge_bucket", "edge_bucket"), ("abs_gap_bucket", "abs_gap_bucket"), ("team", "team"), ("opponent", "opponent")]:
        if col not in d:
            continue
        for key, g in d.groupby(col, dropna=False, observed=True):
            row = stats_row(g, str(key), "grouped_current_model")
            row["group_type"] = group_name
            row["group"] = str(key)
            rows.append(row)
    return pd.DataFrame(rows)


def write_recommendation(comparison: pd.DataFrame, current: pd.DataFrame, clv: pd.DataFrame) -> None:
    overall = stats_row(current, "current", "mixed")
    clv_overall = clv[clv["group_type"].eq("overall")].iloc[0].to_dict() if not clv.empty else {}
    grade = "C. Paper trade only"
    if overall["roi"] > 0.05 and clv_overall.get("avg_clv_pct", -99) > 1 and clv_overall.get("positive_clv_rate", 0) > 0.55 and overall["prob_true_roi_lte_0"] < 0.10:
        grade = "B. Small-stakes live test only"
    if overall["avg_clv_pct"] if "avg_clv_pct" in overall else np.nan:
        pass
    lines = [
        "# Final Recommendation Through 2026-06-29",
        "",
        "## Executive Verdict",
        "",
        f"Grade: **{grade}**.",
        "",
        "There is evidence of a possible strikeout-under edge in the clean 2026 sample through 2026-06-18, but the extension through 2026-06-29 is weaker and the full feature cache needed for a canonical post-06/18 point-in-time rebuild is not preserved in the repo. I would not classify this as deployable real-money infrastructure yet.",
        "",
        "## Current Model Through 2026-06-29",
        "",
        f"- Bets: {overall['bets']}",
        f"- Win rate: {overall['win_rate']:.2%}",
        f"- ROI: {overall['roi']:+.2%}",
        f"- Units: {overall['units']:+.2f}",
        f"- Bootstrap ROI CI: [{overall['bootstrap_roi_ci_lo']:+.2%}, {overall['bootstrap_roi_ci_hi']:+.2%}]",
        f"- Probability true ROI <= 0 by bootstrap: {overall['prob_true_roi_lte_0']:.1%}",
        f"- Max drawdown: {overall['max_drawdown_units']:+.2f} units",
        f"- Longest losing streak: {overall['longest_losing_streak']}",
        "",
        "## CLV",
        "",
        f"- CLV matched: {clv_overall.get('clv_matched', 0)}/{clv_overall.get('bets', 0)}",
        f"- Average CLV: {clv_overall.get('avg_clv_pct', np.nan):+.2f}%",
        f"- Median CLV: {clv_overall.get('median_clv_pct', np.nan):+.2f}%",
        f"- Positive CLV rate: {clv_overall.get('positive_clv_rate', np.nan):.2%}",
        "",
        "## Direct Answers",
        "",
        "- Is there an edge? Possible, but not proven strongly enough for real money.",
        "- Which model should I use? If you must monitor one, use the frozen 2025 `poisson_top120_a2` unders-only rule as a paper/small-token shadow strategy. Do not promote WF yet.",
        "- How confident am I? Low-to-moderate that there is a small edge; high confidence that the repo is not yet production-audit clean through 06/29.",
        "- What can go wrong? Stale feature cache, CLV coverage gaps, best-line optimism, model decay in late June, missing lineup/Statcast timestamps, and variance from under-heavy exposure.",
        "- Should you bet now? No meaningful stakes. Paper trade or token-size only until live CLV and settlement are clean for at least 100 new bets.",
        "",
        "## Production Architecture",
        "",
        "The right architecture is weekly point-in-time retraining with daily feature/odds refreshes, model snapshots, raw input snapshots, and close-line capture. The current evidence does not justify replacing the frozen model with walk-forward because WF underperformed ROI in the locked comparison.",
        "",
        "## Tomorrow Morning Checklist",
        "",
        "1. Fetch probables, lineups, current odds, and save immutable odds snapshot.",
        "2. Rebuild features using only data through yesterday.",
        "3. Score the frozen candidate and any shadow WF/blend candidates.",
        "4. Log every candidate, including skipped bets and reason.",
        "5. Capture closing odds before first pitch for CLV.",
        "6. Do not bet if projected volume spikes, CLV turns negative over the last 30-50 bets, or June-style drawdown continues.",
        "",
        "## Deliverables",
        "",
        "- `model_audit_master_summary.csv`",
        "- `valid_model_comparison.csv`",
        "- `best_candidate_bets_through_2026_06_29.csv`",
        "- `clv_audit.csv`",
        "- `robustness_grid.csv`",
        "- `execution_sensitivity.csv`",
        "- `feature_leakage_audit.csv`",
        "- `artifact_inventory.csv`",
    ]
    (OUT / "final_recommendation.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = inventory_artifacts()
    features = feature_audit()
    current = load_current_frozen_through_0629()
    comparison = make_valid_model_comparison(current)
    clv = clv_audit(current)
    sensitivity = execution_sensitivity(current)
    robust = robustness_grid(WF_SCORED, current)
    grouped = grouped_performance(current)

    inventory.to_csv(OUT / "artifact_inventory.csv", index=False)
    features.to_csv(OUT / "feature_leakage_audit.csv", index=False)
    comparison.to_csv(OUT / "valid_model_comparison.csv", index=False)
    current.to_csv(OUT / "best_candidate_bets_through_2026_06_29.csv", index=False)
    clv.to_csv(OUT / "clv_audit.csv", index=False)
    sensitivity.to_csv(OUT / "execution_sensitivity.csv", index=False)
    robust.to_csv(OUT / "robustness_grid.csv", index=False)
    grouped.to_csv(OUT / "grouped_performance.csv", index=False)

    master = pd.concat(
        [
            comparison.assign(section="valid_model_comparison"),
            robust.assign(section="robustness"),
        ],
        ignore_index=True,
        sort=False,
    )
    master.to_csv(OUT / "model_audit_master_summary.csv", index=False)
    write_recommendation(comparison, current, clv)

    print(f"Wrote master audit deliverables to {OUT}")
    print(comparison.round(4).to_string(index=False))
    print(clv[clv["group_type"].eq("overall")].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
