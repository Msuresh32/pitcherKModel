from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from no_leak_2025_to_2026_model_search import (  # noqa: E402
    EDGE_FILE,
    add_betting_frame,
    filter_strategy,
    fit_model,
    is_feature_col,
    predict_model,
    summarize,
    top_corr_features,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "walk_forward_comparison"

MODEL_NAME = "poisson_top120_a2"
EDGE_MIN = 7.0
EG_MIN = 0.0
SIDE_FILTER = "under"
TEST_START = pd.Timestamp("2026-03-26")
TEST_END = pd.Timestamp("2026-06-18")
TOP_K = 120


@dataclass(frozen=True)
class EvalConfig:
    label: str
    cadence: str


def normalize_pid(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype(str).replace("<NA>", pd.NA)


def estimator() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", PoissonRegressor(alpha=2.0, max_iter=2000)),
        ]
    )


def load_matrix() -> tuple[pd.DataFrame, list[str]]:
    raw = pd.read_csv(EDGE_FILE)
    raw["game_date"] = pd.to_datetime(raw["game_date"])
    raw = raw[raw["market"].eq("strikeouts")].copy()
    raw["pitcher_id"] = normalize_pid(raw["pitcher_id"])
    numeric_cols = set(raw.select_dtypes(include=[np.number]).columns)
    features = [col for col in raw.columns if is_feature_col(col, numeric_cols)]
    return raw, features


def game_rows(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["game_date", "pitcher_id"])
        .drop_duplicates(["game_date", "pitcher_id"], keep="first")
        .reset_index(drop=True)
    )


def fit_poisson(train_games: pd.DataFrame, all_features: list[str], feature_source: pd.DataFrame | None = None):
    source = feature_source if feature_source is not None else train_games
    selected = top_corr_features(source, all_features, TOP_K)
    return fit_model(MODEL_NAME, train_games, selected, estimator())


def frozen_2025_predictions(raw: pd.DataFrame, all_features: list[str], test: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    train_games = game_rows(raw[raw["game_date"].dt.year.eq(2025)])
    feature_source = train_games[train_games["game_date"] < "2025-07-01"].copy()
    model = fit_poisson(train_games, all_features, feature_source=feature_source)
    pred = predict_model(model, test)
    scored = add_betting_frame(test, pred, model.residual_std, "frozen_2025")
    meta = {
        "refits": 1,
        "avg_train_games": len(train_games),
        "features_last": len(model.features),
        "train_start": str(train_games["game_date"].min().date()),
        "train_end": str(train_games["game_date"].max().date()),
    }
    return scored, meta


def retrain_dates(dates: list[pd.Timestamp], cadence: str) -> dict[pd.Timestamp, pd.Timestamp]:
    if cadence == "daily":
        return {d: d for d in dates}
    if cadence == "monthly":
        return {d: pd.Timestamp(year=d.year, month=d.month, day=1) for d in dates}
    if cadence != "weekly":
        raise ValueError(f"Unknown cadence: {cadence}")
    out = {}
    current_anchor = None
    for d in dates:
        week_anchor = d - pd.Timedelta(days=d.weekday())
        if current_anchor is None or week_anchor != current_anchor:
            current_anchor = week_anchor
        out[d] = current_anchor
    return out


def walk_forward_predictions(raw: pd.DataFrame, all_features: list[str], test: pd.DataFrame, cadence: str, rolling_days: int | None = None) -> tuple[pd.DataFrame, dict]:
    test_dates = [pd.Timestamp(d) for d in sorted(test["game_date"].dropna().unique())]
    anchors = retrain_dates(test_dates, cadence)
    scored_parts = []
    fitted_by_anchor = {}
    train_sizes = []

    for d in test_dates:
        anchor = anchors[d]
        # Weekly retraining means the model is fit before the first slate of the
        # week using data available before that week. Daily uses data before d.
        cutoff = min(anchor, d)
        if anchor not in fitted_by_anchor:
            train = raw[raw["game_date"] < cutoff].copy()
            train = train[train["game_date"].dt.year.ge(2025)].copy()
            if rolling_days is not None:
                train = train[train["game_date"] >= cutoff - pd.Timedelta(days=rolling_days)].copy()
            train_games = game_rows(train)
            model = fit_poisson(train_games, all_features)
            fitted_by_anchor[anchor] = model
            train_sizes.append(len(train_games))
        model = fitted_by_anchor[anchor]
        day = test[test["game_date"].eq(d)].copy()
        pred = predict_model(model, day)
        scored = add_betting_frame(day, pred, model.residual_std, f"walk_forward_{cadence}")
        scored["wf_train_cutoff"] = cutoff
        scored["wf_refit_anchor"] = anchor
        scored_parts.append(scored)

    meta = {
        "refits": len(fitted_by_anchor),
        "avg_train_games": float(np.mean(train_sizes)) if train_sizes else 0.0,
        "min_train_games": int(np.min(train_sizes)) if train_sizes else 0,
        "max_train_games": int(np.max(train_sizes)) if train_sizes else 0,
        "features_last": len(next(reversed(fitted_by_anchor.values())).features) if fitted_by_anchor else 0,
        "rolling_days": rolling_days if rolling_days is not None else np.nan,
    }
    return pd.concat(scored_parts, ignore_index=True, sort=False), meta


def blended_predictions(left: pd.DataFrame, right: pd.DataFrame, left_weight: float, label: str) -> tuple[pd.DataFrame, dict]:
    key_cols = ["game_date", "pitcher_id", "market", "line", "over_odds", "under_odds"]
    l = left.copy()
    r = right.copy()
    l["_blend_id"] = np.arange(len(l))
    l_small = l[key_cols + ["_blend_id", "projection", "residual_std_train"]].rename(columns={"projection": "projection_left", "residual_std_train": "std_left"})
    r_small = r[key_cols + ["projection", "residual_std_train"]].rename(columns={"projection": "projection_right", "residual_std_train": "std_right"})
    merged = l_small.merge(r_small, on=key_cols, how="inner")
    merged = merged.drop_duplicates("_blend_id", keep="first")
    base = l.reset_index(drop=True).iloc[merged["_blend_id"].to_numpy(int)].copy().reset_index(drop=True)
    projection = left_weight * merged["projection_left"].to_numpy(float) + (1.0 - left_weight) * merged["projection_right"].to_numpy(float)
    residual_std = float(np.nanmean([merged["std_left"].mean(), merged["std_right"].mean()]))
    scored = add_betting_frame(base.drop(columns=["_row_id"], errors="ignore"), projection, residual_std, label)
    return scored, {"refits": 0, "avg_train_games": np.nan, "features_last": np.nan, "blend_left_weight": left_weight}


def calibration_metrics(scored: pd.DataFrame, bins: int = 10) -> dict:
    d = scored.copy()
    y = (pd.to_numeric(d["strikeouts"], errors="coerce") > pd.to_numeric(d["line"], errors="coerce")).astype(float)
    p = pd.to_numeric(d["over_probability"], errors="coerce").clip(1e-6, 1 - 1e-6)
    mask = y.notna() & p.notna()
    y = y[mask]
    p = p[mask]
    if len(p) == 0:
        return {"brier": np.nan, "log_loss": np.nan, "ece": np.nan, "mce": np.nan}
    brier = float(((p - y) ** 2).mean())
    log_loss = float((-(y * np.log(p) + (1 - y) * np.log(1 - p))).mean())
    cuts = pd.cut(p, bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    ece = 0.0
    mce = 0.0
    for _, idx in p.groupby(cuts, observed=False).groups.items():
        if len(idx) == 0:
            continue
        conf = float(p.loc[idx].mean())
        acc = float(y.loc[idx].mean())
        gap = abs(acc - conf)
        ece += len(idx) / len(p) * gap
        mce = max(mce, gap)
    return {"brier": brier, "log_loss": log_loss, "ece": float(ece), "mce": float(mce)}


def prediction_metrics(scored: pd.DataFrame) -> dict:
    games = (
        scored.sort_values("game_date")
        .drop_duplicates(["game_date", "pitcher_id"], keep="first")
        .copy()
    )
    y = pd.to_numeric(games["strikeouts"], errors="coerce")
    pred = pd.to_numeric(games["projection"], errors="coerce")
    mask = y.notna() & pred.notna()
    return {
        "prediction_rows": int(mask.sum()),
        "mae": float(mean_absolute_error(y[mask], pred[mask])) if mask.any() else np.nan,
        "rmse": float(math.sqrt(mean_squared_error(y[mask], pred[mask]))) if mask.any() else np.nan,
    }


def load_clv_index() -> pd.DataFrame:
    paths = [
        ROOT / "data" / "processed_poisson" / "bt_poisson_2026_full_clv.csv",
        ROOT / "data" / "processed_poisson" / "bt_pois_2026_e20_clv.csv",
        ROOT / "data" / "processed_poisson_wf2025" / "bt_poisson_2025_clv.csv",
    ]
    frames = []
    for path in paths:
        if not path.exists():
            continue
        cols = pd.read_csv(path, nrows=0).columns
        usecols = [c for c in ["game_date", "pitcher_id", "pitcher_name", "market", "line", "best_side", "over_odds", "under_odds", "clv_pct"] if c in cols]
        frame = pd.read_csv(path, usecols=usecols)
        if "clv_pct" not in frame:
            continue
        frame["game_date"] = pd.to_datetime(frame["game_date"])
        frame["pitcher_id"] = normalize_pid(frame["pitcher_id"])
        frame["side"] = frame["best_side"].astype(str).str.lower()
        frame["line"] = pd.to_numeric(frame["line"], errors="coerce")
        frame["clv_pct"] = pd.to_numeric(frame["clv_pct"], errors="coerce")
        frames.append(frame[["game_date", "pitcher_id", "market", "line", "side", "clv_pct"]])
    if not frames:
        return pd.DataFrame(columns=["game_date", "pitcher_id", "market", "line", "side", "clv_pct"])
    out = pd.concat(frames, ignore_index=True, sort=False).dropna(subset=["clv_pct"])
    out = out.sort_values("clv_pct").drop_duplicates(["game_date", "pitcher_id", "market", "line", "side"], keep="last")
    return out


def attach_clv(bets: pd.DataFrame, clv_index: pd.DataFrame) -> pd.DataFrame:
    out = bets.copy()
    if out.empty or clv_index.empty:
        out["clv_pct"] = np.nan
        return out
    out["game_date"] = pd.to_datetime(out["game_date"])
    out["pitcher_id"] = normalize_pid(out["pitcher_id"])
    out["line"] = pd.to_numeric(out["line"], errors="coerce")
    return out.merge(clv_index, on=["game_date", "pitcher_id", "market", "line", "side"], how="left")


def edge_monotonicity(bets: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if bets.empty:
        return pd.DataFrame(), {"edge_spearman": np.nan, "monotonic_steps": 0}
    bins = [7, 10, 15, 20, 30, 50, np.inf]
    labels = ["7-10", "10-15", "15-20", "20-30", "30-50", "50+"]
    d = bets.copy()
    d["edge_bucket"] = pd.cut(d["edge_pct"], bins=bins, labels=labels, right=False)
    rows = []
    for bucket, group in d.groupby("edge_bucket", observed=True):
        rows.append(
            {
                "edge_bucket": str(bucket),
                "avg_edge_pct": float(group["edge_pct"].mean()),
                "bets": len(group),
                "roi": float(group["profit_unit"].mean()),
                "units": float(group["profit_unit"].sum()),
                "win_rate": float(group.loc[~group["push"], "won"].mean()) if (~group["push"]).any() else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    if len(table) >= 3:
        rho = spearmanr(table["avg_edge_pct"], table["roi"], nan_policy="omit").statistic
        steps = int((table["roi"].diff().dropna() >= 0).sum())
    else:
        rho = np.nan
        steps = 0
    return table, {"edge_spearman": float(rho) if pd.notna(rho) else np.nan, "monotonic_steps": steps}


def monthly_table(bets: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame()
    d = bets.copy()
    d["month"] = pd.to_datetime(d["game_date"]).dt.to_period("M").astype(str)
    rows = []
    for month, group in d.groupby("month", sort=True):
        rows.append(
            {
                "month": month,
                "bets": len(group),
                "wins": int(group["won"].sum()),
                "losses": int((~group["won"] & ~group["push"]).sum()),
                "pushes": int(group["push"].sum()),
                "win_rate": float(group.loc[~group["push"], "won"].mean()) if (~group["push"]).any() else np.nan,
                "roi": float(group["profit_unit"].mean()),
                "units": float(group["profit_unit"].sum()),
                "avg_clv_pct": float(group["clv_pct"].mean()) if group["clv_pct"].notna().any() else np.nan,
                "clv_coverage": float(group["clv_pct"].notna().mean()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_mean_ci(values: np.ndarray, n_iter: int = 5000, seed: int = 42) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_iter, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.5, 0.975]))


def daily_diff_test(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    da = a.copy()
    db = b.copy()
    da["date"] = pd.to_datetime(da["game_date"]).dt.date
    db["date"] = pd.to_datetime(db["game_date"]).dt.date
    pa = da.groupby("date")["profit_unit"].sum()
    pb = db.groupby("date")["profit_unit"].sum()
    idx = sorted(set(pa.index) | set(pb.index))
    diff = np.array([pb.get(i, 0.0) - pa.get(i, 0.0) for i in idx], dtype=float)
    ci = bootstrap_mean_ci(diff)
    mean = float(np.mean(diff)) if len(diff) else np.nan
    se = float(np.std(diff, ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else np.nan
    z = mean / se if se and np.isfinite(se) and se > 0 else np.nan
    p_gt0 = float((np.random.default_rng(123).choice(diff, size=(5000, len(diff)), replace=True).mean(axis=1) > 0).mean()) if len(diff) else np.nan
    return {
        "daily_profit_diff_mean_b_minus_a": mean,
        "daily_profit_diff_ci_lo": ci[0],
        "daily_profit_diff_ci_mid": ci[1],
        "daily_profit_diff_ci_hi": ci[2],
        "daily_profit_diff_t": z,
        "bootstrap_pr_b_gt_a": p_gt0,
    }


def summarize_model(label: str, scored: pd.DataFrame, bets: pd.DataFrame, meta: dict) -> dict:
    s = summarize(bets)
    cal = calibration_metrics(scored)
    pred = prediction_metrics(scored)
    _, mono = edge_monotonicity(bets)
    clv = bets["clv_pct"].dropna() if "clv_pct" in bets else pd.Series(dtype=float)
    row = {"model": label, **meta, **s, **pred, **cal, **mono}
    row.update(
        {
            "clv_bets": int(clv.notna().sum()),
            "clv_coverage": float(bets["clv_pct"].notna().mean()) if len(bets) else np.nan,
            "avg_clv_pct": float(clv.mean()) if len(clv) else np.nan,
            "median_clv_pct": float(clv.median()) if len(clv) else np.nan,
            "positive_clv_rate": float((clv > 0).mean()) if len(clv) else np.nan,
        }
    )
    return row


def write_review(summary: pd.DataFrame, diff: dict) -> None:
    daily = summary[summary["model"].eq("walk_forward_daily")].iloc[0]
    frozen = summary[summary["model"].eq("frozen_2025")].iloc[0]
    weekly = summary[summary["model"].eq("walk_forward_weekly")].iloc[0]
    recommend = (
        "replace frozen 2025 with walk-forward"
        if daily["roi"] > frozen["roi"] and daily["mae"] <= frozen["mae"] and daily["avg_clv_pct"] >= frozen["avg_clv_pct"]
        else "do not replace solely on this test"
    )
    lines = [
        "# Walk-Forward Training Review",
        "",
        "## Research Review",
        "",
        "1. Training only on 2025 is likely stale for production. It is excellent for a clean 2026 out-of-sample proof, but it intentionally ignores new pitcher form, role changes, arsenal changes, injuries, lineup context, and league environment updates.",
        "2. An expanding walk-forward model should improve freshness and can improve calibration when the data-generating process drifts. It can also overreact if retrained too often on noisy early-season samples, so the evaluation has to measure calibration, CLV, and month stability, not just ROI.",
        "3. Recommended production cadence: weekly expanding-window retraining, with an emergency/daily refresh only for feature inputs and injuries/lineups. In this repo-sized dataset daily retraining is computationally feasible, but weekly is operationally cleaner and reduces churn from one noisy slate.",
        "4. Computational cost: frozen requires one fit; weekly requires about one fit per week; daily requires one fit per slate. On the current matrix, Poisson top-120 retraining is seconds-level, so the cost is not a blocker. The heavier cost is rebuilding raw Statcast/lineup features, not fitting the GLM.",
        "5. The current repository can support walk-forward modeling at the matrix/backtest level, but production pipeline changes are required: preserve full historical raw feature caches, rebuild features by as-of date, schedule retraining, save model snapshots with cutoffs, and attach open/close odds snapshots for CLV.",
        "6. Correct evaluation: lock the model spec and bet rule, generate predictions sequentially with training data strictly before each game date, compare against the frozen 2025 baseline on the exact same test dates and available markets, and judge ROI, CLV, calibration, prediction error, monthly stability, edge monotonicity, and paired daily P/L differences.",
        "",
        "## Primary Comparison",
        "",
        f"- Test window: {TEST_START.date()} through {TEST_END.date()}",
        f"- Bet rule: unders only, devig edge >= {EDGE_MIN:.0f}%, edge-gap >= {EG_MIN:.0f}",
        f"- Frozen ROI/units: {frozen['roi']:+.2%}, {frozen['profit_units']:+.2f} units on {int(frozen['bets'])} bets",
        f"- Daily WF ROI/units: {daily['roi']:+.2%}, {daily['profit_units']:+.2f} units on {int(daily['bets'])} bets",
        f"- Weekly WF ROI/units: {weekly['roi']:+.2%}, {weekly['profit_units']:+.2f} units on {int(weekly['bets'])} bets",
        f"- Frozen MAE/RMSE: {frozen['mae']:.3f}/{frozen['rmse']:.3f}",
        f"- Daily WF MAE/RMSE: {daily['mae']:.3f}/{daily['rmse']:.3f}",
        f"- Weekly WF MAE/RMSE: {weekly['mae']:.3f}/{weekly['rmse']:.3f}",
        f"- Frozen avg CLV: {frozen['avg_clv_pct']:+.2f}% at {frozen['clv_coverage']:.1%} coverage",
        f"- Daily WF avg CLV: {daily['avg_clv_pct']:+.2f}% at {daily['clv_coverage']:.1%} coverage",
        f"- Weekly WF avg CLV: {weekly['avg_clv_pct']:+.2f}% at {weekly['clv_coverage']:.1%} coverage",
        f"- Paired daily P/L diff, daily WF minus frozen: {diff['daily_profit_diff_mean_b_minus_a']:+.3f} units/day, bootstrap CI [{diff['daily_profit_diff_ci_lo']:+.3f}, {diff['daily_profit_diff_ci_hi']:+.3f}], Pr(WF > frozen) {diff['bootstrap_pr_b_gt_a']:.1%}",
        "",
        "## Recommendation",
        "",
        f"Recommendation from this run: **{recommend}**.",
        "",
        "The professional production architecture should still move to a point-in-time walk-forward framework, because frozen-season training is structurally stale. But replacement of the current betting methodology should require material improvement in the locked comparison, especially CLV and calibration, not only a plausible engineering argument.",
        "",
        "## Files",
        "",
        "- `walk_forward_summary.csv`",
        "- `walk_forward_monthly.csv`",
        "- `walk_forward_edge_buckets.csv`",
        "- `walk_forward_bets.csv`",
        "- `walk_forward_scored_opportunities.csv`",
    ]
    (OUT_DIR / "walk_forward_research_review.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw, all_features = load_matrix()
    test = raw[(raw["game_date"] >= TEST_START) & (raw["game_date"] <= TEST_END)].copy()
    clv_index = load_clv_index()

    runs: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict]] = {}

    frozen_scored, frozen_meta = frozen_2025_predictions(raw, all_features, test)
    frozen_bets = attach_clv(filter_strategy(frozen_scored, EDGE_MIN, EG_MIN, SIDE_FILTER), clv_index)
    runs["frozen_2025"] = (frozen_scored, frozen_bets, frozen_meta)

    for cadence in ["daily", "weekly", "monthly"]:
        scored, meta = walk_forward_predictions(raw, all_features, test, cadence)
        bets = attach_clv(filter_strategy(scored, EDGE_MIN, EG_MIN, SIDE_FILTER), clv_index)
        runs[f"walk_forward_{cadence}"] = (scored, bets, meta)

    for days in [365, 540]:
        scored, meta = walk_forward_predictions(raw, all_features, test, "weekly", rolling_days=days)
        label = f"rolling_{days}d_weekly"
        bets = attach_clv(filter_strategy(scored, EDGE_MIN, EG_MIN, SIDE_FILTER), clv_index)
        runs[label] = (scored, bets, meta)

    weekly_scored = runs["walk_forward_weekly"][0]
    for weight in [0.75, 0.50]:
        label = f"blend_frozen_{int(weight*100)}_weekly_{int((1-weight)*100)}"
        scored, meta = blended_predictions(frozen_scored, weekly_scored, weight, label)
        bets = attach_clv(filter_strategy(scored, EDGE_MIN, EG_MIN, SIDE_FILTER), clv_index)
        runs[label] = (scored, bets, meta)

    summary_rows = []
    monthly_rows = []
    edge_rows = []
    scored_out = []
    bets_out = []
    for label, (scored, bets, meta) in runs.items():
        scored = scored.copy()
        bets = bets.copy()
        scored["eval_model"] = label
        bets["eval_model"] = label
        summary_rows.append(summarize_model(label, scored, bets, meta))
        month = monthly_table(bets)
        if not month.empty:
            month.insert(0, "model", label)
            monthly_rows.append(month)
        edge, _ = edge_monotonicity(bets)
        if not edge.empty:
            edge.insert(0, "model", label)
            edge_rows.append(edge)
        scored_out.append(scored)
        bets_out.append(bets)

    summary = pd.DataFrame(summary_rows)
    monthly = pd.concat(monthly_rows, ignore_index=True, sort=False)
    edge = pd.concat(edge_rows, ignore_index=True, sort=False)
    all_scored = pd.concat(scored_out, ignore_index=True, sort=False)
    all_bets = pd.concat(bets_out, ignore_index=True, sort=False)
    diff = daily_diff_test(runs["frozen_2025"][1], runs["walk_forward_daily"][1])
    diff_df = pd.DataFrame([diff])

    summary.to_csv(OUT_DIR / "walk_forward_summary.csv", index=False)
    monthly.to_csv(OUT_DIR / "walk_forward_monthly.csv", index=False)
    edge.to_csv(OUT_DIR / "walk_forward_edge_buckets.csv", index=False)
    all_bets.to_csv(OUT_DIR / "walk_forward_bets.csv", index=False)
    all_scored.to_csv(OUT_DIR / "walk_forward_scored_opportunities.csv", index=False)
    diff_df.to_csv(OUT_DIR / "walk_forward_daily_diff_test.csv", index=False)
    write_review(summary, diff)

    print("Summary")
    print(summary.round(4).to_string(index=False))
    print("\nPaired daily diff test: walk_forward_daily minus frozen_2025")
    print(diff_df.round(4).to_string(index=False))
    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
