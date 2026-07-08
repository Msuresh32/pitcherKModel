from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from no_leak_2025_to_2026_model_search import (  # noqa: E402
    EDGE_FILE,
    add_betting_frame,
    american_to_decimal,
    filter_strategy,
    fit_model,
    is_feature_col,
    predict_model,
    summarize,
    top_corr_features,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "june_19_30_backfill"
ODDS_FILE = OUT_DIR / "historical_pitcher_props_2026-06-19_2026-06-30.csv"
PITCHER_LOGS_FILE = ROOT / "data" / "raw" / "pitcher_game_logs.csv"
ALL_BETS_OUT = OUT_DIR / "final_model_bets_2026-06-19_2026-06-30.csv"
SUMMARY_OUT = OUT_DIR / "final_model_summary_2026-06-19_2026-06-30.csv"
DAILY_OUT = OUT_DIR / "final_model_daily_2026-06-19_2026-06-30.csv"
REPORT_OUT = OUT_DIR / "final_model_june_19_30_report.md"

START = pd.Timestamp("2026-06-19")
END = pd.Timestamp("2026-06-30")
MODEL_NAME = "poisson_top120_a2"
EDGE_MIN = 7.0
EG_MIN = 0.0
SIDE_FILTER = "under"


def normalize_pitcher_id(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64").astype(str).replace("<NA>", pd.NA)


def american_to_dec_value(odds) -> float:
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    return 1.0 + odds / 100.0 if odds > 0 else 1.0 + 100.0 / abs(odds)


def best_lines(odds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    odds = odds.copy()
    odds = odds[odds["market"].eq("strikeouts")].dropna(subset=["pitcher_id", "line"])
    odds["game_date"] = pd.to_datetime(odds["game_date"])
    odds["pitcher_id"] = normalize_pitcher_id(odds["pitcher_id"])
    odds["over_odds"] = pd.to_numeric(odds["over_odds"], errors="coerce")
    odds["under_odds"] = pd.to_numeric(odds["under_odds"], errors="coerce")

    for keys, group in odds.groupby(["game_date", "pitcher_id", "market", "line"], dropna=False):
        over_row = group.sort_values("over_odds", ascending=False, na_position="last").iloc[0]
        under_row = group.sort_values("under_odds", ascending=False, na_position="last").iloc[0]
        rows.append(
            {
                "game_date": keys[0],
                "pitcher_id": keys[1],
                "market": keys[2],
                "line": float(keys[3]),
                "over_odds": over_row["over_odds"],
                "under_odds": under_row["under_odds"],
                "over_bookmaker": over_row.get("bookmaker", ""),
                "under_bookmaker": under_row.get("bookmaker", ""),
                "player_name": over_row.get("player_name", ""),
                "fetched_at": over_row.get("fetched_at", ""),
                "event_id": over_row.get("event_id", ""),
                "commence_time": over_row.get("commence_time", ""),
            }
        )
    return pd.DataFrame(rows)


def prepare_training_matrix() -> tuple[pd.DataFrame, list[str]]:
    raw = pd.read_csv(EDGE_FILE)
    raw["game_date"] = pd.to_datetime(raw["game_date"])
    raw = raw[raw["market"].eq("strikeouts")].copy()
    raw["pitcher_id"] = normalize_pitcher_id(raw["pitcher_id"])

    numeric_cols = set(raw.select_dtypes(include=[np.number]).columns)
    all_features = [col for col in raw.columns if is_feature_col(col, numeric_cols)]
    train_games = (
        raw[raw["game_date"].dt.year == 2025]
        .sort_values("game_date")
        .drop_duplicates(["game_date", "pitcher_id"], keep="first")
        .reset_index(drop=True)
    )
    train_inner = train_games[train_games["game_date"] < "2025-07-01"].copy()
    top120 = top_corr_features(train_inner, all_features, 120)
    return raw, top120


def fit_final_model(train_matrix: pd.DataFrame, features: list[str]):
    train_games = (
        train_matrix[train_matrix["game_date"].dt.year == 2025]
        .sort_values("game_date")
        .drop_duplicates(["game_date", "pitcher_id"], keep="first")
        .reset_index(drop=True)
    )
    estimator = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", PoissonRegressor(alpha=2.0, max_iter=2000)),
        ]
    )
    return fit_model(MODEL_NAME, train_games, features, estimator)


def build_actual_history(train_matrix: pd.DataFrame, new_logs: pd.DataFrame) -> pd.DataFrame:
    old = (
        train_matrix[
            [
                "game_date",
                "game_pk",
                "pitcher_id",
                "pitcher_name",
                "team",
                "opponent",
                "is_home",
                "strikeouts",
                "walks",
                "hits_allowed",
                "innings_pitched",
                "pitches",
                "strikes",
                "batters_faced",
            ]
        ]
        .sort_values("game_date")
        .drop_duplicates(["game_date", "game_pk", "pitcher_id"], keep="first")
        .copy()
    )
    new = new_logs.copy()
    new["game_date"] = pd.to_datetime(new["game_date"])
    new["pitcher_id"] = normalize_pitcher_id(new["pitcher_id"])
    new = new[(new["game_date"] >= START) & (new["game_date"] <= END)].copy()
    hist = pd.concat([old, new], ignore_index=True, sort=False)
    hist["game_date"] = pd.to_datetime(hist["game_date"])
    hist["pitcher_id"] = normalize_pitcher_id(hist["pitcher_id"])
    return hist.sort_values(["pitcher_id", "game_date", "game_pk"]).reset_index(drop=True)


def trimmed_mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna().sort_values()
    if len(values) >= 5:
        values = values.iloc[1:-1]
    return float(values.mean()) if len(values) else np.nan


def prior_rows(history: pd.DataFrame, pitcher_id: str, game_date: pd.Timestamp) -> pd.DataFrame:
    p = history[(history["pitcher_id"].astype(str) == str(pitcher_id)) & (history["game_date"] < game_date)].copy()
    return p.sort_values("game_date")


def refresh_pitcher_rolling_features(row: pd.Series, prior: pd.DataFrame, features: list[str]) -> pd.Series:
    out = row.copy()
    prior = prior.copy()
    for col in ["strikeouts", "walks", "hits_allowed", "innings_pitched", "pitches", "strikes", "batters_faced"]:
        if col in prior:
            prior[col] = pd.to_numeric(prior[col], errors="coerce")
    prior["k_per_ip"] = prior["strikeouts"] / prior["innings_pitched"].replace(0, np.nan)
    prior["k9"] = prior["strikeouts"] * 9.0 / prior["innings_pitched"].replace(0, np.nan)
    prior["hits_per_ip"] = prior["hits_allowed"] / prior["innings_pitched"].replace(0, np.nan)
    prior["deep_start_6ip"] = prior["innings_pitched"] >= 6
    prior["short_start_under5ip"] = prior["innings_pitched"] < 5

    for w in [3, 5, 10, 20]:
        tail = prior.tail(w)
        mappings = {
            f"p_strikeouts_roll{w}": tail["strikeouts"].mean(),
            f"p_strikeouts_trimmed_roll{w}": trimmed_mean(tail["strikeouts"]),
            f"p_k_per_ip_roll{w}": tail["k_per_ip"].mean(),
            f"p_k9_roll{w}": tail["k9"].mean(),
            f"p_innings_pitched_roll{w}": tail["innings_pitched"].mean(),
            f"p_innings_pitched_max_roll{w}": tail["innings_pitched"].max(),
            f"p_deep_start_rate_6ip_roll{w}": tail["deep_start_6ip"].mean(),
            f"p_short_start_rate_under5ip_roll{w}": tail["short_start_under5ip"].mean(),
            f"p_hits_per_ip_roll{w}": tail["hits_per_ip"].mean(),
        }
        for key, value in mappings.items():
            if key in features:
                out[key] = value

    if len(prior):
        career_ip = prior["innings_pitched"].sum()
        if "p_strikeouts_career_avg_prior" in features:
            out["p_strikeouts_career_avg_prior"] = prior["strikeouts"].mean()
        if career_ip > 0:
            if "p_k_per_ip_career_prior" in features:
                out["p_k_per_ip_career_prior"] = prior["strikeouts"].sum() / career_ip
            if "p_k9_career" in features:
                out["p_k9_career"] = prior["strikeouts"].sum() * 9.0 / career_ip
            if "p_hits_per_ip_career_prior" in features:
                out["p_hits_per_ip_career_prior"] = prior["hits_allowed"].sum() / career_ip
        if "p_deep_start_rate_6ip_career_prior" in features:
            out["p_deep_start_rate_6ip_career_prior"] = prior["deep_start_6ip"].mean()
    return out


def build_extension_games(train_matrix: pd.DataFrame, features: list[str], actual_history: pd.DataFrame) -> pd.DataFrame:
    historical_profiles = (
        train_matrix.sort_values("game_date")
        .drop_duplicates(["game_date", "pitcher_id"], keep="last")
        .copy()
    )
    new_games = actual_history[(actual_history["game_date"] >= START) & (actual_history["game_date"] <= END)].copy()
    rows = []
    for _, actual in new_games.sort_values(["game_date", "pitcher_id"]).iterrows():
        pid = str(actual["pitcher_id"])
        gdate = pd.Timestamp(actual["game_date"])
        profile = historical_profiles[
            (historical_profiles["pitcher_id"].astype(str) == pid)
            & (historical_profiles["game_date"] < gdate)
        ].tail(1)
        if profile.empty:
            base = {feature: np.nan for feature in features}
        else:
            base = profile.iloc[0].to_dict()
        for col in [
            "game_date",
            "game_pk",
            "pitcher_id",
            "pitcher_name",
            "team",
            "opponent",
            "is_home",
            "strikeouts",
            "walks",
            "hits_allowed",
            "innings_pitched",
            "pitches",
            "strikes",
            "batters_faced",
        ]:
            base[col] = actual.get(col)
        prior = prior_rows(actual_history, pid, gdate)
        base = refresh_pitcher_rolling_features(pd.Series(base), prior, features).to_dict()
        rows.append(base)
    out = pd.DataFrame(rows)
    out["game_date"] = pd.to_datetime(out["game_date"])
    out["pitcher_id"] = normalize_pitcher_id(out["pitcher_id"])
    return out


def attach_odds(game_features: pd.DataFrame, odds_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    odds_raw = pd.read_csv(odds_file)
    odds_raw["game_date"] = pd.to_datetime(odds_raw["game_date"])
    open_lines = best_lines(odds_raw[odds_raw["snapshot_type"].eq("open")])
    close_lines = best_lines(odds_raw[odds_raw["snapshot_type"].eq("close")])
    merged = game_features.merge(open_lines, on=["game_date", "pitcher_id"], how="inner", suffixes=("", "_odds"))
    return merged, close_lines


def add_clv(bets: pd.DataFrame, close_lines: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return bets.assign(closing_odds=np.nan, clv_pct=np.nan)
    close = close_lines.rename(columns={"over_odds": "close_over_odds", "under_odds": "close_under_odds"})
    out = bets.merge(
        close[["game_date", "pitcher_id", "market", "line", "close_over_odds", "close_under_odds", "over_bookmaker", "under_bookmaker"]],
        on=["game_date", "pitcher_id", "market", "line"],
        how="left",
        suffixes=("", "_close"),
    )
    out["closing_odds"] = np.where(out["side"].eq("over"), out["close_over_odds"], out["close_under_odds"])
    out["clv_pct"] = [
        (american_to_dec_value(entry) / american_to_dec_value(close) - 1.0) * 100.0
        if pd.notna(entry) and pd.notna(close) and american_to_dec_value(close) > 1
        else np.nan
        for entry, close in zip(out["bet_odds"], out["closing_odds"])
    ]
    return out


def add_reporting_columns(bets: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return bets
    out = bets.copy()
    out["entry_odds"] = out["bet_odds"]
    out["actual_ks"] = out["strikeouts"]
    out["result"] = np.where(out["push"], "Push", np.where(out["won"], "Win", "Loss"))
    out["profit_units"] = out["profit_unit"]
    out["profit_100"] = out["profit_unit"] * 100.0
    out["game_date"] = pd.to_datetime(out["game_date"]).dt.date.astype(str)
    cols = [
        "game_date",
        "pitcher_name",
        "team",
        "opponent",
        "is_home",
        "market",
        "side",
        "line",
        "projection",
        "strikeouts",
        "actual_ks",
        "entry_odds",
        "closing_odds",
        "clv_pct",
        "edge_pct",
        "signed_gap",
        "abs_gap",
        "edge_gap_product",
        "over_probability",
        "under_probability",
        "bet_odds",
        "decimal_odds",
        "result",
        "won",
        "push",
        "profit_units",
        "profit_100",
        "over_bookmaker",
        "under_bookmaker",
        "fetched_at",
        "event_id",
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols + [c for c in out.columns if c not in cols]]


def make_daily_summary(bets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if bets.empty:
        return pd.DataFrame()
    for day, group in bets.groupby("game_date", sort=True):
        valid_clv = group["clv_pct"].dropna()
        rows.append(
            {
                "game_date": day,
                "bets": len(group),
                "wins": int(group["won"].sum()),
                "losses": int((~group["won"] & ~group["push"]).sum()),
                "pushes": int(group["push"].sum()),
                "win_rate": float(group.loc[~group["push"], "won"].mean()) if (~group["push"]).any() else np.nan,
                "profit_units": float(group["profit_unit"].sum()),
                "roi": float(group["profit_unit"].mean()),
                "avg_entry_odds": float(group["bet_odds"].mean()),
                "avg_edge_pct": float(group["edge_pct"].mean()),
                "avg_abs_gap": float(group["abs_gap"].mean()),
                "clv_bets": int(valid_clv.notna().sum()),
                "avg_clv_pct": float(valid_clv.mean()) if len(valid_clv) else np.nan,
                "positive_clv_rate": float((valid_clv > 0).mean()) if len(valid_clv) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_report(bets: pd.DataFrame, daily: pd.DataFrame, model_scores: dict, coverage: dict) -> None:
    valid_clv = bets["clv_pct"].dropna() if not bets.empty and "clv_pct" in bets else pd.Series(dtype=float)
    s = summarize(bets) if not bets.empty else {}
    lines = [
        "# Final Model June 19-30 Backfill",
        "",
        f"Window requested: {START.date()} through {END.date()}",
        f"Model: `{MODEL_NAME}` trained on 2025 pitcher-games only",
        f"Rule: `{SIDE_FILTER}` only, devig edge >= {EDGE_MIN:.0f}%, edge-gap >= {EG_MIN:.0f}",
        "",
        "## Result",
        "",
        f"- Bets: {int(s.get('bets', 0))}",
        f"- Wins/Losses/Pushes: {int(s.get('wins', 0))}/{int(s.get('losses', 0))}/{int(s.get('pushes', 0))}",
        f"- Win rate: {s.get('win_rate', np.nan):.2%}",
        f"- ROI: {s.get('roi', np.nan):+.2%}",
        f"- Units: {s.get('profit_units', 0.0):+.2f}",
        f"- Profit at $100 flat stake: ${s.get('profit_units', 0.0) * 100:+,.2f}",
        f"- Average edge: {s.get('avg_edge_pct', np.nan):.2f}%",
        f"- Average abs gap: {s.get('avg_abs_gap', np.nan):.2f} Ks",
        "",
        "## CLV",
        "",
        f"- CLV matched bets: {len(valid_clv)}/{len(bets)}",
        f"- Average CLV: {valid_clv.mean():+.2f}%" if len(valid_clv) else "- Average CLV: n/a",
        f"- Median CLV: {valid_clv.median():+.2f}%" if len(valid_clv) else "- Median CLV: n/a",
        f"- Positive CLV rate: {(valid_clv > 0).mean():.2%}" if len(valid_clv) else "- Positive CLV rate: n/a",
        "",
        "## Prediction Accuracy",
        "",
        f"- MAE on matched starter-games: {model_scores['mae']:.3f}",
        f"- RMSE on matched starter-games: {model_scores['rmse']:.3f}",
        f"- Scored starter-games with odds: {model_scores['rows']}",
        "",
        "## Data Coverage",
        "",
        f"- Open rows fetched: {coverage['open_rows']}",
        f"- Close rows fetched: {coverage['close_rows']}",
        f"- Open best-line rows matched to starter features: {coverage['matched_open_lines']}",
        "",
        "## Daily Breakdown",
        "",
        "| Date | Bets | W-L-P | ROI | Units | Avg CLV | CLV+ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in daily.iterrows():
        lines.append(
            f"| {row['game_date']} | {int(row['bets'])} | {int(row['wins'])}-{int(row['losses'])}-{int(row['pushes'])} | "
            f"{row['roi']:+.2%} | {row['profit_units']:+.2f} | "
            f"{row['avg_clv_pct']:+.2f}% | {row['positive_clv_rate']:.2%} |"
            if pd.notna(row["avg_clv_pct"])
            else f"| {row['game_date']} | {int(row['bets'])} | {int(row['wins'])}-{int(row['losses'])}-{int(row['pushes'])} | {row['roi']:+.2%} | {row['profit_units']:+.2f} | n/a | n/a |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- The 06/19-06/30 raw feature cache was not present in full when this backfill began.",
            "- Advanced/Statcast/lineup features were carried forward from the last available no-leak research-matrix profile; pitcher rolling K/IP/K9 features were refreshed from newly fetched actual starter logs.",
            "- CLV uses best available entry odds at roughly 4 hours before first pitch versus best available close odds roughly 3 minutes before first pitch.",
            "- Two individual close snapshots returned expired-event 404s from The Odds API; affected bets remain in ROI but can have missing CLV if no same line close was matched.",
        ]
    )
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_matrix, features = prepare_training_matrix()
    model = fit_final_model(train_matrix, features)

    pitcher_logs = pd.read_csv(PITCHER_LOGS_FILE)
    actual_history = build_actual_history(train_matrix, pitcher_logs)
    extension_games = build_extension_games(train_matrix, features, actual_history)
    scored_open, close_lines = attach_odds(extension_games, ODDS_FILE)

    projections = predict_model(model, scored_open)
    scored = add_betting_frame(scored_open, projections, model.residual_std, MODEL_NAME)
    bets = filter_strategy(scored, EDGE_MIN, EG_MIN, SIDE_FILTER)
    bets = add_clv(bets, close_lines)
    bets_report = add_reporting_columns(bets)
    daily = make_daily_summary(bets)

    y = pd.to_numeric(scored_open["strikeouts"], errors="coerce")
    model_scores = {
        "rows": int(len(scored_open.drop_duplicates(["game_date", "pitcher_id"]))),
        "mae": float(mean_absolute_error(y, projections)) if len(y) else np.nan,
        "rmse": float(math.sqrt(mean_squared_error(y, projections))) if len(y) else np.nan,
    }
    odds_raw = pd.read_csv(ODDS_FILE)
    coverage = {
        "open_rows": int(odds_raw["snapshot_type"].eq("open").sum()),
        "close_rows": int(odds_raw["snapshot_type"].eq("close").sum()),
        "matched_open_lines": int(len(scored_open)),
    }

    bets_report.to_csv(ALL_BETS_OUT, index=False)
    pd.DataFrame([summarize(bets)]).to_csv(SUMMARY_OUT, index=False)
    daily.to_csv(DAILY_OUT, index=False)
    write_report(bets, daily, model_scores, coverage)

    print(f"Wrote bets: {ALL_BETS_OUT}")
    print(f"Wrote summary: {SUMMARY_OUT}")
    print(f"Wrote daily: {DAILY_OUT}")
    print(f"Wrote report: {REPORT_OUT}")
    print(pd.DataFrame([summarize(bets)]).round(4).to_string(index=False))
    if not daily.empty:
        print(daily.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
