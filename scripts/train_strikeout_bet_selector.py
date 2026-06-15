import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


FEATURE_CANDIDATES = [
    "edge_pct",
    "ev",
    "line",
    "bet_odds",
    "over_odds",
    "under_odds",
    "over_probability",
    "under_probability",
    "raw_over_probability",
    "fair_over_odds",
    "fair_under_odds",
    "projection_gap",
    "strikeouts_projection",
    "expected_innings_pitched",
    "expected_pitches",
    "expected_batters_faced",
    "days_rest",
    "pitcher_game_number",
    "p_games_prior",
    "p_strikeouts_roll3",
    "p_strikeouts_roll5",
    "p_strikeouts_roll10",
    "p_strikeouts_career_avg_prior",
    "p_innings_pitched_roll3",
    "p_innings_pitched_roll5",
    "p_innings_pitched_roll10",
    "p_innings_pitched_career_avg_prior",
    "p_k_per_ip_roll3",
    "p_k_per_ip_roll5",
    "p_k_per_ip_roll10",
    "p_k_per_ip_career_prior",
    "opp_pitcher_strikeouts_roll3",
    "opp_pitcher_strikeouts_roll5",
    "opp_pitcher_strikeouts_roll10",
    "opp_pitcher_strikeouts_avg_prior",
    "opp_batting_k_rate_roll3",
    "opp_batting_k_rate_roll5",
    "opp_batting_k_rate_roll10",
    "opp_batting_k_rate_prior",
    "opp_lineup_k_rate_prior",
    "opp_lineup_bb_rate_prior",
    "opp_lineup_hit_rate_prior",
    "opp_lineup_left_batters",
    "opp_lineup_right_batters",
    "opp_lineup_switch_batters",
    "opp_lineup_same_hand_batters",
    "opp_lineup_opposite_hand_batters",
    "pitcher_throws_left",
    "pitcher_throws_right",
    "park_so_factor",
    "venue_strikeouts_roll5",
    "venue_strikeouts_avg_prior",
    "umpire_strikeouts_roll5",
    "umpire_strikeouts_avg_prior",
    "sc_statcast_pitches_roll5",
    "sc_avg_release_speed_roll5",
    "sc_swinging_strike_rate_roll5",
    "sc_csw_rate_roll5",
    "sc_zone_rate_roll5",
    "sc_fastball_pct_roll5",
    "sc_slider_pct_roll5",
    "sc_breaking_pct_roll5",
    "sc_offspeed_pct_roll5",
]


def _american_to_decimal(odds: pd.Series) -> pd.Series:
    return np.where(odds > 0, 1 + odds / 100, 1 + 100 / odds.abs())


def _load_candidates(path: str, min_edge: float = 5.0) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["market"] == "strikeouts"].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "pitcher_id", "edge_pct"], ascending=[True, True, False])
    df = df.drop_duplicates(["game_date", "pitcher_id"], keep="first")
    df = df[df["edge_pct"] >= min_edge].copy()
    if df.empty:
        return df

    df["bet_odds"] = np.where(df["best_side"] == "over", df["over_odds"], df["under_odds"])
    df["decimal_odds"] = _american_to_decimal(df["bet_odds"].astype(float))
    df["projection_gap"] = (df["strikeouts_projection"] - df["line"]).abs()
    df["projection_signed_gap"] = df["strikeouts_projection"] - df["line"]
    df["is_over"] = (df["best_side"] == "over").astype(int)
    df["is_under"] = (df["best_side"] == "under").astype(int)
    df["won_int"] = df["won"].astype(int)
    df["unit_profit"] = np.where(
        df["push"],
        0.0,
        np.where(df["won"], df["decimal_odds"] - 1, -1.0),
    )
    return df


def _feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [col for col in FEATURE_CANDIDATES if col in df.columns]
    cols += ["projection_signed_gap", "is_over", "is_under"]
    return [col for col in cols if col in df.columns]


def _make_model(model_type: str, random_state: int) -> Pipeline:
    if model_type == "logistic":
        estimator = LogisticRegression(max_iter=2000, class_weight="balanced")
    else:
        estimator = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_type == "logistic":
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _summarize(df: pd.DataFrame, name: str, mask: pd.Series) -> dict:
    group = df[mask].sort_values("game_date").copy()
    if group.empty:
        return {"strategy": name, "bets": 0}
    group["flat_profit"] = group["unit_profit"] * 100
    cumulative = group["flat_profit"].cumsum()
    drawdown = cumulative - cumulative.cummax()
    return {
        "strategy": name,
        "bets": int(len(group)),
        "wins": int(group["won"].sum()),
        "losses": int((~group["won"] & ~group["push"]).sum()),
        "win_rate": float(group["won"].mean()),
        "profit_100_flat": float(group["flat_profit"].sum()),
        "roi": float(group["flat_profit"].sum() / (len(group) * 100)),
        "max_drawdown": float(drawdown.min()),
        "avg_selector_prob": float(group["selector_win_probability"].mean()),
        "avg_edge_pct": float(group["edge_pct"].mean()),
        "avg_projection_gap": float(group["projection_gap"].mean()),
    }


def _threshold_sweep(scored: pd.DataFrame) -> pd.DataFrame:
    rows = [_summarize(scored, "all edge>=5 candidates", scored["selector_win_probability"] >= 0)]
    for threshold in np.arange(0.50, 0.76, 0.05):
        rows.append(
            _summarize(
                scored,
                f"selector_prob >= {threshold:.2f}",
                scored["selector_win_probability"] >= threshold,
            )
        )
    for count in [25, 50, 75, 100, 150, 200]:
        top = scored.sort_values("selector_win_probability", ascending=False).head(count)
        rows.append(_summarize(scored, f"top {count} selector bets", scored.index.isin(top.index)))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a strikeout bet-selection model.")
    parser.add_argument("--train-bets", default="data/processed/historical_odds_edge5_bets.csv")
    parser.add_argument("--test-bets", default="data/processed/historical_odds_2026_6h_edge5_bets.csv")
    parser.add_argument("--model-type", choices=["random_forest", "logistic"], default="random_forest")
    parser.add_argument("--min-edge", type=float, default=5.0)
    parser.add_argument("--output-prefix", default="data/processed/strikeout_bet_selector")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train = _load_candidates(args.train_bets, min_edge=args.min_edge)
    test = _load_candidates(args.test_bets, min_edge=args.min_edge)
    if train.empty or test.empty:
        raise ValueError(f"Not enough candidate bets. train={len(train)}, test={len(test)}")

    feature_cols = _feature_cols(train)
    model = _make_model(args.model_type, args.random_state)
    model.fit(train[feature_cols], train["won_int"])

    train["selector_win_probability"] = model.predict_proba(train[feature_cols])[:, 1]
    test["selector_win_probability"] = model.predict_proba(test[feature_cols])[:, 1]

    train_auc = roc_auc_score(train["won_int"], train["selector_win_probability"])
    test_auc = roc_auc_score(test["won_int"], test["selector_win_probability"])
    sweep = _threshold_sweep(test)
    meta = pd.DataFrame(
        [
            {
                "model_type": args.model_type,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_win_rate": float(train["won"].mean()),
                "test_win_rate": float(test["won"].mean()),
                "train_auc": float(train_auc),
                "test_auc": float(test_auc),
                "features": ",".join(feature_cols),
            }
        ]
    )

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(prefix.with_name(prefix.name + "_train_scored.csv"), index=False)
    test.to_csv(prefix.with_name(prefix.name + "_test_scored.csv"), index=False)
    sweep.to_csv(prefix.with_name(prefix.name + "_test_summary.csv"), index=False)
    meta.to_csv(prefix.with_name(prefix.name + "_meta.csv"), index=False)

    print(f"Train scored bets saved to {prefix.with_name(prefix.name + '_train_scored.csv')}")
    print(f"Test scored bets saved to {prefix.with_name(prefix.name + '_test_scored.csv')}")
    print(f"Test summary saved to {prefix.with_name(prefix.name + '_test_summary.csv')}")
    print(meta.drop(columns=["features"]).to_string(index=False))
    print(
        sweep[
            [
                "strategy",
                "bets",
                "wins",
                "losses",
                "win_rate",
                "profit_100_flat",
                "roi",
                "max_drawdown",
                "avg_selector_prob",
            ]
        ]
        .head(20)
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
