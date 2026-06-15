import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _summarize(df: pd.DataFrame, name: str, mask: pd.Series) -> Optional[dict]:
    group = df[mask].sort_values("game_date").copy()
    if group.empty:
        return None
    cumulative = group["profit"].cumsum()
    drawdown = cumulative - cumulative.cummax()
    losses = (~group["won"] & ~group["push"]).sum()
    return {
        "filter": name,
        "bets": int(len(group)),
        "wins": int(group["won"].sum()),
        "losses": int(losses),
        "win_rate": float(group["won"].mean()),
        "profit": float(group["profit"].sum()),
        "roi": float(group["profit"].sum() / group["stake"].sum()),
        "avg_edge": float(group["edge_pct"].mean()),
        "avg_odds": float(group["bet_odds"].mean()),
        "max_drawdown": float(drawdown.min()),
        "avg_projection_gap": float(group["projection_gap"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze 2025 strikeout betting filters.")
    parser.add_argument("--bets", default="data/processed/historical_odds_edge5_bets.csv")
    parser.add_argument("--output", default="data/processed/strikeout_filter_study_2025.csv")
    parser.add_argument("--stake", type=float, default=100.0)
    args = parser.parse_args()

    df = pd.read_csv(args.bets)
    df = df[df["market"] == "strikeouts"].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "pitcher_id", "edge_pct"], ascending=[True, True, False])
    df = df.drop_duplicates(["game_date", "pitcher_id"], keep="first")
    df["stake"] = float(args.stake)
    df["profit"] = df["unit_profit"] * df["stake"]
    df["projection_gap"] = (df["strikeouts_projection"] - df["line"]).abs()
    df["ip_volatility_proxy"] = (
        df["p_innings_pitched_roll3"] - df["p_innings_pitched_roll10"]
    ).abs()

    lineup_median = df["opp_lineup_k_rate_prior"].median()
    lineup_q60 = df["opp_lineup_k_rate_prior"].quantile(0.60)
    lineup_q40 = df["opp_lineup_k_rate_prior"].quantile(0.40)

    filters = {
        "baseline edge >=5": df["edge_pct"] >= 5,
        "edge >=8": df["edge_pct"] >= 8,
        "edge >=10": df["edge_pct"] >= 10,
        "overs only": df["best_side"] == "over",
        "unders only": df["best_side"] == "under",
        "expected IP >=5.3": df["expected_innings_pitched"] >= 5.3,
        "expected IP >=5.5": df["expected_innings_pitched"] >= 5.5,
        "recent IP roll5 >=5.5": df["p_innings_pitched_roll5"] >= 5.5,
        "low workload volatility": df["ip_volatility_proxy"] <= 0.75,
        "projection gap >=0.50": df["projection_gap"] >= 0.50,
        "projection gap >=0.75": df["projection_gap"] >= 0.75,
        "projection gap >=1.00": df["projection_gap"] >= 1.00,
        "lineup supports side median": (
            ((df["best_side"] == "over") & (df["opp_lineup_k_rate_prior"] >= lineup_median))
            | ((df["best_side"] == "under") & (df["opp_lineup_k_rate_prior"] <= lineup_median))
        ),
        "lineup supports side 60/40": (
            ((df["best_side"] == "over") & (df["opp_lineup_k_rate_prior"] >= lineup_q60))
            | ((df["best_side"] == "under") & (df["opp_lineup_k_rate_prior"] <= lineup_q40))
        ),
        "CSW roll5 top half": df["sc_csw_rate_roll5"] >= df["sc_csw_rate_roll5"].median(),
        "SwStr roll5 top half": df["sc_swinging_strike_rate_roll5"]
        >= df["sc_swinging_strike_rate_roll5"].median(),
        "over + lineup high": (df["best_side"] == "over")
        & (df["opp_lineup_k_rate_prior"] >= lineup_q60),
        "under + lineup low": (df["best_side"] == "under")
        & (df["opp_lineup_k_rate_prior"] <= lineup_q40),
        "combo: IP+gap+lineup": (
            (df["expected_innings_pitched"] >= 5.3)
            & (df["projection_gap"] >= 0.5)
            & (
                ((df["best_side"] == "over") & (df["opp_lineup_k_rate_prior"] >= lineup_median))
                | ((df["best_side"] == "under") & (df["opp_lineup_k_rate_prior"] <= lineup_median))
            )
        ),
        "combo: overs IP+gap+lineup": (
            (df["best_side"] == "over")
            & (df["expected_innings_pitched"] >= 5.3)
            & (df["projection_gap"] >= 0.5)
            & (df["opp_lineup_k_rate_prior"] >= lineup_median)
        ),
    }

    rows = [_summarize(df, name, mask) for name, mask in filters.items()]
    out = pd.DataFrame([row for row in rows if row is not None]).sort_values("roi", ascending=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"Filter study saved to {output}")
    print(
        out[
            [
                "filter",
                "bets",
                "wins",
                "losses",
                "win_rate",
                "profit",
                "roi",
                "avg_edge",
                "max_drawdown",
            ]
        ]
        .head(15)
        .round({"win_rate": 3, "profit": 2, "roi": 3, "avg_edge": 2, "max_drawdown": 2})
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
