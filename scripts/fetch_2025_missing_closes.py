"""
Fetch DK closing-line snapshots for the 49 dates where we're missing 2025 close data.
Appends to data/odds/hist_2025_dk_close_patch.csv
Then computes full CLV for all 525 qualifying 2025 bets.
"""
import sys, time
from pathlib import Path
import pandas as pd, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.odds.odds_api import (
    fetch_historical_events, fetch_historical_event_odds,
    normalize_event_odds, game_snapshot_time, get_api_key,
    map_odds_to_pitcher_logs,
)
from src.data.loaders import load_pitcher_game_logs
from src.config import load_config

PATCH_FILE = Path("data/odds/hist_2025_dk_close_patch.csv")
CLOSE_FILE  = Path("data/odds/historical_pitcher_props_2025.csv")  # existing
BETS_FILE   = Path("data/processed_2024/thresh_sel_2025_dk_edges.csv")
OUT_CLV     = Path("data/processed_2024/thresh_sel_2025_clv.csv")

MISSING_DATES = [
    "2025-03-27","2025-03-29","2025-03-30","2025-03-31",
    "2025-04-06","2025-04-08","2025-04-14","2025-04-30",
    "2025-05-04","2025-05-06","2025-05-11","2025-05-16","2025-05-17",
    "2025-05-25","2025-05-28",
    "2025-06-03","2025-06-04","2025-06-05","2025-06-09","2025-06-11",
    "2025-06-14","2025-06-18","2025-06-19","2025-06-26",
    "2025-07-08","2025-07-20","2025-07-21","2025-07-22",
    "2025-07-26","2025-07-27",
    "2025-08-02","2025-08-03","2025-08-05","2025-08-10","2025-08-12",
    "2025-08-13","2025-08-21","2025-08-24","2025-08-25","2025-08-28",
    "2025-08-29","2025-08-31",
    "2025-09-01","2025-09-06","2025-09-11","2025-09-18","2025-09-19",
    "2025-09-21","2025-09-26",
]


def day_bounds(day_str):
    ts = pd.Timestamp(day_str, tz="UTC")
    end = ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return ts.isoformat().replace("+00:00","Z"), end.isoformat().replace("+00:00","Z")


def fetch_missing():
    api_key = get_api_key()
    config  = load_config("config/config.yaml")
    pitcher_logs = load_pitcher_game_logs(config["data"]["pitcher_logs_file"])

    # Dates already patched
    already_done = set()
    if PATCH_FILE.exists():
        existing = pd.read_csv(PATCH_FILE)
        already_done = set(existing["game_date"].astype(str).unique())
        print(f"Patch file exists: {len(already_done)} dates already fetched, {len(existing)} rows")

    to_fetch = [d for d in MISSING_DATES if d not in already_done]
    print(f"Dates to fetch: {len(to_fetch)}")

    for di, day_str in enumerate(to_fetch):
        cfrom, cto = day_bounds(day_str)
        disc_snap  = (pd.Timestamp(day_str, tz="UTC") + pd.Timedelta(hours=16)).isoformat().replace("+00:00","Z")

        events, _, _ = fetch_historical_events(api_key, snapshot_date=disc_snap,
                                               commence_time_from=cfrom, commence_time_to=cto)
        print(f"[{di+1}/{len(to_fetch)}] {day_str}: {len(events)} games")

        day_rows = []
        for event in events:
            close_snap = game_snapshot_time(event["commence_time"], hours_before=0.05)
            try:
                event_data, _, payload = fetch_historical_event_odds(
                    api_key=api_key, event_id=event["id"],
                    snapshot_date=close_snap,
                    regions="us", markets=["pitcher_strikeouts"],
                    bookmakers="draftkings",
                )
            except Exception as e:
                print(f"  ERROR {event['id']}: {e}")
                time.sleep(0.5)
                continue
            actual_snap = payload.get("timestamp", close_snap)
            frame = normalize_event_odds(event_data, fetched_at=actual_snap)
            if not frame.empty:
                frame["snapshot_type"]     = "close"
                frame["requested_snapshot"] = close_snap
                frame["historical_snapshot"] = actual_snap
                day_rows.append(frame)
            time.sleep(0.35)

        if not day_rows:
            print(f"  {day_str}: no rows returned")
            continue

        day_df = pd.concat(day_rows, ignore_index=True, sort=False)
        day_df = map_odds_to_pitcher_logs(day_df, pitcher_logs)
        if day_df.empty:
            print(f"  {day_str}: no matched pitchers")
            continue

        day_df.to_csv(PATCH_FILE, mode="a", header=not PATCH_FILE.exists(), index=False)
        print(f"  {day_str}: saved {len(day_df)} rows")

    print(f"\nDone fetching. Patch file: {PATCH_FILE}")


def american_to_decimal(o):
    try:
        o = float(o)
    except: return np.nan
    if pd.isna(o): return np.nan
    return 100/abs(o)+1 if o < 0 else o/100+1


def compute_clv():
    # Load all bets (edge >= 15%, deduped)
    bets = pd.read_csv(BETS_FILE)
    bets = bets[(bets["market"]=="strikeouts") & (bets["edge_pct"]>=15)]
    bets = (bets.sort_values("edge_pct", ascending=False)
               .drop_duplicates(subset=["game_date","pitcher_name","line","best_side"])
               .reset_index(drop=True))
    print(f"\nQualifying bets (edge>=15%, deduped): {len(bets)}")

    # Load DK close data from existing + patch file
    existing = pd.read_csv(CLOSE_FILE)
    existing = existing[(existing["market"]=="strikeouts") &
                        (existing["bookmaker"]=="draftkings") &
                        (existing["snapshot_type"]=="close")]

    patch = pd.DataFrame()
    if PATCH_FILE.exists():
        patch = pd.read_csv(PATCH_FILE)
        patch = patch[(patch["market"]=="strikeouts") &
                      (patch["bookmaker"]=="draftkings") &
                      (patch["snapshot_type"]=="close")]

    close_all = pd.concat([existing, patch], ignore_index=True)
    close_all = close_all.drop_duplicates(subset=["game_date","pitcher_name","line"])
    print(f"DK close rows (combined): {len(close_all)}")

    # Merge entry + close
    merged = bets.merge(
        close_all[["game_date","pitcher_name","line","over_odds","under_odds"]]
                 .rename(columns={"over_odds":"close_over","under_odds":"close_under"}),
        on=["game_date","pitcher_name","line"], how="left"
    )
    print(f"Matched to close: {merged['close_over'].notna().sum()} / {len(merged)}")

    # Compute CLV
    def entry_odds(row):
        return row["over_odds"] if row["best_side"]=="over" else row["under_odds"]
    def close_odds(row):
        return row["close_over"] if row["best_side"]=="over" else row["close_under"]

    merged["entry_decimal"] = merged.apply(lambda r: american_to_decimal(entry_odds(r)), axis=1)
    merged["close_decimal"] = merged.apply(lambda r: american_to_decimal(close_odds(r)), axis=1)
    merged["clv_pct"] = ((merged["entry_decimal"] / merged["close_decimal"]) - 1) * 100

    # Outcome
    merged["won"] = merged.apply(
        lambda r: (r["strikeouts"] > r["line"]) if r["best_side"]=="over"
                  else (r["strikeouts"] < r["line"]), axis=1)

    # Save
    merged.to_csv(OUT_CLV, index=False)
    print(f"CLV output saved -> {OUT_CLV}")

    # Summary
    valid = merged.dropna(subset=["clv_pct"])
    print(f"\n{'='*55}")
    print(f"2025 WALK-FORWARD CLV  (edge>=15%, DK, {len(valid)} bets matched)")
    print(f"{'='*55}")
    print(f"  Mean CLV:          {valid['clv_pct'].mean():+.3f}%")
    print(f"  Median CLV:        {valid['clv_pct'].median():+.3f}%")
    print(f"  % beat close:      {(valid['clv_pct']>0).mean():.1%}  ({(valid['clv_pct']>0).sum()}/{len(valid)})")
    n = len(valid); se = valid["clv_pct"].std()/n**0.5
    t = valid["clv_pct"].mean()/se if se>0 else np.nan
    print(f"  t-stat vs 0:       {t:.2f}  (|t|>2 = sig)")
    print(f"  Win rate:          {merged['won'].mean():.1%}  (n={len(merged)})")

    # CLV+ vs CLV- win rates
    w_pos = valid[valid["clv_pct"]>0]["won"].mean()
    w_neg = valid[valid["clv_pct"]<=0]["won"].mean()
    print(f"  Win rate (CLV>0):  {w_pos:.1%}  (n={(valid['clv_pct']>0).sum()})")
    print(f"  Win rate (CLV<=0): {w_neg:.1%}  (n={(valid['clv_pct']<=0).sum()})")

    # Monthly breakdown
    merged["game_date"] = pd.to_datetime(merged["game_date"])
    merged["month"] = merged["game_date"].dt.to_period("M")
    print(f"\n  Monthly CLV:")
    for m, g in merged.groupby("month"):
        cv = g.dropna(subset=["clv_pct"])
        win = g["won"].mean()
        clv_str = f"{cv['clv_pct'].mean():+.2f}% (n={len(cv)})" if len(cv)>0 else "N/A"
        print(f"    {str(m)}  win={win:.1%}  CLV={clv_str}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--compute-only", action="store_true")
    args = p.parse_args()
    if not args.compute_only:
        fetch_missing()
    compute_clv()
