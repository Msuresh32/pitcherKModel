"""
Quantitative audit checks for the pitcher model backtest.
Tests: duplicates, fill-value leak magnitude, feature distributions,
rolling window correctness, and target correlation.
"""
import pandas as pd, numpy as np
from pathlib import Path

SEP = "=" * 65

# ─────────────────────────────────────────────────────────────────
# 1. DUPLICATE CHECK — raw edges files vs deduplicated
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("1. DUPLICATE CHECK IN EDGES FILES")
print(SEP)

edge_files = [
    ("WF p1 Mar-Apr", "data/processed/wf2026_p1_mar_apr_edges.csv"),
    ("WF p2 May",     "data/processed_apr2026/wf2026_p2_may_edges.csv"),
    ("WF p3 Jun",     "data/processed/wf2026_p3_jun_edges.csv"),
    ("2025 threshsel","data/processed_2024/thresh_sel_2025_dk_edges.csv"),
]

DEDUP_KEYS = ["game_date","pitcher_name","line","best_side"]

for label, path in edge_files:
    if not Path(path).exists():
        print(f"  {label}: FILE NOT FOUND"); continue
    raw = pd.read_csv(path)
    raw = raw[raw["market"]=="strikeouts"]
    dedup = raw.drop_duplicates(subset=DEDUP_KEYS)
    n_raw, n_dedup = len(raw), len(dedup)
    pct = (n_raw - n_dedup) / n_raw * 100 if n_raw > 0 else 0
    print(f"  {label:<22}  raw={n_raw:>5}  deduped={n_dedup:>5}  duplicates={n_raw-n_dedup:>4} ({pct:.1f}%)")

# Show example duplicates from first file
raw = pd.read_csv("data/processed/wf2026_p1_mar_apr_edges.csv")
raw = raw[raw["market"]=="strikeouts"]
dups = raw[raw.duplicated(subset=DEDUP_KEYS, keep=False)]
if len(dups) > 0:
    print(f"\n  Example duplicate rows (first 5 groups):")
    shown = 0
    for keys, grp in dups.groupby(DEDUP_KEYS):
        if shown >= 3: break
        print(f"    {keys[0]} | {keys[1]} | line={keys[2]} | {keys[3]} — {len(grp)} copies, edge_pct={grp['edge_pct'].round(2).tolist()}")
        shown += 1

# ─────────────────────────────────────────────────────────────────
# 2. IMPACT OF DUPLICATES ON REPORTED METRICS
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("2. DUPLICATE IMPACT ON WALK-FORWARD 2026 METRICS")
print(SEP)

def load_edges(pfx, d, edge_min=15.0, dedup=True):
    df = pd.read_csv(f"{d}/{pfx}_edges.csv")
    df = df[df["market"]=="strikeouts"].copy()
    df["won"] = df.apply(lambda r: (r["strikeouts"]>r["line"]) if r["best_side"]=="over"
                          else (r["strikeouts"]<r["line"]), axis=1)
    df["pay"] = df.apply(lambda r:
        (r["over_odds"]/100 if r["over_odds"]>0 else 100/abs(r["over_odds"])) if r["best_side"]=="over"
        else (r["under_odds"]/100 if r["under_odds"]>0 else 100/abs(r["under_odds"])), axis=1)
    df["profit"] = df.apply(lambda r: r["pay"] if r["won"] else -1.0, axis=1)
    if dedup:
        df = df.sort_values("edge_pct", ascending=False).drop_duplicates(subset=DEDUP_KEYS).reset_index(drop=True)
    return df[df["edge_pct"] >= edge_min]

for dedup_flag in [False, True]:
    dfs = [
        load_edges("wf2026_p1_mar_apr","data/processed", dedup=dedup_flag),
        load_edges("wf2026_p2_may","data/processed_apr2026", dedup=dedup_flag),
        load_edges("wf2026_p3_jun","data/processed", dedup=dedup_flag),
    ]
    comb = pd.concat(dfs, ignore_index=True)
    n   = len(comb)
    roi = comb["profit"].mean()
    sh  = roi / comb["profit"].std() * n**0.5
    label = "WITH dedup" if dedup_flag else "WITHOUT dedup (raw)"
    print(f"  {label:<26}  bets={n:>4}  win={comb['won'].mean():.1%}  ROI={roi:>+.1%}  Sharpe={sh:.2f}")

# ─────────────────────────────────────────────────────────────────
# 3. FILL-VALUE LEAK — how different are global vs train-only medians?
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("3. FILL-VALUE LEAK — global vs train-only median comparison")
print(SEP)

import json
fills_path = Path("data/processed/models/fill_values.json")
fills_2024  = Path("data/processed_2024/models/fill_values.json")

if fills_path.exists() and fills_2024.exists():
    with open(fills_path) as f: fills_main = json.load(f)
    with open(fills_2024)  as f: fills_24   = json.load(f)

    sk = fills_main.get("strikeouts", {})
    sk24 = fills_24.get("strikeouts", {})
    common_feats = set(sk.keys()) & set(sk24.keys())
    diffs = {}
    for feat in common_feats:
        v1, v2 = sk.get(feat, 0), sk24.get(feat, 0)
        if v1 != 0:
            diffs[feat] = abs(v1 - v2) / abs(v1) * 100
    top = sorted(diffs.items(), key=lambda x: -x[1])[:10]
    print(f"  Compared {len(common_feats)} features — top 10 median divergences:")
    print(f"  {'Feature':<45} {'Train-2024':>11} {'Main model':>11} {'Diff%':>7}")
    for feat, pct in top:
        v1 = sk.get(feat, 0); v2 = sk24.get(feat, 0)
        print(f"  {feat:<45} {v2:>11.4f} {v1:>11.4f} {pct:>7.1f}%")
    small_diffs = sum(1 for _, d in diffs.items() if d < 5)
    print(f"\n  {small_diffs}/{len(common_feats)} features have <5% median divergence")
else:
    print("  fill_values.json files not found — skipping")

# ─────────────────────────────────────────────────────────────────
# 4. ROLLING WINDOW CORRECTNESS — direct test
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("4. ROLLING WINDOW CORRECTNESS — verify shift(1) in training data")
print(SEP)

# Load a processed predictions file and check if roll3 predates the game
pred_file = Path("data/processed/exp_pred_bk_20260101_20260616_b70_30_predictions.csv")
if pred_file.exists():
    preds = pd.read_csv(pred_file)
    preds["game_date"] = pd.to_datetime(preds["game_date"])

    # For a pitcher with multiple games, check that roll3 on game N doesn't include game N
    if "p_strikeouts_roll3" in preds.columns and "strikeouts" in preds.columns:
        test_pitchers = preds.groupby("pitcher_id").filter(lambda g: len(g) >= 5)
        sample_pid = test_pitchers["pitcher_id"].value_counts().index[0]
        sp = preds[preds["pitcher_id"]==sample_pid].sort_values("game_date")[["game_date","strikeouts","p_strikeouts_roll3"]].head(8)
        print(f"  Sample pitcher ID {sample_pid} — strikeouts vs roll3 (should lag):")
        print(sp.to_string(index=False))

        # Statistical check: roll3 on game N should correlate with strikeouts[N-1:N-4]
        # If there's leakage: roll3 on game N includes strikeouts[N] → perfect same-game correlation
        all_pids = test_pitchers["pitcher_id"].unique()[:50]
        same_game_corrs = []
        for pid in all_pids:
            sub = preds[preds["pitcher_id"]==pid].sort_values("game_date")
            if len(sub) < 5: continue
            corr = sub["strikeouts"].corr(sub["p_strikeouts_roll3"])
            same_game_corrs.append(corr)
        mean_corr = np.nanmean(same_game_corrs)
        print(f"\n  Mean corr(strikeouts_today, roll3_today) across {len(same_game_corrs)} pitchers: {mean_corr:.3f}")
        print(f"  [If leakage: expect ~0.9+. If clean: expect ~0.5-0.7 from autocorrelation]")
    else:
        print("  p_strikeouts_roll3 or strikeouts column not in predictions file")
else:
    print("  Predictions file not found")

# ─────────────────────────────────────────────────────────────────
# 5. FEATURE DISTRIBUTIONS — top correlated features to target
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("5. FEATURE CORRELATIONS TO TARGET (strikeouts)")
print(SEP)

if pred_file.exists():
    preds = pd.read_csv(pred_file)
    feat_cols = [c for c in preds.columns if c.startswith(("p_","sc_","adv_","opp_","venue_","umpire_","lineup_","matchup_"))
                 and preds[c].dtype in [np.float64, np.int64, float, int]]
    if "strikeouts" in preds.columns:
        corrs = {}
        for c in feat_cols:
            if preds[c].std() > 0:
                corrs[c] = abs(preds["strikeouts"].corr(preds[c]))
        top_corr = sorted(corrs.items(), key=lambda x: -x[1])[:15]
        print(f"  Top 15 feature correlations to strikeouts:")
        print(f"  {'Feature':<45} {'|corr|':>7}")
        for feat, c in top_corr:
            flag = " *** CHECK FOR LEAKAGE" if c > 0.7 else ""
            print(f"  {feat:<45} {c:>7.3f}{flag}")

# ─────────────────────────────────────────────────────────────────
# 6. SAME-DAY STATCAST LEAK CHECK
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("6. SAME-DAY STATCAST LEAK — does sc_whiff_rate_roll3 include today?")
print(SEP)

if pred_file.exists():
    preds = pd.read_csv(pred_file)
    if "sc_swinging_strike_rate_roll3" in preds.columns and "strikeouts" in preds.columns:
        # On first game of season: roll3 should be NaN or from prior year
        # After game 1 it should reflect only prior games
        preds["game_date"] = pd.to_datetime(preds["game_date"])
        first_game_data = preds[preds["game_date"] == preds.groupby("pitcher_id")["game_date"].transform("min")]
        print(f"  Pitchers on their first game in dataset: {len(first_game_data)}")
        pct_nan = first_game_data["sc_swinging_strike_rate_roll3"].isna().mean()
        print(f"  % with NaN sc_whiff_rate_roll3 on first game: {pct_nan:.1%}")
        print(f"  [If leakage: first game would have non-NaN values from same day]")

# ─────────────────────────────────────────────────────────────────
# 7. TRAIN/TEST TEMPORAL INTEGRITY
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("7. TRAIN/TEST TEMPORAL INTEGRITY")
print(SEP)

import yaml
for cfg_name in ["config/config.yaml", "config/config_2024only.yaml", "config/config_apr2026.yaml"]:
    if not Path(cfg_name).exists(): continue
    with open(cfg_name) as f: cfg = yaml.safe_load(f)
    tr_end = cfg.get("model",{}).get("train_end") or cfg.get("training",{}).get("train_end","?")
    bt_start = cfg.get("backtest",{}).get("start_date","?")
    bt_end   = cfg.get("backtest",{}).get("end_date","?")
    print(f"  {cfg_name}")
    print(f"    train_end={tr_end}  backtest={bt_start} → {bt_end}")
    if tr_end != "?" and bt_start != "?":
        leak = pd.to_datetime(str(bt_start)) <= pd.to_datetime(str(tr_end))
        print(f"    Temporal integrity: {'*** OVERLAP — test period inside training!' if leak else 'OK (test starts after train_end)'}")

# ─────────────────────────────────────────────────────────────────
# 8. ODDS SNAPSHOT TIMING — entry vs feature availability
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("8. ODDS SNAPSHOT TIMING")
print(SEP)

odds = Path("data/odds/full_2026_odds_dk.csv")
if odds.exists():
    df_odds = pd.read_csv(odds)
    if "snapshot_type" in df_odds.columns:
        print(f"  Snapshot types in full_2026_odds_dk.csv:")
        print(f"  {df_odds['snapshot_type'].value_counts().to_dict()}")
    if "fetched_at" in df_odds.columns:
        df_open = df_odds[df_odds.get("snapshot_type","open")=="open"] if "snapshot_type" in df_odds.columns else df_odds
        sample = df_open.head(5)[["fetched_at","commence_time"]].copy() if "commence_time" in df_open.columns else df_open.head(5)[["fetched_at"]]
        print(f"\n  Sample entry times vs game times:")
        if "commence_time" in df_open.columns:
            df_open = df_open.copy()
            df_open["fetched_at"] = pd.to_datetime(df_open["fetched_at"], utc=True, errors="coerce")
            df_open["commence_time"] = pd.to_datetime(df_open["commence_time"], utc=True, errors="coerce")
            df_open["hrs_before"] = (df_open["commence_time"] - df_open["fetched_at"]).dt.total_seconds()/3600
            hrs = df_open["hrs_before"].dropna()
            print(f"  Hours before game: mean={hrs.mean():.1f}h  min={hrs.min():.1f}h  max={hrs.max():.1f}h  p25={hrs.quantile(.25):.1f}h  p75={hrs.quantile(.75):.1f}h")
            pct_under2 = (hrs < 2).mean()
            print(f"  % of 'open' snapshots taken <2h before game: {pct_under2:.1%}  (should be <5%)")

print(f"\n{SEP}")
print("AUDIT COMPLETE")
print(SEP)
