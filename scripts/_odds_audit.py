import pandas as pd, numpy as np

def load_bets(path, edge_min=15.0):
    df = pd.read_csv(path)
    df = df[df["market"]=="strikeouts"].copy()
    df["won"] = df.apply(lambda r: (r["strikeouts"]>r["line"]) if r["best_side"]=="over"
                          else (r["strikeouts"]<r["line"]), axis=1)
    df["entry_odds"] = df.apply(lambda r: r["over_odds"] if r["best_side"]=="over"
                                 else r["under_odds"], axis=1)
    df["decimal"]    = df["entry_odds"].apply(
        lambda o: o/100+1 if o>0 else 100/abs(o)+1)
    df["implied"]    = 1/df["decimal"]
    df["break_even"] = df["implied"]
    df["profit"]     = df.apply(lambda r: (r["decimal"]-1) if r["won"] else -1.0, axis=1)
    df = (df.sort_values("edge_pct", ascending=False)
            .drop_duplicates(subset=["game_date","pitcher_name","line","best_side"])
            .reset_index(drop=True))
    return df[df["edge_pct"]>=edge_min]

# ── load 2025 + 2026 walk-forward ─────────────────────────────────
bets_25 = load_bets("data/processed_2024/thresh_sel_2025_dk_edges.csv")
bets_26  = pd.concat([
    load_bets("data/processed/wf2026_p1_mar_apr_edges.csv"),
    load_bets("data/processed_apr2026/wf2026_p2_may_edges.csv"),
    load_bets("data/processed/wf2026_p3_jun_edges.csv"),
], ignore_index=True)
bets_26 = (bets_26.sort_values("edge_pct", ascending=False)
               .drop_duplicates(subset=["game_date","pitcher_name","line","best_side"])
               .reset_index(drop=True))

for label, bets in [("2025 (threshold selection)", bets_25),
                    ("2026 walk-forward",           bets_26)]:
    print(f"\n{'='*65}")
    print(f"{label}  —  edge >= 15%,  {len(bets)} bets")
    print(f"{'='*65}")

    for side in ["over","under","ALL"]:
        sub = bets if side=="ALL" else bets[bets["best_side"]==side]
        if len(sub)==0: continue
        n          = len(sub)
        mean_odds  = sub["entry_odds"].mean()
        med_odds   = sub["entry_odds"].median()
        mean_bep   = sub["break_even"].mean()
        win        = sub["won"].mean()
        roi        = sub["profit"].mean()
        edge_over_bep = win - mean_bep
        print(f"\n  {side.upper():5}  n={n:>4}")
        print(f"    Mean entry odds:  {mean_odds:>+.1f}   Median: {med_odds:>+.1f}")
        print(f"    Mean break-even:  {mean_bep:.1%}")
        print(f"    Actual win rate:  {win:.1%}")
        print(f"    Win vs BEP:       {edge_over_bep:>+.1%}")
        print(f"    ROI per bet:      {roi:>+.1%}")

    # Odds bucket breakdown
    print(f"\n  Odds distribution (ALL bets):")
    print(f"  {'Bucket':<18} {'n':>5}  {'mean BEP':>9}  {'win%':>6}  {'ROI':>7}  {'vs BEP':>7}")
    print("  " + "-"*58)
    buckets = [
        ("< -130  (heavy fav)", bets[bets["entry_odds"] <  -130]),
        ("-130 to -110",        bets[(bets["entry_odds"]>=-130) & (bets["entry_odds"]< -110)]),
        ("-110 to -100",        bets[(bets["entry_odds"]>=-110) & (bets["entry_odds"]<= -100)]),
        ("-100 to +100",        bets[(bets["entry_odds"]>  -100) & (bets["entry_odds"]<  100)]),
        ("+100 to +120",        bets[(bets["entry_odds"]>= 100) & (bets["entry_odds"]<= 120)]),
        ("> +120  (big dog)",   bets[bets["entry_odds"] >   120]),
    ]
    for bl, sub in buckets:
        if len(sub)==0: continue
        bep = sub["break_even"].mean()
        win = sub["won"].mean()
        roi = sub["profit"].mean()
        print(f"  {bl:<18} {len(sub):>5}  {bep:>9.1%}  {win:>6.1%}  {roi:>+7.1%}  {win-bep:>+7.1%}")

    # Side x odds combined
    print(f"\n  Over/Under x odds split:")
    print(f"  {'Side + odds':<25} {'n':>5}  {'BEP':>6}  {'win%':>6}  {'ROI':>7}")
    print("  " + "-"*55)
    for side in ["over","under"]:
        for ol, oh, lab in [(-300,-110,"neg odds (<-110)"),(-110,0,"near-even (-110 to -100)"),(0,300,"pos odds (+)")]:
            sub = bets[(bets["best_side"]==side) &
                       (bets["entry_odds"]>=ol) & (bets["entry_odds"]<oh)]
            if len(sub)<5: continue
            bep = sub["break_even"].mean()
            win = sub["won"].mean()
            roi = sub["profit"].mean()
            print(f"  {side+' | '+lab:<25} {len(sub):>5}  {bep:>6.1%}  {win:>6.1%}  {roi:>+7.1%}")
