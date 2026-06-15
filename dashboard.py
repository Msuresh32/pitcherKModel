import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Pitcher K Props", page_icon="⚾", layout="wide")

EXPORTS_DIR = Path("data/exports")
BACKTEST_FILE = Path("data/exports/2026_backtest_extended.csv")
LOGS_FILE = Path("data/raw/pitcher_game_logs.csv")

# ── Helpers ───────────────────────────────────────────────────────────────────
def calc_edge(prob, american_odds):
    if pd.isna(prob) or pd.isna(american_odds) or american_odds == 0:
        return float("nan")
    decimal = 1 + american_odds / 100 if american_odds > 0 else 1 + 100 / abs(american_odds)
    return (prob * (decimal - 1) - (1 - prob)) * 100

def pnl_from_bet(won, american_odds, stake=100):
    if pd.isna(won) or pd.isna(american_odds):
        return float("nan")
    decimal = 1 + american_odds / 100 if american_odds > 0 else 1 + 100 / abs(american_odds)
    return stake * (decimal - 1) if won else -stake

def edge_css(val):
    if pd.isna(val): return ""
    if val >= 15:  return "background-color:#00c853;color:black;font-weight:bold"
    if val >= 10:  return "background-color:#69f0ae;color:black;font-weight:bold"
    if val >= 7:   return "background-color:#fff176;color:black"
    if val >= 0:   return "background-color:#ffb74d;color:black"
    return "background-color:#ef9a9a;color:black"

def gap_label(gap):
    g = abs(gap)
    if g >= 1.2: return "1.2+ (59% WR)"
    if g >= 0.9: return "0.9-1.2 ★ (73% WR)"
    if g >= 0.6: return "0.6-0.9 (55% WR)"
    if g >= 0.3: return "0.3-0.6 ⚠ (44% WR)"
    return "0-0.3 (51% WR)"

def load_projections(target_date):
    path = EXPORTS_DIR / f"daily_pitcher_props_{target_date}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["market"] == "strikeouts"].copy()
    for col in ["strikeouts_projection","line","edge_pct","over_probability",
                "under_probability","over_odds","under_odds"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["pitcher_name"].notna()].reset_index(drop=True)
    df["gap"] = df["strikeouts_projection"] - df["line"]
    df["abs_gap"] = df["gap"].abs()
    df["hit_prob"] = np.where(df["best_side"]=="over", df["over_probability"], df["under_probability"])
    df["model_odds"] = np.where(df["best_side"]=="over", df["over_odds"], df["under_odds"])
    # Deduplicate: one row per pitcher (keep highest edge)
    df = (df.sort_values("edge_pct", ascending=False)
            .drop_duplicates(subset=["pitcher_name"])
            .reset_index(drop=True))
    return df

def load_backtest():
    if not BACKTEST_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(BACKTEST_FILE, low_memory=False)
    for col in ["strikeouts_projection","line","edge_pct","odds_used","won","actual","gap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["pitcher_name"].notna()].reset_index(drop=True)
    df["pnl"] = df.apply(
        lambda r: pnl_from_bet(r["won"], r["odds_used"]) if pd.notna(r["won"]) else float("nan"),
        axis=1
    )
    return df

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_proj, tab_back = st.tabs(["📋 Daily Picks", "📊 Backtest"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — DAILY PICKS
# ════════════════════════════════════════════════════════════════════════════════
with tab_proj:
    st.title("⚾ Pitcher Strikeout Props")

    available = sorted(
        [p.stem.replace("daily_pitcher_props_","") for p in EXPORTS_DIR.glob("daily_pitcher_props_*.csv")],
        reverse=True,
    )
    if not available:
        st.error("No projection files found.")
        st.stop()

    c1, c2, c3, c4 = st.columns([2,1,1,1])
    with c1:
        selected_date = st.selectbox("Date", available, index=0, label_visibility="collapsed")
    with c2:
        min_edge = st.slider("Min edge%", 0, 25, 0)
    with c3:
        dir_filter = st.checkbox("Direction agreement only", value=True)
    with c4:
        sort_by = st.radio("Sort", ["Edge%","Gap"], horizontal=True)

    df = load_projections(selected_date)
    if df.empty:
        st.warning(f"No projections for {selected_date}.")
        st.stop()

    if dir_filter:
        agrees = (((df["best_side"]=="over") & (df["gap"]>0)) |
                  ((df["best_side"]=="under") & (df["gap"]<0)))
        df = df[agrees].copy()

    df = df[df["edge_pct"].fillna(-99) >= min_edge]
    df = df.sort_values("abs_gap" if sort_by=="Gap" else "edge_pct", ascending=False).reset_index(drop=True)

    # Metrics row
    flagged = df[df["edge_pct"] >= 7]
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Pitchers", len(df))
    m2.metric("7%+ edge picks", len(flagged))
    m3.metric("Best edge", f"{df['edge_pct'].max():.1f}%" if len(df) else "—")
    m4.metric("Biggest gap", f"{df['abs_gap'].max():.2f}" if len(df) else "—")

    st.markdown("---")
    st.subheader("✏️ Enter your odds → edge recalculates live")

    display = pd.DataFrame({
        "Pitcher":      df["pitcher_name"].values,
        "Side":         (df["best_side"] + " " + df["line"].astype(str)).values,
        "Proj":         df["strikeouts_projection"].round(2).values,
        "Gap":          df["gap"].round(2).values,
        "P(hit)%":      (df["hit_prob"] * 100).round(1).values,
        "Model Edge%":  df["edge_pct"].round(1).values,
        "Your Odds":    df["model_odds"].fillna(0).astype(int).values,
        "Gap Bucket":   df["abs_gap"].apply(gap_label).values,
    })

    edited = st.data_editor(
        display,
        column_config={
            "Pitcher":      st.column_config.TextColumn(disabled=True),
            "Side":         st.column_config.TextColumn(disabled=True),
            "Proj":         st.column_config.NumberColumn("Proj K", disabled=True, format="%.2f"),
            "Gap":          st.column_config.NumberColumn(disabled=True, format="%+.2f"),
            "P(hit)%":      st.column_config.NumberColumn(disabled=True, format="%.1f"),
            "Model Edge%":  st.column_config.NumberColumn(disabled=True, format="%.1f"),
            "Your Odds":    st.column_config.NumberColumn("Your Odds ✏️", min_value=-500, max_value=1000, step=5),
            "Gap Bucket":   st.column_config.TextColumn(disabled=True),
        },
        hide_index=True,
        use_container_width=True,
        key="odds_editor",
    )

    # Recalc edge at user odds
    probs = df["hit_prob"].values
    your_edges = [calc_edge(probs[i], edited.at[i,"Your Odds"]) for i in range(len(edited))]
    edited["Your Edge%"] = [round(e,1) if not pd.isna(e) else float("nan") for e in your_edges]

    # Color-coded results table
    st.markdown("---")
    result = edited[["Pitcher","Side","Proj","Gap","P(hit)%","Your Odds","Your Edge%","Gap Bucket"]].copy()
    styled = (result.style
              .applymap(edge_css, subset=["Your Edge%"])
              .format({"Proj":"{:.2f}","Gap":"{:+.2f}","P(hit)%":"{:.1f}%","Your Edge%":"{:.1f}%"},
                      na_rep="—"))
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Qualifying plays
    plays = edited[edited["Your Edge%"] >= 7].sort_values("Your Edge%", ascending=False)
    if not plays.empty:
        st.markdown("---")
        st.subheader(f"✅ {len(plays)} plays at 7%+ edge (your odds)")
        for _, row in plays.iterrows():
            e = row["Your Edge%"]
            col = "green" if e >= 15 else ("orange" if e >= 10 else "blue")
            st.markdown(f"**{row['Pitcher']}** {row['Side']} | proj={row['Proj']:.2f} gap={row['Gap']:+.2f} | "
                        f":{col}[**{e:.1f}% edge**] at {int(row['Your Odds']):+d} | {row['Gap Bucket']}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Gap → Historical WR (2026)**")
    st.sidebar.markdown("🟢 0.9–1.2 → **73.3%**")
    st.sidebar.markdown("🟢 1.2+    → 58.8%")
    st.sidebar.markdown("🟡 0.6–0.9 → 54.7%")
    st.sidebar.markdown("🔴 0.3–0.6 → 44.3% ⚠")
    st.sidebar.markdown("⚪ 0–0.3   → 50.6%")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — BACKTEST
# ════════════════════════════════════════════════════════════════════════════════
with tab_back:
    st.title("📊 2026 Backtest — $100/bet")

    bt = load_backtest()
    if bt.empty:
        st.error("Backtest file not found.")
        st.stop()

    resolved = bt[bt["won"].notna()].copy()
    resolved["game_date"] = pd.to_datetime(resolved["game_date"])
    resolved = resolved.sort_values("game_date").reset_index(drop=True)

    # ── Summary metrics ──────────────────────────────────────────────────────
    total_bets  = len(resolved)
    wins        = int(resolved["won"].sum())
    losses      = total_bets - wins
    total_pnl   = resolved["pnl"].sum()
    roi         = total_pnl / (total_bets * 100) * 100
    win_rate    = wins / total_bets * 100

    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Total bets",  total_bets)
    m2.metric("Record",      f"{wins}W / {losses}L")
    m3.metric("Win rate",    f"{win_rate:.1f}%")
    m4.metric("Total P&L",   f"${total_pnl:+,.0f}")
    m5.metric("ROI",         f"{roi:+.1f}%", delta=f"${total_pnl:+,.0f}")

    st.markdown("---")

    # ── Cumulative P&L chart ─────────────────────────────────────────────────
    st.subheader("Cumulative P&L ($100/bet)")
    resolved["cum_pnl"] = resolved["pnl"].cumsum()
    chart_data = resolved.set_index("game_date")[["cum_pnl"]].rename(columns={"cum_pnl":"Cumulative P&L ($)"})
    st.line_chart(chart_data)

    st.markdown("---")

    # ── Gap bucket breakdown ─────────────────────────────────────────────────
    st.subheader("Performance by Gap Bucket")
    bins   = [0, 0.3, 0.6, 0.9, 1.2, 10]
    labels = ["0–0.3","0.3–0.6","0.6–0.9","0.9–1.2","1.2+"]
    resolved["gap_bucket"] = pd.cut(resolved["gap"], bins=bins, labels=labels)

    bucket_rows = []
    for b in labels:
        sub = resolved[resolved["gap_bucket"]==b]
        if len(sub) == 0: continue
        n   = len(sub)
        wr  = sub["won"].mean()*100
        pnl = sub["pnl"].sum()
        r   = pnl / (n*100) * 100
        bucket_rows.append({"Gap":b,"Bets":n,"Win%":round(wr,1),"P&L":round(pnl,0),"ROI%":round(r,1)})

    bucket_df = pd.DataFrame(bucket_rows)
    def color_roi(val):
        if val > 0:  return "background-color:#69f0ae;color:black"
        if val < 0:  return "background-color:#ef9a9a;color:black"
        return ""
    st.dataframe(
        bucket_df.style.applymap(color_roi, subset=["ROI%","P&L"])
                       .format({"Win%":"{:.1f}%","P&L":"${:+,.0f}","ROI%":"{:+.1f}%"}),
        use_container_width=True, hide_index=True
    )

    st.markdown("---")

    # ── Edge bucket breakdown ────────────────────────────────────────────────
    st.subheader("Performance by Edge Bucket")
    ebins   = [7, 10, 15, 20, 100]
    elabels = ["7–10%","10–15%","15–20%","20%+"]
    resolved["edge_bucket"] = pd.cut(resolved["edge_pct"], bins=ebins, labels=elabels)

    edge_rows = []
    for b in elabels:
        sub = resolved[resolved["edge_bucket"]==b]
        if len(sub) == 0: continue
        n   = len(sub)
        wr  = sub["won"].mean()*100
        pnl = sub["pnl"].sum()
        r   = pnl / (n*100) * 100
        edge_rows.append({"Edge":b,"Bets":n,"Win%":round(wr,1),"P&L":round(pnl,0),"ROI%":round(r,1)})

    edge_df = pd.DataFrame(edge_rows)
    st.dataframe(
        edge_df.style.applymap(color_roi, subset=["ROI%","P&L"])
                     .format({"Win%":"{:.1f}%","P&L":"${:+,.0f}","ROI%":"{:+.1f}%"}),
        use_container_width=True, hide_index=True
    )

    st.markdown("---")

    # ── Full pick log ────────────────────────────────────────────────────────
    st.subheader("All Picks Log")
    log = resolved[["game_date","pitcher_name","best_side","line","strikeouts_projection",
                    "gap","edge_pct","odds_used","actual","won","pnl"]].copy()
    log["game_date"] = log["game_date"].dt.strftime("%Y-%m-%d")
    log["gap"]       = log["gap"].round(2)
    log["Result"]    = log["won"].map({1.0:"WIN ✅", True:"WIN ✅", 0.0:"LOSS ❌", False:"LOSS ❌"})
    log = log.rename(columns={
        "game_date":"Date","pitcher_name":"Pitcher","best_side":"Side",
        "strikeouts_projection":"Proj","edge_pct":"Edge%","odds_used":"Odds",
        "actual":"Actual Ks","pnl":"P&L"
    })
    log = log.drop(columns=["won"]).sort_values("Date", ascending=False)

    def color_result(val):
        if "WIN"  in str(val): return "background-color:#69f0ae;color:black"
        if "LOSS" in str(val): return "background-color:#ef9a9a;color:black"
        return ""

    st.dataframe(
        log.style.applymap(color_result, subset=["Result"])
                 .applymap(color_roi, subset=["P&L"])
                 .format({"Proj":"{:.2f}","Edge%":"{:.1f}%","P&L":"${:+,.0f}","gap":"{:+.2f}"}, na_rep="—"),
        use_container_width=True, hide_index=True
    )
