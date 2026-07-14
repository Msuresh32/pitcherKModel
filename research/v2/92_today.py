"""Score today's slate with the configured adaptive pitcher-K head.

Model: count ensemble, July vintage (trained on all games settled before
2026-07-01), pre-2026 isotonic calibration, trailing-90d Platt recalibration
(as of the 2026-07-06 weekly refresh) — identical to the walk-forward system.

Odds: live morning/evening snapshot from data/odds/pitcher_props.csv.
Output: full-board CSV plus a plays-only CSV/HTML with an outcome-blind
qualitative validation for every deployment-qualified signal.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
MS = __import__("20_model_search")

from src.config import load_config
from src.data.loaders import (
    load_batter_game_logs, load_game_context_logs, load_park_factors,
    load_pitcher_game_logs, load_statcast_catcher_framing_daily,
    load_statcast_pitcher_catcher_daily, load_statcast_pitcher_daily,
    load_team_batting_game_logs,
)
from src.features.build_features import build_training_features

import os
TODAY = pd.Timestamp(os.environ.get("SCORE_DATE", pd.Timestamp.now().strftime("%Y-%m-%d")))
# monthly expanding retrain: train on everything before the 1st of the current month
VINTAGE_CUTOFF = TODAY.replace(day=1)
# trailing recalibration refreshes on Mondays
PLATT_ASOF = TODAY - pd.Timedelta(days=(TODAY.weekday()) % 7)
BANKROLL = float(os.environ.get("BANKROLL", "10000"))
OUT = Path("research/v2")
PRED_LOG = Path("data/daily/v2_predictions_log.csv")


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def low_good_score(value, reference):
    """Outcome-blind empirical score: smaller dispersion is better."""
    ref = np.sort(np.asarray(reference, dtype=float))
    ref = ref[np.isfinite(ref)]
    if not np.isfinite(value) or len(ref) == 0:
        return 0.0
    left = np.searchsorted(ref, value, side="left")
    right = np.searchsorted(ref, value, side="right")
    return float(np.clip(1.0 - (left + right) / (2.0 * len(ref)), 0.0, 1.0))


def conviction_fields(probs5, market_row, feature_row, config):
    """Apply the frozen V3 reliability score; this never changes staking."""
    hours = float(market_row.hours_to_start) if pd.notna(market_row.hours_to_start) else 0.0
    # The historical morning-board protocol uses T-12h with a T-6h fallback.
    # A five-hour boundary separates that protocol from the T-4h snapshot.
    timing = "t12" if hours >= 5.0 else "t4"
    timing_cfg = config["timings"][timing]
    refs = timing_cfg["reference"]
    components = {
        "model_agreement": low_good_score(float(np.std(probs5, ddof=0)),
                                           refs["ensemble_prob_std"]),
        "market_agreement": (low_good_score(float(market_row.market_prob_std),
                                             refs["market_prob_std"])
                             if market_row.n_books >= 2 else 0.0),
        "book_breadth": float(np.clip((market_row.n_books - 1.0) / 4.0, 0.0, 1.0)),
        "history_depth": float(np.clip(
            pd.to_numeric(feature_row.p_games_prior, errors="coerce") / 50.0,
            0.0, 1.0)),
        "role_stability": float(np.clip(
            1.0 - abs(pd.to_numeric(feature_row.p_ip_pct_of_career_avg,
                                    errors="coerce") - 1.0) / 0.35,
            0.0, 1.0)),
    }
    components = {k: (v if np.isfinite(v) else 0.0) for k, v in components.items()}
    score = 100.0 * sum(config["weights"][k] * v for k, v in components.items())
    cuts = timing_cfg["grade_cutoffs"]
    grade = "A" if score >= cuts["A"] else ("B" if score >= cuts["B"] else "C")
    return timing, score, grade, components


def active_h3_weights(config, asof):
    """Most recent frozen monthly H3 weights available at score time."""
    if not config:
        return None
    eligible = [v for v in config.get("h3_vintages", [])
                if pd.Timestamp(v["cutoff"]) <= pd.Timestamp(asof)]
    return max(eligible, key=lambda v: v["cutoff"])["weights"] if eligible else None


def enforce_one_signal_per_pitcher(res):
    """Mirror backtest execution: keep only the largest edge per pitcher."""
    if res.empty or "signal" not in res or "pitcher_id" not in res:
        return res
    out = res.copy()
    signal = out.signal.isin(["OVER", "UNDER"])
    candidates = out[signal & out.pitcher_id.notna()].copy()
    if candidates.empty:
        out["suppressed_duplicate"] = False
        return out
    candidates["_abs_edge"] = pd.to_numeric(
        candidates.edge_over, errors="coerce").abs()
    keep = set(candidates.sort_values("_abs_edge", ascending=False)
                .drop_duplicates("pitcher_id", keep="first").index)
    suppressed = signal & ~out.index.isin(keep)
    out["suppressed_duplicate"] = suppressed
    out.loc[suppressed, "signal"] = "-"
    out.loc[suppressed, "tier"] = ""
    for col in ("stake_pct", "stake_usd", "stake_usd_conservative"):
        if col in out:
            out.loc[suppressed, col] = 0.0
    for col in ("paper_challenger", "upgrade_candidate"):
        if col in out:
            out.loc[suppressed, col] = False
    return out


def main():
    config = load_config("config/config_v4_production.yaml")
    deploy = json.loads((OUT / "deploy_config.json").read_text(encoding="utf-8"))
    conviction_path = Path("research/v3/conviction_config.json")
    conviction = (json.loads(conviction_path.read_text(encoding="utf-8"))
                  if conviction_path.exists() else None)
    upgrade_path = Path("research/v3/model_upgrade_config.json")
    upgrade = (json.loads(upgrade_path.read_text(encoding="utf-8"))
               if upgrade_path.exists() else None)
    h3_weights = active_h3_weights(upgrade, TODAY)
    selected_head = deploy.get("probability_head", "H1").upper()
    paper_only = deploy.get("deployment_status", "").startswith("NOT_DEPLOYABLE")
    print(f"Probability head: {selected_head}", flush=True)
    if upgrade:
        print(f"Paper router: {upgrade.get('paper_router')}", flush=True)
    if paper_only:
        print("DEPLOYMENT GUARD: model failed its completed forward gate; "
              "all suggested stakes are forced to $0 (paper tracking only).",
              flush=True)
    logs = load_pitcher_game_logs(config["data"]["pitcher_logs_file"])
    team_bat = load_team_batting_game_logs(config["data"]["team_batting_logs_file"])
    ctx = load_game_context_logs(config["data"]["game_context_logs_file"])
    batter_log = load_batter_game_logs(config["data"]["batter_game_logs_file"])
    statcast = load_statcast_pitcher_daily(config["data"]["statcast_pitcher_daily_file"])
    framing = load_statcast_catcher_framing_daily(config["data"].get("catcher_framing_file", ""))
    pc_map = load_statcast_pitcher_catcher_daily(config["data"].get("pitcher_catcher_file", ""))
    park = load_park_factors(config["data"].get("park_factors_file", ""))
    for _df in (statcast, framing, pc_map):
        _df.drop_duplicates(inplace=True)

    # placeholder rows for today's probables (their features use only PRIOR games)
    probs = pd.read_csv("data/raw/probable_pitchers.csv")
    probs = probs[probs.game_date.astype(str).str[:10] == str(TODAY.date())].copy()
    print(f"Probables today: {len(probs)}", flush=True)
    ph = pd.DataFrame({c: np.nan for c in logs.columns}, index=range(len(probs)))
    for c in ["pitcher_id", "pitcher_name", "team", "opponent", "is_home"]:
        ph[c] = probs[c].values
    ph["game_date"] = TODAY
    for c in logs.columns:
        if c not in ("game_date", "pitcher_id", "pitcher_name", "team", "opponent",
                     "is_home", "game_pk"):
            ph[c] = 0
    # placeholder rows must survive the IP>0 starter filter; their own stats
    # never enter their own features (shift(1) everywhere)
    ph["innings_pitched"] = 5.0
    ph["pitcher_id"] = ph["pitcher_id"].astype(str)
    logs2 = pd.concat([logs, ph], ignore_index=True, sort=False)

    featured, feat_cols, _ = build_training_features(
        logs2, rolling_windows=config["features"]["rolling_windows"],
        min_history_games=config["training"]["min_history_games"],
        min_starter_ip=config["training"].get("min_starter_ip"),
        team_batting_logs=team_bat, game_context_logs=ctx,
        batter_game_logs=batter_log, statcast_pitcher_daily=statcast,
        park_factors=park,
        catcher_framing_daily=framing if not framing.empty else None,
        pitcher_catcher_map=pc_map if not pc_map.empty else None,
        return_before_impute=True)
    featured["game_date"] = pd.to_datetime(featured["game_date"])

    meta = json.loads((OUT / "meta.json").read_text())
    fill = meta["fill_values"]
    feat_cols = [c for c in meta["feature_cols"] if c in featured.columns or c in
                 ("days_into_season", "month", "n_missing_feats")]
    featured["n_missing_feats"] = featured[[c for c in feat_cols if c in featured.columns]].isna().sum(axis=1)
    april1 = pd.to_datetime(featured["game_date"].dt.year.astype(str) + "-04-01")
    featured["days_into_season"] = (featured["game_date"] - april1).dt.days.clip(0, 200)
    featured["month"] = featured["game_date"].dt.month
    for c in feat_cols:
        if c not in featured.columns:
            featured[c] = np.nan
        featured[c] = pd.to_numeric(featured[c], errors="coerce")
        featured[c] = featured[c].fillna(fill.get(c, 0.0))

    count_feat_cols = [c for c in feat_cols if c != "line"]
    today_rows = featured[featured.game_date == TODAY].copy()
    today_rows = today_rows.drop_duplicates(subset=["pitcher_id"], keep="last")
    print(f"Feature rows for today: {len(today_rows)}", flush=True)
    if len(today_rows) == 0:
        print("No scoreable starters today (empty slate or all filtered) — exiting cleanly.",
              flush=True)
        return

    # ---- July vintage training set ----
    tr = featured[(featured.game_date < VINTAGE_CUTOFF)].copy()
    tr["strikeouts"] = pd.to_numeric(tr["strikeouts"], errors="coerce")
    tr = tr.dropna(subset=["strikeouts"])
    tr = tr[tr.game_date >= "2022-01-01"]
    print(f"Vintage training games: {len(tr):,}", flush=True)

    # ---- calibrators: folds A/B synthetic + 2025 real lines (same as 80) ----
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    tr["year"] = tr.game_date.dt.year
    oos_p = {n: [] for n in ["poisson_glm", "xgb_poisson", "lgb_poisson",
                             "hgb_poisson", "cat_poisson"]}
    oos_y = []
    for train_years, test_year in [([2022], 2023), ([2022, 2023], 2024)]:
        trg = tr[tr.year.isin(train_years)]
        teg = tr[tr.year == test_year]
        te_x = MS.expand_lines(teg, feat_cols)
        oos_y.append(te_x["y_over"].values)
        for name, cm in MS.make_count_models().items():
            cm.fit(trg[count_feat_cols], trg["strikeouts"])
            oos_p[name].append(cm.predict_prob(te_x[count_feat_cols], te_x["line"]))
    p25 = pd.read_parquet(OUT / "preds_2025.parquet")
    p25 = p25[p25.outcome_push == 0]
    oos_y.append(p25["outcome_over"].values)
    for n in oos_p:
        oos_p[n].append(p25[f"p_{n}"].values)
    y_cal = np.concatenate(oos_y)
    iso = {}
    for n in oos_p:
        pcat = np.clip(np.concatenate(oos_p[n]), 1e-6, 1 - 1e-6)
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
        ir.fit(pcat, y_cal)
        iso[n] = ir

    # ---- fit vintage models, project today's mu ----
    models = {}
    for name, cm in MS.make_count_models().items():
        cm.fit(tr[count_feat_cols], tr["strikeouts"])
        models[name] = cm
        today_rows[f"mu_{name}"] = cm.predict_mu(today_rows[count_feat_cols])
    today_rows["mu_mean"] = today_rows[[f"mu_{n}" for n in models]].mean(axis=1)

    # ---- trailing Platt from walk-forward rows + settled prediction-log rows ----
    ad = pd.read_parquet(OUT / "adaptive_preds_2026.parquet")
    ad["game_date"] = pd.to_datetime(ad["game_date"])
    trail = ad[(ad.game_date < PLATT_ASOF) &
               (ad.game_date >= PLATT_ASOF - pd.Timedelta(days=90))].dropna(subset=["p_h0"])
    trail = trail[["game_date", "p_h0", "outcome_over"]]
    if PRED_LOG.exists():
        pl = pd.read_csv(PRED_LOG)
        pl["game_date"] = pd.to_datetime(pl["game_date"])
        pl = pl[(pl.game_date < PLATT_ASOF) &
                (pl.game_date >= PLATT_ASOF - pd.Timedelta(days=90)) &
                (pl.game_date > ad.game_date.max())]
        if len(pl):
            settled = pl.merge(
                logs[["game_date", "pitcher_id", "strikeouts"]].assign(
                    pitcher_id=lambda d: pd.to_numeric(d.pitcher_id, errors="coerce")),
                on=["game_date", "pitcher_id"], how="inner")
            settled = settled[settled.strikeouts.notna()]
            settled["outcome_over"] = (settled.strikeouts > settled.line).astype(int)
            settled = settled[settled.strikeouts != settled.line]
            trail = pd.concat([trail, settled[["game_date", "p_h0", "outcome_over"]]],
                              ignore_index=True)
    platt = LogisticRegression(C=1e6, max_iter=1000)
    if len(trail) >= 400:
        platt.fit(logit(trail["p_h0"]).reshape(-1, 1), trail["outcome_over"].astype(int))
        use_platt = True
    else:
        use_platt = False
    print(f"Platt trailing rows: {len(trail)} (recal {'ON' if use_platt else 'OFF - identity'})",
          flush=True)

    # ---- odds: consensus + best price per (pitcher, line) ----
    o = pd.read_csv(os.environ.get("ODDS_OVERRIDE", "data/odds/pitcher_props.csv"))
    # GUARD: drop in-play lines (game already started when odds were fetched) —
    # live lines price the game state; the model only understands pre-game.
    if "commence_time" in o.columns and "fetched_at" in o.columns:
        commence = pd.to_datetime(o["commence_time"], utc=True, errors="coerce")
        fetched = pd.to_datetime(o["fetched_at"], utc=True, errors="coerce")
        live = (commence <= fetched)
        if live.any():
            print(f"GUARD: dropping {live.sum()} in-play odds rows "
                  f"({o.loc[live, 'pitcher_name'].nunique()} pitchers already started)",
                  flush=True)
            o = o[~live]
    o = o[o.game_date.astype(str).str[:10] == str(TODAY.date())].copy()
    o["line"] = pd.to_numeric(o["line"], errors="coerce")
    o = o.dropna(subset=["line", "over_odds", "under_odds"])
    po = np.where(o.over_odds < 0, -o.over_odds/(-o.over_odds+100), 100/(100+o.over_odds))
    pu = np.where(o.under_odds < 0, -o.under_odds/(-o.under_odds+100), 100/(100+o.under_odds))
    o["p_over_novig"] = po / (po + pu)
    if "commence_time" in o and "fetched_at" in o:
        commence = pd.to_datetime(o.commence_time, utc=True, errors="coerce")
        fetched = pd.to_datetime(o.fetched_at, utc=True, errors="coerce")
        o["hours_to_start"] = (commence - fetched).dt.total_seconds() / 3600.0
    else:
        o["hours_to_start"] = np.nan
    cons = o.groupby(["pitcher_id", "line"]).agg(
        p_mkt_over=("p_over_novig", "mean"), n_books=("bookmaker", "nunique"),
        market_prob_std=("p_over_novig", lambda x: float(np.std(x, ddof=0))),
        hours_to_start=("hours_to_start", "median"),
        best_over=("over_odds", "max"), best_under=("under_odds", "max")).reset_index()
    bo = o.loc[o.groupby(["pitcher_id", "line"])["over_odds"].idxmax(),
               ["pitcher_id", "line", "bookmaker"]].rename(columns={"bookmaker": "over_book"})
    bu = o.loc[o.groupby(["pitcher_id", "line"])["under_odds"].idxmax(),
               ["pitcher_id", "line", "bookmaker"]].rename(columns={"bookmaker": "under_book"})
    cons = cons.merge(bo, on=["pitcher_id", "line"]).merge(bu, on=["pitcher_id", "line"])

    today_rows["pitcher_id_n"] = pd.to_numeric(today_rows["pitcher_id"], errors="coerce")
    rows = []
    for _, r in today_rows.iterrows():
        my = cons[cons.pitcher_id == r.pitcher_id_n]
        if my.empty:
            rows.append({"pitcher": r.pitcher_name, "team": r.team, "opp": r.opponent,
                         "proj_ks": r.mu_mean, "line": None})
            continue
        for _, m in my.iterrows():
            probs5 = []
            probs_by_name = {}
            for name, cm in models.items():
                p = MS.prob_over_nb(np.array([r[f"mu_{name}"]]), cm.alpha,
                                    np.array([m.line]))[0]
                p_cal = iso[name].predict([np.clip(p, 1e-6, 1-1e-6)])[0]
                probs5.append(p_cal)
                probs_by_name[name] = p_cal
            p_h0 = float(np.mean(probs5))
            p_h1 = float(platt.predict_proba(logit([p_h0]).reshape(-1, 1))[0, 1]) \
                if use_platt else p_h0
            p_h3 = (float(sum(h3_weights[name] * probs_by_name[name]
                              for name in probs_by_name))
                    if h3_weights else p_h0)
            if conviction:
                conv_timing, conv_score, conv_grade, conv_components = conviction_fields(
                    probs5, m, r, conviction)
            else:
                conv_timing, conv_score, conv_grade, conv_components = (
                    "unavailable", np.nan, "", {})
            route = ((upgrade or {}).get("paper_router", {}).get(conv_timing, {})
                     if upgrade else {})
            row_head = route.get("probability_head", selected_head).upper()
            p_selected = {"H0": p_h0, "H1": p_h1, "H3": p_h3}.get(row_head, p_h0)
            raw_edge = p_selected - m.p_mkt_over
            edge_rule = route.get("edge_rule", "raw_8pp")
            if edge_rule == "lcb_1sd_8pp":
                model_std = float(np.std(probs5, ddof=0))
                edge = float(np.sign(raw_edge) * max(abs(raw_edge) - model_std, 0.0))
            else:
                edge = float(raw_edge)
            p_decision = float(m.p_mkt_over + edge)
            abs_edge = abs(raw_edge)
            paper_challenger = bool(
                conviction and row_head == "H0" and conv_timing == "t12" and
                0.10 <= abs_edge < 0.12 and conv_grade in ("A", "B"))
            upgrade_candidate = abs(edge) >= 0.08
            # shrunk half-Kelly stakes (deploy_config): p_eff = mkt + 0.25*(model-mkt)
            side = "over" if edge > 0 else "under"
            price = m.best_over if side == "over" else m.best_under
            b_dec = (price / 100.0) if price >= 0 else (100.0 / abs(price))
            p_model_side = p_decision if side == "over" else 1 - p_decision
            p_mkt_side = m.p_mkt_over if side == "over" else 1 - m.p_mkt_over
            def kelly_frac(lam):
                pe = p_mkt_side + lam * (p_model_side - p_mkt_side)
                return float(np.clip(0.5 * (pe * (1 + b_dec) - 1) / b_dec, 0, 0.02))
            rows.append({"pitcher": r.pitcher_name, "team": r.team, "opp": r.opponent,
                         "proj_ks": r.mu_mean, "line": m.line,
                         "best_over": m.best_over, "over_book": m.over_book,
                         "best_under": m.best_under, "under_book": m.under_book,
                         "p_model_over": p_selected, "p_decision_over": p_decision,
                         "p_h0": p_h0, "p_h3": p_h3,
                         "model_head": row_head, "edge_rule": edge_rule,
                         "p_mkt_over": m.p_mkt_over,
                         "raw_edge_over": raw_edge, "edge_over": edge,
                         "n_books": m.n_books,
                         "ensemble_prob_std": float(np.std(probs5, ddof=0)),
                         "market_prob_std": m.market_prob_std,
                         "odds_timing": conv_timing,
                         "conviction_score": conv_score,
                         "conviction_grade": conv_grade,
                         "conviction_status": ("CORROBORATED" if conv_grade in ("A", "B")
                                               else "LOW"),
                         "paper_challenger": paper_challenger,
                         "upgrade_candidate": upgrade_candidate,
                         "pitcher_id": r.pitcher_id_n,
                         "signal": "OVER" if edge >= 0.08 else
                                   ("UNDER" if edge <= -0.08 else "-"),
                         "tier": "PAPER-CONV" if paper_challenger else
                                 ("PAPER-V3.1" if upgrade_candidate else
                                 ("CONV" if abs(edge) >= 0.15 else
                                  ("base" if abs(edge) >= 0.08 else ""))),
                         "stake_pct": round(kelly_frac(0.25) * 100, 2),
                         "stake_usd": round(kelly_frac(0.25) * BANKROLL, 0),
                         "stake_usd_conservative": round(kelly_frac(0.15) * BANKROLL, 0)})
    res = pd.DataFrame(rows)
    if "edge_over" in res.columns:
        res = res.sort_values("edge_over", key=lambda s: s.abs(), ascending=False,
                              na_position="last")
    res = enforce_one_signal_per_pitcher(res)
    if paper_only and "stake_usd" in res.columns:
        res["stake_pct"] = 0.0
        res["stake_usd"] = 0.0
        res["stake_usd_conservative"] = 0.0
        # Rows for probables without a matching market intentionally have
        # blank signal/flag fields. Treat those as false, never as selections.
        signal_mask = res["signal"].isin(["OVER", "UNDER"])
        res.loc[signal_mask, "tier"] = "PAPER"
        if "paper_challenger" in res:
            paper_mask = res.paper_challenger.fillna(False).astype(bool)
            res.loc[signal_mask & paper_mask, "tier"] = "PAPER-CONV"
        if "upgrade_candidate" in res:
            upgrade_mask = res.upgrade_candidate.fillna(False).astype(bool)
            paper_mask = res.paper_challenger.fillna(False).astype(bool)
            res.loc[signal_mask & upgrade_mask & ~paper_mask,
                    "tier"] = "PAPER-V3.1"
    outp = Path(f"reports/daily/v2_props_{TODAY.date()}.csv")
    outp.parent.mkdir(parents=True, exist_ok=True)
    res.round(3).to_csv(outp, index=False)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False), flush=True)
    print(f"\nSaved {outp}", flush=True)

    # ---- plays-only view: every row that passed the model gate ----
    # Deployment may deliberately force every stake to $0.  Qualification is
    # therefore defined by the final de-duplicated signal, never by stake.
    if "stake_usd" in res.columns:
        plays = res[res.signal.isin(["OVER", "UNDER"])].copy()
        plays["side"] = plays.signal.str.lower()
        plays["price"] = np.where(plays.side == "over", plays.best_over, plays.best_under)
        plays["book"] = np.where(plays.side == "over", plays.over_book, plays.under_book)
        pcols = [
            "pitcher", "team", "opp", "line", "signal", "tier", "price", "book",
            "proj_ks", "p_model_over", "p_decision_over", "p_mkt_over",
            "edge_over", "n_books", "model_head", "edge_rule", "odds_timing",
            "conviction_score", "conviction_grade", "ensemble_prob_std",
            "market_prob_std", "stake_pct", "stake_usd",
            "stake_usd_conservative", "pitcher_id",
        ]
        plays_out = Path(f"reports/daily/v2_plays_{TODAY.date()}.csv")
        plays[pcols].round(3).to_csv(plays_out, index=False)
        print(f"\n=== PLAYS for {TODAY.date()} (bankroll ${BANKROLL:,.0f}) ===", flush=True)
        if len(plays):
            print(plays[pcols].round(3).to_string(index=False), flush=True)
        else:
            print("No qualifying plays.", flush=True)
        # Attach the independent qualitative score/writeup immediately so a
        # direct 92_today.py run produces the same complete output as the
        # scheduled wrapper.  Failure is visible but never changes the model
        # signal or deployment guard.
        try:
            import a7_qual_brief as qualitative
            q = qualitative.build_qualitative_outputs(str(TODAY.date()))
            print(f"Saved {plays_out}, qualitative writeups for "
                  f"{len(q['plays'])} plays, and reports/daily/v2_plays_latest.html",
                  flush=True)
        except Exception as exc:
            print(f"QUALITATIVE VALIDATION WARNING: {exc}", flush=True)
            print(f"Saved {plays_out}; qualitative layer did not alter any signal.",
                  flush=True)

    # ---- HITS ALLOWED paper-track (frozen: UNDER-only, edge>=0.04) ----
    # 2026 WF verdict (research/v2/98): CLV +1.1pp but ROI -2.3% — signal real,
    # doesn't clear the juice yet. Score + report daily, stake $0 until it
    # earns live sizing via sustained CLV.
    hits_odds_file = Path("data/odds/pitcher_props_hits.csv")
    hits_pkl = OUT / "hits_models.pkl"
    if hits_odds_file.exists() and hits_pkl.exists():
        import pickle
        with open(hits_pkl, "rb") as f:
            hb = pickle.load(f)
        hits_iso = hb["iso"]
        tr_h = tr.copy()
        tr_h["hits_allowed"] = pd.to_numeric(tr_h.get("hits_allowed"), errors="coerce")
        tr_h = tr_h.dropna(subset=["hits_allowed"])
        hmodels = {}
        for name, cm in MS.make_count_models().items():
            cm.fit(tr_h[count_feat_cols], tr_h["hits_allowed"])
            hmodels[name] = cm
        today_rows2 = today_rows.copy()
        for name, cm in hmodels.items():
            today_rows2[f"hmu_{name}"] = cm.predict_mu(today_rows2[count_feat_cols])
        today_rows2["hmu_mean"] = today_rows2[[f"hmu_{n}" for n in hmodels]].mean(axis=1)

        ho = pd.read_csv(hits_odds_file)
        if "commence_time" in ho.columns and "fetched_at" in ho.columns:
            live = (pd.to_datetime(ho["commence_time"], utc=True, errors="coerce") <=
                    pd.to_datetime(ho["fetched_at"], utc=True, errors="coerce"))
            ho = ho[~live]
        ho = ho[ho.game_date.astype(str).str[:10] == str(TODAY.date())].copy()
        ho["line"] = pd.to_numeric(ho["line"], errors="coerce")
        ho = ho.dropna(subset=["line", "over_odds", "under_odds"])
        if len(ho):
            hpo = np.where(ho.over_odds < 0, -ho.over_odds/(-ho.over_odds+100),
                           100/(100+ho.over_odds))
            hpu = np.where(ho.under_odds < 0, -ho.under_odds/(-ho.under_odds+100),
                           100/(100+ho.under_odds))
            ho["p_over_novig"] = hpo / (hpo + hpu)
            hcons = ho.groupby(["pitcher_id", "line"]).agg(
                p_mkt_over=("p_over_novig", "mean"), n_books=("bookmaker", "nunique"),
                best_under=("under_odds", "max")).reset_index()
            hrows = []
            for _, r in today_rows2.iterrows():
                my = hcons[hcons.pitcher_id == r.pitcher_id_n]
                for _, m in my.iterrows():
                    probs5 = [hits_iso[n].predict([np.clip(
                        MS.prob_over_nb(np.array([r[f"hmu_{n}"]]), hmodels[n].alpha,
                                        np.array([m.line]))[0], 1e-6, 1-1e-6)])[0]
                        for n in hmodels]
                    p_h = float(np.mean(probs5))
                    edge_under = (1 - p_h) - (1 - m.p_mkt_over)
                    if edge_under >= 0.04:
                        hrows.append({"pitcher": r.pitcher_name,
                                      "proj_hits": round(float(r.hmu_mean), 2),
                                      "line": m.line, "signal": "UNDER",
                                      "price": m.best_under,
                                      "p_model_under": round(1 - p_h, 3),
                                      "p_mkt_under": round(1 - m.p_mkt_over, 3),
                                      "edge": round(edge_under, 3),
                                      "n_books": int(m.n_books),
                                      "tier": "PAPER", "stake_usd": 0})
            hplays = pd.DataFrame(hrows)
            if len(hplays):
                hplays = (hplays.sort_values("edge", ascending=False)
                          .drop_duplicates(subset=["pitcher"], keep="first"))
                hplays.to_csv(Path(f"reports/daily/v2_hits_paper_{TODAY.date()}.csv"),
                              index=False)
                print(f"\n=== HITS-ALLOWED paper track ({TODAY.date()}) — DO NOT BET, "
                      f"tracking only ===", flush=True)
                print(hplays.to_string(index=False), flush=True)
            else:
                print("\nHITS paper track: no qualifying unders today.", flush=True)

    # ---- append to prediction log (feeds future trailing recalibration) ----
    if "line" in res.columns:
        lg = res.dropna(subset=["line"]).copy()
        lg = lg[["pitcher_id", "line", "p_h0", "p_mkt_over"]].copy()
        lg["game_date"] = str(TODAY.date())
        PRED_LOG.parent.mkdir(parents=True, exist_ok=True)
        if PRED_LOG.exists():
            old = pd.read_csv(PRED_LOG)
            old = old[old.game_date != str(TODAY.date())]
            lg = pd.concat([old, lg], ignore_index=True)
        lg.to_csv(PRED_LOG, index=False)
        print(f"Prediction log: {len(lg)} rows", flush=True)


if __name__ == "__main__":
    main()
