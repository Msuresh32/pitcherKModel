"""Outcome-blind qualitative validation for every deployment-qualified K prop.

This layer is deliberately downstream of the model gate.  It never changes a
signal, stake, model probability, or deployment status.  It uses only data
available before the score date, records both supporting and contradictory
evidence, and emits the required no-confirmation sentence when no independent
baseball support is present.

Inputs
------
reports/daily/v2_props_<date>.csv
data/raw/{pitcher_game_logs,team_batting_game_logs,game_context_logs,
          statcast_pitcher_daily,probable_pitchers,park_factors}.csv

Outputs
-------
reports/daily/v2_plays_<date>.csv
reports/daily/v2_qual_context_<date>.json
reports/daily/v2_qual_analysis_<date>.json
reports/daily/v2_plays_latest.html

An optional reports/daily/v2_qual_external_<date>.json can contribute verified
news/weather/lineup/arsenal findings.  The expected shape is documented in
qual_validation_prompt.md.  No external claim is invented when that file is
absent.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DAILY = ROOT / "reports" / "daily"
RAW = ROOT / "data" / "raw"
NO_CONFIRMATION = "No strong qualitative confirmation found beyond the model edge."
VERDICTS = {
    "Strong qualitative confirmation",
    "Moderate qualitative confirmation",
    "Neutral (model-only edge)",
    "Mixed signals",
    "Qualitative concerns",
}


@dataclass(frozen=True)
class Evidence:
    category: str
    text: str
    points: int


def _num(value, default=np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _read(path: Path, date_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    out = pd.read_csv(path, low_memory=False)
    for col in date_cols:
        if col in out:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _american(value) -> str:
    x = _num(value)
    if not np.isfinite(x):
        return "—"
    return f"{x:+.0f}"


def _pct(value, digits=1) -> str:
    x = _num(value)
    return "—" if not np.isfinite(x) else f"{100 * x:.{digits}f}%"


def _norm_id(value):
    x = _num(value)
    return int(x) if np.isfinite(x) else None


def _qualified_props(day: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    props_path = DAILY / f"v2_props_{day}.csv"
    if not props_path.exists():
        raise FileNotFoundError(f"No props file for {day}: {props_path}")
    props = pd.read_csv(props_path)
    if "signal" not in props:
        raise ValueError(f"{props_path} has no signal column")
    plays = props[props.signal.isin(["OVER", "UNDER"])].copy()
    if "suppressed_duplicate" in plays:
        suppressed = plays.suppressed_duplicate.astype(str).str.lower().eq("true")
        plays = plays[~suppressed]
    if len(plays):
        plays["side"] = plays.signal.str.lower()
        plays["price"] = np.where(
            plays.side.eq("over"), plays.best_over, plays.best_under)
        plays["book"] = np.where(
            plays.side.eq("over"), plays.over_book, plays.under_book)
    return props, plays


def _rate(block: pd.DataFrame, numerator: str) -> tuple[float, int, int]:
    if block.empty or numerator not in block or "plate_appearances" not in block:
        return np.nan, 0, 0
    pa = pd.to_numeric(block.plate_appearances, errors="coerce").sum()
    value = pd.to_numeric(block[numerator], errors="coerce").sum()
    return ((float(value / pa) if pa > 0 else np.nan), int(pa), len(block))


def _window_rates(team_games: pd.DataFrame, league_games: pd.DataFrame,
                  asof: pd.Timestamp) -> dict:
    out = {}
    specs = (("L7", 7, 90), ("L14", 14, 170), ("L30", 30, 320))
    for label, days, minimum_pa in specs:
        cutoff = asof - pd.Timedelta(days=days)
        mine = team_games[team_games.game_date >= cutoff]
        league = league_games[league_games.game_date >= cutoff]
        kr, pa, games = _rate(mine, "strikeouts")
        br, _, _ = _rate(mine, "walks")
        lkr, _, _ = _rate(league, "strikeouts")
        lbr, _, _ = _rate(league, "walks")
        out[label] = {
            "k_rate": kr, "bb_rate": br, "pa": pa, "games": games,
            "league_k_rate": lkr, "league_bb_rate": lbr,
            "usable": pa >= minimum_pa,
        }
    season_start = pd.Timestamp(year=asof.year, month=1, day=1)
    mine = team_games[team_games.game_date >= season_start]
    league = league_games[league_games.game_date >= season_start]
    kr, pa, games = _rate(mine, "strikeouts")
    br, _, _ = _rate(mine, "walks")
    lkr, _, _ = _rate(league, "strikeouts")
    lbr, _, _ = _rate(league, "walks")
    out["Season"] = {
        "k_rate": kr, "bb_rate": br, "pa": pa, "games": games,
        "league_k_rate": lkr, "league_bb_rate": lbr,
        "usable": pa >= 500,
    }
    return out


def _add_directional(evidence: list[Evidence], signal: str,
                     favorable_for_over: bool, category: str,
                     text: str, points: int) -> None:
    supports = (signal == "OVER") == favorable_for_over
    evidence.append(Evidence(category, text, points if supports else -points))


def _load_external(day: str) -> dict:
    path = DAILY / f"v2_qual_external_{day}.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for row in raw.get("plays", []):
        try:
            key = (str(row.get("pitcher")), float(row.get("line")))
        except (TypeError, ValueError):
            continue
        out[key] = row
    return out


class QualitativeValidator:
    """Build independent, pregame evidence without altering model selection."""

    def __init__(self, day: str):
        self.day = str(day)
        self.asof = pd.Timestamp(self.day)
        self.logs = _read(RAW / "pitcher_game_logs.csv", ("game_date",))
        self.team = _read(RAW / "team_batting_game_logs.csv", ("game_date",))
        self.ctx = _read(RAW / "game_context_logs.csv", ("game_date",))
        self.statcast = _read(RAW / "statcast_pitcher_daily.csv", ("game_date",))
        self.probables = _read(RAW / "probable_pitchers.csv", ("game_date",))
        self.parks = _read(RAW / "park_factors.csv")
        self.external = _load_external(self.day)

        # Strictly pregame.  This prevents same-day results, velocity and pitch
        # counts from leaking into the writeup when the script is rerun later.
        for frame in (self.logs, self.team, self.ctx, self.statcast):
            if len(frame) and "game_date" in frame:
                frame.drop(frame[frame.game_date >= self.asof].index, inplace=True)
        if len(self.probables):
            self.probables = self.probables[
                self.probables.game_date.dt.strftime("%Y-%m-%d").eq(self.day)]

        for frame, cols in (
            (self.logs, ("pitcher_id", "team", "opponent", "is_home")),
            (self.team, ("team", "opponent", "is_home")),
            (self.ctx, ("pitcher_id", "team", "opponent")),
            (self.statcast, ("pitcher_id",)),
            (self.probables, ("pitcher_id", "team", "opponent", "is_home")),
        ):
            for col in cols:
                if len(frame) and col in frame:
                    frame[col] = pd.to_numeric(frame[col], errors="coerce")

        self.team_hand = self._team_games_vs_starter_hand()
        self.venue_by_home_team = self._venue_map()

    def _team_games_vs_starter_hand(self) -> pd.DataFrame:
        if self.team.empty or self.ctx.empty or "pitcher_hand" not in self.ctx:
            return pd.DataFrame()
        starters = self.ctx[["game_pk", "opponent", "pitcher_hand"]].copy()
        starters = starters.rename(columns={"opponent": "team"})
        starters = starters.dropna(subset=["game_pk", "team", "pitcher_hand"])
        starters = starters.drop_duplicates(["game_pk", "team"], keep="last")
        return self.team.merge(starters, on=["game_pk", "team"], how="left")

    def _venue_map(self) -> dict[int, int]:
        if self.logs.empty or self.ctx.empty or "venue_id" not in self.ctx:
            return {}
        home = self.logs[pd.to_numeric(self.logs.is_home, errors="coerce").eq(1)]
        home = home[["game_pk", "pitcher_id", "team", "game_date"]]
        venues = self.ctx[["game_pk", "pitcher_id", "venue_id"]].drop_duplicates(
            ["game_pk", "pitcher_id"], keep="last")
        home = home.merge(venues, on=["game_pk", "pitcher_id"], how="inner")
        home = home[home.game_date >= self.asof - pd.Timedelta(days=550)]
        result = {}
        for team_id, group in home.dropna(subset=["team", "venue_id"]).groupby("team"):
            mode = pd.to_numeric(group.venue_id, errors="coerce").dropna().mode()
            if len(mode):
                result[int(team_id)] = int(mode.iloc[0])
        return result

    def _probable(self, play: pd.Series) -> pd.Series | None:
        if self.probables.empty:
            return None
        pid = _norm_id(play.get("pitcher_id"))
        match = self.probables[self.probables.pitcher_id.eq(pid)] if pid else pd.DataFrame()
        if match.empty:
            match = self.probables[
                self.probables.pitcher_name.astype(str).eq(str(play.pitcher))]
        return match.iloc[0] if len(match) else None

    def _pitcher_hand(self, pitcher_id: int | None) -> str | None:
        if pitcher_id is None or self.ctx.empty or "pitcher_hand" not in self.ctx:
            return None
        rows = self.ctx[self.ctx.pitcher_id.eq(pitcher_id)].sort_values("game_date")
        if rows.empty:
            return None
        hand = str(rows.pitcher_hand.dropna().iloc[-1]).upper()
        return hand if hand in ("L", "R") else None

    def _park(self, home_team: int | None) -> tuple[float, str] | None:
        if home_team is None or self.parks.empty:
            return None
        venue = self.venue_by_home_team.get(home_team)
        if venue is None:
            return None
        rows = self.parks[pd.to_numeric(self.parks.venue_id, errors="coerce").eq(venue)]
        rows = rows[pd.to_numeric(rows.factor_year, errors="coerce") <= self.asof.year]
        if rows.empty:
            return None
        row = rows.sort_values("factor_year").iloc[-1]
        factor = _num(row.get("park_so_factor"))
        return (factor, str(row.get("venue_name", "the park"))) if np.isfinite(factor) else None

    def evaluate(self, play: pd.Series) -> tuple[dict, dict]:
        signal = str(play.signal).upper()
        pitcher_id = _norm_id(play.get("pitcher_id"))
        line = _num(play.get("line"))
        probable = self._probable(play)
        team_id = (_norm_id(play.get("team")) if probable is None
                   else _norm_id(probable.get("team")))
        opp_id = (_norm_id(play.get("opp")) if probable is None
                  else _norm_id(probable.get("opponent")))
        is_home = (_num(play.get("is_home")) if probable is None
                   else _num(probable.get("is_home")))
        is_home = int(is_home) if np.isfinite(is_home) else None
        pitcher_hand = self._pitcher_hand(pitcher_id)

        evidence: list[Evidence] = []
        available: list[str] = []
        unavailable = ["confirmed lineup", "injury/news", "weather/roof",
                       "announced umpire", "bullpen rest", "pitch-type matchup"]
        context: dict = {
            "pitcher": str(play.pitcher), "line": line, "signal": signal,
            "pitcher_id": pitcher_id, "team_id": team_id,
            "opponent_team_id": opp_id, "is_home": is_home,
            "pitcher_hand": pitcher_hand,
        }

        # ---- opponent contact and patience ----
        team_games = (self.team[self.team.team.eq(opp_id)].sort_values("game_date")
                      if opp_id is not None and not self.team.empty else pd.DataFrame())
        league_games = self.team.sort_values("game_date") if not self.team.empty else pd.DataFrame()
        if len(team_games):
            available.append("opponent K/BB profile")
            rates = _window_rates(team_games, league_games, self.asof)
            context["opponent_rates"] = rates
            usable = {k: v for k, v in rates.items()
                      if k in ("L7", "L14", "L30") and v["usable"] and
                      np.isfinite(v["k_rate"]) and np.isfinite(v["league_k_rate"])}
            deltas = {k: v["k_rate"] - v["league_k_rate"] for k, v in usable.items()}
            high = [k for k, d in deltas.items() if d >= .018]
            low = [k for k, d in deltas.items() if d <= -.018]
            if len(high) >= 2 or ("L14" in deltas and deltas["L14"] >= .035):
                labels = [k for k in ("L7", "L14", "L30") if k in usable]
                detail = ", ".join(f"{k} {_pct(usable[k]['k_rate'])}" for k in labels)
                base = np.nanmean([usable[k]["league_k_rate"] for k in labels])
                _add_directional(
                    evidence, signal, True, "matchup",
                    f"Opponent strikeout rate is consistently elevated ({detail}; "
                    f"league baseline about {_pct(base)}).", 11)
            elif len(low) >= 2 or ("L14" in deltas and deltas["L14"] <= -.035):
                labels = [k for k in ("L7", "L14", "L30") if k in usable]
                detail = ", ".join(f"{k} {_pct(usable[k]['k_rate'])}" for k in labels)
                base = np.nanmean([usable[k]["league_k_rate"] for k in labels])
                _add_directional(
                    evidence, signal, False, "matchup",
                    f"Opponent has made contact at an above-average rate ({detail}; "
                    f"league strikeout baseline about {_pct(base)}).", 11)

            l14 = rates.get("L14", {})
            if l14.get("usable") and np.isfinite(l14.get("bb_rate", np.nan)) and \
                    np.isfinite(l14.get("league_bb_rate", np.nan)):
                bb_delta = l14["bb_rate"] - l14["league_bb_rate"]
                if bb_delta >= .018:
                    _add_directional(
                        evidence, signal, False, "matchup",
                        f"Opponent's L14 walk rate is {_pct(l14['bb_rate'])} versus "
                        f"a {_pct(l14['league_bb_rate'])} league baseline, creating "
                        "pitch-count pressure that can shorten the starter's runway.", 5)
                elif bb_delta <= -.018:
                    _add_directional(
                        evidence, signal, True, "matchup",
                        f"Opponent's L14 walk rate is only {_pct(l14['bb_rate'])} versus "
                        f"a {_pct(l14['league_bb_rate'])} league baseline, helping the "
                        "starter work efficiently and preserve strikeout opportunities.", 4)

            # Current home/road offense split, only with a real sample and a
            # difference that is meaningful against both its own total and MLB.
            if is_home is not None:
                opp_home = 0 if is_home else 1
                season = team_games[team_games.game_date.dt.year.eq(self.asof.year)]
                loc = season[pd.to_numeric(season.is_home, errors="coerce").eq(opp_home)]
                loc_kr, loc_pa, loc_games = _rate(loc, "strikeouts")
                all_kr, _, _ = _rate(season, "strikeouts")
                league_kr = rates.get("Season", {}).get("league_k_rate", np.nan)
                if loc_games >= 10 and loc_pa >= 320 and np.isfinite(all_kr):
                    delta = loc_kr - all_kr
                    if delta >= .02 and loc_kr - league_kr >= .01:
                        _add_directional(
                            evidence, signal, True, "matchup",
                            f"In the opponent's current {'home' if opp_home else 'road'} "
                            f"split, its strikeout rate rises to {_pct(loc_kr)} across "
                            f"{loc_games} games (season {_pct(all_kr)}).", 5)
                    elif delta <= -.02 and loc_kr - league_kr <= -.01:
                        _add_directional(
                            evidence, signal, False, "matchup",
                            f"In the opponent's current {'home' if opp_home else 'road'} "
                            f"split, its strikeout rate falls to {_pct(loc_kr)} across "
                            f"{loc_games} games (season {_pct(all_kr)}).", 5)

        # Handedness is game-level performance in games started by that hand;
        # it is never mislabeled as pitch-level performance against all relievers.
        if pitcher_hand and not self.team_hand.empty and opp_id is not None:
            season = self.team_hand[
                self.team_hand.game_date.dt.year.eq(self.asof.year) &
                self.team_hand.team.eq(opp_id) &
                self.team_hand.pitcher_hand.astype(str).str.upper().eq(pitcher_hand)]
            league = self.team_hand[
                self.team_hand.game_date.dt.year.eq(self.asof.year) &
                self.team_hand.pitcher_hand.astype(str).str.upper().eq(pitcher_hand)]
            hand_kr, hand_pa, hand_games = _rate(season, "strikeouts")
            league_kr, _, _ = _rate(league, "strikeouts")
            if hand_games >= 8 and hand_pa >= 250 and np.isfinite(league_kr):
                available.append("opponent starter-handedness split")
                context["opponent_vs_starter_hand"] = {
                    "hand": pitcher_hand, "k_rate": hand_kr,
                    "league_k_rate": league_kr, "games": hand_games, "pa": hand_pa,
                }
                delta = hand_kr - league_kr
                if delta >= .02:
                    _add_directional(
                        evidence, signal, True, "matchup",
                        f"In {hand_games} games started by {pitcher_hand}HP this season, "
                        f"the opponent has struck out in {_pct(hand_kr)} of plate "
                        f"appearances (league {_pct(league_kr)}).", 7)
                elif delta <= -.02:
                    _add_directional(
                        evidence, signal, False, "matchup",
                        f"In {hand_games} games started by {pitcher_hand}HP this season, "
                        f"the opponent has struck out in only {_pct(hand_kr)} of plate "
                        f"appearances (league {_pct(league_kr)}).", 7)

        # ---- pitcher form, workload and location ----
        games = (self.logs[self.logs.pitcher_id.eq(pitcher_id)].sort_values("game_date")
                 if pitcher_id is not None and not self.logs.empty else pd.DataFrame())
        if len(games):
            recent = games.tail(10)
            last5 = recent.tail(5)
            ks = pd.to_numeric(last5.strikeouts, errors="coerce")
            side_hits = (ks > line) if signal == "OVER" else (ks < line)
            hit_count = int(side_hits.sum())
            if len(ks) >= 4:
                available.append("pitcher recent form")
            context["recent_form"] = {
                "last_start_date": games.game_date.max().strftime("%Y-%m-%d"),
                "days_since_last_start": int((self.asof - games.game_date.max()).days),
                "ks_last5": [int(x) for x in ks.dropna()],
                "ip_last5": [round(float(x), 2) for x in
                              pd.to_numeric(last5.innings_pitched, errors="coerce").dropna()],
                "pitches_last5": [int(x) for x in
                                   pd.to_numeric(last5.pitches, errors="coerce").dropna()],
                "side_hit_count_last5": hit_count,
            }
            if len(ks) >= 4 and hit_count >= 4:
                evidence.append(Evidence(
                    "pitcher_form",
                    f"The pitcher finished on the {signal.lower()} side of {line:.1f} "
                    f"strikeouts in {hit_count} of his last {len(ks)} starts "
                    f"({', '.join(str(int(x)) for x in ks)} Ks).", 9))
            elif len(ks) >= 4 and hit_count <= 1:
                evidence.append(Evidence(
                    "pitcher_form",
                    f"The pitcher finished on the opposite side of this play in "
                    f"{len(ks) - hit_count} of his last {len(ks)} starts "
                    f"({', '.join(str(int(x)) for x in ks)} Ks).", -9))

            pitches = pd.to_numeric(recent.pitches, errors="coerce")
            innings = pd.to_numeric(recent.innings_pitched, errors="coerce")
            avg_pitches = float(pitches.mean())
            avg_ip = float(innings.mean())
            short = int((innings < 5).sum())
            context["workload"] = {
                "avg_pitches_l10": avg_pitches, "avg_ip_l10": avg_ip,
                "short_starts_under5ip_l10": short,
            }
            if len(recent) >= 4:
                available.append("pitcher workload")
                if avg_pitches >= 94 and avg_ip >= 5.7 and short <= 2:
                    _add_directional(
                        evidence, signal, True, "workload",
                        f"The starter has a durable recent leash: {avg_pitches:.0f} pitches "
                        f"and {avg_ip:.1f} innings per start over his last {len(recent)}, "
                        f"with only {short} start{'s' if short != 1 else ''} under five innings.", 9)
                elif avg_pitches <= 85 or avg_ip <= 5.0 or short >= 4:
                    _add_directional(
                        evidence, signal, False, "workload",
                        f"Recent workload is limited: {avg_pitches:.0f} pitches and "
                        f"{avg_ip:.1f} innings per start over his last {len(recent)}, with "
                        f"{short} start{'s' if short != 1 else ''} under five innings.", 9)

            rest = int((self.asof - games.game_date.max()).days)
            if rest >= 9:
                _add_directional(
                    evidence, signal, False, "workload",
                    f"The pitcher enters on an extended {rest}-day layoff, adding "
                    "uncertainty around sharpness and the expected pitch limit.", 5)
            elif rest <= 4:
                _add_directional(
                    evidence, signal, False, "workload",
                    f"The pitcher is working on only {rest} days since his previous "
                    "start, a modest fatigue/runway concern.", 4)

            if is_home is not None:
                season = games[games.game_date.dt.year.eq(self.asof.year)]
                loc = season[pd.to_numeric(season.is_home, errors="coerce").eq(is_home)]
                other = season[pd.to_numeric(season.is_home, errors="coerce").ne(is_home)]
                if len(loc) >= 5 and len(other) >= 5:
                    loc_k = pd.to_numeric(loc.strikeouts, errors="coerce").mean()
                    other_k = pd.to_numeric(other.strikeouts, errors="coerce").mean()
                    delta = loc_k - other_k
                    if abs(delta) >= 1.0:
                        _add_directional(
                            evidence, signal, delta > 0, "pitcher_split",
                            f"The pitcher's meaningful {'home' if is_home else 'road'} "
                            f"split is {loc_k:.1f} Ks/start across {len(loc)} starts "
                            f"versus {other_k:.1f} in the opposite split.", 5)

        # ---- velocity and bat-missing indicators ----
        sc = (self.statcast[self.statcast.pitcher_id.eq(pitcher_id)]
              .sort_values("game_date") if pitcher_id is not None and
              not self.statcast.empty else pd.DataFrame())
        if len(sc) >= 4:
            available.append("Statcast pitcher trend")
            recent_sc = sc.tail(10)
            last3 = recent_sc.tail(3)
            prior = recent_sc.iloc[:-3]
            velo3 = pd.to_numeric(last3.avg_release_speed, errors="coerce").mean()
            velop = pd.to_numeric(prior.avg_release_speed, errors="coerce").mean()
            swstr = pd.to_numeric(last3.swinging_strike_rate, errors="coerce").mean()
            csw = pd.to_numeric(last3.csw_rate, errors="coerce").mean()
            context["statcast"] = {
                "avg_pitch_velo_l3": _num(velo3),
                "avg_pitch_velo_previous": _num(velop),
                "velo_change": _num(velo3 - velop),
                "swstr_l3": _num(swstr), "csw_l3": _num(csw),
            }
            if np.isfinite(velo3) and np.isfinite(velop) and abs(velo3 - velop) >= .8:
                rising = velo3 > velop
                _add_directional(
                    evidence, signal, rising, "pitcher_stuff",
                    f"Average pitch velocity moved {velo3 - velop:+.1f} mph over the "
                    "last three starts versus the preceding sample, a material "
                    f"{'uptick' if rising else 'decline'} in current stuff.", 7)
            if np.isfinite(swstr) and np.isfinite(csw):
                if swstr >= .13 and csw >= .285:
                    _add_directional(
                        evidence, signal, True, "pitcher_stuff",
                        f"Bat-missing indicators are strong over the last three starts "
                        f"({_pct(swstr)} swinging strikes, {_pct(csw)} CSW).", 9)
                elif swstr <= .09 and csw <= .255:
                    _add_directional(
                        evidence, signal, False, "pitcher_stuff",
                        f"Bat-missing indicators are weak over the last three starts "
                        f"({_pct(swstr)} swinging strikes, {_pct(csw)} CSW).", 9)

        # ---- park context ----
        home_team = team_id if is_home == 1 else opp_id
        park = self._park(home_team)
        if park is not None:
            available.append("park strikeout factor")
            factor, venue_name = park
            context["park"] = {"venue": venue_name, "so_factor": factor}
            if factor >= 1.05:
                _add_directional(
                    evidence, signal, True, "park",
                    f"{venue_name} has a strikeout park factor of {factor:.2f}, a "
                    "meaningful environment boost for pitcher strikeouts.", 5)
            elif factor <= .95:
                _add_directional(
                    evidence, signal, False, "park",
                    f"{venue_name} has a strikeout park factor of {factor:.2f}, a "
                    "meaningful suppressive environment for pitcher strikeouts.", 5)

        # ---- market quality (never used as directional confirmation) ----
        available.append("current market consensus")
        n_books = int(_num(play.get("n_books"), 0))
        market_std = _num(play.get("market_prob_std"))
        context["market"] = {
            "n_books": n_books, "market_probability_std": market_std,
            "opening_move_available": False,
        }
        if n_books <= 2:
            evidence.append(Evidence(
                "market",
                f"Only {n_books} sportsbook{' is' if n_books == 1 else 's are'} "
                "posting this number, so the consensus reference is thin.", -4))
        elif np.isfinite(market_std) and market_std >= .025:
            evidence.append(Evidence(
                "market",
                f"Sportsbooks show material probability disagreement "
                f"(cross-book standard deviation {_pct(market_std)}), weakening "
                "confidence in a single consensus price.", -4))

        # Optional externally verified findings.  These require explicit text;
        # the deterministic layer never manufactures news, weather, lineup or
        # arsenal claims.
        ext = self.external.get((str(play.pitcher), float(line)))
        if ext:
            available.append("verified external context")
            unavailable = [x for x in unavailable if x not in ext.get("covers", [])]
            for text in ext.get("support", []):
                if str(text).strip():
                    evidence.append(Evidence("external", str(text).strip(), 6))
            for text in ext.get("contradictory", []):
                if str(text).strip():
                    evidence.append(Evidence("external", str(text).strip(), -6))
            context["external_sources"] = ext.get("sources", [])

        supporting = [e for e in evidence if e.points > 0]
        contradictory = [e for e in evidence if e.points < 0]
        support_points = sum(e.points for e in supporting)
        concern_points = -sum(e.points for e in contradictory)
        qual_score = int(np.clip(round(50 + support_points - concern_points), 0, 100))

        if not supporting:
            verdict = ("Qualitative concerns" if concern_points >= 8
                       else "Neutral (model-only edge)")
        elif support_points >= 20 and qual_score >= 70 and concern_points <= 7 \
                and len(supporting) >= 3:
            verdict = "Strong qualitative confirmation"
        elif qual_score >= 58 and support_points >= 8 and \
                support_points >= concern_points + 5:
            verdict = "Moderate qualitative confirmation"
        elif concern_points >= support_points + 5:
            verdict = "Qualitative concerns"
        else:
            verdict = "Mixed signals"
        assert verdict in VERDICTS
        flag = {
            "Strong qualitative confirmation": "SUPPORT",
            "Moderate qualitative confirmation": "SUPPORT",
            "Neutral (model-only edge)": "NEUTRAL",
            "Mixed signals": "MIXED",
            "Qualitative concerns": "CONCERN",
        }[verdict]

        available = list(dict.fromkeys(available))
        coverage_label = f"{len(available)} internal/verified areas checked"
        summary = self._model_summary(play)
        analysis = {
            "pitcher": str(play.pitcher), "line": float(line),
            "signal": signal, "qualitative_score": qual_score,
            "qual_score": qual_score, "qual_flag": flag,
            "verdict": verdict, "qual_verdict": verdict,
            "support_points": support_points,
            "concern_points": concern_points,
            "support": [e.text for e in supporting],
            "contradictory": [e.text for e in contradictory],
            "coverage": {
                "label": coverage_label, "available": available,
                "unavailable_not_inferred": unavailable,
            },
            "model_summary": summary,
        }
        analysis["html"] = self._html(analysis)
        context["evidence"] = [e.__dict__ for e in evidence]
        context["coverage"] = analysis["coverage"]
        context["score_breakdown"] = {
            "neutral_start": 50, "support_points": support_points,
            "concern_points": concern_points, "qualitative_score": qual_score,
        }
        return analysis, context

    def _model_summary(self, play: pd.Series) -> str:
        signal = str(play.signal).upper()
        edge = abs(_num(play.get("edge_over")))
        model_p = _num(play.get("p_decision_over", play.get("p_model_over")))
        market_p = _num(play.get("p_mkt_over"))
        if signal == "UNDER":
            model_p, market_p = 1 - model_p, 1 - market_p
        grade = str(play.get("conviction_grade", "")).strip()
        conv = _num(play.get("conviction_score"))
        confidence = (f"model confidence {grade} ({conv:.1f}/100)" if grade and
                      np.isfinite(conv) else str(play.get("tier", "model-qualified")))
        stake = _num(play.get("stake_usd"), 0)
        stake_status = ("PAPER — deployment guard active" if stake <= 0
                        else f"${stake:.0f} suggested stake")
        return (
            f"Projection {_num(play.get('proj_ks')):.2f} Ks · sportsbook line "
            f"{_num(play.get('line')):.1f} · {signal} {_american(play.get('price'))} "
            f"at {play.get('book', '—')} · model edge {100 * edge:.1f}pp · "
            f"model/market {100 * model_p:.1f}%/{100 * market_p:.1f}% · "
            f"{confidence} · {stake_status}"
        )

    @staticmethod
    def _html(analysis: dict) -> str:
        def items(values):
            return "<ul>" + "".join(f"<li>{html.escape(v)}</li>" for v in values) + "</ul>"
        support = (items(analysis["support"]) if analysis["support"] else
                   f"<p><em>{NO_CONFIRMATION}</em></p>")
        contrary = (items(analysis["contradictory"]) if analysis["contradictory"] else
                    "<p>No material contradictory evidence found in the available data.</p>")
        unavailable = ", ".join(analysis["coverage"]["unavailable_not_inferred"])
        return (
            f"<h4>Model Summary</h4><p>{html.escape(analysis['model_summary'])}</p>"
            f"<h4>Qualitative Confirmation</h4>{support}"
            f"<h4>Contradictory Evidence</h4>{contrary}"
            f"<h4>Verdict</h4><p><span class=\"verdict\">"
            f"{html.escape(analysis['verdict'])}</span> · qualitative conviction "
            f"<b>{analysis['qualitative_score']}/100</b></p>"
            f"<p class=\"coverage\">Coverage: "
            f"{html.escape(analysis['coverage']['label'])}. Not available and not "
            f"inferred: {html.escape(unavailable)}. This sanity check does not alter "
            "the qualifying model signal or stake.</p>"
        )


def _write_latest_html(day: str, plays: pd.DataFrame, analyses: list[dict]) -> None:
    amap = {(a["pitcher"], float(a["line"])): a for a in analyses}
    cards = []
    for _, play in plays.iterrows():
        key = (str(play.pitcher), float(play.line))
        a = amap[key]
        flag = a["qual_flag"].lower()
        cards.append(
            f"<article class=\"play {flag}\"><header><div><span class=\"side "
            f"{str(play.signal).lower()}\">{html.escape(str(play.signal))}</span> "
            f"<h2>{html.escape(str(play.pitcher))} {float(play.line):.1f} Ks</h2>"
            f"<p>{html.escape(str(play.get('book', '—')))} "
            f"{_american(play.get('price'))} · edge "
            f"{100 * abs(_num(play.get('edge_over'))):.1f}pp</p></div>"
            f"<div class=\"score\"><b>{a['qualitative_score']}</b><span>QUAL</span>"
            f"</div></header><div class=\"body\">{a['html']}</div></article>"
        )
    body = "".join(cards) if cards else "<p class=\"empty\">No qualifying plays.</p>"
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qualitative validation — {day}</title><style>
:root{{--bg:#07111d;--panel:#0d1c2d;--line:#24384c;--ink:#e8f3ff;--muted:#8ca2b6;
--cyan:#30dce1;--lime:#a5ef38;--amber:#ffb320;--red:#ff6477}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 75% -20%,#173b5b,transparent 42%),var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:30px 18px 70px}}h1{{margin:0;font-size:28px}}.sub{{color:var(--muted);margin:4px 0 24px}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}}
.play{{background:linear-gradient(145deg,var(--panel),#10253a);border:1px solid var(--line);border-radius:13px;overflow:hidden;box-shadow:0 18px 50px #0005}}.play.support{{border-color:#a5ef3855}}.play.concern{{border-color:#ff647755}}.play.mixed{{border-color:#ffb32055}}
header{{display:flex;justify-content:space-between;gap:14px;padding:17px;border-bottom:1px solid var(--line)}}header h2{{display:inline;font-size:17px;margin-left:8px}}header p{{color:var(--muted);margin:7px 0 0}}.side{{font:700 10px monospace;padding:3px 7px;border-radius:12px;border:1px solid var(--line)}}.side.over{{color:var(--cyan)}}.side.under{{color:#ae9cff}}
.score{{width:58px;height:58px;border:1px solid var(--cyan);border-radius:12px;display:grid;place-content:center;text-align:center;color:var(--cyan);flex:none}}.score b{{font-size:23px;line-height:1}}.score span{{font:9px monospace;letter-spacing:.13em}}.body{{padding:4px 17px 17px}}h4{{font:10px monospace;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);margin:15px 0 6px}}p{{margin:6px 0}}ul{{margin:6px 0;padding-left:20px}}li{{margin:5px 0}}.verdict{{font-weight:800;color:var(--lime)}}.concern .verdict{{color:var(--red)}}.mixed .verdict{{color:var(--amber)}}.coverage{{font-size:11px;color:var(--muted);border-top:1px solid var(--line);padding-top:10px;margin-top:12px}}.empty{{color:var(--muted)}}
@media(max-width:800px){{.grid{{grid-template-columns:1fr}}main{{padding:20px 10px 50px}}}}
</style></head><body><main><h1>Pregame qualitative validation</h1>
<p class="sub">{day} · every deployment-qualified strikeout prop · independent sanity check · model gate unchanged</p>
<div class="grid">{body}</div></main></body></html>"""
    (DAILY / "v2_plays_latest.html").write_text(page, encoding="utf-8")


def build_qualitative_outputs(day: str | None = None) -> dict:
    day = str(day or os.environ.get("SCORE_DATE") or
              pd.Timestamp.now().strftime("%Y-%m-%d"))
    props, plays = _qualified_props(day)
    validator = QualitativeValidator(day)
    analyses, contexts = [], []
    for _, play in plays.iterrows():
        analysis, context = validator.evaluate(play)
        analyses.append(analysis)
        contexts.append(context)

    methodology = {
        "selection": "all rows already passing the model deployment threshold",
        "effect_on_model": "none; qualitative output never changes signal or stake",
        "score": "50 neutral + material support points - material concern points",
        "data_cutoff": f"strictly before {day}",
        "required_no_confirmation_text": NO_CONFIRMATION,
    }
    context_payload = {"date": day, "methodology": methodology, "plays": contexts}
    analysis_payload = {"date": day, "methodology": methodology, "plays": analyses}
    (DAILY / f"v2_qual_context_{day}.json").write_text(
        json.dumps(context_payload, indent=2, allow_nan=False), encoding="utf-8")
    (DAILY / f"v2_qual_analysis_{day}.json").write_text(
        json.dumps(analysis_payload, indent=2, allow_nan=False), encoding="utf-8")

    # The plays CSV is the compact machine-readable view used by the artifact.
    amap = {(a["pitcher"], float(a["line"])): a for a in analyses}
    for idx, play in plays.iterrows():
        a = amap[(str(play.pitcher), float(play.line))]
        plays.loc[idx, "qual_score"] = a["qualitative_score"]
        plays.loc[idx, "qual_verdict"] = a["verdict"]
        plays.loc[idx, "qual_flag"] = a["qual_flag"]
        plays.loc[idx, "qual_coverage"] = a["coverage"]["label"]
    keep = [
        "pitcher", "team", "opp", "line", "signal", "tier", "price", "book",
        "proj_ks", "p_model_over", "p_decision_over", "p_mkt_over", "edge_over",
        "n_books", "model_head", "edge_rule", "odds_timing", "conviction_score",
        "conviction_grade", "ensemble_prob_std", "market_prob_std", "stake_pct",
        "stake_usd", "stake_usd_conservative", "pitcher_id", "qual_score",
        "qual_verdict", "qual_flag", "qual_coverage",
    ]
    keep = [c for c in keep if c in plays]
    plays[keep].round(3).to_csv(DAILY / f"v2_plays_{day}.csv", index=False)
    _write_latest_html(day, plays, analyses)
    return analysis_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=os.environ.get("SCORE_DATE"))
    args = parser.parse_args()
    result = build_qualitative_outputs(args.date)
    counts = pd.Series([p["verdict"] for p in result["plays"]]).value_counts()
    detail = ", ".join(f"{k}: {v}" for k, v in counts.items())
    print(f"wrote qualitative validation for {result['date']} "
          f"({len(result['plays'])} plays{'; ' + detail if detail else ''})")


if __name__ == "__main__":
    main()
