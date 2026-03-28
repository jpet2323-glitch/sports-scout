#!/usr/bin/env python3
"""
Daily Sports Betting Scout
Fetches ESPN scores/stats + DraftKings odds, runs analysis across
NBA / NHL / MLB / NCAAB, updates record.json, generates dashboard.html
"""

import json
import os
import sys
import logging
import re
import subprocess
from datetime import datetime, timedelta, date, timezone
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    sys.exit("Missing dependency: pip3 install requests")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
RECORD_FILE  = SCRIPT_DIR / "record.json"
DASHBOARD    = SCRIPT_DIR / "dashboard.html"
LOG_FILE     = SCRIPT_DIR / "scout.log"

# ── Config ─────────────────────────────────────────────────────────────────────
ESPN_BASE    = "https://site.api.espn.com/apis/site/v2/sports"
ODDS_BASE    = "https://api.the-odds-api.com/v4"
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
DK_BASE      = "https://sportsbook-nash.draftkings.com/api/sportscontent/dkusnj/v1/leagues"
DK_HEADERS   = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

SPORTS = {
    "NBA":   {"espn": "basketball/nba",                     "odds": "basketball_nba",  "dk_id": 42648},
    "NHL":   {"espn": "hockey/nhl",                         "odds": "icehockey_nhl",   "dk_id": 42133},
    "MLB":   {"espn": "baseball/mlb",                       "odds": "baseball_mlb",    "dk_id": 84240},
    "NCAAB": {"espn": "basketball/mens-college-basketball",  "odds": "basketball_ncaab","dk_id": 92483},
}

# Min confidence score (0-100) to surface a pick
MIN_CONFIDENCE = 70
# Hard caps — quality over quantity
MAX_PICKS_TOTAL   = 5
MAX_PICKS_PER_SPORT = 2

# Default model weights (overridden by record.json after learning)
DEFAULT_WEIGHTS = {
    "w_win_pct":     30.0,   # season win% contribution (scaled 0-30)
    "w_home_adv":     7.0,   # flat home-court/ice bonus
    "w_quality":      8.0,   # elite/struggling tier bonus
    "w_odds_edge":    1.0,   # multiplier on ML probability edge
    "edge_threshold": 0.04,  # min implied-prob edge to make an odds-based pick
    "stat_diff_min":  8.0,   # min composite score gap for stat-only picks
    "samples":        0,     # total resolved picks used for learning
    "version":        1,
}

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("scout")


# ══════════════════════════════════════════════════════════════════════════════
# Record I/O
# ══════════════════════════════════════════════════════════════════════════════

def load_record() -> dict:
    with open(RECORD_FILE) as f:
        record = json.load(f)
    # Seed model_weights if this is the first run after the feature was added
    if "model_weights" not in record:
        record["model_weights"] = dict(DEFAULT_WEIGHTS)
        log.info("Initialized model weights with defaults.")
    else:
        # Back-fill any new keys added to DEFAULT_WEIGHTS
        for k, v in DEFAULT_WEIGHTS.items():
            record["model_weights"].setdefault(k, v)
    return record


def save_record(record: dict):
    record["meta"]["last_updated"] = date.today().isoformat()
    with open(RECORD_FILE, "w") as f:
        json.dump(record, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# ESPN API helpers
# ══════════════════════════════════════════════════════════════════════════════

def espn_get(path: str, params: dict = None) -> Optional[dict]:
    url = f"{ESPN_BASE}/{path}"
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning(f"ESPN request failed ({url}): {e}")
        return None


def fetch_scoreboard(sport_espn: str, date_str: str) -> list:
    """Return list of game dicts for a given YYYYMMDD date string."""
    data = espn_get(f"{sport_espn}/scoreboard", {"dates": date_str, "limit": 50})
    if not data:
        return []
    events = data.get("events", [])
    games = []
    for event in events:
        comps = event.get("competitions", [{}])
        comp  = comps[0] if comps else {}
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

        def get_record(c):
            for r in c.get("records", []):
                if r.get("name") == "overall":
                    return r.get("summary", "0-0")
            return c.get("records", [{}])[0].get("summary", "0-0") if c.get("records") else "0-0"

        def parse_record(summary: str):
            parts = summary.split("-")
            try:
                return int(parts[0]), int(parts[1])
            except Exception:
                return 0, 0

        home_rec = get_record(home)
        away_rec = get_record(away)
        hw, hl = parse_record(home_rec)
        aw, al = parse_record(away_rec)

        status = comp.get("status", {})
        completed = status.get("type", {}).get("completed", False)

        # ESPN sometimes embeds odds
        espn_odds = comp.get("odds", [])
        embedded_spread = None
        embedded_ou     = None
        if espn_odds:
            o = espn_odds[0]
            embedded_spread = o.get("details")    # e.g. "LAL -3.5"
            embedded_ou     = o.get("overUnder")

        games.append({
            "event_id":       event.get("id"),
            "name":           event.get("name", ""),
            "date":           event.get("date", ""),
            "completed":      completed,
            "home_team":      home.get("team", {}).get("name", ""),
            "home_abbr":      home.get("team", {}).get("abbreviation", ""),
            "home_id":        home.get("team", {}).get("id", ""),
            "home_score":     int(home.get("score", 0) or 0),
            "home_record":    home_rec,
            "home_wins":      hw,
            "home_losses":    hl,
            "away_team":      away.get("team", {}).get("name", ""),
            "away_abbr":      away.get("team", {}).get("abbreviation", ""),
            "away_id":        away.get("team", {}).get("id", ""),
            "away_score":     int(away.get("score", 0) or 0),
            "away_record":    away_rec,
            "away_wins":      aw,
            "away_losses":    al,
            "espn_spread":    embedded_spread,
            "espn_ou":        embedded_ou,
        })
    return games


# ══════════════════════════════════════════════════════════════════════════════
# DraftKings direct odds (no API key required)
# ══════════════════════════════════════════════════════════════════════════════

def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation for fuzzy team matching."""
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


def _parse_american(raw: str) -> Optional[int]:
    """Parse DraftKings american odds string (uses unicode minus − not -)."""
    if raw is None:
        return None
    cleaned = str(raw).replace("\u2212", "-").replace("+", "").strip()
    try:
        return int(cleaned)
    except ValueError:
        return None


def fetch_dk_direct(dk_league_id: int) -> dict:
    """
    Fetch live DraftKings odds directly from their public API.
    Returns a dict keyed by event_id -> {h2h, spread, total} markets.
    Falls back to The Odds API if direct fetch fails.
    """
    try:
        url = f"{DK_BASE}/{dk_league_id}"
        r   = requests.get(url, headers=DK_HEADERS, timeout=12)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning(f"DraftKings direct fetch failed (league {dk_league_id}): {e}")
        return {}

    # Index markets and selections by their IDs
    market_index    = {m["id"]: m for m in data.get("markets", [])}
    selection_index: dict[str, list] = {}
    for sel in data.get("selections", []):
        mid = sel.get("marketId", "")
        selection_index.setdefault(mid, []).append(sel)

    # Build per-event market dict
    events_out: dict[str, dict] = {}
    for event in data.get("events", []):
        eid  = event["id"]
        name = event.get("name", "")
        # DK format: "AWAY @ HOME" or "AWAY vs HOME"
        parts = re.split(r"\s+[@v][s.]?\s+", name, maxsplit=1)
        away_raw = parts[0].strip() if len(parts) == 2 else ""
        home_raw = parts[1].strip() if len(parts) == 2 else ""

        h2h    : dict = {}
        spread : dict = {}
        total  : dict = {}

        for mid, mkt in market_index.items():
            if mkt.get("eventId") != eid:
                continue
            mtype = mkt.get("marketType", {}).get("name", "")
            sels  = selection_index.get(mid, [])
            # Only use "main" lines to avoid alt lines polluting the data
            main_sels = [s for s in sels if s.get("main", False)] or sels

            if mtype == "Moneyline":
                for s in main_sels:
                    team = s.get("label", "")
                    odds = _parse_american(s.get("displayOdds", {}).get("american"))
                    if team and odds is not None:
                        h2h[team] = odds

            elif mtype == "Spread":
                for s in main_sels:
                    team  = s.get("label", "")
                    odds  = _parse_american(s.get("displayOdds", {}).get("american"))
                    point = s.get("points")
                    if team and odds is not None and point is not None:
                        spread[team] = {"point": float(point), "price": odds}

            elif mtype == "Total":
                for s in main_sels:
                    label = s.get("label", "")   # "Over" or "Under"
                    odds  = _parse_american(s.get("displayOdds", {}).get("american"))
                    point = s.get("points")
                    if label and odds is not None and point is not None:
                        total[label] = {"point": float(point), "price": odds}

        if h2h or spread or total:
            events_out[eid] = {
                "dk_event_id": eid,
                "name":        name,
                "away_raw":    away_raw,
                "home_raw":    home_raw,
                "h2h":         h2h,
                "spread":      spread,
                "total":       total,
            }

    log.info(f"  DraftKings direct: {len(events_out)} events with odds (league {dk_league_id})")
    return events_out


def match_dk_event(game: dict, dk_events: dict) -> Optional[dict]:
    """Match an ESPN game to a DraftKings event by fuzzy team name."""
    hn = normalize_name(game["home_team"])
    an = normalize_name(game["away_team"])

    def last(s): return normalize_name(s).split()[-1] if s else ""

    for ev in dk_events.values():
        hr = normalize_name(ev["home_raw"])
        ar = normalize_name(ev["away_raw"])
        # Try full substring match first
        if (hn in hr or hr in hn or last(hn) == last(hr)) and \
           (an in ar or ar in an or last(an) == last(ar)):
            return ev
        # Also try matching ML keys directly
        for team_label in ev["h2h"]:
            tl = normalize_name(team_label)
            if last(hn) in tl and last(an) in " ".join(normalize_name(k) for k in ev["h2h"] if k != team_label):
                return ev
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Analysis engine
# ══════════════════════════════════════════════════════════════════════════════

def american_to_implied(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds >= 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def win_pct(wins: int, losses: int) -> float:
    total = wins + losses
    return wins / total if total > 0 else 0.5


def _game_local_date(espn_date_str: str) -> str:
    """Convert ESPN's UTC date string to the local calendar date (YYYY-MM-DD)."""
    if not espn_date_str:
        return date.today().isoformat()
    try:
        # ESPN format: "2026-03-28T02:30:00Z"
        dt_utc = datetime.strptime(espn_date_str[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        return dt_utc.astimezone().date().isoformat()
    except Exception:
        return date.today().isoformat()


def analyze_game(game: dict, dk_event: Optional[dict], sport: str, weights: dict) -> Optional[dict]:
    """
    Score each team using learned weights and produce a pick.

    Scoring factors:
      • Season win%          — 0 to w_win_pct pts
      • Home-court/ice bonus — w_home_adv pts (flat)
      • Record quality tier  — ±w_quality pts
      • Moneyline value edge — w_odds_edge * edge%  (requires odds)
      • Spread value edge    — w_odds_edge * edge%  (requires odds)

    Weights are loaded from record.json and updated daily via online learning.
    """
    w_win_pct  = weights.get("w_win_pct",  DEFAULT_WEIGHTS["w_win_pct"])
    w_home_adv = weights.get("w_home_adv", DEFAULT_WEIGHTS["w_home_adv"])
    w_quality  = weights.get("w_quality",  DEFAULT_WEIGHTS["w_quality"])
    w_odds     = weights.get("w_odds_edge",DEFAULT_WEIGHTS["w_odds_edge"])
    edge_thr   = weights.get("edge_threshold", DEFAULT_WEIGHTS["edge_threshold"])
    stat_min   = weights.get("stat_diff_min",  DEFAULT_WEIGHTS["stat_diff_min"])

    hw, hl = game["home_wins"], game["home_losses"]
    aw, al = game["away_wins"], game["away_losses"]
    h_pct  = win_pct(hw, hl)
    a_pct  = win_pct(aw, al)

    # Base scores
    h_score = h_pct * w_win_pct
    a_score = a_pct * w_win_pct

    # Home advantage
    h_score += w_home_adv

    # Quality tier bonus/penalty
    h_quality_bonus = w_quality if h_pct >= 0.60 else (-w_quality if h_pct <= 0.38 else 0)
    a_quality_bonus = w_quality if a_pct >= 0.60 else (-w_quality if a_pct <= 0.38 else 0)
    h_score += h_quality_bonus
    a_score += a_quality_bonus

    ml_pick       = None
    ml_confidence = 0
    spread_pick   = None
    spread_line   = None
    spread_conf   = 0
    ou_pick       = None
    ou_line       = None
    ou_conf       = 0
    ml_edge_used  = 0.0
    spread_edge_used = 0.0

    markets = dk_event if dk_event else {"h2h": {}, "spread": {}, "total": {}}

    # ── Moneyline analysis ─────────────────────────────────────────────────
    h2h = markets.get("h2h", {})
    # Match by last word of team name (handles "LA Clippers" vs "Clippers" etc.)
    def find_ml(team_full, team_abbr):
        last = normalize_name(team_full).split()[-1]
        for k, v in h2h.items():
            if normalize_name(k).split()[-1] == last:
                return v
        return h2h.get(team_full) or h2h.get(team_abbr)

    home_ml = find_ml(game["home_team"], game["home_abbr"])
    away_ml = find_ml(game["away_team"], game["away_abbr"])

    if home_ml and away_ml:
        implied_h = american_to_implied(int(home_ml))
        implied_a = american_to_implied(int(away_ml))
        our_h     = h_score / (h_score + a_score)
        our_a     = 1 - our_h
        edge_h    = our_h - implied_h
        edge_a    = our_a - implied_a

        if max(edge_h, edge_a) >= edge_thr:
            if edge_h >= edge_a:
                ml_edge_used  = edge_h
                ml_pick       = f"{game['home_team']} ML ({'+' if home_ml>0 else ''}{home_ml})"
                ml_confidence = min(50 + int(edge_h * w_odds * 150), 95)
                h_score      += edge_h * w_odds * 20
            else:
                ml_edge_used  = edge_a
                ml_pick       = f"{game['away_team']} ML ({'+' if away_ml>0 else ''}{away_ml})"
                ml_confidence = min(50 + int(edge_a * w_odds * 150), 95)
                a_score      += edge_a * w_odds * 20

    # ── Spread analysis ────────────────────────────────────────────────────
    sp = markets.get("spread", {})
    if sp and len(sp) >= 2:
        for team_name, data in sp.items():
            tn      = normalize_name(team_name)
            hn2     = normalize_name(game["home_team"])
            is_home = tn.split()[-1] == hn2.split()[-1] or tn in hn2 or hn2 in tn
            point   = data.get("point", 0)
            price   = data.get("price", -110)
            implied = american_to_implied(int(price))
            our_prob = (h_score / (h_score + a_score)) if is_home else (a_score / (h_score + a_score))
            cover_adj = 0.03 if abs(point) <= 3.5 else 0.0
            our_cover = our_prob - cover_adj
            edge = our_cover - implied
            if edge >= edge_thr and (spread_conf == 0 or edge > spread_edge_used):
                side             = game["home_team"] if is_home else game["away_team"]
                spread_pick      = f"{side} {'+' if point > 0 else ''}{point} ({'+' if price > 0 else ''}{price})"
                spread_line      = point
                spread_conf      = min(50 + int(edge * w_odds * 130), 95)
                spread_edge_used = edge

    # ── Totals analysis ────────────────────────────────────────────────────
    tot = markets.get("total", {})
    if tot:
        over_data  = tot.get("Over")  or next((v for k, v in tot.items() if "over" in k.lower()), None)
        under_data = tot.get("Under") or next((v for k, v in tot.items() if "under" in k.lower()), None)
        if over_data and under_data:
            line = over_data.get("point", 0)
            combined_score_factor = (h_pct + a_pct) / 2
            if sport in ("NBA", "NCAAB"):
                if combined_score_factor > 0.58:
                    ou_pick = f"Over {line}"
                    ou_line = line
                    ou_conf = 60
                elif combined_score_factor < 0.42:
                    ou_pick = f"Under {line}"
                    ou_line = line
                    ou_conf = 58

    # ── Primary pick selection ─────────────────────────────────────────────
    diff = abs(h_score - a_score)

    # Prefer spread pick when available, else ML, else stat-only
    if spread_pick and spread_conf >= MIN_CONFIDENCE:
        primary_pick = spread_pick
        primary_conf = spread_conf
        pick_type    = "Spread"
    elif ml_pick and ml_confidence >= MIN_CONFIDENCE:
        primary_pick = ml_pick
        primary_conf = ml_confidence
        pick_type    = "Moneyline"
    elif diff >= stat_min and not markets.get("h2h"):
        # No odds available — use pure stat edge
        primary_pick = f"{game['home_team'] if h_score > a_score else game['away_team']} (stat edge)"
        primary_conf = min(50 + int(diff * 1.2), 80)
        pick_type    = "Stat Edge"
        if primary_conf < MIN_CONFIDENCE:
            return None  # Stat edge not strong enough
    else:
        return None  # No confident pick

    # Build notes
    notes = []
    if hw + hl > 0:
        notes.append(f"Home: {game['home_team']} {game['home_record']}")
    if aw + al > 0:
        notes.append(f"Away: {game['away_team']} {game['away_record']}")
    if ou_pick:
        notes.append(f"Also watching: {ou_pick} (conf {ou_conf})")

    # ── Build detailed reasoning ───────────────────────────────────────────
    def tier_label(pct):
        if pct >= 0.62: return "Elite"
        if pct >= 0.52: return "Above average"
        if pct >= 0.44: return "Below average"
        return "Struggling"

    picked_team  = primary_pick.split()[0] if primary_pick else ""
    picking_home = game["home_team"].split()[-1] in primary_pick

    reasoning = []

    # Records
    reasoning.append(
        f"📋 <strong>{game['home_team']}</strong> are {game['home_record']} "
        f"({round(h_pct*100,1)}% win rate) — {tier_label(h_pct)} at home this season."
    )
    reasoning.append(
        f"📋 <strong>{game['away_team']}</strong> are {game['away_record']} "
        f"({round(a_pct*100,1)}% win rate) — {tier_label(a_pct)} on the road this season."
    )

    # Home advantage
    reasoning.append(
        f"🏠 Home-court/ice advantage gives <strong>{game['home_team']}</strong> "
        f"a +7 pt scoring bonus in our model."
    )

    # Record quality tier
    if h_pct >= 0.60:
        reasoning.append(f"⭐ <strong>{game['home_team']}</strong> qualify as an elite team (≥60% win rate), earning a +8 quality bonus.")
    elif h_pct <= 0.38:
        reasoning.append(f"⚠️ <strong>{game['home_team']}</strong> are a struggling team (≤38% win rate), applying a −8 quality penalty.")
    if a_pct >= 0.60:
        reasoning.append(f"⭐ <strong>{game['away_team']}</strong> qualify as an elite team (≥60% win rate), earning a +8 quality bonus.")
    elif a_pct <= 0.38:
        reasoning.append(f"⚠️ <strong>{game['away_team']}</strong> are a struggling team (≤38% win rate), applying a −8 quality penalty.")

    # Odds edge
    if pick_type in ("Moneyline", "Spread") and home_ml and away_ml:
        implied_h = american_to_implied(int(home_ml))
        implied_a = american_to_implied(int(away_ml))
        our_h = h_score / (h_score + a_score)
        our_a = 1 - our_h
        if picking_home:
            reasoning.append(
                f"📈 DraftKings implies <strong>{game['home_team']}</strong> win probability at "
                f"{round(implied_h*100,1)}%. Our model estimates {round(our_h*100,1)}% — "
                f"a +{round((our_h-implied_h)*100,1)}% edge."
            )
        else:
            reasoning.append(
                f"📈 DraftKings implies <strong>{game['away_team']}</strong> win probability at "
                f"{round(implied_a*100,1)}%. Our model estimates {round(our_a*100,1)}% — "
                f"a +{round((our_a-implied_a)*100,1)}% edge."
            )

    if pick_type == "Spread" and spread_line is not None:
        direction = "giving" if spread_line < 0 else "getting"
        reasoning.append(
            f"📐 Spread pick: {picked_team} {direction} {abs(spread_line)} points. "
            f"Our cover probability exceeds the implied probability by the required {int(edge_thr*100)}% threshold."
        )

    # Stat edge (no odds)
    if pick_type == "Stat Edge":
        score_gap = round(abs(h_score - a_score), 1)
        reasoning.append(
            f"📊 No DraftKings line available. Pure stat model gives "
            f"{'home' if picking_home else 'away'} team a composite score gap of {score_gap} pts "
            f"(threshold for pick: 8 pts)."
        )
        reasoning.append(
            "💡 Tip: Set ODDS_API_KEY in .env to unlock spread/moneyline edge analysis."
        )

    # O/U note
    if ou_pick:
        reasoning.append(
            f"🔢 Totals lean: <strong>{ou_pick}</strong> — combined team win rates suggest "
            f"{'higher' if 'Over' in ou_pick else 'lower'} scoring output vs. the line."
        )

    # Confidence summary
    reasoning.append(
        f"🎯 <strong>Confidence: {primary_conf}%</strong> — "
        f"{'High conviction, consider 2u.' if primary_conf >= 75 else 'Standard 1u play.' if primary_conf >= 62 else 'Lean only, small play.'}"
    )

    picking_home = (pick_type == "Spread" and game["home_team"].split()[-1] in primary_pick) or \
                   (pick_type == "Moneyline" and game["home_team"].split()[-1] in primary_pick) or \
                   (pick_type == "Stat Edge" and h_score > a_score)

    return {
        "sport":       sport,
        "game":        game["name"],
        "home":        game["home_team"],
        "away":        game["away_team"],
        "pick":        primary_pick,
        "pick_type":   pick_type,
        "confidence":  primary_conf,
        "notes":       " | ".join(notes),
        "reasoning":   reasoning,
        "ou_pick":     ou_pick,
        "ou_line":     ou_line,
        "ou_conf":     ou_conf,
        "result":      "pending",
        "date":        _game_local_date(game.get("date", "")),
        "event_id":    game.get("event_id"),
        "home_score":  None,
        "away_score":  None,
        # Feature vector stored for model learning
        "features": {
            "h_win_pct":       round(h_pct, 4),
            "a_win_pct":       round(a_pct, 4),
            "pct_diff":        round((h_pct - a_pct) if picking_home else (a_pct - h_pct), 4),
            "picking_home":    int(picking_home),
            "h_quality_bonus": round(h_quality_bonus, 2),
            "a_quality_bonus": round(a_quality_bonus, 2),
            "quality_edge":    round((h_quality_bonus - a_quality_bonus) if picking_home else (a_quality_bonus - h_quality_bonus), 2),
            "ml_edge":         round(ml_edge_used, 4),
            "spread_edge":     round(spread_edge_used, 4),
            "had_dk_odds":     int(bool(markets.get("h2h"))),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# Results updater — checks yesterday's pending picks
# ══════════════════════════════════════════════════════════════════════════════

def update_results(record: dict):
    """Match yesterday's pending picks to ESPN final scores, update record."""
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y%m%d")
    pending   = [p for p in record["picks_history"] if p.get("result") == "pending"]
    if not pending:
        log.info("No pending picks to resolve.")
        return

    log.info(f"Resolving {len(pending)} pending picks from {yesterday}...")

    # Fetch yesterday's results for all sports
    results_map = {}  # event_id -> game
    for sport, cfg in SPORTS.items():
        games = fetch_scoreboard(cfg["espn"], yesterday)
        for g in games:
            if g["completed"]:
                results_map[g["event_id"]] = g

    updated = 0
    for pick in pending:
        event_id = pick.get("event_id")
        result_game = results_map.get(event_id)

        if not result_game:
            # Skip — never fall back to name matching across days.
            # Same teams play back-to-back series (common in MLB) so name
            # matching would silently grade tomorrow's picks against today's results.
            log.info(f"  No result found for {pick['game']} (event {event_id}) — leaving pending.")
            continue

        pick["home_score"] = result_game["home_score"]
        pick["away_score"] = result_game["away_score"]
        sport = pick["sport"]

        # Determine outcome
        pick_text = pick["pick"]
        h_score   = result_game["home_score"]
        a_score   = result_game["away_score"]
        home_won  = h_score > a_score
        away_won  = a_score > h_score

        outcome = "loss"  # default

        if pick["pick_type"] == "Moneyline":
            # Check if the team we picked actually won — use stored home/away names
            picking_home = pick.get("home", "").lower() in pick_text.lower()
            if (picking_home and home_won) or (not picking_home and away_won):
                outcome = "win"
        elif pick["pick_type"] == "Spread":
            # Parse spread: "Team +/-X.X (odds)"
            m = re.search(r"([+-]?\d+\.?\d*)\s*\(", pick_text)
            if m:
                spread = float(m.group(1))
                # Determine which team the spread is on
                if result_game["home_team"].split()[-1] in pick_text:
                    adjusted = h_score + spread
                    if adjusted > a_score:
                        outcome = "win"
                    elif adjusted == a_score:
                        outcome = "push"
                else:
                    adjusted = a_score + spread
                    if adjusted > h_score:
                        outcome = "win"
                    elif adjusted == h_score:
                        outcome = "push"
        elif pick["pick_type"] == "Stat Edge":
            # Use stored pick["home"] — same name used when pick was written,
            # avoids mismatches when ESPN returns different name formats on re-fetch
            picking_home = pick.get("home", "").lower() in pick_text.lower()
            outcome = "win" if (picking_home and home_won) or (not picking_home and away_won) else "loss"

        pick["result"] = outcome
        record["overall"][{"win": "wins", "loss": "losses", "push": "pushes"}[outcome]] += 1
        record["overall"]["units_wagered"] += 1
        if outcome == "win":
            record["overall"]["units_net"] += 0.91  # standard -110 payout
        elif outcome == "loss":
            record["overall"]["units_net"] -= 1.0

        sport_key = sport if sport in record["by_sport"] else "NBA"
        record["by_sport"][sport_key][{"win": "wins", "loss": "losses", "push": "pushes"}[outcome]] += 1

        log.info(f"  {pick['game']} → {outcome.upper()} ({h_score}-{a_score})")
        updated += 1

    # Recalculate ROI
    uw = record["overall"]["units_wagered"]
    un = record["overall"]["units_net"]
    record["overall"]["roi_pct"] = round((un / uw * 100) if uw > 0 else 0, 1)
    log.info(f"Updated {updated} picks. Record: {record['overall']['wins']}-{record['overall']['losses']}")

    # Trigger learning after results are resolved
    if updated > 0:
        update_model_weights(record)


def update_model_weights(record: dict):
    """
    Online learning: adjust model weights based on resolved picks.

    Algorithm: additive gradient update (perceptron-style).
    For each newly resolved pick (win/loss/push):
      - reward = +1 (win), -1 (loss), 0 (push)
      - Each weight nudged proportional to how much its feature
        contributed to picking the winning/losing side.
      - Learning rate decays slightly as sample count grows
        (conservative early exploration, stable later).
    Weights are clipped to sensible bounds after each update.
    """
    weights = record["model_weights"]
    lr_base = 0.04

    newly_resolved = [
        p for p in record["picks_history"]
        if p.get("result") in ("win", "loss") and p.get("features")
        and p.get("_weight_updated") is not True
    ]

    if not newly_resolved:
        return

    for pick in newly_resolved:
        feat   = pick["features"]
        result = pick["result"]
        reward = 1 if result == "win" else -1

        # Decay learning rate as we accumulate more samples
        n  = max(1, weights.get("samples", 0))
        lr = lr_base / (1 + n / 200)   # halves at 200 samples, quarters at 600

        # Win% weight: reward scales with how large the record gap was
        weights["w_win_pct"] += lr * reward * feat.get("pct_diff", 0) * 15
        # Home advantage: reward if picking home and correct
        weights["w_home_adv"] += lr * reward * feat.get("picking_home", 0) * 1.5
        # Quality tier: reward proportional to quality edge
        weights["w_quality"] += lr * reward * feat.get("quality_edge", 0) * 0.5
        # Odds edge weight: reward when a large odds edge led to correct call
        combined_edge = feat.get("ml_edge", 0) + feat.get("spread_edge", 0)
        if combined_edge > 0:
            weights["w_odds_edge"] += lr * reward * combined_edge * 10

        # Clip weights to sensible ranges
        weights["w_win_pct"]    = round(max(10.0, min(55.0, weights["w_win_pct"])),  3)
        weights["w_home_adv"]   = round(max(2.0,  min(15.0, weights["w_home_adv"])), 3)
        weights["w_quality"]    = round(max(1.0,  min(18.0, weights["w_quality"])),  3)
        weights["w_odds_edge"]  = round(max(0.3,  min(3.0,  weights["w_odds_edge"])),3)

        weights["samples"] = n + 1
        pick["_weight_updated"] = True

    weights["version"]      = weights.get("version", 1) + 1
    weights["last_updated"] = date.today().isoformat()
    log.info(
        f"Model updated on {len(newly_resolved)} picks → "
        f"w_win_pct={weights['w_win_pct']} w_home={weights['w_home_adv']} "
        f"w_quality={weights['w_quality']} w_odds={weights['w_odds_edge']} "
        f"(samples={weights['samples']})"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard HTML generator
# ══════════════════════════════════════════════════════════════════════════════

SPORT_EMOJI = {"NBA": "🏀", "NHL": "🏒", "MLB": "⚾", "NCAAB": "🏀"}
SPORT_COLOR = {
    "NBA":   ("#c8102e", "#1d428a"),
    "NHL":   ("#000000", "#a2aaad"),
    "MLB":   ("#002d72", "#d50032"),
    "NCAAB": ("#ff6600", "#003087"),
}

def conf_color(conf: int) -> str:
    if conf >= 75: return "#22c55e"
    if conf >= 62: return "#f59e0b"
    return "#94a3b8"


def conf_label(conf: int) -> str:
    if conf >= 78: return "🔥 HIGH"
    if conf >= 65: return "✅ GOOD"
    return "📊 LEAN"


def _weight_bar(current: float, default: float, lo: float, hi: float) -> str:
    """Render a small inline bar showing weight vs default."""
    pct = int((current - lo) / (hi - lo) * 100)
    pct = max(0, min(100, pct))
    color = "#22c55e" if current >= default else "#f59e0b"
    return (
        f'<div style="display:flex;align-items:center;gap:8px">'
        f'<div style="flex:1;background:#334155;border-radius:4px;height:6px">'
        f'<div style="width:{pct}%;background:{color};height:6px;border-radius:4px"></div></div>'
        f'<span style="font-size:0.8rem;min-width:36px;text-align:right">{current:.2f}</span></div>'
    )


def generate_dashboard(record: dict, picks: list):
    o   = record["overall"]
    tot = o["wins"] + o["losses"] + o["pushes"]
    win_rate = round(o["wins"] / tot * 100, 1) if tot else 0
    roi  = o.get("roi_pct", 0)
    unet = round(o.get("units_net", 0), 2)

    by_sport_rows = ""
    for sp, stats in record["by_sport"].items():
        st = stats["wins"] + stats["losses"] + stats["pushes"]
        wr = round(stats["wins"] / st * 100, 1) if st else 0
        emoji = SPORT_EMOJI.get(sp, "🎯")
        by_sport_rows += f"""
        <tr>
          <td>{emoji} {sp}</td>
          <td class="num">{stats['wins']}-{stats['losses']}{'-' + str(stats['pushes']) + 'P' if stats['pushes'] else ''}</td>
          <td class="num">{wr}%</td>
        </tr>"""

    picks_html = ""
    for i, p in enumerate(sorted(picks, key=lambda x: -x["confidence"])):
        cc  = conf_color(p["confidence"])
        lbl = conf_label(p["confidence"])
        sp  = p["sport"]
        card_id = f"card-{i}"
        reasoning_items = "".join(
            f'<li>{r}</li>' for r in p.get("reasoning", [])
        )
        picks_html += f"""
      <div class="pick-card" style="border-left:4px solid {cc}" onclick="toggleCard('{card_id}')">
        <div class="pick-header">
          <span class="sport-badge">{SPORT_EMOJI.get(sp,'🎯')} {sp}</span>
          <span class="conf-badge" style="color:{cc}">{lbl} {p['confidence']}%</span>
          <span class="expand-icon" id="icon-{card_id}">▼ Why?</span>
        </div>
        <div class="pick-game">{p['game']}</div>
        <div class="pick-line">
          <strong>PICK:</strong> {p['pick']}
          <span class="pick-type">{p['pick_type']}</span>
        </div>
        {f'<div class="pick-ou">Also: {p["ou_pick"]} (conf {p["ou_conf"]}%)</div>' if p.get("ou_pick") else ''}
        <div class="pick-notes">{p['notes']}</div>
        <div class="reasoning-panel" id="{card_id}">
          <ul class="reasoning-list">{reasoning_items}</ul>
        </div>
      </div>"""

    if not picks_html:
        picks_html = '<div class="no-picks">No high-confidence picks identified for today. Check back after morning lines firm up.</div>'

    # Build history grouped by day (newest first, excluding today's pending picks)
    from collections import defaultdict, OrderedDict
    days: dict = defaultdict(list)
    today_iso = date.today().isoformat()
    for p in record["picks_history"]:
        if p.get("result") != "pending" or p.get("date") != today_iso:
            days[p.get("date", "unknown")].append(p)

    history_by_day_html = ""
    for day in sorted(days.keys(), reverse=True):
        day_picks = days[day]
        day_wins   = sum(1 for p in day_picks if p.get("result") == "win")
        day_losses = sum(1 for p in day_picks if p.get("result") == "loss")
        day_pushes = sum(1 for p in day_picks if p.get("result") == "push")
        day_pend   = sum(1 for p in day_picks if p.get("result") == "pending")
        day_units  = round(day_wins * 0.91 - day_losses, 2)
        try:
            day_label = datetime.strptime(day, "%Y-%m-%d").strftime("%A, %B %d %Y")
        except Exception:
            day_label = day

        rec_color = "var(--green)" if day_wins > day_losses else ("var(--red)" if day_losses > day_wins else "var(--muted)")
        rec_str   = f"{day_wins}-{day_losses}" + (f"-{day_pushes}P" if day_pushes else "") + (f" · {day_pend} pending" if day_pend else "")
        units_str = (f"+{day_units}u" if day_units >= 0 else f"{day_units}u")

        rows = ""
        for p in day_picks:
            r = p.get("result", "pending")
            score_str = f"{p['home_score']}-{p['away_score']}" if p.get("home_score") is not None else "—"
            r_icon = {"win": "✅", "loss": "❌", "push": "🔄", "pending": "⏳"}.get(r, "—")
            rows += f"""
            <tr>
              <td>{SPORT_EMOJI.get(p['sport'],'🎯')} {p['sport']}</td>
              <td class="game-cell">{p['game']}</td>
              <td class="pick-cell">{p['pick']}</td>
              <td class="num" style="color:var(--muted)">{p.get('confidence',0)}%</td>
              <td class="num">{score_str}</td>
              <td class="result {r}" style="text-align:center">{r_icon}</td>
            </tr>"""

        history_by_day_html += f"""
        <div class="day-block">
          <div class="day-header" onclick="toggleDay(this)">
            <div>
              <span class="day-label">{day_label}</span>
              <span class="day-record" style="color:{rec_color}">{rec_str}</span>
            </div>
            <div style="display:flex;align-items:center;gap:16px">
              <span style="font-size:0.9rem;font-weight:600;color:{'var(--green)' if day_units >= 0 else 'var(--red)'}">{'+' if day_units >= 0 else ''}{units_str}</span>
              <span class="day-toggle">▼</span>
            </div>
          </div>
          <div class="day-content">
            <table>
              <thead><tr>
                <th>Sport</th><th>Game</th><th>Pick</th>
                <th class="num">Conf</th><th class="num">Score</th><th style="text-align:center">Result</th>
              </tr></thead>
              <tbody>{rows}</tbody>
            </table>
          </div>
        </div>"""

    if not history_by_day_html:
        history_by_day_html = '<div style="text-align:center;color:var(--muted);padding:40px">No completed picks yet — check back after your first results come in.</div>'

    now_str = datetime.now().strftime("%A, %B %d %Y — %I:%M %p")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sports Betting Scout — {date.today().isoformat()}</title>
<style>
  :root {{
    --bg: #0f172a; --surface: #1e293b; --surface2: #273549;
    --text: #e2e8f0; --muted: #64748b; --border: #334155;
    --green: #22c55e; --red: #ef4444; --yellow: #f59e0b; --blue: #3b82f6;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 14px; line-height: 1.6; }}
  .header {{ background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%); padding: 24px 32px; display:flex; justify-content:space-between; align-items:center; }}
  .header h1 {{ font-size: 1.8rem; font-weight: 700; letter-spacing: -0.5px; }}
  .header .subtitle {{ color: #90caf9; font-size: 0.9rem; margin-top: 2px; }}
  .header .timestamp {{ color: #90caf9; font-size: 0.85rem; text-align:right; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 24px 20px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }}
  .grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }}
  .card h2 {{ font-size: 0.75rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }}
  .stat-big {{ font-size: 2.4rem; font-weight: 700; line-height: 1; }}
  .stat-sub {{ font-size: 0.85rem; color: var(--muted); margin-top: 4px; }}
  .green {{ color: var(--green); }} .red {{ color: var(--red); }} .yellow {{ color: var(--yellow); }} .blue {{ color: var(--blue); }}
  .section-title {{ font-size: 1.1rem; font-weight: 600; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }}
  .picks-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 16px; margin-bottom: 32px; }}
  .pick-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }}
  .pick-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
  .sport-badge {{ font-size: 0.8rem; font-weight: 600; color: var(--muted); }}
  .conf-badge {{ font-size: 0.85rem; font-weight: 700; }}
  .pick-game {{ font-size: 0.95rem; color: var(--muted); margin-bottom: 8px; }}
  .pick-line {{ font-size: 1.05rem; margin-bottom: 6px; }}
  .pick-type {{ display: inline-block; background: var(--surface2); border-radius: 4px; padding: 1px 7px; font-size: 0.75rem; color: var(--muted); margin-left: 8px; }}
  .pick-ou {{ font-size: 0.85rem; color: var(--yellow); margin-bottom: 4px; }}
  .pick-notes {{ font-size: 0.8rem; color: var(--muted); }}
  .expand-icon {{ margin-left: auto; font-size: 0.75rem; color: var(--muted); white-space: nowrap; transition: color 0.2s; }}
  .pick-card {{ cursor: pointer; transition: background 0.15s; }}
  .pick-card:hover {{ background: #243044; }}
  .pick-card:hover .expand-icon {{ color: var(--blue); }}
  .reasoning-panel {{ display: none; margin-top: 14px; padding-top: 14px; border-top: 1px solid var(--border); cursor: default; }}
  .reasoning-panel.open {{ display: block; }}
  .reasoning-list {{ list-style: none; display: flex; flex-direction: column; gap: 8px; }}
  .reasoning-list li {{ font-size: 0.85rem; color: #cbd5e1; line-height: 1.5; padding: 8px 10px; background: var(--surface2); border-radius: 6px; }}
  .no-picks {{ background: var(--surface); border: 1px dashed var(--border); border-radius: 10px; padding: 32px; text-align: center; color: var(--muted); }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ text-align: left; font-size: 0.75rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; padding: 8px 12px; border-bottom: 1px solid var(--border); }}
  td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: var(--surface2); }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .game-cell {{ font-size: 0.85rem; color: var(--muted); }}
  .pick-cell {{ font-size: 0.88rem; }}
  .result {{ text-align: center; font-weight: 700; border-radius: 4px; }}
  .result.win {{ color: var(--green); }}
  .result.loss {{ color: var(--red); }}
  .result.push {{ color: var(--yellow); }}
  .result.pending {{ color: var(--muted); }}
  .footer {{ text-align: center; color: var(--muted); font-size: 0.8rem; padding: 24px; }}
  /* ── Tabs ── */
  .tab-bar {{ display: flex; gap: 4px; border-bottom: 2px solid var(--border); margin-bottom: 24px; }}
  .tab-btn {{ background: none; border: none; color: var(--muted); font-size: 0.95rem; font-weight: 600; padding: 10px 20px; cursor: pointer; border-bottom: 3px solid transparent; margin-bottom: -2px; transition: color 0.15s; }}
  .tab-btn.active {{ color: var(--text); border-bottom-color: var(--blue); }}
  .tab-btn:hover {{ color: var(--text); }}
  .tab-pane {{ display: none; }}
  .tab-pane.active {{ display: block; }}
  /* ── History ── */
  .day-block {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; margin-bottom: 12px; overflow: hidden; }}
  .day-header {{ display: flex; justify-content: space-between; align-items: center; padding: 14px 18px; cursor: pointer; user-select: none; }}
  .day-header:hover {{ background: var(--surface2); }}
  .day-label {{ font-weight: 600; font-size: 0.95rem; margin-right: 12px; }}
  .day-record {{ font-size: 0.85rem; font-weight: 600; }}
  .day-toggle {{ color: var(--muted); font-size: 0.8rem; transition: transform 0.2s; }}
  .day-content {{ display: none; border-top: 1px solid var(--border); }}
  .day-content.open {{ display: block; }}
  .day-block.open .day-toggle {{ transform: rotate(180deg); }}
  @media (max-width: 768px) {{ .grid-2, .grid-4 {{ grid-template-columns: 1fr 1fr; }} .header {{ flex-direction: column; gap: 8px; }} }}
  @media (max-width: 500px) {{ .grid-2, .grid-4 {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>🎯 Sports Betting Scout</h1>
    <div class="subtitle">NBA · NHL · MLB · NCAAB · Powered by DraftKings Lines</div>
  </div>
  <div class="timestamp">{now_str}</div>
</div>

<div class="container">

  <!-- KPI row -->
  <div class="grid-4">
    <div class="card">
      <h2>Overall Record</h2>
      <div class="stat-big {'green' if o['wins'] > o['losses'] else 'red'}">{o['wins']}-{o['losses']}{'-' + str(o['pushes']) + 'P' if o['pushes'] else ''}</div>
      <div class="stat-sub">Win Rate: {win_rate}%</div>
    </div>
    <div class="card">
      <h2>ROI</h2>
      <div class="stat-big {'green' if roi >= 0 else 'red'}">{'+' if roi >= 0 else ''}{roi}%</div>
      <div class="stat-sub">Since tracking began</div>
    </div>
    <div class="card">
      <h2>Units Net</h2>
      <div class="stat-big {'green' if unet >= 0 else 'red'}">{'+' if unet >= 0 else ''}{unet}u</div>
      <div class="stat-sub">{o['units_wagered']} wagered</div>
    </div>
    <div class="card">
      <h2>Today's Picks</h2>
      <div class="stat-big blue">{len(picks)}</div>
      <div class="stat-sub">{sum(1 for p in picks if p['confidence'] >= 72)} high confidence</div>
    </div>
  </div>

  <!-- Tab bar -->
  <div class="tab-bar">
    <button class="tab-btn active" onclick="switchTab('today', this)">📋 Today's Picks</button>
    <button class="tab-btn" onclick="switchTab('history', this)">📅 History</button>
    <button class="tab-btn" onclick="switchTab('stats', this)">📊 Stats</button>
  </div>

  <!-- TODAY TAB -->
  <div id="tab-today" class="tab-pane active">
    <div class="section-title">📋 {date.today().strftime('%A, %B %d %Y')} — {len(picks)} Pick{'s' if len(picks) != 1 else ''}</div>
    <div class="picks-grid">
      {picks_html}
    </div>
  </div>

  <!-- HISTORY TAB -->
  <div id="tab-history" class="tab-pane">
    <div class="section-title" style="margin-bottom:20px">📅 Results by Day</div>
    {history_by_day_html}
  </div>

  <!-- STATS TAB -->
  <div id="tab-stats" class="tab-pane">

    <!-- By sport -->
    <div class="grid-2" style="margin-bottom:20px">
      <div class="card">
        <div class="section-title" style="margin-bottom:12px">By Sport</div>
        <table>
          <thead><tr><th>Sport</th><th class="num">Record</th><th class="num">Win%</th></tr></thead>
          <tbody>{by_sport_rows}</tbody>
        </table>
      </div>
      <div class="card">
        <div class="section-title" style="margin-bottom:12px">Quick Stats</div>
        <table>
          <tbody>
            <tr><td>Total picks tracked</td><td class="num">{len(record['picks_history'])}</td></tr>
            <tr><td>Avg confidence</td><td class="num">{round(sum(p.get('confidence',0) for p in record['picks_history']) / len(record['picks_history']), 0) if record['picks_history'] else 0}%</td></tr>
            <tr><td>Pending picks</td><td class="num yellow">{sum(1 for p in record['picks_history'] if p.get('result')=='pending')}</td></tr>
            <tr><td>Imported opening record</td><td class="num">{record['meta'].get('note','')}</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Model weights -->
    <div class="card">
      <div class="section-title" style="margin-bottom:12px">🧠 Self-Learning Model — Weight State</div>
      <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:16px">
        <div>
          <div style="font-size:0.75rem;color:var(--muted);margin-bottom:4px">Win% Weight <span style="color:#64748b">(default {DEFAULT_WEIGHTS["w_win_pct"]})</span></div>
          {_weight_bar(record["model_weights"]["w_win_pct"], DEFAULT_WEIGHTS["w_win_pct"], 10, 55)}
        </div>
        <div>
          <div style="font-size:0.75rem;color:var(--muted);margin-bottom:4px">Home Advantage <span style="color:#64748b">(default {DEFAULT_WEIGHTS["w_home_adv"]})</span></div>
          {_weight_bar(record["model_weights"]["w_home_adv"], DEFAULT_WEIGHTS["w_home_adv"], 2, 15)}
        </div>
        <div>
          <div style="font-size:0.75rem;color:var(--muted);margin-bottom:4px">Quality Tier Bonus <span style="color:#64748b">(default {DEFAULT_WEIGHTS["w_quality"]})</span></div>
          {_weight_bar(record["model_weights"]["w_quality"], DEFAULT_WEIGHTS["w_quality"], 1, 18)}
        </div>
        <div>
          <div style="font-size:0.75rem;color:var(--muted);margin-bottom:4px">Odds Edge Multiplier <span style="color:#64748b">(default {DEFAULT_WEIGHTS["w_odds_edge"]})</span></div>
          {_weight_bar(record["model_weights"]["w_odds_edge"], DEFAULT_WEIGHTS["w_odds_edge"], 0.3, 3.0)}
        </div>
      </div>
      <div style="margin-top:12px;font-size:0.8rem;color:var(--muted)">
        Trained on <strong style="color:var(--text)">{record["model_weights"].get("samples",0)}</strong> resolved picks &nbsp;·&nbsp;
        Model v{record["model_weights"].get("version",1)} &nbsp;·&nbsp;
        Last updated {record["model_weights"].get("last_updated","never")}
      </div>
    </div>

  </div>

</div>
<div class="footer">
  Auto-generated by Sports Scout · Odds via DraftKings / The Odds API · Data via ESPN · Not financial advice
</div>
<script>
  function toggleCard(id) {{
    const panel = document.getElementById(id);
    const icon  = document.getElementById('icon-' + id);
    const open  = panel.classList.toggle('open');
    icon.textContent = open ? '▲ Close' : '▼ Why?';
    icon.style.color = open ? 'var(--blue)' : '';
  }}

  function switchTab(name, btn) {{
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    btn.classList.add('active');
    localStorage.setItem('scout-tab', name);
  }}

  function toggleDay(header) {{
    const block   = header.parentElement;
    const content = block.querySelector('.day-content');
    block.classList.toggle('open');
    content.classList.toggle('open');
  }}

  // Restore last active tab
  const savedTab = localStorage.getItem('scout-tab');
  if (savedTab) {{
    const btn = document.querySelector(`.tab-btn[onclick*="${{savedTab}}"]`);
    if (btn) switchTab(savedTab, btn);
  }}

  // Auto-open today's day block in history if navigating there
  document.addEventListener('DOMContentLoaded', () => {{
    const first = document.querySelector('#tab-history .day-block');
    if (first) {{
      first.classList.add('open');
      first.querySelector('.day-content').classList.add('open');
    }}
  }});
</script>
</body>
</html>"""

    with open(DASHBOARD, "w") as f:
        f.write(html)
    log.info(f"Dashboard written → {DASHBOARD}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info(f"Sports Scout starting — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    record = load_record()

    # 1. Resolve yesterday's pending picks
    update_results(record)
    save_record(record)

    # 2. Fetch today's games + odds, run analysis
    today_str = date.today().strftime("%Y%m%d")
    today_picks = []

    weights = record["model_weights"]
    log.info(f"Model weights: win_pct={weights['w_win_pct']} home={weights['w_home_adv']} "
             f"quality={weights['w_quality']} odds={weights['w_odds_edge']} (v{weights.get('version',1)}, n={weights.get('samples',0)})")

    for sport, cfg in SPORTS.items():
        log.info(f"Fetching {sport} games...")
        games = fetch_scoreboard(cfg["espn"], today_str)
        if not games:
            log.info(f"  No {sport} games today.")
            continue
        log.info(f"  Found {len(games)} {sport} games.")

        dk_events = fetch_dk_direct(cfg["dk_id"])

        sport_picks = []
        for game in games:
            if game["completed"]:
                continue
            dk = match_dk_event(game, dk_events)
            pick = analyze_game(game, dk, sport, weights)
            if pick:
                sport_picks.append(pick)

        # Keep only the top MAX_PICKS_PER_SPORT by confidence for this sport
        sport_picks.sort(key=lambda p: -p["confidence"])
        for pick in sport_picks[:MAX_PICKS_PER_SPORT]:
            today_picks.append(pick)
            log.info(f"  ✓ Pick: {pick['pick']} (conf {pick['confidence']}%)")
        skipped = len(sport_picks) - min(len(sport_picks), MAX_PICKS_PER_SPORT)
        if skipped:
            log.info(f"  Skipped {skipped} lower-confidence {sport} picks.")

    # Keep only the top MAX_PICKS_TOTAL overall
    today_picks.sort(key=lambda p: -p["confidence"])
    today_picks = today_picks[:MAX_PICKS_TOTAL]
    log.info(f"Final slate: {len(today_picks)} picks (cap: {MAX_PICKS_TOTAL})")

    # 3. Save new picks to record (avoid duplicates by event_id)
    existing_ids = {p.get("event_id") for p in record["picks_history"]}
    new_picks = [p for p in today_picks if p.get("event_id") not in existing_ids]
    record["picks_history"].extend(new_picks)
    save_record(record)
    log.info(f"Saved {len(new_picks)} new picks.")

    # 4. Generate dashboard
    generate_dashboard(record, today_picks)
    log.info(f"Done. {len(today_picks)} picks today, overall record {record['overall']['wins']}-{record['overall']['losses']}.")

    # 5. Publish to GitHub Pages (if repo is configured)
    publish_to_github()


def publish_to_github():
    """
    Commit dashboard.html + record.json and push to GitHub Pages.
    Silently skips if the repo has no remote (not yet set up).
    """
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=SCRIPT_DIR, capture_output=True, text=True
        )
        if result.returncode != 0:
            log.info("GitHub Pages: no remote configured, skipping publish.")
            return

        today = date.today().isoformat()
        cmds = [
            ["git", "add", "dashboard.html", "record.json"],
            ["git", "commit", "--allow-empty", "-m", f"scout: {today}"],
            ["git", "push", "origin", "main"],
        ]
        for cmd in cmds:
            r = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True)
            if r.returncode != 0 and "nothing to commit" not in r.stdout + r.stderr:
                log.warning(f"GitHub push step failed: {' '.join(cmd)}\n{r.stderr.strip()}")
                return

        log.info("✓ Dashboard published to GitHub Pages.")
    except Exception as e:
        log.warning(f"GitHub Pages publish skipped: {e}")


if __name__ == "__main__":
    main()
