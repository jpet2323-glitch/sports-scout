#!/usr/bin/env python3
"""
ESPN Fantasy Football League Viewer
Fetches standings and current-week matchups for a private ESPN
Fantasy Football league using authenticated cookies (espn_s2 + SWID).
"""

import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    sys.exit("Missing dependency: pip3 install requests")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
OUT_FILE     = SCRIPT_DIR / "fantasy.json"

# ── Config ─────────────────────────────────────────────────────────────────────
LEAGUE_ID    = int(os.getenv("FANTASY_LEAGUE_ID", "5814407"))
ESPN_S2      = os.getenv("ESPN_S2", "")
ESPN_SWID    = os.getenv("ESPN_SWID", "")
FANTASY_BASE = "https://fantasy.espn.com/apis/v3/games/ffl/seasons"


def current_season_year() -> int:
    """NFL season is labeled by the year it starts (Sept-Feb)."""
    today = date.today()
    return today.year if today.month >= 3 else today.year - 1


def fantasy_get(views: list, season: Optional[int] = None) -> Optional[dict]:
    season = season or current_season_year()
    url = f"{FANTASY_BASE}/{season}/segments/0/leagues/{LEAGUE_ID}"
    params = [("view", v) for v in views]
    cookies = {}
    if ESPN_S2 and ESPN_SWID:
        cookies = {"espn_s2": ESPN_S2, "SWID": ESPN_SWID}

    try:
        r = requests.get(url, params=params, cookies=cookies, timeout=10)
    except requests.exceptions.RequestException as e:
        print(f"ESPN Fantasy request failed: {e}")
        return None

    if r.status_code == 401 or r.status_code == 403:
        print(f"ESPN rejected the request ({r.status_code}) — this league is private.")
        print("Set ESPN_S2 and ESPN_SWID in your local .env file (see .env.example),")
        print("then run: source .env && python3 fantasy.py")
        return None
    if r.status_code == 404:
        print(f"League {LEAGUE_ID} not found for season {season}.")
        print("Double-check FANTASY_LEAGUE_ID and that the season year is right.")
        return None

    try:
        r.raise_for_status()
        return r.json()
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"ESPN Fantasy request failed: {e}")
        return None


def team_display_name(t: dict) -> str:
    name = t.get("name")
    if name:
        return name
    combo = f"{t.get('location', '')} {t.get('nickname', '')}".strip()
    return combo or f"Team {t.get('id')}"


def parse_standings(data: dict) -> list:
    teams = []
    for t in data.get("teams", []):
        record = t.get("record", {}).get("overall", {})
        teams.append({
            "team_id":        t.get("id"),
            "name":           team_display_name(t),
            "wins":           record.get("wins", 0),
            "losses":         record.get("losses", 0),
            "ties":           record.get("ties", 0),
            "points_for":     record.get("pointsFor", 0),
            "points_against": record.get("pointsAgainst", 0),
        })
    teams.sort(key=lambda t: (-t["wins"], -t["points_for"]))
    return teams


def parse_matchups(data: dict) -> list:
    team_names = {t.get("id"): team_display_name(t) for t in data.get("teams", [])}
    current_week = data.get("status", {}).get("currentMatchupPeriod") or data.get("scoringPeriodId")

    matchups = []
    for m in data.get("schedule", []):
        if m.get("matchupPeriodId") != current_week:
            continue
        home = m.get("home", {}) or {}
        away = m.get("away")
        entry = {
            "week":       current_week,
            "home":       team_names.get(home.get("teamId"), "?"),
            "home_score": round(home.get("totalPoints", 0) or 0, 1),
            "away":       None,
            "away_score": None,
        }
        if away:
            entry["away"] = team_names.get(away.get("teamId"), "?")
            entry["away_score"] = round(away.get("totalPoints", 0) or 0, 1)
        matchups.append(entry)
    return matchups


def main():
    data = fantasy_get(["mTeam", "mStandings", "mSettings", "mMatchupScore"])
    if not data:
        sys.exit(1)

    league_name = data.get("settings", {}).get("name", f"League {LEAGUE_ID}")
    standings = parse_standings(data)
    matchups = parse_matchups(data)

    print(f"═══ {league_name} ({LEAGUE_ID}) ═══\n")

    print("STANDINGS")
    print(f"{'Team':<25}{'W':>4}{'L':>4}{'T':>4}{'PF':>10}{'PA':>10}")
    for t in standings:
        print(f"{t['name']:<25}{t['wins']:>4}{t['losses']:>4}{t['ties']:>4}"
              f"{t['points_for']:>10.1f}{t['points_against']:>10.1f}")

    if matchups:
        print(f"\nWEEK {matchups[0]['week']} MATCHUPS")
        for m in matchups:
            if m["away"] is None:
                print(f"  {m['home']} — BYE")
            else:
                print(f"  {m['home']:<25}{m['home_score']:>7.1f}   vs   "
                      f"{m['away_score']:<7.1f}{m['away']}")

    with open(OUT_FILE, "w") as f:
        json.dump({
            "league_id": LEAGUE_ID,
            "league_name": league_name,
            "standings": standings,
            "matchups": matchups,
        }, f, indent=2)
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
