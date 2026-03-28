#!/usr/bin/env python3
"""
Regrade all completed picks in record.json from scratch using live ESPN scores.
Recalculates wins/losses/units and prints a before/after comparison.
Run once manually: python3 regrade.py
"""

import json, re, requests
from pathlib import Path
from datetime import datetime

SCRIPT_DIR  = Path(__file__).parent
RECORD_FILE = SCRIPT_DIR / "record.json"
ESPN_BASE   = "https://site.api.espn.com/apis/site/v2/sports"

SPORTS_ESPN = {
    "NBA":   "basketball/nba",
    "NHL":   "hockey/nhl",
    "MLB":   "baseball/mlb",
    "NCAAB": "basketball/mens-college-basketball",
}

IMPORTED_WINS   = 11  # your 11-6 opening record
IMPORTED_LOSSES = 6

def fetch_score(sport_espn: str, event_id: str):
    """Return (home_score, away_score, home_team, away_team) for a completed game."""
    try:
        url = f"{ESPN_BASE}/{sport_espn}/summary"
        r = requests.get(url, params={"event": event_id}, timeout=10)
        r.raise_for_status()
        d = r.json()
        comp = d.get("header", {}).get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
        return (
            int(home.get("score", 0) or 0),
            int(away.get("score", 0) or 0),
            home.get("team", {}).get("name", ""),
            away.get("team", {}).get("name", ""),
        )
    except Exception:
        return None, None, None, None


def grade_pick(pick: dict, h_score: int, a_score: int) -> str:
    if h_score is None:
        return pick.get("result", "pending")

    pick_text  = pick.get("pick", "")
    pick_type  = pick.get("pick_type", "Stat Edge")
    home_won   = h_score > a_score
    away_won   = a_score > h_score
    picking_home = pick.get("home", "").lower() in pick_text.lower()

    if pick_type in ("Moneyline", "Stat Edge"):
        return "win" if (picking_home and home_won) or (not picking_home and away_won) else "loss"

    elif pick_type == "Spread":
        m = re.search(r"([+-]?\d+\.?\d*)\s*\(", pick_text)
        if not m:
            return "loss"
        spread = float(m.group(1))
        if picking_home:
            adj = h_score + spread
            if adj > a_score:   return "win"
            if adj == a_score:  return "push"
            return "loss"
        else:
            adj = a_score + spread
            if adj > h_score:   return "win"
            if adj == h_score:  return "push"
            return "loss"

    return "loss"


def main():
    with open(RECORD_FILE) as f:
        record = json.load(f)

    picks = record["picks_history"]
    graded = [p for p in picks if p.get("result") not in ("pending", None)]

    print(f"\n{'='*65}")
    print(f"  REGRADING {len(graded)} completed picks")
    print(f"{'='*65}\n")

    changes = 0
    for pick in graded:
        old_result = pick.get("result")
        sport      = pick.get("sport", "NBA")
        event_id   = pick.get("event_id")

        # Re-fetch score from ESPN
        sport_espn = SPORTS_ESPN.get(sport, "basketball/nba")
        hs, as_, ht, at = fetch_score(sport_espn, event_id)

        # Use stored scores if ESPN can't return them (old games may expire)
        if hs is None:
            hs = pick.get("home_score")
            as_ = pick.get("away_score")

        new_result = grade_pick(pick, hs, as_)

        status = "  OK " if new_result == old_result else "✗ FIX"
        if new_result != old_result:
            changes += 1
            print(f"{status} {pick['date']} {sport:5} | {pick['game']}")
            print(f"       Pick: {pick['pick']}")
            print(f"       Score: home {hs} – away {as_}")
            print(f"       Was: {old_result.upper()}  →  Now: {new_result.upper()}\n")

        # Apply correction
        pick["result"]     = new_result
        pick["home_score"] = hs
        pick["away_score"] = as_

    if changes == 0:
        print("  All picks graded correctly — no changes needed.\n")

    # Rebuild totals from scratch
    wins = losses = pushes = 0
    by_sport = {s: {"wins": 0, "losses": 0, "pushes": 0} for s in SPORTS_ESPN}

    for pick in graded:
        r = pick.get("result")
        if r == "win":    wins   += 1
        elif r == "loss": losses += 1
        elif r == "push": pushes += 1
        sp = pick.get("sport", "NBA")
        if sp in by_sport:
            if r == "win":    by_sport[sp]["wins"]   += 1
            elif r == "loss": by_sport[sp]["losses"] += 1
            elif r == "push": by_sport[sp]["pushes"] += 1

    # Add imported opening record
    total_wins   = IMPORTED_WINS   + wins
    total_losses = IMPORTED_LOSSES + losses
    total_wagered = IMPORTED_WINS + IMPORTED_LOSSES + wins + losses + pushes
    # Approx units: imported record assumed flat -110
    units_net = round(
        (IMPORTED_WINS * 0.91 - IMPORTED_LOSSES) +
        (wins * 0.91 - losses),
        2
    )
    roi = round(units_net / total_wagered * 100, 1) if total_wagered else 0

    record["overall"]["wins"]          = total_wins
    record["overall"]["losses"]        = total_losses
    record["overall"]["pushes"]        = pushes
    record["overall"]["units_wagered"] = total_wagered
    record["overall"]["units_net"]     = units_net
    record["overall"]["roi_pct"]       = roi
    record["by_sport"]                 = by_sport
    record["meta"]["last_updated"]     = datetime.today().date().isoformat()

    with open(RECORD_FILE, "w") as f:
        json.dump(record, f, indent=2)

    print(f"{'='*65}")
    print(f"  CORRECTED RECORD")
    print(f"{'='*65}")
    print(f"  Overall:   {total_wins}-{total_losses}  ({roi}% ROI)")
    for sp, st in by_sport.items():
        t = st["wins"] + st["losses"] + st["pushes"]
        pct = round(st["wins"]/t*100, 1) if t else 0
        print(f"  {sp:6}:    {st['wins']}-{st['losses']}  ({pct}%)")
    print(f"\n  {changes} picks corrected. record.json updated.\n")


if __name__ == "__main__":
    main()
