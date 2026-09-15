#!/usr/bin/env python3
"""
ESPN Fantasy Football — Weekly Power Rankings
Computes a blended power score (record + season scoring + recent form)
for each team, tracks week-over-week movement, and emails the report.
"""

import json
import os
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path

import fantasy  # reuses fantasy_get / team_display_name / LEAGUE_ID

SCRIPT_DIR   = Path(__file__).parent
HISTORY_FILE = SCRIPT_DIR / "power_rankings.json"

GMAIL_ADDRESS      = os.getenv("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
RECIPIENT_EMAILS   = [e.strip() for e in os.getenv("RECIPIENT_EMAILS", GMAIL_ADDRESS).split(",") if e.strip()]

# Power score weights (sum to 100)
W_RECORD = 45   # season win %
W_SEASON = 35   # season points-per-week, relative to the league's best
W_RECENT = 20   # last-3-weeks points-per-week, relative to the league's best


def completed_weeks(data: dict) -> dict:
    """matchupPeriodId -> list of matchup dicts, for weeks with a final score on both sides."""
    weeks = {}
    for m in data.get("schedule", []):
        home, away = m.get("home") or {}, m.get("away")
        if not away:
            continue  # bye
        if not home.get("totalPoints") or not away.get("totalPoints"):
            continue  # not played yet
        wk = m.get("matchupPeriodId")
        weeks.setdefault(wk, []).append(m)
    return weeks


def build_team_stats(data: dict):
    """Return (team_stats dict keyed by team_id, last_completed_week or None)."""
    weeks = completed_weeks(data)
    if not weeks:
        return {}, None
    last_week = max(weeks.keys())

    points_by_team = {}  # team_id -> list of (week, points)
    for wk, matchups in weeks.items():
        for m in matchups:
            for side in ("home", "away"):
                s = m.get(side) or {}
                tid = s.get("teamId")
                if tid is not None:
                    points_by_team.setdefault(tid, []).append((wk, s.get("totalPoints", 0)))

    stats = {}
    for t in data.get("teams", []):
        tid = t.get("id")
        record = t.get("record", {}).get("overall", {})
        history = sorted(points_by_team.get(tid, []))
        scores = [p for _, p in history]
        recent = scores[-3:] if scores else []
        last_week_points = next((p for wk, p in history if wk == last_week), None)

        stats[tid] = {
            "team_id":          tid,
            "name":             fantasy.team_display_name(t),
            "wins":             record.get("wins", 0),
            "losses":           record.get("losses", 0),
            "ties":             record.get("ties", 0),
            "streak_type":      record.get("streakType"),
            "streak_length":    record.get("streakLength", 0),
            "season_avg":       round(sum(scores) / len(scores), 1) if scores else 0.0,
            "recent_avg":       round(sum(recent) / len(recent), 1) if recent else 0.0,
            "last_week_points": round(last_week_points, 1) if last_week_points is not None else None,
        }
    return stats, last_week


def score_teams(stats: dict) -> list:
    max_season = max((s["season_avg"] for s in stats.values()), default=1) or 1
    max_recent = max((s["recent_avg"] for s in stats.values()), default=1) or 1

    ranked = []
    for s in stats.values():
        games = s["wins"] + s["losses"] + s["ties"]
        win_pct = (s["wins"] + 0.5 * s["ties"]) / games if games else 0.0
        power = (
            W_RECORD * win_pct
            + W_SEASON * (s["season_avg"] / max_season)
            + W_RECENT * (s["recent_avg"] / max_recent)
        )
        ranked.append({**s, "power_score": round(power, 1)})

    ranked.sort(key=lambda s: -s["power_score"])
    for i, s in enumerate(ranked, start=1):
        s["rank"] = i
    return ranked


def load_history() -> dict:
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return {"league_id": fantasy.LEAGUE_ID, "weeks": {}}


def save_history(history: dict, week: int, ranked: list):
    history["weeks"][str(week)] = ranked
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def apply_movement(ranked: list, history: dict, week: int) -> list:
    prev = history.get("weeks", {}).get(str(week - 1))
    prev_rank = {t["team_id"]: t["rank"] for t in prev} if prev else {}
    for t in ranked:
        old = prev_rank.get(t["team_id"])
        t["movement"] = (old - t["rank"]) if old else 0
    return ranked


def blurb(t: dict) -> str:
    if t["recent_avg"] and t["season_avg"] and t["recent_avg"] >= t["season_avg"] * 1.15:
        note = "trending up \U0001F525"
    elif t["recent_avg"] and t["season_avg"] and t["recent_avg"] <= t["season_avg"] * 0.85:
        note = "cooling off ❄️"
    else:
        note = "steady"
    if t.get("streak_type") == "WIN" and t.get("streak_length", 0) >= 2:
        note += f" — {t['streak_length']}-game win streak"
    elif t.get("streak_type") == "LOSS" and t.get("streak_length", 0) >= 2:
        note += f" — {t['streak_length']}-game skid"
    return note


def render_html(league_name: str, week: int, ranked: list) -> str:
    medals = {1: "\U0001F947", 2: "\U0001F948", 3: "\U0001F949"}
    rows = []
    for t in ranked:
        arrow = "▲" if t["movement"] > 0 else "▼" if t["movement"] < 0 else "–"
        color = "#1a7f37" if t["movement"] > 0 else "#cf222e" if t["movement"] < 0 else "#6e7781"
        rank_label = f"{medals.get(t['rank'], '')} {t['rank']}".strip()
        last_wk = f"{t['last_week_points']:.1f}" if t["last_week_points"] is not None else "—"
        record = f"{t['wins']}-{t['losses']}" + (f"-{t['ties']}" if t["ties"] else "")
        rows.append(f"""
          <tr>
            <td style="padding:8px 12px;border-bottom:1px solid #e1e4e8;font-weight:600;">{rank_label}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #e1e4e8;color:{color};">{arrow}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #e1e4e8;">
              <div style="font-weight:600;">{t['name']}</div>
              <div style="font-size:12px;color:#6e7781;">{blurb(t)}</div>
            </td>
            <td style="padding:8px 12px;border-bottom:1px solid #e1e4e8;text-align:center;">{record}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #e1e4e8;text-align:right;font-weight:600;">{t['power_score']}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #e1e4e8;text-align:right;">{t['season_avg']}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #e1e4e8;text-align:right;">{last_wk}</td>
          </tr>""")

    return f"""\
<html>
  <body style="font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:#f6f8fa;padding:24px;">
    <div style="max-width:640px;margin:0 auto;background:#ffffff;border-radius:8px;overflow:hidden;border:1px solid #e1e4e8;">
      <div style="background:#0d1117;padding:20px 24px;">
        <h1 style="color:#ffffff;margin:0;font-size:20px;">\U0001F3C8 {league_name} — Week {week} Power Rankings</h1>
      </div>
      <table style="width:100%;border-collapse:collapse;font-size:14px;color:#24292f;">
        <thead>
          <tr style="background:#f6f8fa;">
            <th style="padding:8px 12px;text-align:left;">Rank</th>
            <th style="padding:8px 12px;"></th>
            <th style="padding:8px 12px;text-align:left;">Team</th>
            <th style="padding:8px 12px;">Record</th>
            <th style="padding:8px 12px;text-align:right;">Power</th>
            <th style="padding:8px 12px;text-align:right;">Season Avg</th>
            <th style="padding:8px 12px;text-align:right;">Last Wk</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      <div style="padding:12px 24px;font-size:12px;color:#6e7781;">
        Power score = 45% record + 35% season scoring + 20% recent form (last 3 weeks).
      </div>
    </div>
  </body>
</html>"""


def send_email(subject: str, html_body: str):
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        print("GMAIL_ADDRESS / GMAIL_APP_PASSWORD not set — printing report instead of emailing.\n")
        print(html_body)
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = ", ".join(RECIPIENT_EMAILS)
    msg.set_content("This email requires HTML support to view the power rankings.")
    msg.add_alternative(html_body, subtype="html")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        smtp.send_message(msg)
    print(f"Emailed power rankings to {', '.join(RECIPIENT_EMAILS)}")


def main():
    data = fantasy.fantasy_get(["mTeam", "mStandings", "mSettings", "mMatchupScore"])
    if not data:
        sys.exit(1)

    league_name = data.get("settings", {}).get("name", f"League {fantasy.LEAGUE_ID}")
    stats, week = build_team_stats(data)
    if week is None:
        print("No completed weeks yet this season — nothing to rank.")
        return

    ranked = score_teams(stats)
    history = load_history()
    ranked = apply_movement(ranked, history, week)

    subject = f"\U0001F3C8 {league_name} — Week {week} Power Rankings"
    html_body = render_html(league_name, week, ranked)
    send_email(subject, html_body)

    save_history(history, week, ranked)
    print(f"Saved week {week} rankings to {HISTORY_FILE}")


if __name__ == "__main__":
    main()
