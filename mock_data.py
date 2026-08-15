"""
Générateur de données synthétiques pour faire tourner l'application sans
aucune clé API : utile pour tester l'interface, faire une démo, ou développer
hors-ligne. Simule un historique de saison réaliste (forces d'équipes
tirées aléatoirement) puis génère de vrais matchs à venir avec des cotes
bookmaker cohérentes (calculées à partir des probabilités "vraies" du
simulateur, additionnées d'une marge bookmaker) — avec, volontairement,
quelques écarts injectés pour illustrer des value bets en mode démo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import LEAGUES

_TEAMS_BY_LEAGUE = {
    "Premier League": [
        "Arsenal", "Manchester City", "Liverpool", "Chelsea", "Manchester United",
        "Tottenham", "Newcastle", "Aston Villa", "Brighton", "West Ham",
    ],
    "La Liga": [
        "Real Madrid", "Barcelona", "Atletico Madrid", "Real Sociedad", "Villarreal",
        "Athletic Bilbao", "Real Betis", "Sevilla", "Valencia", "Girona",
    ],
    "Bundesliga": [
        "Bayern Munich", "Bayer Leverkusen", "RB Leipzig", "Borussia Dortmund", "Union Berlin",
        "Freiburg", "Eintracht Frankfurt", "Wolfsburg", "Mainz", "Stuttgart",
    ],
    "Serie A": [
        "Inter Milan", "AC Milan", "Juventus", "Napoli", "Atalanta",
        "AS Roma", "Lazio", "Fiorentina", "Bologna", "Torino",
    ],
    "Ligue 1": [
        "Paris Saint-Germain", "Monaco", "Marseille", "Lille", "Lyon",
        "Lens", "Rennes", "Nice", "Toulouse", "Reims",
    ],
}


def _simulate_team_strengths(teams: list[str], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {t: {"attack": rng.normal(0, 0.35), "defense": rng.normal(0, 0.35)} for t in teams}


def generate_history(league: str, n_rounds: int = 26, seed: int = 42) -> pd.DataFrame:
    """Simule `n_rounds` journées de championnat (chaque équipe joue une fois par journée)."""
    teams = _TEAMS_BY_LEAGUE[league]
    strengths = _simulate_team_strengths(teams, seed=hash(league) % (2**31))
    rng = np.random.default_rng(hash(league) % (2**31))
    home_adv = 0.25

    rows = []
    today = pd.Timestamp.today().normalize()
    for rnd in range(n_rounds):
        shuffled = teams.copy()
        rng.shuffle(shuffled)
        pairs = list(zip(shuffled[: len(shuffled) // 2], shuffled[len(shuffled) // 2 :]))
        match_date = today - pd.Timedelta(days=(n_rounds - rnd) * 7)
        for home, away in pairs:
            lam = np.exp(strengths[home]["attack"] - strengths[away]["defense"] + home_adv)
            mu = np.exp(strengths[away]["attack"] - strengths[home]["defense"])
            hg = rng.poisson(lam)
            ag = rng.poisson(mu)
            rows.append(
                {
                    "date": match_date,
                    "home_team": home,
                    "away_team": away,
                    "home_goals": int(hg),
                    "away_goals": int(ag),
                    "status": "FT",
                }
            )
    return pd.DataFrame(rows)


def generate_upcoming_fixtures_with_odds(league: str, n_matches: int = 5, seed: int = 7) -> pd.DataFrame:
    """
    Génère de faux matchs à venir + des cotes bookmaker "réalistes" : on part
    des vraies probabilités du simulateur, on ajoute la marge bookmaker
    (overround ~6%), puis on bruite légèrement pour émuler le fait qu'un
    bookmaker n'a jamais une évaluation parfaite du marché — ce qui fait
    naturellement apparaître quelques value bets.
    """
    teams = _TEAMS_BY_LEAGUE[league]
    strengths = _simulate_team_strengths(teams, seed=hash(league) % (2**31))
    rng = np.random.default_rng(seed + hash(league) % (2**31))
    home_adv = 0.25

    shuffled = teams.copy()
    rng.shuffle(shuffled)
    pairs = list(zip(shuffled[::2], shuffled[1::2]))[:n_matches]

    rows = []
    kickoff = pd.Timestamp.today().normalize() + pd.Timedelta(days=1, hours=15)
    for i, (home, away) in enumerate(pairs):
        lam = np.exp(strengths[home]["attack"] - strengths[away]["defense"] + home_adv)
        mu = np.exp(strengths[away]["attack"] - strengths[home]["defense"])

        # probabilités "vraies" approximées par simulation Monte Carlo rapide
        sims_h = rng.poisson(lam, 20000)
        sims_a = rng.poisson(mu, 20000)
        p_home = float(np.mean(sims_h > sims_a))
        p_draw = float(np.mean(sims_h == sims_a))
        p_away = float(np.mean(sims_h < sims_a))
        p_over = float(np.mean((sims_h + sims_a) > 2.5))
        p_under = 1 - p_over

        overround = 1.06
        noise = lambda: rng.normal(0, 0.018)  # léger bruit du bookmaker vs la "vraie" proba

        def to_odds(p):
            p_book = min(max(p + noise(), 0.06), 0.95)
            return round(min(overround / p_book, 12.0), 2)

        rows.append(
            {
                "fixture_id": f"demo-{league}-{i}",
                "date": kickoff + pd.Timedelta(hours=i * 2),
                "commence_time": kickoff + pd.Timedelta(hours=i * 2),
                "league": league,
                "home_team": home,
                "away_team": away,
                "home_goals": None,
                "away_goals": None,
                "status": "NS",
                "odds_home": to_odds(p_home),
                "odds_draw": to_odds(p_draw),
                "odds_away": to_odds(p_away),
                "odds_over25": to_odds(p_over),
                "odds_under25": to_odds(p_under),
                "n_bookmakers": int(rng.integers(4, 12)),
            }
        )
    return pd.DataFrame(rows)


def available_demo_leagues() -> list[str]:
    return [l for l in LEAGUES if l in _TEAMS_BY_LEAGUE]
