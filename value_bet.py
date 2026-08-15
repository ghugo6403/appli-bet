"""
Détection de "Value Bets" : compare la probabilité réelle calculée par le
modèle Dixon-Coles à la probabilité implicite de la cote moyenne du marché.

Formule de l'edge (avantage attendu) pour une mise unitaire :

    edge = (probabilité_modèle * cote) - 1

Interprétation : si edge = 0.08, cela signifie que pour 1€ misé, l'espérance
de gain du pari est de +0.08€ *si notre probabilité modèle est correcte*.
Un edge > 0 indique que le bookmaker sous-évalue implicitement la probabilité
de l'issue (sa cote est "trop généreuse"). On ne signale un Value Bet que si
cet edge dépasse un seuil (VALUE_BET_EDGE_THRESHOLD, 5% par défaut) pour
laisser une marge de sécurité face à l'incertitude du modèle.
"""
from __future__ import annotations

import pandas as pd

from config import VALUE_BET_EDGE_THRESHOLD

# (libellé affiché, colonne probabilité modèle, colonne cote bookmaker)
MARKETS = [
    ("1 (Domicile)", "p_home", "odds_home"),
    ("X (Nul)", "p_draw", "odds_draw"),
    ("2 (Extérieur)", "p_away", "odds_away"),
    ("Over 2.5", "p_over25", "odds_over25"),
    ("Under 2.5", "p_under25", "odds_under25"),
]


def implied_probability(odds: float | None) -> float | None:
    """Probabilité implicite par la cote, hors marge bookmaker (1/cote)."""
    if not odds or odds <= 0:
        return None
    return 1 / odds


def compute_edge(model_prob: float | None, odds: float | None) -> float | None:
    if model_prob is None or odds is None:
        return None
    return round((model_prob * odds) - 1, 4)


def detect_value_bets(match_row: dict, threshold: float = VALUE_BET_EDGE_THRESHOLD) -> list[dict]:
    """
    Parcourt les marchés (1X2, Over/Under 2.5) d'un match et retourne la
    liste des value bets détectés (edge > threshold), triés par edge
    décroissant.
    """
    value_bets = []
    for label, prob_col, odds_col in MARKETS:
        prob = match_row.get(prob_col)
        odds = match_row.get(odds_col)
        edge = compute_edge(prob, odds)
        if edge is not None and edge > threshold:
            implied = implied_probability(odds)
            value_bets.append(
                {
                    "market": label,
                    "model_probability": prob,
                    "implied_probability": round(implied, 4) if implied else None,
                    "odds": odds,
                    "edge": edge,
                    "reliable": match_row.get("reliable", True),
                }
            )
    return sorted(value_bets, key=lambda v: v["edge"], reverse=True)


def build_value_bets_table(matches_df: pd.DataFrame, threshold: float = VALUE_BET_EDGE_THRESHOLD) -> pd.DataFrame:
    """Construit un tableau plat (une ligne par value bet détecté) pour l'affichage."""
    rows = []
    for _, row in matches_df.iterrows():
        row_dict = row.to_dict()
        for vb in detect_value_bets(row_dict, threshold):
            rows.append(
                {
                    "date": row_dict.get("date") or row_dict.get("commence_time"),
                    "league": row_dict.get("league"),
                    "home_team": row_dict.get("home_team"),
                    "away_team": row_dict.get("away_team"),
                    **vb,
                }
            )
    columns = [
        "date", "league", "home_team", "away_team", "market",
        "model_probability", "implied_probability", "odds", "edge", "reliable",
    ]
    return pd.DataFrame(rows, columns=columns)
