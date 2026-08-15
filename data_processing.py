"""
Fusion des données API-Football (fixtures, forme, historique) avec les
cotes The-Odds-API.

Les deux fournisseurs ne nomment pas toujours les équipes de la même façon
("Paris Saint Germain" vs "Paris Saint-Germain", "Man United" vs "Manchester
United"...). On applique donc un matching approché (difflib) plutôt qu'une
jointure stricte sur le nom, avec une normalisation basique en amont.
"""
from __future__ import annotations

import difflib

import pandas as pd

_SUFFIXES = [" fc", " cf", " afc", " sc", " ac"]


def normalize_name(name: str) -> str:
    n = (name or "").lower().strip()
    n = n.replace(".", "").replace("-", " ")
    for suffix in _SUFFIXES:
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    return " ".join(n.split())


def match_team_name(name: str, choices: list[str], cutoff: float = 0.6) -> str | None:
    """Retrouve, parmi `choices`, le nom le plus proche de `name`."""
    if not choices:
        return None
    normalized_choices = {normalize_name(c): c for c in choices}
    best = difflib.get_close_matches(normalize_name(name), list(normalized_choices.keys()), n=1, cutoff=cutoff)
    return normalized_choices[best[0]] if best else None


def merge_fixtures_with_odds(fixtures_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Associe chaque match à venir (API-Football) à ses cotes moyennes
    (The-Odds-API) en tolérant les différences de nommage des équipes.
    Les matchs sans cote correspondante sont exclus (rien à comparer).
    """
    if fixtures_df.empty or odds_df.empty:
        return pd.DataFrame()

    odds_home_teams = odds_df["home_team"].dropna().unique().tolist()
    merged_rows = []
    for _, row in fixtures_df.iterrows():
        matched_home = match_team_name(row["home_team"], odds_home_teams)
        if matched_home is None:
            continue
        candidates = odds_df[odds_df["home_team"] == matched_home]
        if candidates.empty:
            continue
        # en cas d'ambiguïté (plusieurs matchs pour la même équipe à domicile
        # sur la fenêtre récupérée), on vérifie aussi l'équipe à l'extérieur
        if len(candidates) > 1:
            matched_away = match_team_name(row["away_team"], candidates["away_team"].tolist())
            if matched_away is not None:
                candidates = candidates[candidates["away_team"] == matched_away]
        odds_row = candidates.iloc[0]
        merged_rows.append({**row.to_dict(), **odds_row.to_dict()})

    return pd.DataFrame(merged_rows)


def build_team_history(past_fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare l'historique au format attendu par le modèle Dixon-Coles :
    colonnes date / home_team / away_team / home_goals / away_goals,
    limité aux matchs réellement terminés avec un score connu.
    """
    if past_fixtures_df.empty:
        return past_fixtures_df
    df = past_fixtures_df[past_fixtures_df["status"].isin(["FT", "FINISHED"])].copy()
    df = df.dropna(subset=["home_goals", "away_goals"])
    return df[["date", "home_team", "away_team", "home_goals", "away_goals"]]
