"""
Client d'ingestion pour l'API-Football (https://www.api-football.com/).

Fournit :
  - les matchs à venir d'une ligue (pour construire le tableau de bord)
  - l'historique des matchs joués d'une ligue (pour calibrer le modèle Dixon-Coles)
  - les statistiques d'une équipe (forme, buts marqués/encaissés, xG si disponible)
  - les confrontations directes (head-to-head)

Toutes les réponses brutes de l'API sont converties en DataFrame pandas pour
être directement exploitables par les modules de modélisation.
"""
from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

from config import API_FOOTBALL_HOST, API_FOOTBALL_KEY, CURRENT_SEASON, PAST_FIXTURES_LOOKBACK


class APIFootballClient:
    def __init__(self, api_key: str = API_FOOTBALL_KEY):
        self.base_url = f"https://{API_FOOTBALL_HOST}"
        self.headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": API_FOOTBALL_HOST,
        }
        self.enabled = bool(api_key)

    # -- appel générique -----------------------------------------------
    def _get(self, endpoint: str, params: dict | None = None) -> list:
        if not self.enabled:
            return []
        try:
            resp = requests.get(
                f"{self.base_url}/{endpoint}", headers=self.headers, params=params, timeout=15
            )
            resp.raise_for_status()
            payload = resp.json()
            errors = payload.get("errors")
            if errors:
                st.warning(f"API-Football ({endpoint}) a renvoyé une erreur : {errors}")
            return payload.get("response", [])
        except requests.RequestException as exc:
            st.warning(f"Erreur réseau API-Football ({endpoint}) : {exc}")
            return []

    # -- endpoints --------------------------------------------------------
    def get_upcoming_fixtures(self, league_id: int, season: int = CURRENT_SEASON, next_n: int = 20) -> list:
        """Prochains matchs programmés d'une ligue (utilisé pour le dashboard)."""
        return self._get("fixtures", {"league": league_id, "season": season, "next": next_n})

    def get_past_fixtures(
        self, league_id: int, season: int = CURRENT_SEASON, last_n: int = PAST_FIXTURES_LOOKBACK
    ) -> list:
        """Matchs terminés d'une ligue (utilisé pour calibrer le modèle Dixon-Coles)."""
        return self._get(
            "fixtures", {"league": league_id, "season": season, "last": last_n, "status": "FT"}
        )

    def get_team_statistics(self, team_id: int, league_id: int, season: int = CURRENT_SEASON) -> dict:
        """Stats agrégées d'une équipe sur la saison : forme, buts pour/contre, etc."""
        data = self._get(
            "teams/statistics", {"team": team_id, "league": league_id, "season": season}
        )
        return data if isinstance(data, dict) else {}

    def get_head_to_head(self, team1_id: int, team2_id: int, last_n: int = 10) -> list:
        """Historique des confrontations directes entre deux équipes."""
        return self._get("fixtures/headtohead", {"h2h": f"{team1_id}-{team2_id}", "last": last_n})

    def get_fixture_statistics(self, fixture_id: int) -> list:
        """
        Statistiques détaillées d'un match déjà joué (dont les xG si le plan
        API-Football souscrit les fournit). Utile pour enrichir le modèle plus
        tard ; peut renvoyer une liste vide selon le plan tarifaire.
        """
        return self._get("fixtures/statistics", {"fixture": fixture_id})

    # -- helpers de conversion ------------------------------------------------
    @staticmethod
    def fixtures_to_dataframe(fixtures: list) -> pd.DataFrame:
        rows = []
        for f in fixtures:
            try:
                rows.append(
                    {
                        "fixture_id": f["fixture"]["id"],
                        "date": f["fixture"]["date"],
                        "league_id": f["league"]["id"],
                        "round": f["league"].get("round"),
                        "home_team": f["teams"]["home"]["name"],
                        "home_team_id": f["teams"]["home"]["id"],
                        "away_team": f["teams"]["away"]["name"],
                        "away_team_id": f["teams"]["away"]["id"],
                        "home_goals": f["goals"]["home"],
                        "away_goals": f["goals"]["away"],
                        "status": f["fixture"]["status"]["short"],
                    }
                )
            except (KeyError, TypeError):
                continue
        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    @staticmethod
    def extract_form(team_stats: dict) -> str:
        """Chaîne de forme du type 'WWDLW' (5 derniers résultats)."""
        return team_stats.get("form", "") or ""

    @staticmethod
    def extract_goals_avg(team_stats: dict) -> dict:
        """Moyennes de buts marqués/encaissés à domicile et à l'extérieur."""
        goals = team_stats.get("goals", {})
        try:
            return {
                "avg_scored_home": float(goals["for"]["average"]["home"] or 0),
                "avg_scored_away": float(goals["for"]["average"]["away"] or 0),
                "avg_conceded_home": float(goals["against"]["average"]["home"] or 0),
                "avg_conceded_away": float(goals["against"]["average"]["away"] or 0),
            }
        except (KeyError, TypeError, ValueError):
            return {
                "avg_scored_home": None,
                "avg_scored_away": None,
                "avg_conceded_home": None,
                "avg_conceded_away": None,
            }


@st.cache_data(ttl=3600, show_spinner=False)
def cached_upcoming_fixtures(league_id: int, season: int, next_n: int = 20) -> pd.DataFrame:
    client = APIFootballClient()
    return client.fixtures_to_dataframe(client.get_upcoming_fixtures(league_id, season, next_n))


@st.cache_data(ttl=3600, show_spinner=False)
def cached_past_fixtures(league_id: int, season: int, last_n: int = PAST_FIXTURES_LOOKBACK) -> pd.DataFrame:
    client = APIFootballClient()
    return client.fixtures_to_dataframe(client.get_past_fixtures(league_id, season, last_n))
