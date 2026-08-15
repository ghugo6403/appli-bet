"""
Client d'ingestion pour The-Odds-API (https://the-odds-api.com/).

Récupère les cotes 1X2 (marché "h2h") et Over/Under 2.5 buts (marché
"totals") pour un championnat donné, puis calcule la cote moyenne du marché
en agrégeant tous les bookmakers renvoyés par l'API. C'est cette moyenne de
marché qui sert de référence "cote bookmaker" pour la détection de value bets
(une moyenne est plus robuste qu'un bookmaker isolé, qui peut avoir une cote
erronée ou décalée).
"""
from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

from config import MAX_ODDS_CONSIDERED, ODDS_API_BASE_URL, ODDS_API_KEY, ODDS_MARKETS, ODDS_REGIONS


class OddsAPIClient:
    def __init__(self, api_key: str = ODDS_API_KEY):
        self.api_key = api_key
        self.base_url = ODDS_API_BASE_URL
        self.enabled = bool(api_key)

    def get_odds(self, sport_key: str) -> list:
        """Renvoie la liste brute des événements à venir avec leurs cotes."""
        if not self.enabled:
            return []
        url = f"{self.base_url}/sports/{sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": ODDS_REGIONS,
            "markets": ODDS_MARKETS,
            "oddsFormat": "decimal",
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            st.warning(f"Erreur réseau The-Odds-API ({sport_key}) : {exc}")
            return []

    @staticmethod
    def _avg(values: list[float]) -> float | None:
        clean = [v for v in values if v and 1.0 < v <= MAX_ODDS_CONSIDERED]
        return round(sum(clean) / len(clean), 3) if clean else None

    def events_to_dataframe(self, events: list) -> pd.DataFrame:
        """
        Transforme la réponse brute en DataFrame avec une ligne par match et
        les cotes moyennes 1X2 / Over-Under 2.5.
        """
        rows = []
        for event in events:
            home, away = event.get("home_team"), event.get("away_team")
            home_odds, draw_odds, away_odds = [], [], []
            over_odds, under_odds = [], []
            bookmakers = event.get("bookmakers", [])
            for bm in bookmakers:
                for market in bm.get("markets", []):
                    if market["key"] == "h2h":
                        for outcome in market["outcomes"]:
                            if outcome["name"] == home:
                                home_odds.append(outcome["price"])
                            elif outcome["name"] == away:
                                away_odds.append(outcome["price"])
                            elif outcome["name"] == "Draw":
                                draw_odds.append(outcome["price"])
                    elif market["key"] == "totals":
                        for outcome in market["outcomes"]:
                            if outcome.get("point") == 2.5:
                                if outcome["name"] == "Over":
                                    over_odds.append(outcome["price"])
                                elif outcome["name"] == "Under":
                                    under_odds.append(outcome["price"])
            rows.append(
                {
                    "commence_time": event.get("commence_time"),
                    "home_team": home,
                    "away_team": away,
                    "odds_home": self._avg(home_odds),
                    "odds_draw": self._avg(draw_odds),
                    "odds_away": self._avg(away_odds),
                    "odds_over25": self._avg(over_odds),
                    "odds_under25": self._avg(under_odds),
                    "n_bookmakers": len(bookmakers),
                }
            )
        df = pd.DataFrame(rows)
        if not df.empty:
            df["commence_time"] = pd.to_datetime(df["commence_time"])
        return df


@st.cache_data(ttl=900, show_spinner=False)  # cache court : les cotes bougent vite
def cached_odds(sport_key: str) -> pd.DataFrame:
    client = OddsAPIClient()
    return client.events_to_dataframe(client.get_odds(sport_key))
