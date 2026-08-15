from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

from config import FOOTBALL_DATA_API_KEY, FOOTBALL_DATA_BASE_URL


class FootballDataClient:
    def __init__(self, api_key: str = FOOTBALL_DATA_API_KEY):
        self.base_url = FOOTBALL_DATA_BASE_URL
        self.headers = {"X-Auth-Token": api_key}
        self.enabled = bool(api_key)

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        if not self.enabled:
            return {}
        try:
            resp = requests.get(
                f"{self.base_url}/{endpoint}", headers=self.headers, params=params, timeout=15
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            st.warning(f"Erreur réseau football-data.org ({endpoint}) : {exc}")
            return {}

    def get_upcoming_matches(self, competition_code, date_from, date_to):
        data = self._get(
            f"competitions/{competition_code}/matches",
            {"status": "SCHEDULED", "dateFrom": date_from, "dateTo": date_to},
        )
        return data.get("matches", [])

    def get_finished_matches(self, competition_code, date_from, date_to):
        data = self._get(
            f"competitions/{competition_code}/matches",
            {"status": "FINISHED", "dateFrom": date_from, "dateTo": date_to},
        )
        return data.get("matches", [])

    @staticmethod
    def matches_to_dataframe(matches):
        rows = []
        for m in matches:
            try:
                rows.append({
                    "fixture_id": m["id"],
                    "date": m["utcDate"],
                    "home_team": m["homeTeam"]["name"],
                    "away_team": m["awayTeam"]["name"],
                    "home_goals": m.get("score", {}).get("fullTime", {}).get("home"),
                    "away_goals": m.get("score", {}).get("fullTime", {}).get("away"),
                    "status": m.get("status"),
                })
            except (KeyError, TypeError):
                continue
        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df


@st.cache_data(ttl=3600, show_spinner=False)
def cached_upcoming_matches(competition_code, days_ahead=21):
    client = FootballDataClient()
    today = pd.Timestamp.now(tz="UTC").normalize()
    date_from = today.strftime("%Y-%m-%d")
    date_to = (today + pd.Timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    matches = client.get_upcoming_matches(competition_code, date_from, date_to)
    return client.matches_to_dataframe(matches)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_past_matches(competition_code, days_back=365):
    client = FootballDataClient()
    today = pd.Timestamp.now(tz="UTC").normalize()
    date_from = (today - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d")
    date_to = today.strftime("%Y-%m-%d")
    matches = client.get_finished_matches(competition_code, date_from, date_to)
    return client.matches_to_dataframe(matches)
