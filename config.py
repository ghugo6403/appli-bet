"""
Configuration centrale de l'application.

Toutes les clés API et tous les paramètres "métier" (seuils, ligues suivies,
saison en cours, etc.) sont définis ici pour être importés partout ailleurs.
Les clés sont lues depuis les variables d'environnement (voir .env.example) :
en local via un fichier .env (python-dotenv), et en déploiement Streamlit
Cloud via st.secrets (voir la petite passerelle plus bas).
"""
import os

from dotenv import load_dotenv

load_dotenv()  # charge le fichier .env s'il existe (ne fait rien en prod)


def _get_secret(key: str, default: str = "") -> str:
    """
    Va chercher une clé d'abord dans les variables d'environnement, puis dans
    st.secrets si l'app tourne sur Streamlit Cloud. Ne plante jamais si aucune
    des deux sources n'existe : l'app doit pouvoir démarrer en mode démo.
    """
    value = os.getenv(key)
    if value:
        return value
    try:
        import streamlit as st  # import local pour ne pas dépendre de streamlit hors app

        return st.secrets.get(key, default)
    except Exception:
        return default


# --- Clés API -----------------------------------------------------------
API_FOOTBALL_KEY = _get_secret("API_FOOTBALL_KEY")
API_FOOTBALL_HOST = "v3.football.api-sports.io"

FOOTBALL_DATA_API_KEY = _get_secret("FOOTBALL_DATA_API_KEY")
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"

ODDS_API_KEY = _get_secret("ODDS_API_KEY")
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

LEAGUES = {
    "Premier League": {"api_football_id": 39, "football_data_code": "PL", "odds_api_key": "soccer_epl", "country": "Angleterre"},
    "La Liga": {"api_football_id": 140, "football_data_code": "PD", "odds_api_key": "soccer_spain_la_liga", "country": "Espagne"},
    "Bundesliga": {"api_football_id": 78, "football_data_code": "BL1", "odds_api_key": "soccer_germany_bundesliga", "country": "Allemagne"},
    "Serie A": {"api_football_id": 135, "football_data_code": "SA", "odds_api_key": "soccer_italy_serie_a", "country": "Italie"},
    "Ligue 1": {"api_football_id": 61, "football_data_code": "FL1", "odds_api_key": "soccer_france_ligue_one", "country": "France"},
    "Ligue des Champions": {"api_football_id": 2, "football_data_code": "CL", "odds_api_key": "soccer_uefa_champs_league", "country": "Europe"},
}
# --- Paramètres saison / historique --------------------------------------
# API-Football désigne une saison par son année de démarrage
# (ex : 2025 pour la saison 2025-2026). A ajuster à la volée si besoin.
CURRENT_SEASON = int(_get_secret("CURRENT_SEASON", "2025"))
PAST_FIXTURES_LOOKBACK = 200  # nb de matchs passés max récupérés pour calibrer le modèle
MIN_HISTORICAL_MATCHES = 6  # matchs minimum joués par une équipe pour juger le modèle fiable

# --- Paramètres de cotes ---------------------------------------------------
ODDS_REGIONS = "eu"
ODDS_MARKETS = "h2h,totals"

# --- Détection de value bets ------------------------------------------
VALUE_BET_EDGE_THRESHOLD = 0.05  # edge minimum (5%) pour déclencher une alerte
MAX_ODDS_CONSIDERED = 15.0  # ignore les cotes aberrantes (illiquides / erreurs de flux)

# --- Modèle Dixon-Coles ---------------------------------------------------
DC_TIME_DECAY_XI = 0.0018  # pondération temporelle (plus la valeur est grande, plus le passé récent pèse)
DC_MAX_GOALS = 8  # taille de la matrice de scores simulés (0..8 buts par équipe)
