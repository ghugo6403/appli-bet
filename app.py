"""
Value Bets Finder — Dashboard Streamlit
=========================================

Point d'entrée de l'application. Pour chaque championnat sélectionné :
  1. récupère (ou simule) l'historique de la saison et calibre un modèle
     Dixon-Coles (Poisson modifiée) équipe par équipe ;
  2. récupère les prochains matchs et les cotes moyennes du marché ;
  3. calcule les probabilités "réelles" (1X2, Over/Under 2.5) et les compare
     aux cotes pour détecter les Value Bets (edge > seuil) ;
  4. affiche un tableau de bord : matchs à venir, cotes, probabilités,
     alertes Value Bets, et le détail (matrice de scores) de chaque match.

Lancer avec :  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import mock_data
from football_data import FootballDataClient, cached_past_matches, cached_upcoming_matches
from odds_api import OddsAPIClient, cached_odds
from poisson_model import DixonColesModel
from value_bet import MARKETS, build_value_bets_table, detect_value_bets

# --- Palette (voir la skill dataviz — palette de référence validée) --------
COLOR_SERIES_1 = "#2a78d6"   # bleu — Domicile / Over
COLOR_SERIES_2 = "#eb6834"   # orange — Nul / Under
COLOR_SERIES_3 = "#1baf7a"   # aqua — Extérieur
COLOR_GOOD = "#0ca30c"       # vert statut "good" — value bet détecté
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
SEQUENTIAL_BLUE = [
    "#fcfcfb", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#0d366b",
]

st.set_page_config(page_title="Value Bets Finder — 5 Grands Championnats", layout="wide")


# ---------------------------------------------------------------------------
# Chargement des données (réel ou démo) avec calibration du modèle
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False, hash_funcs={pd.DataFrame: lambda df: pd.util.hash_pandas_object(df).sum()})
def fit_model_for_league(history_df: pd.DataFrame) -> DixonColesModel:
    model = DixonColesModel()
    model.fit(history_df)
    return model


def load_league_data(league: str, demo_mode: bool, n_upcoming: int):
    league_cfg = LEAGUES[league]

    if demo_mode:
        history_df = mock_data.generate_history(league)
        matches_with_odds = mock_data.generate_upcoming_fixtures_with_odds(league, n_matches=n_upcoming)
    else:
        past_fixtures = cached_past_matches(league_cfg["football_data_code"])
        history_df = build_team_history(past_fixtures)

        upcoming_fixtures = cached_upcoming_matches(league_cfg["football_data_code"])
        odds_events = cached_odds(league_cfg["odds_api_key"])
        matches_with_odds = merge_fixtures_with_odds(upcoming_fixtures, odds_events)
        if not matches_with_odds.empty:
            matches_with_odds["league"] = league

    if history_df.empty or matches_with_odds.empty:
        return None, matches_with_odds

    model = fit_model_for_league(history_df)
    return model, matches_with_odds


def enrich_with_predictions(model: DixonColesModel, matches_df: pd.DataFrame) -> pd.DataFrame:
    enriched_rows = []
    matrices = {}
    for _, row in matches_df.iterrows():
        pred = model.predict_match(row["home_team"], row["away_team"])
        if pred is None:
            continue
        matrix = pred.pop("score_matrix")
        matrices[(row["home_team"], row["away_team"])] = matrix
        enriched_rows.append({**row.to_dict(), **pred})
    df = pd.DataFrame(enriched_rows)
    return df, matrices


# ---------------------------------------------------------------------------
# Composants graphiques
# ---------------------------------------------------------------------------
def probability_vs_market_chart(row: pd.Series) -> go.Figure:
    labels = ["Domicile (1)", "Nul (X)", "Extérieur (2)"]
    model_probs = [row["p_home"], row["p_draw"], row["p_away"]]
    implied_probs = [
        1 / row["odds_home"] if row.get("odds_home") else None,
        1 / row["odds_draw"] if row.get("odds_draw") else None,
        1 / row["odds_away"] if row.get("odds_away") else None,
    ]

    fig = go.Figure()
    fig.add_bar(name="Probabilité modèle", x=labels, y=model_probs, marker_color=COLOR_SERIES_1)
    fig.add_bar(name="Probabilité implicite (cote)", x=labels, y=implied_probs, marker_color=COLOR_SERIES_2)
    fig.update_layout(
        barmode="group",
        yaxis_title="Probabilité",
        yaxis_tickformat=".0%",
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=40, b=20, l=10, r=10),
        height=320,
    )
    fig.update_yaxes(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID)
    fig.update_xaxes(showgrid=False)
    return fig


def score_heatmap(matrix: np.ndarray, home_team: str, away_team: str, max_display: int = 5) -> go.Figure:
    sub = matrix[: max_display + 1, : max_display + 1]
    fig = go.Figure(
        data=go.Heatmap(
            z=sub,
            x=[str(i) for i in range(max_display + 1)],
            y=[str(i) for i in range(max_display + 1)],
            colorscale=SEQUENTIAL_BLUE,
            text=[[f"{v:.1%}" for v in row_] for row_ in sub],
            texttemplate="%{text}",
            textfont={"size": 11},
            hovertemplate=f"{home_team} %{{y}} - %{{x}} {away_team}<br>Probabilité : %{{z:.2%}}<extra></extra>",
            colorbar=dict(title="Proba", tickformat=".0%"),
        )
    )
    fig.update_layout(
        xaxis_title=f"Buts {away_team}",
        yaxis_title=f"Buts {home_team}",
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        margin=dict(t=20, b=20, l=10, r=10),
        height=380,
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Paramètres")

football_data_client = FootballDataClient()
odds_client = OddsAPIClient()
keys_missing = not (football_data_client.enabled and odds_client.enabled)

demo_mode = st.sidebar.toggle(
    "Mode démo (données simulées)",
    value=keys_missing,
    help="Active ce mode si tu n'as pas encore renseigné tes clés API-Football / The-Odds-API "
    "(voir .env.example). Les données sont alors simulées pour tester l'interface.",
)
if keys_missing and not demo_mode:
    st.sidebar.warning("Clés API manquantes : ajoute-les dans .env pour désactiver le mode démo.")

selected_leagues = st.sidebar.multiselect(
    "Championnats à analyser", options=list(LEAGUES.keys()), default=list(LEAGUES.keys())[:3]
)
n_upcoming = st.sidebar.slider("Nombre de matchs à venir par championnat", 3, 20, 8)
edge_threshold = st.sidebar.slider(
    "Seuil de Value Bet (edge minimum)", min_value=0.0, max_value=0.30,
    value=VALUE_BET_EDGE_THRESHOLD, step=0.01, format="%.2f",
)
st.sidebar.caption(f"Un pari est signalé « Value Bet » si edge > {edge_threshold:.0%}.")
show_unreliable = st.sidebar.checkbox(
    "Inclure les équipes avec peu d'historique", value=False,
    help="Décoche pour ignorer les prédictions basées sur moins de matchs que le minimum fiable.",
)

# ---------------------------------------------------------------------------
# Corps du dashboard
# ---------------------------------------------------------------------------
st.title("⚽ Value Bets Finder — 5 Grands Championnats")
st.caption(
    "Détection de value bets à partir d'un modèle Dixon-Coles (Poisson modifiée) "
    "calibré sur l'historique de chaque championnat, comparé aux cotes moyennes du marché."
)

if not selected_leagues:
    st.info("Sélectionne au moins un championnat dans le menu de gauche pour commencer.")
    st.stop()

all_matches = []
all_matrices = {}

with st.spinner("Chargement des données et calibration des modèles..."):
    for league in selected_leagues:
        model, matches_df = load_league_data(league, demo_mode, n_upcoming)
        if model is None or matches_df.empty:
            st.warning(f"Pas assez de données pour **{league}** (historique ou cotes manquants).")
            continue
        enriched_df, matrices = enrich_with_predictions(model, matches_df)
        if enriched_df.empty:
            continue
        enriched_df["league"] = league
        if not show_unreliable:
            enriched_df = enriched_df[enriched_df["reliable"]]
        all_matches.append(enriched_df)
        all_matrices.update({(league, h, a): m for (h, a), m in matrices.items()})

if not all_matches:
    st.error("Aucun match exploitable trouvé. Essaie le mode démo ou vérifie tes clés API.")
    st.stop()

matches_df = pd.concat(all_matches, ignore_index=True)
value_bets_df = build_value_bets_table(matches_df, threshold=edge_threshold)

# -- Indicateurs clés --------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Matchs analysés", len(matches_df))
col2.metric("Value Bets détectés", len(value_bets_df))
col3.metric(
    "Edge moyen des Value Bets",
    f"{value_bets_df['edge'].mean():.1%}" if not value_bets_df.empty else "—",
)

# -- Alertes Value Bets -------------------------------------------------------
st.subheader("💎 Alertes Value Bets")
if value_bets_df.empty:
    st.info("Aucun value bet détecté avec le seuil actuel. Essaie de baisser le seuil dans la barre latérale.")
else:
    display_vb = value_bets_df.copy()
    display_vb["date"] = pd.to_datetime(display_vb["date"]).dt.strftime("%a %d/%m %H:%M")
    display_vb["match"] = display_vb["home_team"] + " vs " + display_vb["away_team"]
    st.dataframe(
        display_vb[["date", "league", "match", "market", "odds", "model_probability", "implied_probability", "edge"]],
        column_config={
            "model_probability": st.column_config.NumberColumn("Proba modèle", format="%.1f%%"),
            "implied_probability": st.column_config.NumberColumn("Proba implicite", format="%.1f%%"),
            "edge": st.column_config.ProgressColumn("Edge", format="%.1f%%", min_value=0, max_value=float(max(display_vb["edge"].max(), 0.3))),
            "odds": st.column_config.NumberColumn("Cote moyenne", format="%.2f"),
        },
        hide_index=True,
        use_container_width=True,
    )

# -- Tableau des matchs à venir -----------------------------------------------
st.subheader("📅 Matchs à venir")
table = matches_df.copy()
table["date"] = pd.to_datetime(table["date"] if "date" in table else table["commence_time"]).dt.strftime("%a %d/%m %H:%M")
table["match"] = table["home_team"] + " vs " + table["away_team"]
table["value_bet"] = table.apply(lambda r: len(detect_value_bets(r.to_dict(), edge_threshold)) > 0, axis=1)

st.dataframe(
    table[[
        "date", "league", "match", "odds_home", "odds_draw", "odds_away",
        "p_home", "p_draw", "p_away", "odds_over25", "odds_under25",
        "p_over25", "p_under25", "value_bet",
    ]].rename(columns={
        "league": "Championnat", "odds_home": "Cote 1", "odds_draw": "Cote X", "odds_away": "Cote 2",
        "p_home": "Proba 1", "p_draw": "Proba X", "p_away": "Proba 2",
        "odds_over25": "Cote O2.5", "odds_under25": "Cote U2.5",
        "p_over25": "Proba O2.5", "p_under25": "Proba U2.5", "value_bet": "Value Bet ?",
    }),
    column_config={
        "Proba 1": st.column_config.NumberColumn(format="%.1f%%"),
        "Proba X": st.column_config.NumberColumn(format="%.1f%%"),
        "Proba 2": st.column_config.NumberColumn(format="%.1f%%"),
        "Proba O2.5": st.column_config.NumberColumn(format="%.1f%%"),
        "Proba U2.5": st.column_config.NumberColumn(format="%.1f%%"),
    },
    hide_index=True,
    use_container_width=True,
)

# -- Détail par match ----------------------------------------------------------
st.subheader("🔍 Détail d'un match")
match_options = {
    f"{r.league} — {r.home_team} vs {r.away_team}": (r.league, r.home_team, r.away_team)
    for r in matches_df.itertuples()
}
selected_label = st.selectbox("Choisir un match", options=list(match_options.keys()))
league_sel, home_sel, away_sel = match_options[selected_label]
row_sel = matches_df[
    (matches_df["league"] == league_sel) & (matches_df["home_team"] == home_sel) & (matches_df["away_team"] == away_sel)
].iloc[0]

vb_for_match = detect_value_bets(row_sel.to_dict(), edge_threshold)
if vb_for_match:
    st.success(
        "Value Bet(s) détecté(s) : "
        + ", ".join(f"{vb['market']} (edge {vb['edge']:.1%})" for vb in vb_for_match)
    )
else:
    st.caption("Aucun value bet sur ce match au seuil actuel.")

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(probability_vs_market_chart(row_sel), use_container_width=True)
with c2:
    matrix = all_matrices.get((league_sel, home_sel, away_sel))
    if matrix is not None:
        st.plotly_chart(score_heatmap(matrix, home_sel, away_sel), use_container_width=True)

st.caption(
    f"xG modèle : {home_sel} {row_sel['xg_home']:.2f} — {away_sel} {row_sel['xg_away']:.2f} · "
    f"{'⚠️ historique limité pour au moins une équipe' if not row_sel.get('reliable', True) else ''}"
)

st.divider()
st.caption(
    "⚠️ Prototype à but éducatif. Les probabilités sont estimées par un modèle statistique et ne "
    "constituent pas un conseil de paris. Parie de façon responsable."
)
