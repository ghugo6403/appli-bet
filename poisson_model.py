"""
Modèle de Dixon & Coles (1997) : une distribution de Poisson bivariée
"modifiée", pensée spécifiquement pour le football.

Pourquoi "modifiée" et pas une simple double Poisson indépendante ?
Un modèle de Poisson indépendant pour les buts à domicile et à l'extérieur
sous-estime systématiquement la fréquence réelle des scores serrés et
faibles (0-0, 1-0, 0-1, 1-1), parce qu'en réalité les buts des deux équipes
sont légèrement corrélés en fin de match (petit score => les deux équipes
jouent différemment). Dixon & Coles corrigent cela avec une fonction tau qui
ajuste seulement ces 4 scores, sans toucher au reste de la matrice.

Le modèle apprend, pour chaque équipe, une force offensive (attack) et une
force défensive (defense), ainsi qu'un avantage du terrain global
(home_advantage) et le paramètre de corrélation (rho), par maximum de
vraisemblance (MLE) sur l'historique des résultats. Un poids de décroissance
temporelle (xi) donne plus d'importance aux matchs récents.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from config import DC_MAX_GOALS, DC_TIME_DECAY_XI, MIN_HISTORICAL_MATCHES


class DixonColesModel:
    def __init__(self, xi: float = DC_TIME_DECAY_XI):
        self.xi = xi
        self.teams: list[str] = []
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.home_advantage: float = 0.0
        self.rho: float = 0.0
        self.fitted = False
        self.team_match_counts: dict[str, int] = {}

    # -- fonction de correction pour les scores faibles ----------------
    @staticmethod
    def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
        if x == 0 and y == 0:
            return 1 - lam * mu * rho
        if x == 0 and y == 1:
            return 1 + lam * rho
        if x == 1 and y == 0:
            return 1 + mu * rho
        if x == 1 and y == 1:
            return 1 - rho
        return 1.0

    # -- log-vraisemblance négative à minimiser -------------------------
    def _negative_log_likelihood(self, params: np.ndarray, matches: list[dict], n_teams: int) -> float:
        attack = dict(zip(self.teams, params[:n_teams]))
        defense = dict(zip(self.teams, params[n_teams : 2 * n_teams]))
        home_adv = params[-2]
        rho = params[-1]

        log_lik = 0.0
        for m in matches:
            lam = np.exp(attack[m["home"]] + defense[m["away"]] + home_adv)
            mu = np.exp(attack[m["away"]] + defense[m["home"]])
            tau = max(self._tau(m["hg"], m["ag"], lam, mu, rho), 1e-10)
            log_lik += m["weight"] * (
                np.log(tau) + poisson.logpmf(m["hg"], lam) + poisson.logpmf(m["ag"], mu)
            )

        # Pénalité douce pour centrer la moyenne des forces offensives sur 0 :
        # sans cette contrainte, le système est sur-paramétré (on peut décaler
        # toutes les attaques de +c et toutes les défenses de -c sans rien
        # changer à la vraisemblance). Une pénalité quadratique légère suffit
        # à ancrer une solution unique sans dénaturer l'optimum.
        penalty = 1000.0 * float(np.mean(params[:n_teams])) ** 2
        return -log_lik + penalty

    def fit(self, history_df: pd.DataFrame) -> "DixonColesModel":
        """
        history_df : colonnes ['date', 'home_team', 'away_team', 'home_goals', 'away_goals']
        """
        df = history_df.dropna(subset=["home_goals", "away_goals"]).copy()
        if df.empty:
            raise ValueError("Historique vide : impossible d'entraîner le modèle.")

        df["date"] = pd.to_datetime(df["date"])
        # on neutralise le fuseau horaire pour pouvoir soustraire des dates sans erreur
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
        max_date = df["date"].max()
        df["days_ago"] = (max_date - df["date"]).dt.days
        df["weight"] = np.exp(-self.xi * df["days_ago"])

        self.teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        n_teams = len(self.teams)
        self.team_match_counts = (
            pd.concat([df["home_team"], df["away_team"]]).value_counts().to_dict()
        )

        matches = [
            {
                "home": r.home_team,
                "away": r.away_team,
                "hg": int(r.home_goals),
                "ag": int(r.away_goals),
                "weight": float(r.weight),
            }
            for r in df.itertuples()
        ]

        init_params = np.concatenate(
            [
                np.zeros(n_teams),  # attack
                np.zeros(n_teams),  # defense
                [0.25],  # home advantage (log-échelle)
                [0.0],  # rho
            ]
        )
        bounds = [(-3.0, 3.0)] * n_teams + [(-3.0, 3.0)] * n_teams + [(-2.0, 2.0)] + [(-0.9, 0.9)]

        result = minimize(
            self._negative_log_likelihood,
            init_params,
            args=(matches, n_teams),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 300},
        )

        self.attack = dict(zip(self.teams, result.x[:n_teams]))
        self.defense = dict(zip(self.teams, result.x[n_teams : 2 * n_teams]))
        self.home_advantage = float(result.x[-2])
        self.rho = float(result.x[-1])
        self.fitted = True
        return self

    def team_is_reliable(self, team: str) -> bool:
        """Un minimum de matchs joués est requis pour faire confiance aux paramètres estimés."""
        return self.team_match_counts.get(team, 0) >= MIN_HISTORICAL_MATCHES

    def predict_score_matrix(self, home_team: str, away_team: str, max_goals: int = DC_MAX_GOALS):
        if not self.fitted:
            raise RuntimeError("Le modèle doit être entraîné (fit) avant toute prédiction.")
        if home_team not in self.teams or away_team not in self.teams:
            return None

        lam = np.exp(self.attack[home_team] + self.defense[away_team] + self.home_advantage)
        mu = np.exp(self.attack[away_team] + self.defense[home_team])

        home_probs = poisson.pmf(np.arange(max_goals + 1), lam)
        away_probs = poisson.pmf(np.arange(max_goals + 1), mu)
        matrix = np.outer(home_probs, away_probs)

        for x in range(2):
            for y in range(2):
                matrix[x, y] *= self._tau(x, y, lam, mu, self.rho)

        matrix = np.clip(matrix, 0, None)
        matrix /= matrix.sum()
        return matrix, lam, mu

    def predict_match(self, home_team: str, away_team: str, max_goals: int = DC_MAX_GOALS) -> dict | None:
        """
        Retourne les probabilités 1X2 et Over/Under 2.5, les xG attendus par
        le modèle, et la matrice de scores complète (utile pour la heatmap).
        """
        result = self.predict_score_matrix(home_team, away_team, max_goals)
        if result is None:
            return None
        matrix, lam, mu = result

        p_home = float(np.tril(matrix, -1).sum())
        p_draw = float(np.trace(matrix))
        p_away = float(np.triu(matrix, 1).sum())

        goals_grid = np.add.outer(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
        p_over25 = float(matrix[goals_grid > 2.5].sum())
        p_under25 = float(matrix[goals_grid <= 2.5].sum())

        return {
            "p_home": round(p_home, 4),
            "p_draw": round(p_draw, 4),
            "p_away": round(p_away, 4),
            "p_over25": round(p_over25, 4),
            "p_under25": round(p_under25, 4),
            "xg_home": round(float(lam), 2),
            "xg_away": round(float(mu), 2),
            "score_matrix": matrix,
            "reliable": self.team_is_reliable(home_team) and self.team_is_reliable(away_team),
        }
