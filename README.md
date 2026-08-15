# ⚽ Value Bets Finder — 5 Grands Championnats

Prototype d'application Streamlit qui détecte des value bets sur les 5 grands
championnats européens (Premier League, La Liga, Bundesliga, Serie A, Ligue 1),
en comparant des probabilités calculées par un modèle statistique (Dixon-Coles)
aux cotes moyennes du marché.

## Architecture

```
value_bets_app/
├── app.py                  # Dashboard Streamlit (point d'entrée)
├── config.py                # Clés API, ligues suivies, seuils, paramètres du modèle
├── requirements.txt
├── .env.example              # Modèle de fichier de secrets
└── src/
    ├── api_football.py       # Ingestion API-Football (fixtures, forme, stats)
    ├── odds_api.py            # Ingestion The-Odds-API (cotes moyennes du marché)
    ├── data_processing.py      # Fusion des deux sources + matching des noms d'équipes
    ├── poisson_model.py         # Modèle Dixon-Coles (Poisson bivariée modifiée)
    ├── value_bet.py               # Calcul de l'edge et détection des value bets
    └── mock_data.py                # Générateur de données de démo (sans clé API)
```

Le flux de données suit exactement les 4 étapes que tu avais décrites :

1. **Ingestion** (`src/api_football.py`, `src/odds_api.py`) — deux clients HTTP
   indépendants, chacun avec son propre cache (`st.cache_data`) pour respecter
   les quotas des plans gratuits : 1h de cache pour les données API-Football
   (peu volatiles), 15 min pour les cotes (qui bougent vite).
2. **Modélisation** (`src/poisson_model.py`) — voir section dédiée ci-dessous.
3. **Détection de value** (`src/value_bet.py`) — calcule pour chaque marché
   (1X2, Over/Under 2.5) l'edge `= (probabilité_modèle × cote) - 1`, et
   remonte tout ce qui dépasse le seuil configuré (5 % par défaut).
4. **Interface** (`app.py`) — tableau de bord avec métriques clés, tableau des
   value bets, tableau des matchs à venir, et une vue détaillée par match
   (comparatif probabilité modèle vs. probabilité implicite, matrice de
   scores).

## Le modèle : Dixon-Coles (Poisson "modifiée")

C'est le modèle académique de référence pour le football (Dixon & Coles,
1997). Une simple double loi de Poisson indépendante sous-estime la
fréquence réelle des scores serrés (0-0, 1-0, 0-1, 1-1) : le modèle corrige
ce biais avec une fonction `tau` appliquée uniquement à ces 4 scores.

Pour chaque équipe, le modèle estime par maximum de vraisemblance (MLE, via
`scipy.optimize.minimize`) :

- une **force offensive** (`attack`)
- une **force défensive** (`defense`)

... plus un **avantage du terrain** global et un paramètre de corrélation
`rho`. Les matchs récents pèsent plus que les anciens grâce à une
pondération à décroissance exponentielle (`DC_TIME_DECAY_XI` dans
`config.py`).

À partir de ces paramètres, pour un match Domicile vs. Extérieur :

```
λ (buts attendus à domicile) = exp(attack_domicile + defense_extérieur + home_advantage)
μ (buts attendus à l'extérieur) = exp(attack_extérieur + defense_domicile)
```

On construit ensuite la matrice complète des probabilités de score (0-0,
1-0, 0-1, ... jusqu'à 8-8), corrigée par `tau`, et on somme les cases
pertinentes pour obtenir P(1), P(X), P(2), P(Over 2.5), P(Under 2.5).

> Tu avais aussi mentionné une alternative scikit-learn (ex :
> `PoissonRegressor` entraîné sur des variables muettes équipe/domicile).
> C'est une évolution naturelle du projet (section "Pour aller plus loin"
> ci-dessous) mais Dixon-Coles a été choisi comme modèle principal car
> c'est littéralement la "distribution de Poisson modifiée" que tu décrivais,
> et il ne nécessite pas de données d'entraînement massives pour être
> pertinent dès la première saison suivie.

## Détection des value bets

```
edge = (probabilité_modèle × cote_moyenne_marché) - 1
```

Un edge de +8 % signifie que, si notre probabilité est correcte, la cote
proposée par le marché est "trop généreuse" : l'espérance de gain est
positive. On ne remonte que les edges au-dessus du seuil (5 % par défaut,
réglable dans la barre latérale) pour absorber une partie de l'incertitude
du modèle — un edge de 1-2 % est trop souvent du bruit statistique.

La cote de référence est la **moyenne des cotes de tous les bookmakers**
renvoyés par The-Odds-API pour un marché donné (plus robuste qu'un seul
bookmaker).

## Installation

```bash
cd value_bets_app
python -m venv .venv && source .venv/bin/activate  # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # puis renseigne tes clés API-Football / The-Odds-API
streamlit run app.py
```

Sans clés API, l'app démarre automatiquement en **mode démo** (données
simulées) — pratique pour explorer l'interface avant de brancher les vraies
APIs. Bascule le mode dans la barre latérale une fois tes clés en place.

### Où obtenir les clés API

- **API-Football** : https://www.api-football.com/ (plan gratuit : 100
  requêtes/jour, largement suffisant pour ce prototype avec le cache activé).
- **The-Odds-API** : https://the-odds-api.com/ (plan gratuit : 500
  requêtes/mois).

## Limites connues du prototype

- Les **xG bruts** (Expected Goals fournis par API-Football) ne sont
  disponibles que sur certains plans payants au niveau `fixtures/statistics` ;
  le modèle Dixon-Coles s'en passe et infère sa propre notion de force
  offensive/défensive directement à partir des résultats — mais
  `src/api_football.py` expose déjà `get_fixture_statistics()` pour les
  intégrer facilement si ton plan les fournit (par exemple en les injectant
  comme variable de contrôle dans une régression Poisson scikit-learn).
- Le **matching des noms d'équipes** entre les deux APIs est fait par
  similarité de chaînes (`difflib`) : robuste dans la majorité des cas mais
  pas infaillible sur des clubs aux noms très proches. À surveiller si tu
  observes des matchs mal associés.
- Le modèle ne tient pas compte des **blessures, suspensions, ou
  motivation** (enjeu de fin de saison, coupe d'Europe la même semaine...) —
  des signaux qui améliorent significativement un modèle de paris en
  production.
- Prototype **éducatif** : aucune des probabilités calculées ne constitue un
  conseil de pari. Les paris sportifs comportent un risque de perte
  financière ; ce projet est un exercice de data science, pas un système de
  gains garantis.

## Pour aller plus loin

- Remplacer/compléter Dixon-Coles par un `sklearn.linear_model.PoissonRegressor`
  entraîné sur des features enrichies (xG, forme sur 5 matchs, repos entre
  matchs, importance de la rencontre) pour comparer les deux approches.
- Ajouter un **backtesting** : rejouer le modèle sur une saison passée et
  calculer la rentabilité réelle (ROI) d'une stratégie "miser systématiquement
  sur tout edge > seuil" avant de faire confiance au signal en conditions
  réelles.
- Suivre l'évolution des cotes dans le temps (les figer dès l'ouverture du
  marché) pour détecter du "steam" (mouvement de cote) en plus du value bet
  statique.
- Génération automatique d'alertes (email/Slack) quand un nouveau value bet
  dépasse un edge élevé.
