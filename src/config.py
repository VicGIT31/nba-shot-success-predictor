"""Configuration centrale du projet : chemins, saisons, features, hyperparamètres.

Tout ce qui est susceptible d'être ajusté d'un run à l'autre vit ici, pour garder les
modules métier (ingestion / features / train / evaluate) propres et reproductibles.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------------------
# Chemins
# --------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"               # cache brut (par équipe + agrégat saison)
PROCESSED_DIR = DATA_DIR / "processed"   # dataset de features prêt à l'entraînement
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Crée les dossiers au besoin (idempotent).
for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Saisons
# --------------------------------------------------------------------------------------
# Source = ShotChartDetail (tir par tir avec coordonnées de terrain), disponible à
# partir de la saison 1996-97. Voir le README pour pourquoi ce n'est pas le tracking
# shot log (défenseur / shot clock / dribbles), retiré de l'API publique NBA.
EARLIEST_SEASON = "1996-97"

# Saisons utilisées par défaut si aucune n'est passée en CLI.
DEFAULT_SEASONS = ["2022-23", "2023-24"]

SEASON_TYPE = "Regular Season"  # ou "Playoffs"

# --------------------------------------------------------------------------------------
# Ingestion / rate limiting (cf. README — l'API NBA est non officielle et fragile)
# --------------------------------------------------------------------------------------
REQUEST_DELAY = 0.6     # secondes entre deux appels API (politesse / anti rate-limit)
REQUEST_TIMEOUT = 45    # timeout par requête (s) — les tirs d'une équipe = grosse réponse
MAX_RETRIES = 4         # tentatives par appel avant abandon
BACKOFF_FACTOR = 1.8    # délai = BACKOFF_FACTOR ** tentative

# --------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------
# Colonnes numériques (utilisées telles quelles par le modèle).
NUMERIC_FEATURES = [
    "shot_distance",          # SHOT_DISTANCE — distance du panier (pieds)
    "loc_x",                  # LOC_X         — position horizontale (0.1 pied)
    "loc_y",                  # LOC_Y         — position verticale (0.1 pied)
    "period",                 # PERIOD        — quart-temps
    "period_time_remaining",  # MIN*60+SEC    — secondes restantes dans le quart-temps
    "shot_value",             # SHOT_TYPE     — 2 ou 3 points
]
# Colonnes catégorielles (one-hot encodées dans features.py).
CATEGORICAL_FEATURES = [
    "shot_zone_basic",        # SHOT_ZONE_BASIC — Restricted Area, Mid-Range, Above Break 3…
    "shot_zone_area",         # SHOT_ZONE_AREA  — Left/Center/Right
    "shot_zone_range",        # SHOT_ZONE_RANGE — <8ft, 8-16ft, 16-24ft, 24+ft
    "action_type",            # ACTION_TYPE     — Jump Shot, Layup, Dunk, Pull-Up… (bucketé)
]
TARGET_COLUMN = "shot_made"   # SHOT_MADE_FLAG — 1 = réussi, 0 = manqué

# Colonnes conservées pour l'EDA / le notebook mais JAMAIS données au modèle.
METADATA_COLUMNS = [
    "game_id", "game_date", "player_id", "player_name", "team_id", "season",
]

# action_type a beaucoup de modalités rares : on ne garde que les N plus fréquentes,
# le reste est regroupé dans "Other" (limite la largeur du one-hot, évite les
# catégories vues à l'entraînement mais absentes du test).
TOP_ACTION_TYPES = 15

# --------------------------------------------------------------------------------------
# Modèle
# --------------------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 4

# Espace de recherche d'hyperparamètres (RandomizedSearchCV dans train.py).
PARAM_DISTRIBUTIONS = {
    "n_estimators": [200, 400, 600, 800],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 10],
    "gamma": [0, 0.1, 0.3, 0.5],
    "reg_lambda": [1.0, 2.0, 5.0],
}
N_SEARCH_ITER = 30  # nombre de combinaisons testées par RandomizedSearchCV

# Artefacts persistés
MODEL_PATH = MODELS_DIR / "xgb_shot_model.json"
MODEL_META_PATH = MODELS_DIR / "model_meta.json"
PROCESSED_DATASET_PATH = PROCESSED_DIR / "shots_features.parquet"
