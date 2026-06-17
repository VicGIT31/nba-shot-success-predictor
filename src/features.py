"""Feature engineering : transforme les tirs bruts en matrice prête à l'entraînement.

Lit les agrégats ``data/raw/shots_<season>.parquet`` produits par ``data_ingestion``,
construit les features et one-hot encode les variables catégorielles, puis écrit
``data/processed/shots_features.parquet``.

Mapping colonnes brutes (ShotChartDetail) → features :
    SHOT_DISTANCE                          → shot_distance
    LOC_X / LOC_Y                          → loc_x / loc_y
    PERIOD                                 → period
    MINUTES_REMAINING, SECONDS_REMAINING   → period_time_remaining (en secondes)
    SHOT_TYPE ("2PT/3PT Field Goal")       → shot_value (2 / 3)
    SHOT_ZONE_BASIC / _AREA / _RANGE       → catégorielles (one-hot)
    ACTION_TYPE                            → action_type (bucketé top-N, one-hot)
    SHOT_MADE_FLAG                         → shot_made (cible)
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("features")


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Construit la matrice (features brutes + cible + métadonnées) à partir des tirs."""
    df = pd.DataFrame(index=raw.index)

    # --- Numériques ---
    df["shot_distance"] = pd.to_numeric(raw["SHOT_DISTANCE"], errors="coerce")
    df["loc_x"] = pd.to_numeric(raw["LOC_X"], errors="coerce")
    df["loc_y"] = pd.to_numeric(raw["LOC_Y"], errors="coerce")
    df["period"] = pd.to_numeric(raw["PERIOD"], errors="coerce")

    minutes = pd.to_numeric(raw["MINUTES_REMAINING"], errors="coerce")
    seconds = pd.to_numeric(raw["SECONDS_REMAINING"], errors="coerce")
    df["period_time_remaining"] = minutes * 60 + seconds

    # SHOT_TYPE = "2PT Field Goal" / "3PT Field Goal" → 2 / 3
    df["shot_value"] = raw["SHOT_TYPE"].astype(str).str.extract(r"(\d)").astype(float)

    # --- Catégorielles (valeurs brutes ; one-hot encodées plus bas) ---
    df["shot_zone_basic"] = raw["SHOT_ZONE_BASIC"].astype(str)
    df["shot_zone_area"] = raw["SHOT_ZONE_AREA"].astype(str)
    df["shot_zone_range"] = raw["SHOT_ZONE_RANGE"].astype(str)
    df["action_type"] = raw["ACTION_TYPE"].astype(str)

    # --- Cible ---
    df["shot_made"] = pd.to_numeric(raw["SHOT_MADE_FLAG"], errors="coerce")

    # --- Métadonnées (non utilisées par le modèle) ---
    meta_map = {
        "GAME_ID": "game_id", "GAME_DATE": "game_date", "PLAYER_ID": "player_id",
        "PLAYER_NAME": "player_name", "TEAM_ID": "team_id", "SEASON": "season",
    }
    for src_col, dst_col in meta_map.items():
        if src_col in raw.columns:
            df[dst_col] = raw[src_col].values

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Filtre les lignes aberrantes / incomplètes."""
    before = len(df)

    df = df.dropna(subset=["shot_made", "shot_distance", "shot_value",
                           "period_time_remaining"])
    df = df[df["shot_made"].isin([0, 1])]
    df = df[df["shot_value"].isin([2, 3])]
    df = df[df["shot_distance"].between(0, 60)]  # garde-fou bruit de saisie

    df = df.copy()
    df["shot_made"] = df["shot_made"].astype(int)
    df["shot_value"] = df["shot_value"].astype(int)

    logger.info("Nettoyage : %d → %d lignes (%d retirées)", before, len(df), before - len(df))
    return df.reset_index(drop=True)


def bucket_action_type(df: pd.DataFrame) -> pd.DataFrame:
    """Regroupe les ACTION_TYPE rares dans 'Other' (limite la largeur du one-hot)."""
    top = df["action_type"].value_counts().nlargest(config.TOP_ACTION_TYPES).index
    df = df.copy()
    df["action_type"] = df["action_type"].where(df["action_type"].isin(top), "Other")
    logger.info("action_type : %d modalités conservées + 'Other'", len(top))
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode les features catégorielles ; conserve numériques, cible, métadonnées."""
    encoded = pd.get_dummies(
        df,
        columns=config.CATEGORICAL_FEATURES,
        prefix=config.CATEGORICAL_FEATURES,
    )
    # Les colonnes one-hot sont des booléens → entiers (XGBoost / parquet friendly).
    bool_cols = encoded.select_dtypes(include="bool").columns
    encoded[bool_cols] = encoded[bool_cols].astype("int8")
    return encoded


def load_raw_seasons(seasons: list[str]) -> pd.DataFrame:
    """Charge et concatène les agrégats bruts d'une ou plusieurs saisons."""
    frames = []
    for season in seasons:
        path = config.RAW_DIR / f"shots_{season}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} introuvable. Lance d'abord : python -m src.data_ingestion "
                f"--seasons {season}"
            )
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def build_dataset(seasons: list[str]) -> pd.DataFrame:
    """Pipeline complet : raw → features → clean → bucket → one-hot, et persiste."""
    raw = load_raw_seasons(seasons)
    logger.info("Chargé %d tirs bruts sur %d saison(s).", len(raw), len(seasons))

    features = build_features(raw)
    features = clean(features)
    features = bucket_action_type(features)
    features = encode(features)

    features.to_parquet(config.PROCESSED_DATASET_PATH, index=False)
    logger.info("Dataset features → %s (%d lignes, %d colonnes, FG%% global = %.3f)",
                config.PROCESSED_DATASET_PATH.name, len(features), features.shape[1],
                features["shot_made"].mean())
    return features


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Colonnes données au modèle = tout sauf la cible et les métadonnées."""
    excluded = set(config.METADATA_COLUMNS) | {config.TARGET_COLUMN}
    return [c for c in df.columns if c not in excluded]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Construction de la matrice de features.")
    p.add_argument("--seasons", nargs="+", default=config.DEFAULT_SEASONS,
                   help="Saisons à inclure. Défaut : %(default)s")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    build_dataset(args.seasons)


if __name__ == "__main__":
    main()
