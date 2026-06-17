"""Entraînement du modèle : XGBClassifier + tuning d'hyperparamètres par CV.

Lit ``data/processed/shots_features.parquet`` (produit par ``features.py``), effectue
un split train/test stratifié, recherche les meilleurs hyperparamètres par
``RandomizedSearchCV`` (scoring = ROC AUC), réentraîne sur tout le train et persiste :
    * le modèle              → models/xgb_shot_model.json
    * les métadonnées        → models/model_meta.json (features, params, métriques test)

Usage :
    python -m src.train
    python -m src.train --n-iter 50 --cv 5
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier

from src import config
from src.features import get_feature_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("train")


def load_dataset() -> pd.DataFrame:
    if not config.PROCESSED_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{config.PROCESSED_DATASET_PATH} introuvable. Lance d'abord : "
            "python -m src.features"
        )
    return pd.read_parquet(config.PROCESSED_DATASET_PATH)


def split_xy(df: pd.DataFrame):
    """Sépare features / cible et le jeu train / test (stratifié sur la cible).

    Les colonnes de features sont dérivées dynamiquement du dataset (one-hot inclus),
    donc le pipeline reste valide quel que soit le nombre de modalités catégorielles.
    """
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].astype(float)
    y = df[config.TARGET_COLUMN].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    logger.info("Split : %d train / %d test (FG%% train=%.3f)", len(X_train), len(X_test),
                y_train.mean())
    return X_train, X_test, y_train, y_test


def build_search(n_iter: int, cv: int) -> RandomizedSearchCV:
    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    return RandomizedSearchCV(
        estimator=base,
        param_distributions=config.PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )


def evaluate_on_test(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "log_loss": float(log_loss(y_test, proba)),
        "baseline_fg_pct": float(y_test.mean()),
        "n_test": int(len(y_test)),
    }
    logger.info("Test — AUC=%.4f | log-loss=%.4f | baseline FG%%=%.3f",
                metrics["roc_auc"], metrics["log_loss"], metrics["baseline_fg_pct"])
    return metrics


def train(n_iter: int, cv: int) -> None:
    df = load_dataset()
    feature_cols = get_feature_columns(df)
    X_train, X_test, y_train, y_test = split_xy(df)

    logger.info("RandomizedSearchCV : %d combinaisons × %d folds…", n_iter, cv)
    search = build_search(n_iter, cv)
    search.fit(X_train, y_train)

    logger.info("Meilleur AUC CV = %.4f", search.best_score_)
    logger.info("Meilleurs params : %s", json.dumps(search.best_params_, indent=2))

    best_model = search.best_estimator_
    metrics = evaluate_on_test(best_model, X_test, y_test)

    # Persistance du modèle (format natif XGBoost, portable).
    best_model.save_model(config.MODEL_PATH)
    logger.info("Modèle → %s", config.MODEL_PATH)

    meta = {
        "feature_columns": feature_cols,
        "target_column": config.TARGET_COLUMN,
        "best_params": search.best_params_,
        "cv_best_auc": float(search.best_score_),
        "test_metrics": metrics,
        "n_train": int(len(X_train)),
        "random_state": config.RANDOM_STATE,
    }
    config.MODEL_META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("Métadonnées → %s", config.MODEL_META_PATH)
    logger.info("✅ Entraînement terminé.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entraînement XGBoost + tuning CV.")
    p.add_argument("--n-iter", type=int, default=config.N_SEARCH_ITER,
                   help="Itérations RandomizedSearchCV. Défaut : %(default)s")
    p.add_argument("--cv", type=int, default=config.CV_FOLDS,
                   help="Nombre de folds CV. Défaut : %(default)s")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train(args.n_iter, args.cv)


if __name__ == "__main__":
    main()
