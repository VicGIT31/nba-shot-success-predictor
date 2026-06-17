"""Évaluation du modèle entraîné : métriques, calibration, ROC et interprétabilité SHAP.

Recharge le modèle (models/) et le dataset (data/processed/), reconstruit le MÊME split
test que ``train.py`` (random_state identique), puis produit dans ``reports/`` :
    * calibration_curve.png   — fiabilité des probabilités prédites + histogramme
    * roc_curve.png           — courbe ROC
    * shap_summary.png        — beeswarm SHAP (impact + direction par feature)
    * shap_importance.png     — importance moyenne |SHAP| (barres)
    * feature_importance.png  — importance par gain (interne XGBoost)
    * metrics.json            — récapitulatif chiffré

Usage :
    python -m src.evaluate
"""
from __future__ import annotations

import json
import logging

import matplotlib
matplotlib.use("Agg")  # backend non interactif (sauvegarde fichier, pas d'affichage)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

from src import config
from src.features import get_feature_columns
from src.train import load_dataset, split_xy

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("evaluate")


def load_model() -> XGBClassifier:
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{config.MODEL_PATH} introuvable. Lance d'abord : python -m src.train"
        )
    model = XGBClassifier()
    model.load_model(config.MODEL_PATH)
    return model


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def plot_calibration(y_test, proba) -> None:
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7),
                                   gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax1.plot([0, 1], [0, 1], "k--", label="Parfaitement calibré")
    ax1.plot(mean_pred, frac_pos, "o-", label="XGBoost")
    ax1.set_ylabel("Fréquence réelle de réussite")
    ax1.set_title("Courbe de calibration — P(tir réussi)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.hist(proba, bins=20, color="steelblue", alpha=0.8)
    ax2.set_xlabel("Probabilité prédite")
    ax2.set_ylabel("Nb de tirs")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = config.REPORTS_DIR / "calibration_curve.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    logger.info("Figure → %s", out.name)


def plot_roc(y_test, proba, auc: float) -> None:
    fpr, tpr, _ = roc_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"XGBoost (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Aléatoire")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbe ROC")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = config.REPORTS_DIR / "roc_curve.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    logger.info("Figure → %s", out.name)


def plot_xgb_importance(model: XGBClassifier, feature_cols: list[str]) -> None:
    importances = model.feature_importances_
    order = np.argsort(importances)[-20:]  # top 20 (one-hot peut être large)
    feats = np.array(feature_cols)[order]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(feats, importances[order], color="darkorange")
    ax.set_title("Importance des features (gain XGBoost, top 20)")
    ax.set_xlabel("Importance relative")
    fig.tight_layout()
    out = config.REPORTS_DIR / "feature_importance.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    logger.info("Figure → %s", out.name)


def shap_analysis(model: XGBClassifier, X_sample: pd.DataFrame) -> None:
    """Calcule les valeurs SHAP et trace beeswarm + importance moyenne."""
    import shap

    logger.info("Calcul SHAP sur %d tirs…", len(X_sample))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Beeswarm — impact et direction de chaque feature (max 15 affichées).
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15,
                      feature_names=list(X_sample.columns))
    plt.title("SHAP — impact des features sur P(tir réussi)")
    plt.tight_layout()
    out = config.REPORTS_DIR / "shap_summary.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info("Figure → %s", out.name)

    # Importance moyenne |SHAP| — barres.
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=15,
                      feature_names=list(X_sample.columns))
    plt.title("SHAP — importance moyenne (|valeur SHAP|)")
    plt.tight_layout()
    out = config.REPORTS_DIR / "shap_importance.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info("Figure → %s", out.name)


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def evaluate() -> dict:
    df = load_dataset()
    feature_cols = get_feature_columns(df)
    _, X_test, _, y_test = split_xy(df)  # même split que l'entraînement (random_state fixe)
    model = load_model()

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    metrics = {
        "roc_auc": float(auc),
        "log_loss": float(log_loss(y_test, proba)),
        "brier_score": float(brier_score_loss(y_test, proba)),
        "accuracy_0.5": float(((proba >= 0.5).astype(int) == y_test).mean()),
        "baseline_fg_pct": float(y_test.mean()),
        "n_test": int(len(y_test)),
    }
    logger.info("Métriques test : %s", json.dumps(metrics, indent=2))
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plot_calibration(y_test, proba)
    plot_roc(y_test, proba, auc)
    plot_xgb_importance(model, feature_cols)

    # SHAP : on échantillonne pour limiter le coût de calcul sur gros datasets.
    sample_n = min(5000, len(X_test))
    X_sample = X_test.sample(sample_n, random_state=config.RANDOM_STATE)
    try:
        shap_analysis(model, X_sample)
    except Exception as err:  # SHAP peut être capricieux selon les versions
        logger.warning("Analyse SHAP ignorée (%s : %s)", type(err).__name__, err)

    logger.info("✅ Évaluation terminée — figures dans %s/", config.REPORTS_DIR.name)
    return metrics


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
