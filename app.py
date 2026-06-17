"""Interface graphique (Streamlit) du NBA Shot Success Predictor.

Trois usages :
    1. 🎯 Prédire — décrire un tir et obtenir P(tir réussi) en temps réel.
    2. 📊 Performance — métriques du modèle + courbes ROC / calibration.
    3. 🔍 Interprétabilité — SHAP + importance des features.
    4. 🏀 Données — shot chart et distributions du dataset.

Lancement :
    streamlit run app.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Arc, Circle, Rectangle
import matplotlib.pyplot as plt

from src import config
from src.features import get_feature_columns

# --------------------------------------------------------------------------------------
# Configuration de la page
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="NBA Shot Success Predictor", page_icon="🏀", layout="wide")


# --------------------------------------------------------------------------------------
# Chargement (mis en cache) des artefacts
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """Charge le modèle XGBoost entraîné. Renvoie None s'il n'existe pas encore."""
    if not config.MODEL_PATH.exists():
        return None
    from xgboost import XGBClassifier

    model = XGBClassifier()
    model.load_model(config.MODEL_PATH)
    return model


@st.cache_data(show_spinner=False)
def load_meta() -> dict | None:
    if config.MODEL_META_PATH.exists():
        return json.loads(config.MODEL_META_PATH.read_text())
    return None


@st.cache_data(show_spinner=False)
def load_metrics() -> dict | None:
    path = config.REPORTS_DIR / "metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame | None:
    if config.PROCESSED_DATASET_PATH.exists():
        return pd.read_parquet(config.PROCESSED_DATASET_PATH)
    return None


# --------------------------------------------------------------------------------------
# Helpers features : reconstruire un vecteur conforme au modèle
# --------------------------------------------------------------------------------------
def categorical_groups(feature_cols: list[str]) -> dict[str, list[str]]:
    """Pour chaque feature catégorielle, liste les modalités one-hot disponibles."""
    groups: dict[str, list[str]] = {}
    for g in config.CATEGORICAL_FEATURES:
        prefix = g + "_"
        groups[g] = sorted(c[len(prefix):] for c in feature_cols if c.startswith(prefix))
    return groups


def build_input_row(feature_cols: list[str], numeric: dict, chosen_cat: dict) -> pd.DataFrame:
    """Construit la ligne (1×N) attendue par le modèle à partir des choix utilisateur."""
    row = {c: 0.0 for c in feature_cols}
    for name, value in numeric.items():
        if name in row:
            row[name] = float(value)
    for group, value in chosen_cat.items():
        col = f"{group}_{value}"
        if col in row:
            row[col] = 1.0
    return pd.DataFrame([row])[feature_cols]


def proba_color(p: float) -> str:
    """Vert si tir favorable, orange moyen, rouge difficile."""
    if p >= 0.55:
        return "#2e9e3f"
    if p >= 0.40:
        return "#e08a00"
    return "#d23b3b"


def draw_court(ax, color="black", lw=1.5):
    """Demi-terrain NBA (coordonnées stats.nba : 1 unité = 0.1 pied)."""
    elements = [
        Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False),
        Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color),
        Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color, fill=False),
        Arc((0, 142.5), 120, 120, theta1=0, theta2=180, linewidth=lw, color=color),
        Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw, color=color),
        Rectangle((-220, -47.5), 0, 140, linewidth=lw, color=color),
        Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color),
        Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw, color=color),
        Rectangle((-250, -47.5), 500, 470, linewidth=lw, color=color, fill=False),
    ]
    for el in elements:
        ax.add_patch(el)
    return ax


# --------------------------------------------------------------------------------------
# En-tête
# --------------------------------------------------------------------------------------
st.title("🏀 NBA Shot Success Predictor")
st.caption("Estimation de la probabilité qu'un tir NBA soit réussi — modèle XGBoost "
           "entraîné sur des données réelles (`nba_api` / ShotChartDetail).")

model = load_model()
meta = load_meta()
metrics = load_metrics()
data = load_data()

# Garde-fou : pipeline pas encore exécuté.
if model is None or meta is None:
    st.warning(
        "⚠️ **Modèle introuvable.** Lance d'abord le pipeline (dans le venv activé) :\n\n"
        "```bash\n"
        "python -m src.data_ingestion --seasons 2023-24\n"
        "python -m src.features --seasons 2023-24\n"
        "python -m src.train\n"
        "python -m src.evaluate\n"
        "```"
    )
    st.stop()

feature_cols = meta["feature_columns"]
cat_groups = categorical_groups(feature_cols)

# Bandeau de métriques clés (toujours visible).
test_m = (metrics or meta.get("test_metrics") or {})
c1, c2, c3, c4 = st.columns(4)
c1.metric("ROC AUC", f"{test_m.get('roc_auc', float('nan')):.3f}",
          help="Capacité à distinguer un tir réussi d'un tir manqué (0.5 = hasard, 1 = parfait).")
c2.metric("Log-loss", f"{test_m.get('log_loss', float('nan')):.3f}",
          help="Erreur sur les probabilités prédites (plus bas = mieux).")
c3.metric("Brier score", f"{test_m.get('brier_score', float('nan')):.3f}",
          help="Qualité de calibration des probabilités (plus bas = mieux).")
c4.metric("FG% moyen (référence)", f"{test_m.get('baseline_fg_pct', float('nan')):.1%}",
          help="Taux de réussite moyen — le point de comparaison naïf.")

st.divider()

# --------------------------------------------------------------------------------------
# Onglets
# --------------------------------------------------------------------------------------
tab_predict, tab_perf, tab_shap, tab_data = st.tabs(
    ["🎯 Prédire un tir", "📊 Performance", "🔍 Interprétabilité", "🏀 Données"]
)

# ============================== 1. PRÉDICTION ==============================
with tab_predict:
    st.subheader("Décris un tir, obtiens sa probabilité de réussite")

    left, right = st.columns([1, 1])

    with left:
        st.markdown("**Paramètres principaux**")
        shot_distance = st.slider("Distance du panier (pieds)", 0, 35, 15)
        shot_value = st.radio("Type de tir", [2, 3], horizontal=True,
                              format_func=lambda v: f"{v} points")

        action_opts = cat_groups.get("action_type", [])
        default_action = action_opts.index("Jump Shot") if "Jump Shot" in action_opts else 0
        action_type = st.selectbox("Action", action_opts, index=default_action) \
            if action_opts else None

        with st.expander("⚙️ Contexte avancé (zone, quart-temps)"):
            zone_basic = st.selectbox("Zone (basic)", cat_groups.get("shot_zone_basic", []))
            zone_area = st.selectbox("Zone (côté)", cat_groups.get("shot_zone_area", []))
            zone_range = st.selectbox("Zone (distance)", cat_groups.get("shot_zone_range", []))
            period = st.selectbox("Quart-temps", [1, 2, 3, 4])
            time_left = st.slider("Temps restant dans le quart (s)", 0, 720, 360)

    # Valeurs par défaut si l'expander n'a pas été ouvert (Streamlit garde l'état,
    # mais on sécurise avec des valeurs sensées issues des données).
    period = locals().get("period", 1)
    time_left = locals().get("time_left", 360)
    zone_basic = locals().get("zone_basic", cat_groups.get("shot_zone_basic", [""])[0])
    zone_area = locals().get("zone_area", cat_groups.get("shot_zone_area", [""])[0])
    zone_range = locals().get("zone_range", cat_groups.get("shot_zone_range", [""])[0])

    # Géométrie : tir supposé centré → loc_x=0, loc_y ≈ distance (en 0.1 pied).
    numeric = {
        "shot_distance": shot_distance,
        "loc_x": 0.0,
        "loc_y": shot_distance * 10.0,
        "period": period,
        "period_time_remaining": time_left,
        "shot_value": shot_value,
    }
    chosen_cat = {
        "action_type": action_type,
        "shot_zone_basic": zone_basic,
        "shot_zone_area": zone_area,
        "shot_zone_range": zone_range,
    }

    X = build_input_row(feature_cols, numeric, chosen_cat)
    p = float(model.predict_proba(X)[0, 1])

    with right:
        st.markdown("**Probabilité de réussite estimée**")
        color = proba_color(p)
        st.markdown(
            f"<div style='font-size:64px;font-weight:700;color:{color};"
            f"line-height:1.1'>{p:.0%}</div>",
            unsafe_allow_html=True,
        )
        st.progress(p)

        baseline = test_m.get("baseline_fg_pct", 0.47)
        delta = p - baseline
        verdict = "tir favorable ✅" if p >= 0.55 else (
            "tir difficile ❌" if p < 0.40 else "tir moyen ➖")
        st.markdown(
            f"**Verdict : {verdict}**  \n"
            f"{'+' if delta >= 0 else ''}{delta:.0%} vs la moyenne ligue "
            f"({baseline:.0%})."
        )
        st.caption("La géométrie suppose un tir centré (loc_x = 0). Ajuste la zone dans "
                   "« Contexte avancé » pour un tir en coin / aile.")

# ============================== 2. PERFORMANCE ==============================
with tab_perf:
    st.subheader("Performance du modèle sur le jeu de test")
    st.markdown(
        f"- **{test_m.get('n_test', '?')}** tirs de test\n"
        f"- **AUC {test_m.get('roc_auc', float('nan')):.3f}** : à distance et type de tir "
        "identiques, le modèle classe correctement réussite vs échec dans "
        f"~{test_m.get('roc_auc', 0.65):.0%} des cas.\n"
        "- Un AUC ~0.65 est cohérent sans donnée défensive (distance du défenseur "
        "indisponible dans l'API publique)."
    )
    g1, g2 = st.columns(2)
    roc_png = config.REPORTS_DIR / "roc_curve.png"
    cal_png = config.REPORTS_DIR / "calibration_curve.png"
    if roc_png.exists():
        g1.image(str(roc_png), caption="Courbe ROC", use_container_width=True)
    if cal_png.exists():
        g2.image(str(cal_png), caption="Calibration des probabilités", use_container_width=True)
    if not roc_png.exists():
        st.info("Lance `python -m src.evaluate` pour générer les figures.")

    with st.expander("Hyperparamètres retenus (tuning par cross-validation)"):
        st.json(meta.get("best_params", {}))

# ============================== 3. SHAP ==============================
with tab_shap:
    st.subheader("Qu'est-ce qui fait rentrer un tir ?")
    st.markdown(
        "SHAP mesure la contribution de chaque feature à la prédiction. "
        "Plus une barre est longue, plus la feature pèse dans la décision du modèle."
    )
    s1, s2 = st.columns(2)
    shap_imp = config.REPORTS_DIR / "shap_importance.png"
    shap_sum = config.REPORTS_DIR / "shap_summary.png"
    feat_imp = config.REPORTS_DIR / "feature_importance.png"
    if shap_imp.exists():
        s1.image(str(shap_imp), caption="Importance moyenne (|SHAP|)", use_container_width=True)
    if shap_sum.exists():
        s2.image(str(shap_sum), caption="Impact + direction (beeswarm)", use_container_width=True)
    if feat_imp.exists():
        st.image(str(feat_imp), caption="Importance par gain (XGBoost)", use_container_width=True)
    if not shap_imp.exists():
        st.info("Lance `python -m src.evaluate` pour générer les figures SHAP.")

# ============================== 4. DONNÉES ==============================
with tab_data:
    st.subheader("Exploration du dataset")
    if data is None:
        st.info("Dataset introuvable — lance `python -m src.features`.")
    else:
        st.markdown(f"**{len(data):,}** tirs · FG% global **{data['shot_made'].mean():.1%}**")

        d1, d2 = st.columns([1.1, 1])
        with d1:
            st.markdown("**Shot chart** (échantillon)")
            sample = data.sample(min(4000, len(data)), random_state=config.RANDOM_STATE)
            made = sample[sample["shot_made"] == 1]
            missed = sample[sample["shot_made"] == 0]
            fig, ax = plt.subplots(figsize=(6, 5.6))
            ax.scatter(missed["loc_x"], missed["loc_y"], s=6, c="#d23b3b", alpha=0.35, label="Manqué")
            ax.scatter(made["loc_x"], made["loc_y"], s=6, c="#2e9e3f", alpha=0.45, label="Réussi")
            draw_court(ax)
            ax.set_xlim(-250, 250); ax.set_ylim(-50, 425)
            ax.set_aspect("equal"); ax.axis("off"); ax.legend(loc="upper right", fontsize=8)
            st.pyplot(fig)

        with d2:
            st.markdown("**FG% par distance**")
            bins = np.arange(0, 35, 2)
            fg = data.groupby(pd.cut(data["shot_distance"], bins))["shot_made"].mean()
            fig2, ax2 = plt.subplots(figsize=(5, 3.2))
            ax2.plot([b.mid for b in fg.index], fg.values, marker="o")
            ax2.set_xlabel("Distance (pieds)"); ax2.set_ylabel("FG%"); ax2.grid(alpha=0.3)
            st.pyplot(fig2)

            st.markdown("**FG% : 2 pts vs 3 pts**")
            st.bar_chart(data.groupby("shot_value")["shot_made"].mean())

st.divider()
st.caption("Données © NBA via stats.nba.com (API non officielle) — usage éducatif. "
           "Modèle : XGBoost + SHAP.")
