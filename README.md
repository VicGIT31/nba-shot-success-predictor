# 🏀 NBA Shot Success Predictor

Modèle de classification **probabiliste** qui estime `P(tir réussi)` — *un tir va-t-il
rentrer, oui/non* — à partir de données de tir réelles de la NBA récupérées via
[`nba_api`](https://github.com/swar/nba_api).

Pipeline ML complet et reproductible : ingestion des données → feature engineering →
entraînement XGBoost avec tuning par cross-validation → évaluation (AUC, log-loss,
calibration) → interprétabilité SHAP.

---

## 🎯 Objectif

Prédire la probabilité qu'un tir soit réussi en fonction de son contexte : distance au
panier, position sur le terrain, zone, type de tir (layup / dunk / jump shot…) et
moment dans le quart-temps.

La sortie est une **probabilité calibrée**, pas une simple classe 0/1 — c'est ce qui
rend le modèle exploitable (qualité de tir, *expected FG%*, etc.).

---

## 🧱 Stack technique

| Rôle | Outil |
|---|---|
| Langage | Python 3.10+ |
| Données NBA | `nba_api` |
| Manipulation de données | `pandas`, `numpy` |
| Cache | `pyarrow` (Parquet) |
| Modèle | `xgboost` (`XGBClassifier`) |
| Tuning / métriques | `scikit-learn` |
| Interprétabilité | `shap` |
| Visualisation | `matplotlib` (+ `seaborn`) |

---

## 📊 Source des données & choix d'architecture

### ⚠️ Important : ce qui est (et n'est plus) disponible dans l'API NBA

Le cahier des charges initial visait des features de *tracking par tir* — **distance du
défenseur, shot clock restant, nombre de dribbles**. Vérification faite sur l'API :
**ces données ne sont plus accessibles par tir.** Le *tracking shot log* qui les
exposait a été retiré de l'API publique NBA (~2017).

| Endpoint testé | Granularité | Défenseur / shot clock / dribbles |
|---|---|---|
| `PlayerDashPtShots` | **Agrégats** (FG% par tranche, tables séparées) | Oui, mais **pas par tir** → non joignable à l'échelle du tir |
| **`ShotChartDetail`** ✅ | **Tir par tir** | Non |

➡️ Le projet est donc bâti sur **`ShotChartDetail`**, seule source fournissant un vrai
dataset *par tir* exploitable en classification. Bonus : il fournit les **coordonnées
de terrain** (`LOC_X` / `LOC_Y`) et la **zone** de tir, qui se révèlent très prédictives.

### Contraintes de l'API (à connaître avant l'ingestion)

1. **Profondeur historique** — `ShotChartDetail` remonte à la saison **1996-97**.
2. **Ingestion par équipe** — On récupère **tous les tirs d'une équipe en un appel**
   (`player_id=0`) → **~30 requêtes par saison** (et non ~450 par joueur).
3. **API non officielle & fragile** — `stats.nba.com` n'a pas de SLA, *timeout* et
   bloque parfois (surtout depuis des IP datacenter/cloud → préférer une IP
   résidentielle). Le module gère : délai configurable entre appels (`REQUEST_DELAY`),
   **retry avec backoff exponentiel**, et **cache Parquet** (rien n'est re-téléchargé).

   ⏱️ Ordre de grandeur : **~1 min par saison** (30 appels), quasi instantané ensuite.

---

## 📂 Structure du projet

```
nba-shot-success-predictor/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py                   # interface graphique Streamlit (prédiction + dashboards)
├── src/
│   ├── config.py            # constantes : chemins, saisons, hyperparams, features
│   ├── data_ingestion.py    # tirs via ShotChartDetail (par équipe) → data/raw/*.parquet
│   ├── features.py          # nettoyage + features + one-hot → data/processed/
│   ├── train.py             # XGBClassifier + tuning CV → models/
│   └── evaluate.py          # AUC, log-loss, calibration, SHAP → reports/
├── notebooks/
│   └── exploration.ipynb    # EDA + shot chart
├── data/
│   ├── raw/                 # cache brut par équipe + agrégat saison (Parquet)
│   └── processed/           # dataset features prêt à l'entraînement
├── models/                  # modèle entraîné + métadonnées (features, params, métriques)
└── reports/                 # figures (calibration, ROC, SHAP, importances) + metrics.json
```

---

## ⚙️ Installation

```bash
cd nba-shot-success-predictor

# Environnement virtuel (indispensable : Python système verrouillé depuis 3.12 / PEP 668)
python3 -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

pip install -r requirements.txt
```

> 💡 Une fois le venv activé (`(.venv)` dans le prompt), la commande `python` existe et
> `pip install` fonctionne. À refaire dans chaque nouveau terminal.

---

## 🚀 Utilisation (lancer les étapes **une par une**, dans l'ordre)

Le pipeline est séquentiel : chaque étape lit le fichier produit par la précédente.

```bash
# 1) Ingestion — télécharge les tirs (mis en cache ensuite)
python -m src.data_ingestion --seasons 2022-23 2023-24

#    Variantes utiles :
python -m src.data_ingestion --seasons 2023-24 --max-teams 5     # échantillon rapide
python -m src.data_ingestion --seasons 2023-24 --delay 1.5       # API plus prudente

# 2) Features — construit le dataset d'entraînement → data/processed/
python -m src.features --seasons 2022-23 2023-24

# 3) Entraînement — XGBoost + tuning CV → models/
python -m src.train

# 4) Évaluation — métriques + figures → reports/
python -m src.evaluate
```

---

## 🖥️ Interface graphique

Une fois le pipeline exécuté (modèle + figures présents), lance l'app web :

```bash
streamlit run app.py
```

Elle s'ouvre dans le navigateur (http://localhost:8501) avec 4 onglets :

| Onglet | Contenu |
|---|---|
| 🎯 **Prédire un tir** | sliders (distance, type, action, zone…) → `P(tir réussi)` en direct + verdict |
| 📊 **Performance** | métriques (AUC, log-loss, Brier) + courbes ROC & calibration |
| 🔍 **Interprétabilité** | figures SHAP + importance des features |
| 🏀 **Données** | shot chart + FG% par distance / type de tir |

---

## 🧪 Features utilisées

| Feature | Source (`ShotChartDetail`) | Description |
|---|---|---|
| `shot_distance` | `SHOT_DISTANCE` | Distance au panier (pieds) |
| `loc_x`, `loc_y` | `LOC_X`, `LOC_Y` | Position du tir sur le terrain |
| `period` | `PERIOD` | Quart-temps |
| `period_time_remaining` | `MINUTES_REMAINING`,`SECONDS_REMAINING` | Secondes restantes dans le quart |
| `shot_value` | `SHOT_TYPE` | 2 ou 3 points |
| `shot_zone_basic/area/range` | `SHOT_ZONE_*` | Zone de tir (one-hot) |
| `action_type` | `ACTION_TYPE` | Type de tir : Jump Shot, Layup, Dunk… (one-hot, top-15 + Other) |

**Cible :** `shot_made` (1 = réussi, 0 = manqué), depuis `SHOT_MADE_FLAG`.

---

## 📈 Résultats

> Valeurs obtenues sur un échantillon de test (5 équipes, saison 2023-24, recherche
> d'hyperparamètres réduite). Relance `python -m src.evaluate` après un entraînement
> complet pour les chiffres définitifs.

| Métrique | Valeur (échantillon) |
|---|---|
| ROC AUC | ~0.65 |
| Log-loss | ~0.65 |
| Brier score | ~0.23 |
| Accuracy (@0.5) | ~0.63 |
| Baseline (FG% global) | ~0.48 |

> ℹ️ Un AUC autour de **0.65** est cohérent avec la littérature pour un modèle de
> réussite de tir *sans* données défensives : la position du défenseur, absente de
> l'API publique, est précisément le facteur qui ferait grimper la performance.

**Figures générées dans `reports/` :**
- `calibration_curve.png` — fiabilité des probabilités prédites
- `roc_curve.png` — courbe ROC
- `shap_summary.png` — impact + direction des features (beeswarm)
- `shap_importance.png` — importance moyenne |SHAP|
- `feature_importance.png` — gain XGBoost (top 20)

**Lecture des résultats (confirmée par SHAP)** : `shot_distance` domine très largement,
suivi du type de tir (layups / Restricted Area = forte proba) et de la position
(`loc_x/loc_y`). Le moment dans le quart-temps a un impact marginal.

---

## 🛠️ Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| `command not found: python` | venv non activé | `source .venv/bin/activate` |
| `externally-managed-environment` | `pip` sur le Python système (PEP 668) | activer le venv puis `pip install` |
| `FileNotFoundError: shots_features.parquet` | étape précédente non exécutée | lancer les étapes **dans l'ordre**, une par une |
| `ReadTimeout` / appels qui pendouillent | rate limit `stats.nba.com` | augmenter `--delay`, relancer (le cache reprend) |
| Réponses vides / 403 systématiques | IP datacenter/VPN bloquée | lancer depuis une IP résidentielle |

---

## 📜 Notes

- Données © NBA via `stats.nba.com` (API non officielle) — usage éducatif / recherche.
- Reproductibilité : `RANDOM_STATE` fixé dans `src/config.py` ; `evaluate.py`
  reconstruit exactement le même split test que `train.py`.
