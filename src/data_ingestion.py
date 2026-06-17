"""Ingestion des données de tir NBA via ``nba_api``.

Source : ``ShotChartDetail`` — le détail *tir par tir*, avec coordonnées de terrain
(LOC_X / LOC_Y), distance, type de tir, zone, quart-temps et issue (réussi/manqué).
Disponible à partir de la saison 1996-97.

⚠️ Pourquoi pas les features défenseur / shot clock / dribbles ? Le *tracking shot log*
qui les exposait par tir a été retiré de l'API publique NBA (~2017). L'endpoint
``PlayerDashPtShots`` ne fournit plus que des agrégats marginaux (FG% par tranche),
non joignables à l'échelle du tir individuel. ShotChartDetail est donc la seule source
permettant un vrai dataset de classification par tir. (Cf. README.)

Stratégie d'ingestion :
    1. Pour chaque équipe NBA (liste statique, 30 équipes, sans appel réseau),
       récupérer TOUS ses tirs de la saison en un seul appel (player_id=0).
    2. Concaténer les 30 réponses → tous les tirs de la ligue pour la saison.
    3. Mettre en cache au format Parquet dans ``data/raw/``.

→ ~30 requêtes par saison (contre ~450 en itérant par joueur).

Robustesse (l'API stats.nba.com est non officielle et fragile) :
    * délai configurable entre les appels (anti rate-limit) ;
    * retry avec backoff exponentiel sur timeout / erreur réseau ;
    * cache à deux niveaux — par équipe ET agrégat saison — pour ne jamais
      re-télécharger ce qui est déjà sur disque.

Usage :
    python -m src.data_ingestion --seasons 2022-23 2023-24
    python -m src.data_ingestion --seasons 2023-24 --max-teams 5 --delay 1.0
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from src import config

# nba_api est importé paresseusement dans les fonctions qui en ont besoin, pour que
# `import src.data_ingestion` ne plante pas si la lib n'est pas installée (tests, docs).

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("data_ingestion")


# --------------------------------------------------------------------------------------
# Appel API générique avec retry / backoff
# --------------------------------------------------------------------------------------
def _call_with_retry(endpoint_cls, *, label: str, **kwargs) -> pd.DataFrame | None:
    """Instancie un endpoint nba_api et renvoie son premier DataFrame, avec retry.

    Renvoie ``None`` si toutes les tentatives échouent (on logge et on continue plutôt
    que de faire tomber tout le batch pour une équipe).
    """
    last_err: Exception | None = None
    for attempt in range(config.MAX_RETRIES):
        try:
            endpoint = endpoint_cls(timeout=config.REQUEST_TIMEOUT, **kwargs)
            return endpoint.get_data_frames()[0]
        except Exception as err:  # timeouts, JSONDecodeError, 403, etc.
            last_err = err
            wait = config.BACKOFF_FACTOR ** attempt
            logger.warning(
                "%s — échec %d/%d (%s). Nouvel essai dans %.1fs",
                label, attempt + 1, config.MAX_RETRIES, type(err).__name__, wait,
            )
            time.sleep(wait)
    logger.error("%s — abandon après %d tentatives : %s", label, config.MAX_RETRIES, last_err)
    return None


# --------------------------------------------------------------------------------------
# 1. Liste statique des équipes NBA (aucun appel réseau)
# --------------------------------------------------------------------------------------
def get_teams() -> list[dict]:
    """Renvoie la liste des 30 équipes NBA : dicts avec 'id', 'full_name', 'abbreviation'."""
    from nba_api.stats.static import teams as static_teams

    return static_teams.get_teams()


# --------------------------------------------------------------------------------------
# 2. Tirs d'une équipe sur une saison
# --------------------------------------------------------------------------------------
def fetch_team_shots(team_id: int, season: str) -> pd.DataFrame | None:
    """Télécharge (avec cache) tous les tirs d'une équipe pour une saison.

    Cache : ``data/raw/shots/<season>/<team_id>.parquet``. Un fichier vide (0 ligne) est
    tout de même écrit pour mémoriser le résultat et éviter de re-tenter l'appel.
    """
    cache_dir = config.RAW_DIR / "shots" / season
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{team_id}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    from nba_api.stats.endpoints import shotchartdetail

    df = _call_with_retry(
        shotchartdetail.ShotChartDetail,
        label=f"ShotChartDetail[{season}/{team_id}]",
        team_id=team_id,
        player_id=0,                       # 0 = tous les joueurs de l'équipe
        season_nullable=season,
        season_type_all_star=config.SEASON_TYPE,
        context_measure_simple="FGA",      # tous les tirs tentés (réussis + manqués)
    )
    time.sleep(config.REQUEST_DELAY)
    if df is None:
        return None  # échec réseau : ne PAS cacher, on retentera au prochain run

    df["SEASON"] = season
    df.to_parquet(cache, index=False)
    return df


# --------------------------------------------------------------------------------------
# 3. Agrégation d'une saison complète
# --------------------------------------------------------------------------------------
def ingest_season(season: str, max_teams: int | None = None) -> Path:
    """Télécharge les tirs de toutes les équipes d'une saison et écrit l'agrégat Parquet.

    Renvoie le chemin du fichier agrégé ``data/raw/shots_<season>.parquet``.
    """
    if season < config.EARLIEST_SEASON:
        logger.warning(
            "⚠️  %s est antérieure à %s : ShotChartDetail risque de ne rien renvoyer.",
            season, config.EARLIEST_SEASON,
        )

    agg_path = config.RAW_DIR / f"shots_{season}.parquet"
    if agg_path.exists() and max_teams is None:
        logger.info("Saison %s : agrégat déjà présent → %s", season, agg_path.name)
        return agg_path

    teams = get_teams()
    if max_teams is not None:
        teams = teams[:max_teams]
        logger.info("Saison %s : limité à %d équipes (--max-teams).", season, max_teams)

    try:
        from tqdm import tqdm
        iterator = tqdm(teams, desc=f"{season}")
    except ImportError:  # tqdm optionnel
        iterator = teams

    frames: list[pd.DataFrame] = []
    n_fail = 0
    for team in iterator:
        df = fetch_team_shots(int(team["id"]), season)
        if df is None:
            n_fail += 1
            logger.warning("Équipe %s : échec, ignorée.", team["abbreviation"])
            continue
        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError(
            f"Aucun tir récupéré pour {season} "
            f"(saison < 1996-97 ? IP bloquée ? {n_fail} échecs réseau)."
        )

    season_df = pd.concat(frames, ignore_index=True)
    season_df.to_parquet(agg_path, index=False)
    logger.info(
        "Saison %s : %d tirs / %d équipes (%d échecs réseau) → %s",
        season, len(season_df), len(frames), n_fail, agg_path.name,
    )
    return agg_path


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingestion des tirs NBA (ShotChartDetail) via nba_api.")
    p.add_argument(
        "--seasons", nargs="+", default=config.DEFAULT_SEASONS,
        help='Saisons au format "2023-24" (≥ 1996-97). Défaut : %(default)s',
    )
    p.add_argument(
        "--max-teams", type=int, default=None,
        help="Limiter le nombre d'équipes par saison (échantillon rapide / test).",
    )
    p.add_argument(
        "--delay", type=float, default=None,
        help=f"Délai entre appels API en s (défaut {config.REQUEST_DELAY}).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.delay is not None:
        config.REQUEST_DELAY = args.delay
        logger.info("Délai inter-requêtes fixé à %.2fs", config.REQUEST_DELAY)

    for season in args.seasons:
        logger.info("=" * 60)
        logger.info("Ingestion saison %s", season)
        ingest_season(season, max_teams=args.max_teams)

    logger.info("✅ Ingestion terminée.")


if __name__ == "__main__":
    main()
