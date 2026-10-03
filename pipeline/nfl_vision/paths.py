"""Localização dos dados. NFL_VISION_DATA sobrescreve o padrão <repo>/data."""

import os
from pathlib import Path

_PADRAO = Path(__file__).resolve().parents[2] / "data"


def dados_dir() -> Path:
    return Path(os.environ.get("NFL_VISION_DATA", _PADRAO))


def runs_dir() -> Path:
    return dados_dir() / "runs"


def cache_dir() -> Path:
    return dados_dir() / "cache"


def datasets_dir() -> Path:
    return dados_dir() / "datasets"


def avaliacoes_dir() -> Path:
    return dados_dir() / "avaliacoes"
