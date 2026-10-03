"""Siglas e cores oficiais dos times (nflverse), com cache em parquet."""

from pathlib import Path

import polars as pl

ALIASES = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "LVR": "LV"}


class TimeDesconhecido(ValueError):
    pass


def normalizar(sigla: str) -> str:
    s = sigla.strip().upper()
    return ALIASES.get(s, s)


def carregar_times(cache_dir: Path) -> pl.DataFrame:
    arquivo = cache_dir / "teams.parquet"
    if arquivo.exists():
        return pl.read_parquet(arquivo)
    import nflreadpy as nfl

    df = nfl.load_teams().select("team_abbr", "team_color", "team_color2")
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(arquivo)
    return df


def validar(sigla: str, times_df: pl.DataFrame) -> str:
    s = normalizar(sigla)
    if s not in set(times_df["team_abbr"].to_list()):
        raise TimeDesconhecido(f"time '{sigla}' não existe no nflverse")
    return s


def cores(sigla: str, times_df: pl.DataFrame) -> list[str]:
    linha = times_df.filter(pl.col("team_abbr") == normalizar(sigla))
    if linha.height == 0:
        raise TimeDesconhecido(f"time '{sigla}' não existe no nflverse")
    return [c for c in (linha["team_color"][0], linha["team_color2"][0]) if c]
