"""Etapa 5: posição e nome pelo roster semanal (temporada + semana + time + número)."""

import os
from pathlib import Path
from typing import Iterable

import polars as pl

from nfl_vision import paths, teams
from nfl_vision.schemas import Contexto, Correcao, NumeroDet, RosterDet, RosterOut, TimeDet

COLUNAS = ["season", "week", "team", "jersey_number", "position", "full_name", "gsis_id", "status"]


class RosterIndisponivel(RuntimeError):
    pass


def _baixar(temporada: int) -> pl.DataFrame:
    import nflreadpy as nfl

    df = nfl.load_rosters_weekly(seasons=[temporada]).select(COLUNAS)
    return df.with_columns(
        pl.col("jersey_number").cast(pl.Int64, strict=False),
        pl.col("team").str.strip_chars().str.to_uppercase().replace(teams.ALIASES),
    )


def _gravar(df: pl.DataFrame, arquivo: Path) -> None:
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    tmp = arquivo.with_name(f"{arquivo.stem}.tmp.parquet")
    df.write_parquet(tmp)
    os.replace(tmp, arquivo)


def carregar_roster(temporada: int, semana: int, cache_dir: Path) -> pl.DataFrame:
    """Usa o cache se já cobre a semana; senão baixa de novo (temporada em andamento)."""
    arquivo = cache_dir / "rosters" / f"{temporada}.parquet"
    cache = pl.read_parquet(arquivo) if arquivo.exists() else None
    if cache is not None and cache.height and cache["week"].max() >= semana:
        return cache
    try:
        df = _baixar(temporada)
    except Exception as exc:
        if cache is not None:
            raise RosterIndisponivel(
                f"cache do roster de {temporada} desatualizado (sem a semana {semana}) "
                f"e sem rede: {exc}"
            ) from exc
        raise RosterIndisponivel(
            f"roster de {temporada} indisponível (sem cache e sem rede?): {exc}"
        ) from exc
    _gravar(df, arquivo)
    if semana not in set(df["week"].to_list()):
        raise RosterIndisponivel(f"semana {semana} de {temporada} não está no roster")
    return df


def buscar(df: pl.DataFrame, temporada: int, semana: int, time: str,
           numero: int) -> tuple[str | None, str | None, str | None, str]:
    linhas = df.filter(
        (pl.col("season") == temporada) & (pl.col("week") == semana)
        & (pl.col("team") == time) & (pl.col("jersey_number") == numero)
    )
    if linhas.height == 0:
        return None, None, None, "numero_fora_do_roster"
    if linhas.height > 1:
        linhas = linhas.filter(pl.col("status") == "ACT")
        if linhas.height != 1:
            return None, None, None, "ambiguo"
    r = linhas.row(0, named=True)
    return r["position"], r["full_name"], r["gsis_id"], "ok"


def aplicar_correcoes(times: Iterable[TimeDet], numeros: Iterable[NumeroDet],
                      correcoes: list[Correcao]) -> tuple[dict[int, str | None], dict[int, int | None]]:
    time_por_det = {t.det_id: t.time for t in times if not t.arbitro}
    num_por_det = {n.det_id: n.numero for n in numeros}
    for c in correcoes:
        if c.time is not None:
            time_por_det[c.det_id] = teams.normalizar(c.time)
        if c.numero is not None:
            num_por_det[c.det_id] = c.numero
    return time_por_det, num_por_det


def resolver(df: pl.DataFrame, ctx: Contexto, time_por_det: dict[int, str | None],
             num_por_det: dict[int, int | None]) -> list[RosterDet]:
    itens = []
    for det_id in sorted(time_por_det):
        time, numero = time_por_det[det_id], num_por_det.get(det_id)
        if time is None:
            itens.append(RosterDet(det_id=det_id, motivo="sem_time"))
        elif numero is None:
            itens.append(RosterDet(det_id=det_id, motivo="sem_numero"))
        else:
            posicao, nome, gsis_id, motivo = buscar(df, ctx.temporada, ctx.semana, time, numero)
            itens.append(RosterDet(det_id=det_id, posicao=posicao, nome=nome,
                                   gsis_id=gsis_id, motivo=motivo))
    return itens


def executar(estado) -> RosterOut:
    df = carregar_roster(estado.contexto.temporada, estado.contexto.semana, paths.cache_dir())
    time_por_det, num_por_det = aplicar_correcoes(
        estado.saidas["team"].itens, estado.saidas["jersey"].itens, estado.correcoes()
    )
    return RosterOut(itens=resolver(df, estado.contexto, time_por_det, num_por_det))
