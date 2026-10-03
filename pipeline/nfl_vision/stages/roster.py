"""Etapa 5: posição e nome pelo roster semanal (temporada + semana + time + número)."""

from pathlib import Path
from typing import Iterable

import polars as pl

from nfl_vision import paths, teams
from nfl_vision.schemas import Contexto, Correcao, NumeroDet, RosterDet, RosterOut, TimeDet

COLUNAS = ["season", "week", "team", "jersey_number", "position", "full_name", "gsis_id", "status"]


class RosterIndisponivel(RuntimeError):
    pass


def carregar_roster(temporada: int, cache_dir: Path) -> pl.DataFrame:
    arquivo = cache_dir / "rosters" / f"{temporada}.parquet"
    if arquivo.exists():
        return pl.read_parquet(arquivo)
    try:
        import nflreadpy as nfl

        df = nfl.load_rosters_weekly(seasons=[temporada]).select(COLUNAS)
    except Exception as exc:
        raise RosterIndisponivel(
            f"roster de {temporada} indisponível (sem cache e sem rede?): {exc}"
        ) from exc
    df = df.with_columns(
        pl.col("jersey_number").cast(pl.Int64, strict=False),
        pl.col("team").map_elements(teams.normalizar, return_dtype=pl.String),
    )
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(arquivo)
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
    df = carregar_roster(estado.contexto.temporada, paths.cache_dir())
    time_por_det, num_por_det = aplicar_correcoes(
        estado.saidas["team"].itens, estado.saidas["jersey"].itens, estado.correcoes()
    )
    return RosterOut(itens=resolver(df, estado.contexto, time_por_det, num_por_det))
