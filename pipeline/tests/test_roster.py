import pytest

from nfl_vision import paths
from nfl_vision.schemas import Contexto, Correcao, NumeroDet, TimeDet
from nfl_vision.stages.roster import (
    RosterIndisponivel, aplicar_correcoes, buscar, carregar_roster, resolver,
)

CTX = Contexto(temporada=2025, semana=11, times=("KC", "BUF"))


@pytest.fixture
def df(dados):
    return carregar_roster(2025, paths.cache_dir())


def test_busca_simples(df):
    assert buscar(df, 2025, 11, "KC", 87) == ("TE", "Jogador KC 87", "00-1", "ok")


def test_duplicado_prefere_ativo(df):
    assert buscar(df, 2025, 11, "BUF", 14)[1] == "Jogador BUF 14 Ativo"


def test_duplicado_ativo_e_ambiguo(df):
    assert buscar(df, 2025, 11, "KC", 10) == (None, None, None, "ambiguo")


def test_numero_fora_do_roster(df):
    assert buscar(df, 2025, 11, "KC", 99)[3] == "numero_fora_do_roster"


def test_resolver_motivos(df):
    itens = resolver(df, CTX, {0: "KC", 1: None, 2: "BUF"}, {0: 87, 1: 15, 2: None})
    assert [(r.det_id, r.motivo) for r in itens] == [(0, "ok"), (1, "sem_time"), (2, "sem_numero")]


def test_correcoes_sobrescrevem_em_ordem():
    times = [TimeDet(det_id=0, time="KC", confianca=0.9),
             TimeDet(det_id=1, time=None, confianca=1.0, arbitro=True)]
    numeros = [NumeroDet(det_id=0, numero=None, confianca=0.0)]
    correcoes = [Correcao(det_id=0, numero=15, timestamp="t1"),
                 Correcao(det_id=0, time="buf", timestamp="t2")]

    time_por_det, num_por_det = aplicar_correcoes(times, numeros, correcoes)

    assert time_por_det == {0: "BUF"}
    assert num_por_det == {0: 15}


def test_sem_cache_e_sem_rede(tmp_path, monkeypatch):
    import nflreadpy

    def sem_rede(**kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(nflreadpy, "load_rosters_weekly", sem_rede)
    with pytest.raises(RosterIndisponivel, match="2024"):
        carregar_roster(2024, tmp_path)
