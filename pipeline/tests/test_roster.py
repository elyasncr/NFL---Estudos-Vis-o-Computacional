import polars as pl
import pytest

from nfl_vision import paths
from nfl_vision.schemas import Contexto, Correcao, NumeroDet, TimeDet
from nfl_vision.stages.roster import (
    COLUNAS, RosterIndisponivel, aplicar_correcoes, buscar, carregar_roster, resolver,
)

CTX = Contexto(temporada=2025, semana=11, times=("KC", "BUF"))


@pytest.fixture
def df(dados):
    return carregar_roster(2025, 11, paths.cache_dir())


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
        carregar_roster(2024, 1, tmp_path)


def _roster(semanas, temporada=2025):
    n = len(semanas)
    return pl.DataFrame({
        "season": [temporada] * n, "week": semanas, "team": ["KC"] * n,
        "jersey_number": list(range(1, n + 1)), "position": ["WR"] * n,
        "full_name": [f"J{i}" for i in range(n)], "gsis_id": [f"00-{i}" for i in range(n)],
        "status": ["ACT"] * n,
    })


def _cache(tmp_path, df, temporada=2025):
    arquivo = tmp_path / "rosters" / f"{temporada}.parquet"
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(arquivo)
    return arquivo


def _loader(monkeypatch, resposta):
    import nflreadpy

    chamadas = []

    def carregar(**kwargs):
        chamadas.append(kwargs)
        if isinstance(resposta, Exception):
            raise resposta
        return resposta

    monkeypatch.setattr(nflreadpy, "load_rosters_weekly", carregar)
    return chamadas


def test_cache_atual_nao_baixa(tmp_path, monkeypatch):
    _cache(tmp_path, _roster([3, 4]))
    chamadas = _loader(monkeypatch, ConnectionError("não devia baixar"))

    df = carregar_roster(2025, 4, tmp_path)

    assert chamadas == []
    assert df["week"].max() == 4


def test_cache_velho_e_atualizado(tmp_path, monkeypatch):
    arquivo = _cache(tmp_path, _roster([1, 2]))
    chamadas = _loader(monkeypatch, _roster([1, 2, 3, 4]))

    df = carregar_roster(2025, 4, tmp_path)

    assert len(chamadas) == 1
    assert df["week"].max() == 4
    assert pl.read_parquet(arquivo)["week"].max() == 4
    assert not list(arquivo.parent.glob("*.tmp.parquet"))


def test_semana_ausente_apos_baixar(tmp_path, monkeypatch):
    _loader(monkeypatch, _roster([1, 2]))
    with pytest.raises(RosterIndisponivel, match="semana 5 de 2025"):
        carregar_roster(2025, 5, tmp_path)


def test_sem_rede_com_cache_velho(tmp_path, monkeypatch):
    _cache(tmp_path, _roster([1, 2]))
    _loader(monkeypatch, ConnectionError("offline"))
    with pytest.raises(RosterIndisponivel, match="desatualizado"):
        carregar_roster(2025, 5, tmp_path)


def test_download_normaliza_tipos_e_siglas(tmp_path, monkeypatch):
    bruto = pl.DataFrame({
        "season": pl.Series([2025, 2025, 2025], dtype=pl.Int32),
        "week": pl.Series([11, 11, 11], dtype=pl.Int32),
        "team": [" lar", "KC", "KC"],
        "jersey_number": pl.Series([10, None, 87], dtype=pl.Int32),
        "position": ["WR", "OL", "TE"],
        "full_name": ["Jogador LA 10", "Sem Numero", "Jogador KC 87"],
        "gsis_id": ["00-a", "00-b", "00-c"],
        "status": ["ACT", "ACT", "ACT"],
        "headshot_url": ["x", "y", "z"],
        "birth_date": ["2000-01-01"] * 3,
    })
    _loader(monkeypatch, bruto)

    carregar_roster(2025, 11, tmp_path)
    cache = pl.read_parquet(tmp_path / "rosters" / "2025.parquet")

    assert cache.columns == COLUNAS
    assert cache["team"].to_list() == ["LA", "KC", "KC"]
    assert cache.schema["jersey_number"] == pl.Int64
    assert cache["jersey_number"].to_list() == [10, None, 87]
    assert buscar(cache, 2025, 11, "LA", 10) == ("WR", "Jogador LA 10", "00-a", "ok")


def _linhas(*linhas):
    return pl.DataFrame(
        [dict(season=2025, week=11, team="KC", jersey_number=n, position="WR",
              full_name=nome, gsis_id=nome, status=st) for n, nome, st in linhas]
    )


def test_cortado_sozinho_fica_fora_do_roster():
    df = _linhas((12, "Cortado", "CUT"))
    assert buscar(df, 2025, 11, "KC", 12)[3] == "numero_fora_do_roster"


def test_elevado_vence_cortado_com_mesmo_numero():
    df = _linhas((12, "Cortado", "CUT"), (12, "Elevado", "DEV"))
    assert buscar(df, 2025, 11, "KC", 12)[1:] == ("Elevado", "Elevado", "ok")


def test_inativos_nao_contam_para_ambiguidade():
    df = _linhas((12, "Ativo", "ACT"), (12, "Aposentado", "RET"), (12, "Trocado", "TRD"))
    assert buscar(df, 2025, 11, "KC", 12)[1] == "Ativo"


def test_correcoes_de_det_inexistente_ou_arbitro_sao_ignoradas():
    times = [TimeDet(det_id=0, time="KC", confianca=0.9),
             TimeDet(det_id=1, time=None, confianca=1.0, arbitro=True)]
    numeros = [NumeroDet(det_id=0, numero=87, confianca=0.9)]
    correcoes = [Correcao(det_id=7, numero=15, timestamp="t1"),
                 Correcao(det_id=1, time="BUF", numero=3, timestamp="t2")]

    time_por_det, num_por_det = aplicar_correcoes(times, numeros, correcoes)

    assert time_por_det == {0: "KC"}
    assert num_por_det == {0: 87}
