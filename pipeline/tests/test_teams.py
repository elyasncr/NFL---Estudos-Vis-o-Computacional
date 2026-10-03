import nflreadpy
import pytest

from nfl_vision import paths, teams


def test_normalizar_aliases():
    assert teams.normalizar(" lar ") == "LA"
    assert teams.normalizar("JAC") == "JAX"
    assert teams.normalizar("kc") == "KC"
    assert teams.normalizar("OAK") == "LV"
    assert teams.normalizar("SD") == "LAC"
    assert teams.normalizar("STL") == "LA"


def test_validar_e_cores_usando_cache(dados):
    df = teams.carregar_times(paths.cache_dir())
    assert teams.validar("LAR", df) == "LA"
    assert teams.cores("KC", df) == ["#E31837", "#FFB612"]
    with pytest.raises(teams.TimeDesconhecido, match="XYZ"):
        teams.validar("XYZ", df)


def test_times_indisponiveis_sem_cache_e_sem_rede(tmp_path, monkeypatch):
    def falha():
        raise ConnectionError("sem rede")

    monkeypatch.setattr(nflreadpy, "load_teams", falha)

    with pytest.raises(teams.TimesIndisponiveis, match="sem rede"):
        teams.carregar_times(tmp_path / "cache_vazio")
