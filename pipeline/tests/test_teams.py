import pytest

from nfl_vision import paths, teams


def test_normalizar_aliases():
    assert teams.normalizar(" lar ") == "LA"
    assert teams.normalizar("JAC") == "JAX"
    assert teams.normalizar("kc") == "KC"


def test_validar_e_cores_usando_cache(dados):
    df = teams.carregar_times(paths.cache_dir())
    assert teams.validar("LAR", df) == "LA"
    assert teams.cores("KC", df) == ["#E31837", "#FFB612"]
    with pytest.raises(teams.TimeDesconhecido, match="XYZ"):
        teams.validar("XYZ", df)
