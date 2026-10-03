import polars as pl
import pytest

TIMES = pl.DataFrame({
    "team_abbr": ["KC", "BUF", "LA"],
    "team_color": ["#E31837", "#00338D", "#003594"],
    "team_color2": ["#FFB612", "#C60C30", "#FFA300"],
})


@pytest.fixture
def dados(tmp_path, monkeypatch):
    """Diretório de dados temporário com caches prontos (sem rede)."""
    monkeypatch.setenv("NFL_VISION_DATA", str(tmp_path))
    cache = tmp_path / "cache"
    cache.mkdir()
    TIMES.write_parquet(cache / "teams.parquet")
    return tmp_path
