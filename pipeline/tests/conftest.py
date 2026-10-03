import polars as pl
import pytest

TIMES = pl.DataFrame({
    "team_abbr": ["KC", "BUF", "LA"],
    "team_color": ["#E31837", "#00338D", "#003594"],
    "team_color2": ["#FFB612", "#C60C30", "#FFA300"],
})

ROSTER = pl.DataFrame({
    "season": [2025] * 7,
    "week": [11] * 7,
    "team": ["KC", "KC", "BUF", "BUF", "BUF", "KC", "KC"],
    "jersey_number": [87, 15, 17, 14, 14, 10, 10],
    "position": ["TE", "QB", "QB", "WR", "WR", "WR", "RB"],
    "full_name": ["Jogador KC 87", "Jogador KC 15", "Jogador BUF 17",
                  "Jogador BUF 14 Ativo", "Jogador BUF 14 Reserva",
                  "Jogador KC 10 A", "Jogador KC 10 B"],
    "gsis_id": ["00-1", "00-2", "00-3", "00-4", "00-5", "00-6", "00-7"],
    "status": ["ACT", "ACT", "ACT", "ACT", "RES", "ACT", "ACT"],
})


@pytest.fixture
def dados(tmp_path, monkeypatch):
    """Diretório de dados temporário com caches prontos (sem rede)."""
    monkeypatch.setenv("NFL_VISION_DATA", str(tmp_path))
    cache = tmp_path / "cache"
    (cache / "rosters").mkdir(parents=True)
    TIMES.write_parquet(cache / "teams.parquet")
    ROSTER.write_parquet(cache / "rosters" / "2025.parquet")
    return tmp_path
