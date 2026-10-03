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


@pytest.fixture
def foto_sintetica(tmp_path):
    """Foto com 2 KC, 2 BUF, 1 árbitro e 1 pessoa na arquibancada; caixas por det_id."""
    import cv2

    from sintetico import AZUL_BUF, VERMELHO_KC, arbitro, campo, jogador

    img = campo()
    caixas = [
        jogador(img, 100, 300, VERMELHO_KC),
        jogador(img, 200, 300, VERMELHO_KC),
        jogador(img, 400, 300, AZUL_BUF),
        jogador(img, 500, 300, AZUL_BUF),
        arbitro(img, 650, 300),
        jogador(img, 300, 10, VERMELHO_KC),
    ]
    caminho = tmp_path / "foto.png"
    cv2.imwrite(str(caminho), img)
    return caminho, caixas


class LeitorFalso:
    def __init__(self, respostas):
        self.respostas = list(respostas)

    def ler(self, img):
        return self.respostas.pop(0) if self.respostas else []


@pytest.fixture
def modelos_falsos(monkeypatch, foto_sintetica, tmp_path):
    """Substitui YOLO e PaddleOCR. Conta as chamadas ao detector."""
    from types import SimpleNamespace

    from nfl_vision.schemas import Deteccao
    from nfl_vision.stages import detect, jersey

    _, caixas = foto_sintetica
    chamadas = {"detect": 0}
    pesos = tmp_path / "pesos_falsos.pt"
    pesos.write_bytes(b"pesos falsos")

    def detectar(img, cfg):
        chamadas["detect"] += 1
        return [Deteccao(det_id=i, bbox=b, confianca=0.9) for i, b in enumerate(caixas)]

    leitor = LeitorFalso([[("87", 0.95)], [("15", 0.90)], [("17", 0.92)], [("3x", 0.99)]])
    monkeypatch.setattr(detect, "detectar_pessoas", detectar)
    monkeypatch.setattr(detect, "_modelo", lambda p: SimpleNamespace(ckpt_path=str(pesos)))
    monkeypatch.setattr(jersey, "leitor_padrao", lambda device: leitor)
    return chamadas
