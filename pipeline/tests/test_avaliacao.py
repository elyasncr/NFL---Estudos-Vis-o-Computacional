from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from nfl_vision.config import Config
from nfl_vision.eval import detect as eval_detect
from nfl_vision.eval import jersey as eval_jersey
from nfl_vision.eval import preditores
from nfl_vision.eval.datasets import AmostraDeteccao
from nfl_vision.schemas import Deteccao
from nfl_vision.stages.jersey import Leitura

JOGADOR = (10.0, 10.0, 50.0, 90.0)
ARBITRO = (60.0, 10.0, 100.0, 90.0)


class PreditorFalso:
    nome = "falso"

    def __init__(self, caixas):
        self.caixas = caixas

    def prever(self, imagem):
        return [(0.9, c) for c in self.caixas]


def test_avaliar_deteccao():
    amostras = [AmostraDeteccao(Path("x.jpg"), {"player": [JOGADOR], "referee": [ARBITRO]})]

    so_jogador = eval_detect.avaliar(PreditorFalso([JOGADOR]), amostras)
    com_arbitro = eval_detect.avaliar(PreditorFalso([JOGADOR, ARBITRO]), amostras)

    assert so_jogador == {"preditor": "falso", "imagens": 1, "map50": 1.0, "arbitros_como_jogador": 0.0}
    assert com_arbitro["map50"] == 1.0  # árbitro não é GT de jogador, vira FP de menor rank
    assert com_arbitro["arbitros_como_jogador"] == 1.0


def test_avaliar_deteccao_sem_arbitros_no_dataset():
    amostras = [AmostraDeteccao(Path("x.jpg"), {"player": [JOGADOR]})]
    assert eval_detect.avaliar(PreditorFalso([JOGADOR]), amostras)["arbitros_como_jogador"] is None


class LeitorFalso:
    def __init__(self, respostas):
        self.respostas = list(respostas)

    def ler(self, img):
        return self.respostas.pop(0)


def test_avaliar_ocr(tmp_path):
    amostras = []
    for i, rotulo in enumerate(["87", "9", "15", "-1", "07", "00"]):
        caminho = tmp_path / f"{i}.jpg"
        Image.new("RGB", (20, 20)).save(caminho)
        amostras.append((caminho, rotulo))
    leitor = LeitorFalso([[Leitura("87", 0.9)], [Leitura("8", 0.9)], [Leitura("15", 0.3)]])

    r = eval_jersey.avaliar(leitor, amostras, limiar=0.60)

    assert r == {
        "amostras": 3,                  # "-1", "07" e "00" não são números legíveis
        "taxa_null": pytest.approx(1 / 3),
        "acuracia_entre_lidos": 0.5,
        "acuracia_geral": pytest.approx(1 / 3),
    }


def _detectar_caixas(monkeypatch, caixas):
    def detectar(img, cfg):
        return [Deteccao(det_id=i, bbox=b, confianca=0.9) for i, b in enumerate(caixas)]

    monkeypatch.setattr(preditores, "detectar_pessoas", detectar)


def test_preditor_nosso_descarta_arbitro_e_fora_de_campo(foto_sintetica, monkeypatch):
    caminho, caixas = foto_sintetica
    _detectar_caixas(monkeypatch, caixas)

    saida = preditores.PreditorNosso(Config()).prever(caminho)

    assert [caixa for _, caixa in saida] == caixas[:4]  # sem árbitro (4) e arquibancada (5)


def test_preditor_nosso_mantem_quem_nao_da_para_classificar(foto_sintetica, monkeypatch):
    caminho, caixas = foto_sintetica
    _detectar_caixas(monkeypatch, caixas)
    monkeypatch.setattr(preditores, "pixels_uteis", lambda recorte, cfg: np.empty((0, 3)))
    monkeypatch.setattr(preditores, "eh_arbitro", lambda recorte: True)

    saida = preditores.PreditorNosso(Config()).prever(caminho)

    assert [caixa for _, caixa in saida] == caixas[:5]  # poucos pixels: não é árbitro


def test_rfdetr_filtra_pessoa(tmp_path):
    from types import SimpleNamespace

    imagem = tmp_path / "x.jpg"
    Image.new("RGB", (20, 20)).save(imagem)
    dets = SimpleNamespace(
        xyxy=np.array([[0, 0, 5, 5], [1, 1, 6, 6]], float), confidence=np.array([0.8, 0.7]),
        class_id=np.array([1, 18]), data={"class_name": np.array(["person", "dog"], dtype=object)},
    )
    p = object.__new__(preditores.PreditorRFDETR)
    p.modelo = SimpleNamespace(predict=lambda im, threshold: dets)
    p.conf, p.ids_pessoa = 0.25, {1}

    assert p.prever(imagem) == [(0.8, (0.0, 0.0, 5.0, 5.0))]


def test_roboflow_nfl_converte_caixas(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("ROBOFLOW_API_KEY", "chave")
    p = preditores.PreditorRoboflowNFL("nfl/1")
    resposta = {"predictions": [
        {"class": "player", "confidence": 0.9, "x": 10, "y": 20, "width": 4, "height": 8},
        {"class": "referee", "confidence": 0.9, "x": 10, "y": 20, "width": 4, "height": 8},
        {"class": "player", "confidence": 0.1, "x": 10, "y": 20, "width": 4, "height": 8},
    ]}
    p.cliente = SimpleNamespace(infer=lambda caminho, model_id: resposta)

    assert p.prever(Path("x.jpg")) == [(0.9, (8.0, 16.0, 12.0, 24.0))]


def test_roboflow_nfl_exige_chave(monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ROBOFLOW_API_KEY"):
        preditores.PreditorRoboflowNFL("nfl/1")
