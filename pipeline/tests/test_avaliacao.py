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
    pos_processamento = "nenhum"

    def __init__(self, caixas):
        self.caixas = caixas

    def prever(self, imagem):
        return [(0.9, c) for c in self.caixas]


def test_avaliar_deteccao():
    amostras = [AmostraDeteccao(Path("x.jpg"), {"player": [JOGADOR], "referee": [ARBITRO]})]

    so_jogador = eval_detect.avaliar(PreditorFalso([JOGADOR]), amostras)
    com_arbitro = eval_detect.avaliar(PreditorFalso([JOGADOR, ARBITRO]), amostras)

    assert so_jogador == {"preditor": "falso", "pos_processamento": "nenhum", "imagens": 1,
                          "map50": 1.0, "arbitros_como_jogador": 0.0}
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
        "amostras": 3,
        "excluidas": 3,                 # "-1", "07" e "00" não são números legíveis
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
    from nfl_vision.stages import team

    caminho, caixas = foto_sintetica
    _detectar_caixas(monkeypatch, caixas)
    monkeypatch.setattr(team, "pixels_uteis", lambda recorte, cfg: np.empty((0, 3)))
    monkeypatch.setattr(team, "eh_arbitro", lambda recorte: True)

    saida = preditores.PreditorNosso(Config()).prever(caminho)

    assert [caixa for _, caixa in saida] == caixas[:5]  # poucos pixels: não é árbitro


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


def test_preditor_yolo_bruto_nao_filtra(foto_sintetica, monkeypatch):
    caminho, caixas = foto_sintetica
    _detectar_caixas(monkeypatch, caixas)

    p = preditores.PreditorYoloBruto(Config())

    assert [caixa for _, caixa in p.prever(caminho)] == caixas  # árbitro e arquibancada incluídos
    assert p.pos_processamento == "nenhum"


def test_pos_processamento_de_cada_preditor():
    assert preditores.PreditorNosso.pos_processamento == "filtros de campo + remoção de árbitro"
    assert preditores.PreditorRFDETR.pos_processamento == "nenhum"
    assert preditores.PreditorRoboflowNFL.pos_processamento == "modelo treinado em NFL (classe player)"


def test_roboflow_nfl_envia_limiar_ao_servidor(monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "chave")
    p = preditores.PreditorRoboflowNFL("nfl/1", conf=0.4)
    assert p.cliente.inference_configuration.confidence_threshold == 0.4


def test_rotulo_dos_pesos():
    assert preditores.rotulo_pesos("yolo11m.pt") == "yolo11m"
    assert preditores.rotulo_pesos("../data/treinos/player-v1/weights/best.pt") == "player-v1"
    assert preditores.rotulo_pesos("/x/treinos/player-v1/weights/last.pt") == "player-v1/last"
    assert preditores.rotulo_pesos("pesos/meu-detector.pt") == "meu-detector"


def test_nomes_dos_preditores_indicam_o_detector():
    padrao = Config()
    res = padrao.detector_resolucao
    assert preditores.PreditorNosso(padrao).nome == f"nosso (rfdetr-base@{res} + filtros + árbitro)"
    assert preditores.PreditorYoloBruto(padrao).nome == "yolo11m bruto (COCO, pessoa)"
    assert preditores.PreditorRFDETR(padrao).nome == f"rfdetr bruto (rfdetr-base@{res}, COCO, pessoa)"
    assert preditores.PreditorRFDETR(Config(detector_resolucao=560)).nome == (
        "rfdetr bruto (rfdetr-base@560, COCO, pessoa)")
    yolo = Config(detector_tipo="yolo")
    assert preditores.PreditorNosso(yolo).nome == "nosso (yolo11m + filtros + árbitro)"
    ajustado = Config(detector_tipo="yolo", detector_pesos="data/treinos/player-v1/weights/best.pt")
    assert preditores.PreditorNosso(ajustado).nome == "nosso (player-v1 + filtros + árbitro)"
    assert preditores.PreditorYoloBruto(ajustado).nome == "yolo bruto (player-v1)"


def _espiar_detector(monkeypatch, caixas):
    """Substitui detectar_pessoas e guarda a config com que cada preditor o chamou."""
    configs = []

    def detectar(img, cfg):
        configs.append(cfg)
        return [Deteccao(det_id=i, bbox=b, confianca=0.9) for i, b in enumerate(caixas)]

    monkeypatch.setattr(preditores, "detectar_pessoas", detectar)
    return configs


def test_preditor_nosso_segue_o_tipo_da_config(foto_sintetica, monkeypatch):
    caminho, caixas = foto_sintetica
    configs = _espiar_detector(monkeypatch, caixas)

    preditores.PreditorNosso(Config()).prever(caminho)
    preditores.PreditorNosso(Config(detector_tipo="yolo")).prever(caminho)

    assert [c.detector_tipo for c in configs] == ["rfdetr", "yolo"]


def test_preditor_rfdetr_usa_o_detector_compartilhado_sem_filtros(foto_sintetica, monkeypatch):
    caminho, caixas = foto_sintetica
    configs = _espiar_detector(monkeypatch, caixas)
    cfg = Config(detector_tipo="yolo", detector_resolucao=560, detector_conf=0.01)

    p = preditores.PreditorRFDETR(cfg)

    assert [caixa for _, caixa in p.prever(caminho)] == caixas  # árbitro e arquibancada incluídos
    assert p.pos_processamento == "nenhum"
    assert (configs[0].detector_tipo, configs[0].detector_resolucao, configs[0].detector_conf) == (
        "rfdetr", 560, 0.01)
    assert cfg.detector_tipo == "yolo"  # a config recebida não é alterada


def test_preditor_yolo_bruto_forca_yolo(foto_sintetica, monkeypatch):
    caminho, caixas = foto_sintetica
    configs = _espiar_detector(monkeypatch, caixas)

    preditores.PreditorYoloBruto(Config()).prever(caminho)

    assert configs[0].detector_tipo == "yolo"


def test_preditor_rfdetr_usa_imagem_com_orientacao_exif(tmp_path, monkeypatch):
    imagem = tmp_path / "x.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # girada 90°
    Image.new("RGB", (100, 50)).save(imagem, exif=exif)
    formas = []

    def detectar(img, cfg):
        formas.append(img.shape)
        return []

    monkeypatch.setattr(preditores, "detectar_pessoas", detectar)

    assert preditores.PreditorRFDETR(Config()).prever(imagem) == []
    assert formas == [(100, 50, 3)]  # carregar_imagem aplica o EXIF, como no pipeline
