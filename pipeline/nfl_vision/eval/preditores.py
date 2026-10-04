"""Preditores comparados na avaliação: o nosso, YOLO bruto, RF-DETR bruto e o modelo NFL do Roboflow."""

from pathlib import Path
from typing import Protocol

from nfl_vision.config import Config
from nfl_vision.eval.datasets import chave_roboflow
from nfl_vision.schemas import BBox
from nfl_vision.stages.detect import aplicar_filtros, detectar_pessoas, rotulo_detector, rotulo_pesos
from nfl_vision.stages.ingest import carregar_imagem
from nfl_vision.stages.team import eh_arbitro_deteccao

Predicao = tuple[float, BBox]


class Preditor(Protocol):
    nome: str
    pos_processamento: str

    def prever(self, imagem: Path) -> list[Predicao]: ...


class PreditorNosso:
    pos_processamento = "filtros de campo + remoção de árbitro"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.nome = f"nosso ({rotulo_detector(cfg)} + filtros + árbitro)"

    def prever(self, imagem: Path) -> list[Predicao]:
        img = carregar_imagem(imagem)
        saida = []
        for d in aplicar_filtros(detectar_pessoas(img, self.cfg), img, self.cfg):
            if d.descartado:
                continue
            if eh_arbitro_deteccao(img, d.bbox, self.cfg):
                continue
            saida.append((d.confianca, d.bbox))
        return saida


class PreditorYoloBruto:
    """O YOLO do pipeline, sem filtros de campo nem remoção de árbitro."""

    pos_processamento = "nenhum"

    def __init__(self, cfg: Config):
        self.cfg = cfg.model_copy(update={"detector_tipo": "yolo"})
        padrao = cfg.detector_pesos == Config().detector_pesos
        self.nome = ("yolo11m bruto (COCO, pessoa)" if padrao
                     else f"yolo bruto ({rotulo_pesos(cfg.detector_pesos)})")

    def prever(self, imagem: Path) -> list[Predicao]:
        return [(d.confianca, d.bbox) for d in detectar_pessoas(carregar_imagem(imagem), self.cfg)]


class PreditorRFDETR:
    """O RF-DETR do pipeline (mesma função de detecção), sem filtros nem remoção de árbitro."""

    pos_processamento = "nenhum"

    def __init__(self, cfg: Config):
        self.cfg = cfg.model_copy(update={"detector_tipo": "rfdetr"})
        self.nome = f"rfdetr bruto ({rotulo_detector(self.cfg)}, COCO, pessoa)"

    def prever(self, imagem: Path) -> list[Predicao]:
        return [(d.confianca, d.bbox) for d in detectar_pessoas(carregar_imagem(imagem), self.cfg)]


class PreditorRoboflowNFL:
    pos_processamento = "modelo treinado em NFL (classe player)"

    def __init__(self, modelo_id: str, conf: float = 0.25):
        from inference_sdk import InferenceConfiguration, InferenceHTTPClient

        chave = chave_roboflow()
        self.cliente = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=chave)
        # o servidor aplica o mesmo limiar dos outros preditores (o padrão dele é outro)
        self.cliente.configure(InferenceConfiguration(confidence_threshold=conf))
        self.modelo_id = modelo_id
        self.conf = conf
        self.nome = f"roboflow {modelo_id} (NFL, player)"

    def prever(self, imagem: Path) -> list[Predicao]:
        resposta = self.cliente.infer(str(imagem), model_id=self.modelo_id)
        saida = []
        for p in resposta.get("predictions", []):
            if p.get("class") != "player" or p["confidence"] < self.conf:
                continue
            x, y, w, h = p["x"], p["y"], p["width"], p["height"]
            saida.append((float(p["confidence"]), (x - w / 2, y - h / 2, x + w / 2, y + h / 2)))
        return saida
