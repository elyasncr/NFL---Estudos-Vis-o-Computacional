"""Preditores comparados na avaliação: o nosso, YOLO bruto, RF-DETR local e o modelo NFL do Roboflow."""

import os
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageOps

from nfl_vision.config import Config
from nfl_vision.schemas import BBox
from nfl_vision.stages.detect import aplicar_filtros, detectar_pessoas
from nfl_vision.stages.ingest import carregar_imagem
from nfl_vision.stages.team import MIN_PIXELS, eh_arbitro, pixels_uteis, recorte_tronco

Predicao = tuple[float, BBox]


class Preditor(Protocol):
    nome: str
    pos_processamento: str

    def prever(self, imagem: Path) -> list[Predicao]: ...


class PreditorNosso:
    nome = "nosso (yolo11m + filtros + árbitro)"
    pos_processamento = "filtros de campo + remoção de árbitro"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def prever(self, imagem: Path) -> list[Predicao]:
        img = carregar_imagem(imagem)
        saida = []
        for d in aplicar_filtros(detectar_pessoas(img, self.cfg), img, self.cfg):
            if d.descartado:
                continue
            # mesma ordem de team.classificar: sem pixels suficientes é jogador sem time,
            # não árbitro
            recorte = recorte_tronco(img, d.bbox)
            if len(pixels_uteis(recorte, self.cfg)) >= MIN_PIXELS and eh_arbitro(recorte):
                continue
            saida.append((d.confianca, d.bbox))
        return saida


class PreditorYoloBruto:
    """O mesmo detector do pipeline, sem filtros de campo nem remoção de árbitro."""

    nome = "yolo11m bruto (COCO, pessoa)"
    pos_processamento = "nenhum"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def prever(self, imagem: Path) -> list[Predicao]:
        return [(d.confianca, d.bbox) for d in detectar_pessoas(carregar_imagem(imagem), self.cfg)]


def _classes_coco() -> dict[int, str]:
    try:
        from rfdetr.assets.coco_classes import COCO_CLASSES
    except ImportError:  # rfdetr < 1.9
        from rfdetr.util.coco_classes import COCO_CLASSES
    return COCO_CLASSES


class PreditorRFDETR:
    nome = "rf-detr base (COCO, pessoa)"
    pos_processamento = "nenhum"

    def __init__(self, conf: float = 0.25):
        from rfdetr import RFDETRBase

        self.modelo = RFDETRBase()
        self.conf = conf
        self.ids_pessoa = {i for i, nome in _classes_coco().items() if nome == "person"}

    def _eh_pessoa(self, dets) -> list[bool]:
        nomes = dets.data.get("class_name")
        if nomes is not None:
            return [n == "person" for n in nomes]
        return [int(k) in self.ids_pessoa for k in dets.class_id]

    def prever(self, imagem: Path) -> list[Predicao]:
        with Image.open(imagem) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB")  # como carregar_yolo e o pipeline
        dets = self.modelo.predict(rgb, threshold=self.conf)
        return [
            (float(c), tuple(float(v) for v in caixa))
            for caixa, c, pessoa in zip(dets.xyxy, dets.confidence, self._eh_pessoa(dets))
            if pessoa
        ]


class PreditorRoboflowNFL:
    pos_processamento = "modelo treinado em NFL (classe player)"

    def __init__(self, modelo_id: str, conf: float = 0.25):
        from inference_sdk import InferenceConfiguration, InferenceHTTPClient

        chave = os.environ.get("ROBOFLOW_API_KEY")
        if not chave:
            raise RuntimeError("defina ROBOFLOW_API_KEY no .env (app.roboflow.com/settings/api)")
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
