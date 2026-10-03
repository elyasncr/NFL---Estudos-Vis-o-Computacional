"""Etapa 2: detecção de pessoas (YOLO) e descarte de quem está fora de campo."""

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

from nfl_vision.config import Config
from nfl_vision.cores import fracao_gramado
from nfl_vision.schemas import Deteccao, DetectOut


def faixa_dos_pes(img: np.ndarray, bbox) -> np.ndarray:
    """Faixa logo abaixo da caixa (10% da altura), recortada pelos limites da imagem."""
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, x2 = max(x1, 0), min(x2, img.shape[1])
    altura = max(1, int(round((y2 - y1) * 0.10)))
    return img[y2:min(y2 + altura, img.shape[0]), x1:x2]


def toca_borda_inferior(bbox, altura_img: int) -> bool:
    return bbox[3] >= altura_img - 2


def _fora_de_campo(img: np.ndarray, bbox, cfg: Config) -> bool:
    """Gramado insuficiente sob os pés; jogador cortado pela borda inferior é mantido."""
    if toca_borda_inferior(bbox, img.shape[0]):
        return False
    faixa = faixa_dos_pes(img, bbox)
    return fracao_gramado(faixa, cfg.gramado_hsv_min, cfg.gramado_hsv_max) < cfg.filtro_gramado_min


def aplicar_filtros(deteccoes: list[Deteccao], img: np.ndarray, cfg: Config) -> list[Deteccao]:
    if not deteccoes:
        return []
    alturas = [d.bbox[3] - d.bbox[1] for d in deteccoes]
    mediana = float(np.median(alturas))
    saida = []
    for d, h in zip(deteccoes, alturas):
        motivo = None
        if h < cfg.filtro_altura_rel * mediana:
            motivo = "pequeno"
        elif _fora_de_campo(img, d.bbox, cfg):
            motivo = "fora_de_campo"
        saida.append(d.model_copy(update={"descartado": motivo is not None,
                                          "motivo_descarte": motivo}))
    return saida


@lru_cache(maxsize=2)
def _modelo(pesos: str):
    from ultralytics import YOLO

    return YOLO(pesos)


def resolver_device(device: str) -> str:
    if device.startswith("cuda"):
        import torch

        return device if torch.cuda.is_available() else "cpu"
    return device


def detectar_pessoas(img: np.ndarray, cfg: Config) -> list[Deteccao]:
    resultado = _modelo(cfg.detector_pesos).predict(
        img, imgsz=cfg.detector_imgsz, conf=cfg.detector_conf, classes=[0],
        device=resolver_device(cfg.device), verbose=False,
    )[0]
    caixas = resultado.boxes.xyxy.cpu().numpy()
    confs = resultado.boxes.conf.cpu().numpy()
    return [
        Deteccao(det_id=i, bbox=tuple(float(v) for v in caixa), confianca=float(conf))
        for i, (caixa, conf) in enumerate(zip(caixas, confs))
    ]


def caminho_pesos(pesos: str) -> str:
    """Arquivo que o ultralytics realmente carregou (pode ter baixado para outro lugar)."""
    return getattr(_modelo(pesos), "ckpt_path", None) or pesos


def sha256_pesos(pesos: str) -> str | None:
    caminho = Path(pesos)
    return hashlib.sha256(caminho.read_bytes()).hexdigest() if caminho.exists() else None


def executar(estado) -> DetectOut:
    img = estado.imagem()
    cfg = estado.config
    return DetectOut(
        deteccoes=aplicar_filtros(detectar_pessoas(img, cfg), img, cfg),
        pesos_sha256=sha256_pesos(caminho_pesos(cfg.detector_pesos)),
    )
