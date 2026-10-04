"""Etapa 2: detecção de pessoas (YOLO) e descarte de quem está fora de campo."""

import hashlib
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from nfl_vision.config import Config
from nfl_vision.cores import mascara_gramado
from nfl_vision.schemas import Deteccao, DetectOut


LADO_MAX_CAMPO = 640


def regiao_do_campo(img: np.ndarray, cfg: Config) -> np.ndarray | None:
    """Polígono (envoltória convexa) do campo na imagem inteira, ou None se não há campo.

    Máscara de gramado numa cópia reduzida; fechamento morfológico une as faixas de
    grama separadas por linhas de jarda e abertura remove ruído. Ficam os componentes
    com área >= `campo_area_min` da imagem; pintura, letras, linhas e sombras dentro
    do campo caem dentro da envoltória.
    """
    altura, largura = img.shape[:2]
    escala = min(1.0, LADO_MAX_CAMPO / max(altura, largura))
    reduzida = img if escala == 1.0 else cv2.resize(
        img, (max(1, round(largura * escala)), max(1, round(altura * escala))),
        interpolation=cv2.INTER_AREA)
    mascara = mascara_gramado(reduzida, cfg.gramado_hsv_min, cfg.gramado_hsv_max)
    mascara = mascara.astype(np.uint8)
    lado = max(3, round(0.02 * max(reduzida.shape[:2])) | 1)
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lado, lado))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, nucleo)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, nucleo)
    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    area_min = cfg.campo_area_min * mascara.size
    grandes = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= area_min]
    if not grandes:
        return None
    ys, xs = np.nonzero(np.isin(rotulos, grandes))
    pontos = np.stack([xs, ys], axis=1).astype(np.float32)
    # centros dos pixels reduzidos -> coordenadas da imagem original
    pontos = (pontos + 0.5) / escala - 0.5
    return cv2.convexHull(pontos)


def toca_borda_inferior(bbox, altura_img: int) -> bool:
    return bbox[3] >= altura_img - 2


def _fora_de_campo(poligono: np.ndarray | None, bbox, shape, cfg: Config) -> bool:
    """Pés (centro da base da caixa) fora do campo por mais que a margem.

    Sem campo detectado (close) ninguém é descartado; jogador cortado pela borda
    inferior é mantido.
    """
    if poligono is None or toca_borda_inferior(bbox, shape[0]):
        return False
    pes = (float((bbox[0] + bbox[2]) / 2), float(bbox[3]))
    margem = cfg.campo_margem_rel * float(np.hypot(shape[0], shape[1]))
    return cv2.pointPolygonTest(poligono, pes, True) < -margem


def aplicar_filtros(deteccoes: list[Deteccao], img: np.ndarray, cfg: Config) -> list[Deteccao]:
    if not deteccoes:
        return []
    alturas = [d.bbox[3] - d.bbox[1] for d in deteccoes]
    mediana = float(np.median(alturas))
    # close (jogadores ocupando boa parte da altura): não há arquibancada na mesma
    # escala e o gramado visível é só retalho entre pernas; o filtro de campo é pulado
    close = mediana > cfg.campo_close_altura_rel * img.shape[0]
    poligono = None if close else regiao_do_campo(img, cfg)
    saida = []
    for d, h in zip(deteccoes, alturas):
        motivo = None
        if h < cfg.filtro_altura_rel * mediana:
            motivo = "pequeno"
        elif _fora_de_campo(poligono, d.bbox, img.shape, cfg):
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
