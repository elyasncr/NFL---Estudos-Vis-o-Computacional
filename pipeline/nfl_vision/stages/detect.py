"""Etapa 2: detecção de pessoas (RF-DETR ou YOLO) e descarte de quem está fora de campo."""

import hashlib
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from nfl_vision.config import Config
from nfl_vision.cores import mascara_gramado
from nfl_vision.schemas import Deteccao, DetectOut


LADO_MAX_CAMPO = 640
DETECTORES = ("rfdetr", "yolo")


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


def rotulo_pesos(pesos: str) -> str:
    """Nome curto dos pesos YOLO: `.../player-v1/weights/best.pt` vira `player-v1`."""
    caminho = Path(pesos)
    if caminho.parent.name == "weights" and caminho.stem in ("best", "last"):
        treino = caminho.parent.parent.name
        return treino if caminho.stem == "best" else f"{treino}/last"
    return caminho.stem


def rotulo_detector(cfg: Config) -> str:
    """Nome curto do detector da config: `rfdetr-base@896` ou o rótulo dos pesos YOLO."""
    if cfg.detector_tipo == "rfdetr":
        return f"rfdetr-{cfg.detector_modelo_rfdetr}@{cfg.detector_resolucao}"
    return rotulo_pesos(cfg.detector_pesos)


@lru_cache(maxsize=2)
def _modelo(pesos: str):
    from ultralytics import YOLO

    return YOLO(pesos)


@lru_cache(maxsize=2)
def _modelo_rfdetr(variante: str, resolucao: int, device: str):
    import rfdetr

    classe = getattr(rfdetr, f"RFDETR{variante.capitalize()}", None)
    if classe is None:
        raise ValueError(f"variante do RF-DETR desconhecida: '{variante}'")
    return classe(resolution=resolucao, device=device)


def resolver_device(device: str) -> str:
    if device.startswith("cuda"):
        import torch

        return device if torch.cuda.is_available() else "cpu"
    return device


def _modelo_rfdetr_da_config(cfg: Config):
    return _modelo_rfdetr(cfg.detector_modelo_rfdetr, cfg.detector_resolucao,
                          resolver_device(cfg.device))


def _classes_coco() -> dict[int, str]:
    try:
        from rfdetr.assets.coco_classes import COCO_CLASSES
    except ImportError:  # rfdetr < 1.9
        from rfdetr.util.coco_classes import COCO_CLASSES
    return COCO_CLASSES


def _eh_pessoa(dets) -> list[bool]:
    nomes = (dets.data or {}).get("class_name")
    if nomes is not None:
        return [n == "person" for n in nomes]
    ids_pessoa = {i for i, nome in _classes_coco().items() if nome == "person"}
    return [int(k) in ids_pessoa for k in dets.class_id]


def _detectar_rfdetr(img: np.ndarray, cfg: Config) -> list[Deteccao]:
    # o pipeline trabalha em BGR (OpenCV); o RF-DETR espera RGB
    rgb = Image.fromarray(np.ascontiguousarray(img[:, :, ::-1]))
    dets = _modelo_rfdetr_da_config(cfg).predict(rgb, threshold=cfg.detector_conf)
    pessoas = sorted(
        ((float(conf), tuple(float(v) for v in caixa))
         for caixa, conf, pessoa in zip(dets.xyxy, dets.confidence, _eh_pessoa(dets)) if pessoa),
        key=lambda p: -p[0],  # como o YOLO: det_id 0 é a mais confiante
    )
    return [Deteccao(det_id=i, bbox=caixa, confianca=conf)
            for i, (conf, caixa) in enumerate(pessoas)]


def _detectar_yolo(img: np.ndarray, cfg: Config) -> list[Deteccao]:
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


def detectar_pessoas(img: np.ndarray, cfg: Config) -> list[Deteccao]:
    """Pessoas detectadas, sem filtros; o detector vem de `cfg.detector_tipo`."""
    if cfg.detector_tipo == "rfdetr":
        return _detectar_rfdetr(img, cfg)
    return _detectar_yolo(img, cfg)


def caminho_pesos(cfg: Config) -> str | None:
    """Arquivo de pesos que o detector realmente carregou (pode ter sido baixado para outro lugar).

    YOLO: `ckpt_path` do ultralytics. RF-DETR: `model_config.pretrain_weights`
    (ex.: `~/.roboflow/models/rf-detr-base.pth`); None se o pacote não expuser o caminho.
    """
    if cfg.detector_tipo == "rfdetr":
        config_modelo = getattr(_modelo_rfdetr_da_config(cfg), "model_config", None)
        caminho = getattr(config_modelo, "pretrain_weights", None)
        return str(caminho) if caminho else None
    return getattr(_modelo(cfg.detector_pesos), "ckpt_path", None) or cfg.detector_pesos


def sha256_pesos(pesos: str | None) -> str | None:
    if not pesos:
        return None
    caminho = Path(pesos)
    return hashlib.sha256(caminho.read_bytes()).hexdigest() if caminho.exists() else None


def executar(estado) -> DetectOut:
    img = estado.imagem()
    cfg = estado.config
    return DetectOut(
        deteccoes=aplicar_filtros(detectar_pessoas(img, cfg), img, cfg),
        pesos_sha256=sha256_pesos(caminho_pesos(cfg)),
        detector=rotulo_detector(cfg),
    )
