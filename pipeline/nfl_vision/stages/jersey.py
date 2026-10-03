"""Etapa 4: leitura do número da camisa com OCR."""

import os
import re
from functools import lru_cache
from typing import Protocol

import cv2
import numpy as np

from nfl_vision.config import Config
from nfl_vision.geometria import caixa_inteira
from nfl_vision.schemas import Deteccao, JerseyOut, NumeroDet, TimeDet

PADRAO = re.compile(r"^\d{1,2}$")

# Evita um bug do PaddlePaddle 3.3.1 em CPU no Windows: o run_mode "mkldnn"
# (padrão) usa o executor PIR com oneDNN e quebra na detecção de texto com
# `NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
# ArrayAttribute<DoubleAttribute>`. Precisa ser definido antes do primeiro
# `import paddlex` (direto ou via paddleocr), que lê a env var no import.
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")


class LeitorOCR(Protocol):
    def ler(self, img: np.ndarray) -> list[tuple[str, float]]: ...


def ampliar(img: np.ndarray, altura_min: int) -> np.ndarray:
    if img.size == 0 or img.shape[0] >= altura_min:
        return img
    fator = altura_min / img.shape[0]
    return cv2.resize(img, None, fx=fator, fy=fator, interpolation=cv2.INTER_CUBIC)


def recorte_numero(img: np.ndarray, bbox, altura_min: int) -> np.ndarray:
    """Região do número: 15–60% da altura e 10–90% da largura da caixa."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    xa, ya, xb, yb = caixa_inteira(
        (x1 + 0.10 * w, y1 + 0.15 * h, x1 + 0.90 * w, y1 + 0.60 * h), img.shape)
    return ampliar(img[ya:yb, xa:xb], altura_min)


def escolher_numero(leituras: list[tuple[str, float]], limiar: float) -> tuple[int | None, float, str | None]:
    validas = [(t.strip(), s) for t, s in leituras if PADRAO.match(t.strip())]
    if not validas:
        bruto = max(leituras, key=lambda x: x[1])[0] if leituras else None
        return None, 0.0, bruto
    texto, score = max(validas, key=lambda x: x[1])
    if score < limiar:
        return None, float(score), texto
    return int(texto), float(score), texto


class PaddleLeitor:
    def __init__(self, device: str):
        # No Windows, importar paddleocr (→ modelscope → torch) antes do torch dá WinError 127 (shm.dll).
        import torch  # noqa: F401
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=device,
        )

    def ler(self, img: np.ndarray) -> list[tuple[str, float]]:
        leituras = []
        for r in self._ocr.predict(img):
            leituras.extend(zip(r["rec_texts"], (float(s) for s in r["rec_scores"])))
        return leituras


@lru_cache(maxsize=2)
def leitor_padrao(device: str) -> LeitorOCR:
    return PaddleLeitor(device)


def ler_numeros(img: np.ndarray, deteccoes: list[Deteccao], times: list[TimeDet],
                leitor: LeitorOCR, cfg: Config) -> list[NumeroDet]:
    arbitros = {t.det_id for t in times if t.arbitro}
    itens = []
    for d in deteccoes:
        if d.descartado or d.det_id in arbitros:
            continue
        recorte = recorte_numero(img, d.bbox, cfg.numero_altura_min)
        leituras = leitor.ler(recorte) if recorte.size else []
        numero, conf, bruto = escolher_numero(leituras, cfg.limiar_numero)
        itens.append(NumeroDet(det_id=d.det_id, numero=numero, confianca=round(conf, 4),
                               texto_bruto=bruto))
    return itens


def executar(estado) -> JerseyOut:
    leitor = leitor_padrao(estado.config.ocr_device)
    return JerseyOut(itens=ler_numeros(
        estado.imagem(), estado.saidas["detect"].deteccoes,
        estado.saidas["team"].itens, leitor, estado.config,
    ))
