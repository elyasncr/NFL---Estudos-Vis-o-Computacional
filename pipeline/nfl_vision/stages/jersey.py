"""Etapa 4: leitura do número da camisa com OCR."""

import os
import re
from functools import lru_cache
from itertools import combinations
from typing import NamedTuple, Protocol

import cv2
import numpy as np

from nfl_vision.config import Config
from nfl_vision.geometria import caixa_inteira
from nfl_vision.schemas import Deteccao, JerseyOut, NumeroDet, TimeDet

# 0–99 sem zero à esquerda: "07" e "00" não são números da NFL ("0" é).
PADRAO = re.compile(r"^(?:\d|[1-9]\d)$")
# Um número de 2 dígitos vence um de 1 dígito contido nele se a confiança
# não for mais que isto abaixo (o OCR costuma ler só metade do número).
MARGEM_NUMERO_COMPLETO = 0.15

Caixa = tuple[float, float, float, float]


class Leitura(NamedTuple):
    texto: str
    score: float
    caixa: Caixa | None = None  # xyxy no recorte

# Evita um bug do PaddlePaddle 3.3.1 em CPU no Windows: o run_mode "mkldnn"
# (padrão) usa o executor PIR com oneDNN e quebra na detecção de texto com
# `NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
# ArrayAttribute<DoubleAttribute>`. Precisa ser definido antes do primeiro
# `import paddlex` (direto ou via paddleocr), que lê a env var no import.
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")


class LeitorOCR(Protocol):
    def ler(self, img: np.ndarray) -> list[Leitura]: ...


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


def normalizar_texto(texto: str) -> str:
    return re.sub(r"\s+", "", texto).strip(".#-")


def _vizinhos(a: Caixa, b: Caixa) -> bool:
    """`a` à esquerda de `b`, na mesma linha, alturas parecidas e separados por pouco."""
    ha, hb = a[3] - a[1], b[3] - b[1]
    h = max(ha, hb)
    if h <= 0 or abs(ha - hb) > 0.4 * h:
        return False
    if abs((a[1] + a[3]) / 2 - (b[1] + b[3]) / 2) >= 0.5 * h:
        return False
    return b[0] - a[2] < 1.0 * h


def juntar_digitos(leituras: list[Leitura]) -> list[Leitura]:
    """Acrescenta às leituras os pares de dígitos soltos vizinhos (ex.: "8" + "7" → "87")."""
    digitos = sorted(
        (l for l in leituras if l.caixa is not None and len(l.texto) == 1 and l.texto.isdigit()),
        key=lambda l: l.caixa[0],
    )
    juntas = []
    for a, b in combinations(digitos, 2):
        if b.caixa[0] > a.caixa[0] and _vizinhos(a.caixa, b.caixa):
            caixa = (min(a.caixa[0], b.caixa[0]), min(a.caixa[1], b.caixa[1]),
                     max(a.caixa[2], b.caixa[2]), max(a.caixa[3], b.caixa[3]))
            juntas.append(Leitura(a.texto + b.texto, min(a.score, b.score), caixa))
    return list(leituras) + juntas


def escolher_numero(leituras: list[Leitura], limiar: float) -> tuple[int | None, float, str | None]:
    normalizadas = [l._replace(texto=normalizar_texto(l.texto)) for l in leituras]
    validas = [l for l in juntar_digitos(normalizadas) if PADRAO.match(l.texto)]
    if not validas:
        bruto = max(leituras, key=lambda l: l.score).texto if leituras else None
        return None, 0.0, bruto
    melhor = max(validas, key=lambda l: l.score)
    if len(melhor.texto) == 1:
        completas = [l for l in validas if len(l.texto) == 2 and melhor.texto in l.texto
                     and l.score >= melhor.score - MARGEM_NUMERO_COMPLETO]
        if completas:
            melhor = max(completas, key=lambda l: l.score)
    texto, score = melhor.texto, float(melhor.score)
    if score < limiar:
        return None, score, texto
    return int(texto), score, texto


def _caixas(r, n: int) -> list[Caixa | None]:
    """Caixas xyxy das linhas reconhecidas (PaddleOCR 3.x: `rec_boxes` ou `rec_polys`)."""
    boxes = r.get("rec_boxes")
    if boxes is not None and len(boxes) == n:
        return [tuple(float(v) for v in b[:4]) for b in boxes]
    polys = r.get("rec_polys")
    if polys is not None and len(polys) == n:
        caixas = []
        for p in polys:
            pts = np.asarray(p, dtype=float).reshape(-1, 2)
            caixas.append((pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()))
        return [tuple(float(v) for v in c) for c in caixas]
    return [None] * n


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

    def ler(self, img: np.ndarray) -> list[Leitura]:
        leituras = []
        for r in self._ocr.predict(img):
            textos, scores = r["rec_texts"], r["rec_scores"]
            for t, s, c in zip(textos, scores, _caixas(r, len(textos))):
                leituras.append(Leitura(t, float(s), c))
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
