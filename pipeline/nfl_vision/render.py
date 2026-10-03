"""Desenha caixas e rótulos sobre a foto."""

import cv2
import numpy as np

from nfl_vision.schemas import BBox, Jogador

COR_DESCONHECIDO = (128, 116, 107)  # #6B7480 em BGR (confidence/unknown)
FONTE = cv2.FONT_HERSHEY_SIMPLEX


def rotulo(j: Jogador) -> str:
    partes = [j.time or "?", "?" if j.numero is None else str(j.numero)]
    if j.posicao:
        partes.append(j.posicao)
    return " ".join(partes)


def desenhar(img: np.ndarray, jogadores: list[Jogador], caixas: dict[int, BBox],
             cores_times: dict[str, tuple[int, int, int]]) -> np.ndarray:
    saida = img.copy()
    for j in jogadores:
        x1, y1, x2, y2 = (int(round(v)) for v in caixas[j.track_id])
        conhecido = j.time is not None and j.numero is not None
        cor = cores_times.get(j.time, COR_DESCONHECIDO) if conhecido else COR_DESCONHECIDO
        cv2.rectangle(saida, (x1, y1), (x2, y2), cor, 2)
        texto = rotulo(j)
        (tw, th), _ = cv2.getTextSize(texto, FONTE, 0.5, 1)
        topo = max(y1 - th - 6, 0)
        cv2.rectangle(saida, (x1, topo), (x1 + tw + 4, topo + th + 6), cor, -1)
        cv2.putText(saida, texto, (x1 + 2, topo + th + 2), FONTE, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return saida
