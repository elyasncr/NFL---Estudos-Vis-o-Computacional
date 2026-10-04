"""Desenha caixas e rótulos sobre a foto."""

import cv2
import numpy as np

from nfl_vision.cores import delta_e, hex_para_bgr, hex_para_lab
from nfl_vision.schemas import BBox, Jogador

COR_DESCONHECIDO = (128, 116, 107)  # #6B7480 em BGR (confidence/unknown): jogador sem time
COR_RESERVA = (255, 255, 255)       # branco: time cujas cores somem no gramado ou repetem o rival
COR_CONTORNO = (20, 20, 20)
FONTE = cv2.FONT_HERSHEY_SIMPLEX

GRAMADO = hex_para_lab("#2E6B3F")    # field/turf do design system
CONTRASTE_GRAMADO_MIN = 30.0         # ΔE mínimo entre a cor da caixa e o gramado
LUMINANCIA_MIN = 60.0                # cores quase pretas somem sobre o contorno escuro
DIFERENCA_TIMES_MIN = 25.0           # ΔE mínimo entre as cores dos dois times


def _luminancia(cor_hex: str) -> float:
    b, g, r = hex_para_bgr(cor_hex)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _visivel(cor_hex: str) -> bool:
    return (delta_e(hex_para_lab(cor_hex), GRAMADO) >= CONTRASTE_GRAMADO_MIN
            and _luminancia(cor_hex) >= LUMINANCIA_MIN)


def cores_de_exibicao(paletas: dict[str, list[str]]) -> dict[str, tuple[int, int, int]]:
    """Cor da caixa de cada time (BGR), na ordem dos times.

    Usa a primeira cor oficial visível sobre o gramado e diferente das já escolhidas;
    se nenhuma servir, o time fica com a cor reserva (branco).
    """
    escolhidas: dict[str, tuple[int, int, int]] = {}
    labs: list[np.ndarray] = []
    for time, cores in paletas.items():
        escolha = None
        for c in cores:
            lab = hex_para_lab(c)
            if _visivel(c) and all(delta_e(lab, outro) >= DIFERENCA_TIMES_MIN for outro in labs):
                escolha = c
                break
        if escolha is None:
            escolhidas[time] = COR_RESERVA
            labs.append(hex_para_lab("#FFFFFF"))
        else:
            escolhidas[time] = hex_para_bgr(escolha)
            labs.append(hex_para_lab(escolha))
    return escolhidas


def rotulo(j: Jogador) -> str:
    partes = [j.time or "?", "?" if j.numero is None else str(j.numero)]
    if j.posicao:
        partes.append(j.posicao)
    return " ".join(partes)


def cor_do_texto(cor: tuple[int, int, int]) -> tuple[int, int, int]:
    """Preto sobre cores claras, branco sobre escuras (cor em BGR)."""
    b, g, r = cor
    return (0, 0, 0) if 0.299 * r + 0.587 * g + 0.114 * b > 150 else (255, 255, 255)


def desenhar(img: np.ndarray, jogadores: list[Jogador], caixas: dict[int, BBox],
             cores_times: dict[str, tuple[int, int, int]]) -> np.ndarray:
    saida = img.copy()
    lado = max(saida.shape[:2])
    espessura = max(2, round(lado / 640))
    escala = max(0.5, lado / 1600 * 0.55)
    traco_texto = max(1, round(escala * 1.6))
    for j in jogadores:
        x1, y1, x2, y2 = (int(round(v)) for v in caixas[j.track_id])
        # o "?" do rótulo já indica número desconhecido; a cor mostra o time
        cor = cores_times.get(j.time, COR_DESCONHECIDO) if j.time else COR_DESCONHECIDO
        cv2.rectangle(saida, (x1, y1), (x2, y2), COR_CONTORNO, espessura + 2)
        cv2.rectangle(saida, (x1, y1), (x2, y2), cor, espessura)
        texto = rotulo(j)
        (tw, th), _ = cv2.getTextSize(texto, FONTE, escala, traco_texto)
        pad = max(2, espessura)
        topo = max(y1 - th - 2 * pad, 0)
        lx = max(0, min(x1, saida.shape[1] - tw - 2 * pad))  # não corta na borda direita
        cv2.rectangle(saida, (lx, topo), (lx + tw + 2 * pad, topo + th + 2 * pad), cor, -1)
        cv2.putText(saida, texto, (lx + pad, topo + th + pad), FONTE, escala, cor_do_texto(cor),
                    traco_texto, cv2.LINE_AA)
    return saida
