"""IoU, AP@0.5 (interpolação de todos os pontos) e fração de alvos casados."""

import numpy as np

from nfl_vision.schemas import BBox


def iou(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    uniao = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / uniao if uniao > 0 else 0.0


def average_precision(predicoes: list[tuple[int, float, BBox]], gts: dict[int, list[BBox]],
                      limiar_iou: float = 0.5) -> float:
    """predicoes: (id_imagem, confiança, caixa). gts: id_imagem -> caixas verdadeiras."""
    total = sum(len(v) for v in gts.values())
    if total == 0 or not predicoes:
        return 0.0
    usados = {k: [False] * len(v) for k, v in gts.items()}
    ordenadas = sorted(predicoes, key=lambda p: -p[1])
    tp = np.zeros(len(ordenadas))
    for i, (img_id, _, caixa) in enumerate(ordenadas):
        candidatos = gts.get(img_id, [])
        ious = [iou(caixa, g) for g in candidatos]
        j = int(np.argmax(ious)) if ious else -1
        if j >= 0 and ious[j] >= limiar_iou and not usados[img_id][j]:
            tp[i] = 1
            usados[img_id][j] = True
    tp_acum = np.cumsum(tp)
    recall = tp_acum / total
    precisao = tp_acum / np.arange(1, len(ordenadas) + 1)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precisao, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def fracao_casada(alvos: dict[int, list[BBox]], predicoes: dict[int, list[BBox]],
                  limiar_iou: float = 0.5) -> float:
    """Fração das caixas-alvo que têm alguma predição com IoU >= limiar."""
    total = casados = 0
    for img_id, caixas in alvos.items():
        for alvo in caixas:
            total += 1
            if any(iou(alvo, p) >= limiar_iou for p in predicoes.get(img_id, [])):
                casados += 1
    return casados / total if total else 0.0
