"""Avaliação de detecção: mAP@0.5 de jogador e árbitros mantidos como jogador."""

from nfl_vision.eval.datasets import AmostraDeteccao
from nfl_vision.eval.metricas import average_precision, fracao_casada
from nfl_vision.eval.preditores import Preditor


def avaliar(preditor: Preditor, amostras: list[AmostraDeteccao],
            classe_alvo: str = "player", classe_arbitro: str = "referee") -> dict:
    predicoes, gts, arbitros, mantidas = [], {}, {}, {}
    for i, amostra in enumerate(amostras):
        gts[i] = amostra.caixas.get(classe_alvo, [])
        arbitros[i] = amostra.caixas.get(classe_arbitro, [])
        saida = preditor.prever(amostra.imagem)
        mantidas[i] = [caixa for _, caixa in saida]
        predicoes.extend((i, conf, caixa) for conf, caixa in saida)
    tem_arbitros = any(arbitros.values())
    return {
        "preditor": preditor.nome,
        "pos_processamento": preditor.pos_processamento,
        "imagens": len(amostras),
        "map50": round(average_precision(predicoes, gts), 4),
        "arbitros_como_jogador": round(fracao_casada(arbitros, mantidas), 4) if tem_arbitros else None,
    }
