"""Avaliação do OCR em recortes de números rotulados por pasta."""

from pathlib import Path

from nfl_vision.stages.ingest import carregar_imagem
from nfl_vision.stages.jersey import PADRAO, LeitorOCR, ampliar, escolher_numero


def avaliar(leitor: LeitorOCR, amostras: list[tuple[Path, str]], limiar: float,
            altura_min: int = 128) -> dict:
    # só rótulos que são números válidos da NFL ("-1", "07", "00" ficam de fora)
    legiveis = [(p, int(r)) for p, r in amostras if PADRAO.match(r)]
    acertos = nulos = 0
    for caminho, esperado in legiveis:
        img = ampliar(carregar_imagem(caminho), altura_min)
        numero, _, _ = escolher_numero(leitor.ler(img), limiar)
        if numero is None:
            nulos += 1
        elif numero == esperado:
            acertos += 1
    n = len(legiveis)
    lidos = n - nulos
    return {
        "amostras": n,
        "excluidas": len(amostras) - n,
        "taxa_null": nulos / n if n else None,
        "acuracia_entre_lidos": acertos / lidos if lidos else None,
        "acuracia_geral": acertos / n if n else None,
    }
