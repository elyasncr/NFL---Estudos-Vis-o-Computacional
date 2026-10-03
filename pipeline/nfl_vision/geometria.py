"""Geometria de caixas: coordenadas float -> índices inteiros dentro da imagem."""


def caixa_inteira(bbox, shape) -> tuple[int, int, int, int]:
    """Arredonda (x1, y1, x2, y2) e limita a [0, largura] e [0, altura]."""
    altura, largura = shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    return (min(max(x1, 0), largura), min(max(y1, 0), altura),
            min(max(x2, 0), largura), min(max(y2, 0), altura))
