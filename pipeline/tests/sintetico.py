"""Imagens sintéticas: campo verde, arquibancada cinza, jogadores e árbitro."""

import numpy as np

VERDE = (40, 140, 40)
CINZA = (128, 128, 128)
BRANCO = (255, 255, 255)
VERMELHO_KC = (55, 24, 227)   # #E31837 em BGR
AZUL_BUF = (141, 51, 0)       # #00338D em BGR


def campo(largura=800, altura=600):
    img = np.full((altura, largura, 3), VERDE, np.uint8)
    img[:150] = CINZA  # arquibancada
    return img


def jogador(img, x, y, cor, w=60, h=120):
    img[y:y + h, x:x + w] = cor
    return (float(x), float(y), float(x + w), float(y + h))


def arbitro(img, x, y, w=60, h=120):
    for i in range(w):
        img[y:y + h, x + i] = (0, 0, 0) if (i // 4) % 2 == 0 else (255, 255, 255)
    return (float(x), float(y), float(x + w), float(y + h))
VERDE_GB = (49, 55, 32)       # #203731 em BGR
