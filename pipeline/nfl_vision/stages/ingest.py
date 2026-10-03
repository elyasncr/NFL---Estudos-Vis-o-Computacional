"""Etapa 1: leitura da foto com orientação EXIF e hash."""

import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from nfl_vision.schemas import IngestOut

FORMATOS = {".jpg", ".jpeg", ".png"}


class FormatoNaoSuportado(ValueError):
    pass


def validar_formato(caminho: Path) -> None:
    if caminho.suffix.lower() not in FORMATOS:
        raise FormatoNaoSuportado(
            f"formato '{caminho.suffix}' não suportado; use JPG ou PNG (vídeo ainda não é aceito)"
        )


def carregar_imagem(caminho: Path) -> np.ndarray:
    validar_formato(caminho)
    with Image.open(caminho) as img:
        rgb = np.asarray(ImageOps.exif_transpose(img).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def executar(estado) -> IngestOut:
    caminho = estado.caminho_imagem
    img = estado.imagem()
    return IngestOut(
        caminho=str(caminho),
        largura=img.shape[1],
        altura=img.shape[0],
        sha256=hashlib.sha256(caminho.read_bytes()).hexdigest(),
    )
