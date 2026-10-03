"""Download de datasets do Roboflow e leitura nos formatos YOLO e por pastas."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image

from nfl_vision.schemas import BBox

IMAGENS = {".jpg", ".jpeg", ".png"}


@dataclass
class AmostraDeteccao:
    imagem: Path
    caixas: dict[str, list[BBox]]  # classe -> caixas em pixels


def baixar(workspace: str, projeto: str, versao: int, formato: str, destino: Path) -> Path:
    chave = os.environ.get("ROBOFLOW_API_KEY")
    if not chave:
        raise RuntimeError("defina ROBOFLOW_API_KEY no .env (app.roboflow.com/settings/api)")
    alvo = destino / f"{projeto}-v{versao}-{formato}"
    if alvo.exists():
        return alvo
    from roboflow import Roboflow

    Roboflow(api_key=chave).workspace(workspace).project(projeto).version(versao).download(
        formato, location=str(alvo)
    )
    return alvo


def carregar_yolo(raiz: Path, split: str = "test") -> list[AmostraDeteccao]:
    nomes = yaml.safe_load((raiz / "data.yaml").read_text("utf-8"))["names"]
    if isinstance(nomes, dict):
        nomes = [nomes[k] for k in sorted(nomes)]
    amostras = []
    for caminho in sorted((raiz / split / "images").iterdir()):
        if caminho.suffix.lower() not in IMAGENS:
            continue
        with Image.open(caminho) as im:
            w, h = im.size
        caixas: dict[str, list[BBox]] = {}
        rotulos = raiz / split / "labels" / f"{caminho.stem}.txt"
        if rotulos.exists():
            for linha in rotulos.read_text().splitlines():
                partes = linha.split()
                if len(partes) != 5:  # polígonos e linhas vazias ficam de fora
                    continue
                classe = nomes[int(partes[0])]
                cx, cy, bw, bh = map(float, partes[1:])
                caixas.setdefault(classe, []).append(
                    ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
                )
        amostras.append(AmostraDeteccao(caminho, caixas))
    return amostras


def carregar_pastas(raiz: Path, split: str = "test") -> list[tuple[Path, str]]:
    """Formato de classificação: <split>/<rótulo>/<imagem>."""
    return [(p, p.parent.name) for p in sorted((raiz / split).glob("*/*"))
            if p.suffix.lower() in IMAGENS]
