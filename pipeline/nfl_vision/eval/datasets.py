"""Download de datasets do Roboflow e leitura nos formatos YOLO e por pastas."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image, ImageOps

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


def splits_existentes(raiz: Path) -> list[str]:
    return sorted(p.name for p in raiz.iterdir() if p.is_dir()) if raiz.is_dir() else []


def nomes_das_classes(raiz: Path) -> list[str]:
    arquivo = raiz / "data.yaml"
    if not arquivo.exists():
        raise FileNotFoundError(f"data.yaml não encontrado em {raiz} (esperado um dataset em formato YOLO)")
    dados = yaml.safe_load(arquivo.read_text("utf-8")) or {}
    nomes = dados.get("names") if isinstance(dados, dict) else None
    if not nomes:
        raise ValueError(f"{arquivo} não define 'names' (os nomes das classes)")
    if isinstance(nomes, dict):
        nomes = [nomes[k] for k in sorted(nomes)]
    return list(nomes)


def _caixa_da_linha(partes: list[str], w: int, h: int) -> BBox | None:
    """Caixa em pixels de uma linha YOLO: `cx cy w h` ou polígono `x1 y1 x2 y2 ...`."""
    valores = list(map(float, partes[1:]))
    if len(valores) == 4:
        cx, cy, bw, bh = valores
        return ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
    if len(valores) >= 6 and len(valores) % 2 == 0:  # segmentação: caixa que envolve o polígono
        xs, ys = valores[0::2], valores[1::2]
        return (min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h)
    return None


def carregar_yolo(raiz: Path, split: str = "test") -> list[AmostraDeteccao]:
    nomes = nomes_das_classes(raiz)
    pasta_imagens = raiz / split / "images"
    if not pasta_imagens.is_dir():
        existentes = ", ".join(splits_existentes(raiz)) or "nenhuma"
        raise FileNotFoundError(
            f"pasta {pasta_imagens} não encontrada; pastas em {raiz}: {existentes} "
            "(exports do Roboflow usam 'valid', não 'val')")
    amostras = []
    for caminho in sorted(pasta_imagens.iterdir()):
        if caminho.suffix.lower() not in IMAGENS:
            continue
        with Image.open(caminho) as im:
            w, h = ImageOps.exif_transpose(im).size
        caixas: dict[str, list[BBox]] = {}
        rotulos = raiz / split / "labels" / f"{caminho.stem}.txt"
        if rotulos.exists():
            for n, linha in enumerate(rotulos.read_text().splitlines(), start=1):
                partes = linha.split()
                if not partes:
                    continue
                try:
                    caixa = _caixa_da_linha(partes, w, h)
                    indice = int(partes[0])
                except ValueError:
                    caixa = None
                if caixa is None:
                    raise ValueError(f"{rotulos}, linha {n}: linha YOLO inválida: {linha!r}")
                if not 0 <= indice < len(nomes):
                    raise ValueError(
                        f"{rotulos}, linha {n}: classe {indice} fora de names ({len(nomes)} classes)")
                caixas.setdefault(nomes[indice], []).append(caixa)
        amostras.append(AmostraDeteccao(caminho, caixas))
    return amostras


def carregar_pastas(raiz: Path, split: str = "test") -> list[tuple[Path, str]]:
    """Formato de classificação: <split>/<rótulo>/<imagem>."""
    return [(p, p.parent.name) for p in sorted((raiz / split).glob("*/*"))
            if p.suffix.lower() in IMAGENS]
