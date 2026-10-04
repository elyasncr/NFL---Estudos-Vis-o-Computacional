"""Montagem do dataset de treino do detector: conversão para `player`, splits sem vazamento e manifest."""

import hashlib
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from nfl_vision.eval.datasets import IMAGENS, ler_rotulos, nomes_das_classes
from nfl_vision.pipeline import _versao
from nfl_vision.runner import gravar_json, gravar_texto
from nfl_vision.treino.fontes import (
    CLIPE_VALID, PREFIXO_TESTE, Fonte, split_base, validar_mapeamento,
)

SPLITS = ("train", "valid", "test")
FRACAO_VALID_EXTERNOS = 0.15
PASTA_PADRAO = "treino-player-v1"
MOTIVO_BASE = "base: jogo de teste (cin_cle) e treino"


@dataclass(frozen=True)
class Decisao:
    fonte: Fonte
    aprovada: bool
    motivo: str
    raiz: Path | None = None  # pasta baixada; None para fonte rejeitada


def imagens_da_fonte(raiz: Path) -> list[tuple[Path, Path]]:
    """(imagem, rótulo) de todos os splits do export (train, valid, test...), em ordem estável."""
    pares = []
    for pasta in sorted(p for p in raiz.glob("*/images") if p.is_dir()):
        for imagem in sorted(pasta.iterdir()):
            if imagem.suffix.lower() in IMAGENS:
                pares.append((imagem, pasta.parent / "labels" / f"{imagem.stem}.txt"))
    return pares


def converter(rotulos: Path, nomes: list, fonte: Fonte) -> tuple[list[str], Counter]:
    """Linhas YOLO `0 cx cy w h` das caixas de player e a contagem das descartadas por classe.

    Polígonos viram a caixa que os envolve; caixas são cortadas aos limites da imagem.
    """
    linhas: list[str] = []
    descartadas: Counter = Counter()
    if not rotulos.exists():
        return linhas, descartadas
    for classe, caixa in ler_rotulos(rotulos, nomes):
        if str(classe) not in fonte.classes_player:
            descartadas[str(classe)] += 1
            continue
        x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in caixa)
        if x2 <= x1 or y2 <= y1:
            descartadas["(caixa degenerada)"] += 1
            continue
        linhas.append(f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
    return linhas, descartadas


def grupo_externo(nome_arquivo: str) -> str:
    """Imagem original: o Roboflow exporta cópias aumentadas como `<original>.rf.<hash>.<ext>`."""
    return nome_arquivo.split(".rf.")[0]


def split_externo(fonte: str, nome_arquivo: str, semente: int,
                  fracao: float = FRACAO_VALID_EXTERNOS) -> str:
    """Sorteio reprodutível por imagem original (independe da ordem dos arquivos)."""
    chave = f"{semente}:{fonte}:{grupo_externo(nome_arquivo)}"
    sorteio = int(hashlib.sha256(chave.encode()).hexdigest()[:12], 16) / 16**12
    return "valid" if sorteio < fracao else "train"


def _descrever(d: Decisao) -> dict:
    f = d.fonte
    return {
        "nome": f.nome, "workspace": f.workspace, "projeto": f.projeto, "versao": f.versao,
        "aprovada": d.aprovada, "motivo": d.motivo,
        "mapeamento": {c: "player" for c in f.classes_player},
        "pasta": str(d.raiz) if d.raiz is not None else None,
    }


def construir(saida: Path, decisoes: list[Decisao], semente: int = 0) -> dict:
    """Monta o dataset YOLO de uma classe em `saida`, grava data.yaml e manifest.json e devolve o manifest."""
    if not any(not d.fonte.externa and d.aprovada for d in decisoes):
        raise ValueError("a fonte base é obrigatória: é dela que vem o split de teste")
    existentes = [n for n in (*SPLITS, "data.yaml") if (saida / n).exists()]
    if existentes:
        raise FileExistsError(
            f"{saida} já tem um dataset ({', '.join(existentes)}); "
            "apague essas pastas ou use outra --saida")
    aprovadas = [d for d in decisoes if d.aprovada]
    nomes_por_fonte = {}
    for d in aprovadas:  # valida tudo antes de gravar qualquer arquivo
        if d.raiz is None:
            raise ValueError(f"fonte aprovada '{d.fonte.nome}' sem pasta baixada")
        nomes = nomes_das_classes(d.raiz)
        validar_mapeamento(d.fonte, nomes)
        nomes_por_fonte[d.fonte.nome] = nomes

    for split in SPLITS:
        for sub in ("images", "labels"):
            (saida / split / sub).mkdir(parents=True, exist_ok=True)
    contagens: dict[str, dict] = {split: {} for split in SPLITS}
    descartadas: dict[str, dict] = {}
    sem_player: dict[str, int] = {}
    for d in aprovadas:
        fonte, nomes = d.fonte, nomes_por_fonte[d.fonte.nome]
        descartes: Counter = Counter()
        sem = 0
        for imagem, rotulos in imagens_da_fonte(d.raiz):
            linhas, desc = converter(rotulos, nomes, fonte)
            descartes.update(desc)
            if not linhas:
                sem += 1
                continue
            split = (split_externo(fonte.nome, imagem.name, semente) if fonte.externa
                     else split_base(imagem.name))
            nome = f"{fonte.nome}__{imagem.name}"
            if any((saida / s / "images" / nome).exists() for s in SPLITS):
                raise ValueError(f"fonte '{fonte.nome}': imagem repetida no export: {imagem.name}")
            alvo = saida / split / "images" / nome
            shutil.copy2(imagem, alvo)
            gravar_texto(saida / split / "labels" / f"{alvo.stem}.txt", "\n".join(linhas) + "\n")
            c = contagens[split].setdefault(fonte.nome, {"imagens": 0, "caixas": 0})
            c["imagens"] += 1
            c["caixas"] += len(linhas)
        descartadas[fonte.nome] = dict(sorted(descartes.items()))
        sem_player[fonte.nome] = sem

    gravar_texto(saida / "data.yaml", yaml.safe_dump({
        # path absoluto: o Ultralytics não depende do diretório atual nem do seu datasets_dir
        "path": str(saida.resolve()), "train": "train/images", "val": "valid/images",
        "test": "test/images", "nc": 1, "names": ["player"],
    }, sort_keys=False, allow_unicode=True))
    manifest = {
        "versao": _versao("nfl-vision"),
        "criado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "semente": semente,
        "fracao_valid_externos": FRACAO_VALID_EXTERNOS,
        "regras_de_split": {
            "test": f"clipes {PREFIXO_TESTE}* da base",
            "valid": (f"clipe {CLIPE_VALID} da base + {FRACAO_VALID_EXTERNOS:.0%} de cada externa "
                      "aprovada (sorteio por imagem original)"),
            "train": "o restante",
        },
        "fontes": [_descrever(d) for d in decisoes],
        "contagens": contagens,
        "descartadas": descartadas,
        "sem_player": sem_player,
    }
    gravar_json(saida / "manifest.json", manifest)
    return manifest
