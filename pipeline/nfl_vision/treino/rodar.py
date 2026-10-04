"""Ajuste fino do YOLO11m para `player` com o Ultralytics e o manifest do treino."""

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from nfl_vision.eval.datasets import IMAGENS, nomes_das_classes
from nfl_vision.pipeline import _versao
from nfl_vision.runner import gravar_json

MODELO_INICIAL = "yolo11m.pt"
PARAMETROS = {
    "imgsz": 1280,
    "epochs": 100,
    "patience": 20,
    "batch": 8,            # fixo: o AutoBatch (-1) mede errado no Windows e cai para 1
    "amp": True,
    "single_cls": True,
    "close_mosaic": 10,
    "workers": 2,          # estabilidade dos dataloaders no Windows
    "seed": 0,
    "deterministic": True,
}
SPLITS = ("train", "valid", "test")
_NOME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class EntradaInvalida(Exception):
    """Erro de entrada (dataset, nome, retomada): vira mensagem sem traceback na CLI."""


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _yolo(pesos: str):
    from ultralytics import YOLO

    return YOLO(pesos)


def _gpu() -> str | None:
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def _pasta(nome: str, destino: Path) -> Path:
    if not _NOME.fullmatch(nome):
        raise EntradaInvalida(
            f"nome de treino inválido: {nome!r} (use letras, números, '.', '_' ou '-')")
    return (destino / nome).resolve()


def validar_dataset(dataset: Path) -> Path:
    """data.yaml de um dataset de uma classe com imagens em train, valid e test."""
    try:
        nomes = nomes_das_classes(dataset)
    except (FileNotFoundError, ValueError) as exc:
        raise EntradaInvalida(str(exc)) from exc
    if [str(n) for n in nomes] != ["player"]:
        raise EntradaInvalida(
            f"o dataset deve ter só a classe player (names: {nomes}); "
            "gere-o com nfl-vision treino preparar")
    for split in SPLITS:
        pasta = dataset / split / "images"
        if not pasta.is_dir() or not any(p.suffix.lower() in IMAGENS for p in pasta.iterdir()):
            raise EntradaInvalida(f"split '{split}' sem imagens em {pasta}")
    return dataset / "data.yaml"


def _metricas(modelo) -> dict:
    brutas = getattr(getattr(modelo, "trainer", None), "metrics", None) or {}
    saida = {}
    for chave, valor in brutas.items():
        try:
            saida[chave] = round(float(valor), 5)
        except (TypeError, ValueError):
            continue
    return saida


def _executar(pasta: Path, manifest: dict, treino: Callable[[], object]) -> None:
    """Roda o treino registrando status, duração acumulada, best.pt e métricas no manifest."""
    arquivo = pasta / "manifest.json"
    manifest.update(status="em_andamento", mensagem=None)
    gravar_json(arquivo, manifest)
    t0 = time.perf_counter()

    def somar_duracao() -> None:
        manifest["duracao_s"] = round(manifest.get("duracao_s", 0.0) + time.perf_counter() - t0, 1)

    try:
        modelo = treino()
        best = pasta / "weights" / "best.pt"
        if not best.exists():
            raise RuntimeError(f"o Ultralytics terminou sem gerar {best}")
    except BaseException as exc:
        somar_duracao()
        manifest["status"] = "interrompido" if isinstance(exc, KeyboardInterrupt) else "erro"
        manifest["mensagem"] = f"{type(exc).__name__}: {exc}"
        gravar_json(arquivo, manifest)
        raise
    somar_duracao()
    manifest.update(status="concluido", best={"caminho": str(best), "sha256": _sha256(best)},
                    metricas=_metricas(modelo))
    gravar_json(arquivo, manifest)


def treinar(dataset: Path, nome: str, destino: Path, parametros: dict | None = None,
            modelo: str = MODELO_INICIAL) -> Path:
    """Treino novo em `<destino>/<nome>`; devolve a pasta do treino."""
    pasta = _pasta(nome, destino)
    if pasta.exists():
        raise EntradaInvalida(
            f"o treino '{nome}' já existe em {pasta}; use --retomar (se houver weights/last.pt), "
            "apague a pasta ou escolha outro --nome")
    data_yaml = validar_dataset(dataset).resolve()
    params = {**PARAMETROS, **(parametros or {})}
    manifest_dataset = dataset / "manifest.json"
    pasta.mkdir(parents=True)
    manifest = {
        "nome": nome,
        "dataset": str(dataset.resolve()),
        "data_yaml_sha256": _sha256(data_yaml),
        "dataset_manifest_sha256": _sha256(manifest_dataset) if manifest_dataset.exists() else None,
        "modelo_inicial": modelo,
        "parametros": params,
        "versoes": {p: _versao(p) for p in ("nfl-vision", "ultralytics", "torch")},
        "gpu": _gpu(),
        "inicio": _agora(),
        "duracao_s": 0.0,
        "retomadas": [],
    }

    def treino():
        m = _yolo(modelo)
        # project absoluto (senão vai para runs/detect/); exist_ok=True porque a pasta já foi
        # criada acima, depois de recusar nome repetido (exist_ok=False criaria "<nome>2").
        m.train(data=str(data_yaml), project=str(pasta.parent), name=pasta.name, exist_ok=True,
                **params)
        return m

    _executar(pasta, manifest, treino)
    return pasta


def retomar(nome: str, destino: Path) -> Path:
    """Continua o treino `<destino>/<nome>` a partir de weights/last.pt."""
    pasta = _pasta(nome, destino)
    last = pasta / "weights" / "last.pt"
    if not last.exists():
        raise EntradaInvalida(f"não há {last} para retomar; confira o --nome ou comece um treino novo")
    arquivo = pasta / "manifest.json"
    manifest = json.loads(arquivo.read_text("utf-8")) if arquivo.exists() else {"nome": nome}
    if manifest.get("status") == "concluido":
        raise EntradaInvalida(f"o treino '{nome}' já terminou; não há o que retomar")
    manifest.setdefault("retomadas", []).append(_agora())

    def treino():
        m = _yolo(str(last))
        m.train(resume=True)  # dataset, parâmetros e pasta vêm do checkpoint
        return m

    _executar(pasta, manifest, treino)
    return pasta
