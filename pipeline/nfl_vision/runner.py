"""Executa etapas em sequência, gravando a saída de cada uma em disco."""

import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
from pydantic import BaseModel

from nfl_vision.config import Config
from nfl_vision.schemas import Contexto, Correcao


class EtapaFalhou(Exception):
    def __init__(self, etapa: str, mensagem: str, analise_id: str | None = None):
        super().__init__(f"etapa '{etapa}' falhou: {mensagem}")
        self.etapa = etapa
        self.mensagem = mensagem
        self.analise_id = analise_id


@dataclass
class Estado:
    run_dir: Path
    contexto: Contexto
    config: Config
    saidas: dict[str, BaseModel] = field(default_factory=dict)
    _imagem: np.ndarray | None = None

    @property
    def caminho_imagem(self) -> Path:
        return next(self.run_dir.glob("input.*"))

    def imagem(self) -> np.ndarray:
        if self._imagem is None:
            from nfl_vision.stages.ingest import carregar_imagem

            self._imagem = carregar_imagem(self.caminho_imagem)
        return self._imagem

    def correcoes(self) -> list[Correcao]:
        arquivo = self.run_dir / "corrections.json"
        if not arquivo.exists():
            return []
        return [Correcao.model_validate(c) for c in json.loads(arquivo.read_text("utf-8"))]


@dataclass(frozen=True)
class Etapa:
    nome: str
    saida: type[BaseModel]
    executar: Callable[[Estado], BaseModel]


def proximo_id(runs_dir: Path, hoje: date | None = None) -> str:
    prefixo = (hoje or date.today()).isoformat()
    existentes = list(runs_dir.glob(f"{prefixo}-*")) if runs_dir.exists() else []
    n = max((int(p.name.rsplit("-", 1)[1]) for p in existentes), default=0) + 1
    return f"{prefixo}-{n:03d}"


def ler_manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text("utf-8"))


def _gravar_json(caminho: Path, dados) -> None:
    caminho.write_text(json.dumps(dados, indent=2, ensure_ascii=False), encoding="utf-8")


class Runner:
    def __init__(self, etapas: list[Etapa], runs_dir: Path):
        self.etapas = etapas
        self.runs_dir = runs_dir

    def nova_analise(self, imagem: Path, contexto: Contexto, config: Config,
                     versoes: dict[str, str]) -> Path:
        run_dir = self.runs_dir / proximo_id(self.runs_dir)
        run_dir.mkdir(parents=True)
        shutil.copy2(imagem, run_dir / f"input{imagem.suffix.lower()}")
        _gravar_json(run_dir / "manifest.json", {
            "analise_id": run_dir.name,
            "contexto": contexto.model_dump(mode="json"),
            "config": config.model_dump(mode="json"),
            "versoes": versoes,
            "etapas": {},
        })
        return run_dir

    def carregar_estado(self, run_dir: Path) -> Estado:
        manifest = ler_manifest(run_dir)
        return Estado(
            run_dir=run_dir,
            contexto=Contexto.model_validate(manifest["contexto"]),
            config=Config.model_validate(manifest["config"]),
        )

    def executar(self, run_dir: Path, a_partir_de: str | None = None) -> Estado:
        nomes = [e.nome for e in self.etapas]
        if a_partir_de is not None and a_partir_de not in nomes:
            raise ValueError(f"etapa desconhecida: {a_partir_de}. Opções: {', '.join(nomes)}")
        inicio = nomes.index(a_partir_de) if a_partir_de else 0
        estado = self.carregar_estado(run_dir)
        manifest = ler_manifest(run_dir)

        for etapa in self.etapas[inicio:]:
            manifest["etapas"].pop(etapa.nome, None)
            arquivo = run_dir / f"{etapa.nome}.json"
            if arquivo.exists():
                arquivo.unlink()
        _gravar_json(run_dir / "manifest.json", manifest)

        for i, etapa in enumerate(self.etapas):
            arquivo = run_dir / f"{etapa.nome}.json"
            if i < inicio:
                status = manifest["etapas"].get(etapa.nome, {}).get("status")
                if not arquivo.exists() or status != "ok":
                    raise EtapaFalhou(etapa.nome, "artefato ausente ou inválido", run_dir.name)
                estado.saidas[etapa.nome] = etapa.saida.model_validate_json(arquivo.read_text("utf-8"))
                continue

            t0 = time.perf_counter()
            try:
                saida = etapa.executar(estado)
            except Exception as exc:
                manifest["etapas"][etapa.nome] = {
                    "status": "erro", "mensagem": str(exc),
                    "duracao_s": round(time.perf_counter() - t0, 3),
                }
                _gravar_json(run_dir / "manifest.json", manifest)
                raise EtapaFalhou(etapa.nome, str(exc), run_dir.name) from exc

            arquivo.write_text(saida.model_dump_json(indent=2), encoding="utf-8")
            estado.saidas[etapa.nome] = saida
            manifest["etapas"][etapa.nome] = {
                "status": "ok", "mensagem": None,
                "duracao_s": round(time.perf_counter() - t0, 3),
            }
            _gravar_json(run_dir / "manifest.json", manifest)

        return estado
