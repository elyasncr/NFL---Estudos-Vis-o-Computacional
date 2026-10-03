"""Pipeline concreto: etapas na ordem, análise nova, reprocessamento e correção."""

import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cv2

from nfl_vision import paths, teams
from nfl_vision.config import Config
from nfl_vision.cores import hex_para_bgr
from nfl_vision.montagem import montar
from nfl_vision.render import desenhar
from nfl_vision.runner import Estado, Etapa, Runner, atualizar_manifest, gravar_json, gravar_texto
from nfl_vision.schemas import (
    Analise, Contexto, Correcao, DetectOut, IngestOut, JerseyOut, RosterOut, TeamOut,
)
from nfl_vision.stages import detect, ingest, jersey, roster, team

ETAPAS = [
    Etapa("ingest", IngestOut, ingest.executar),
    Etapa("detect", DetectOut, detect.executar),
    Etapa("team", TeamOut, team.executar),
    Etapa("jersey", JerseyOut, jersey.executar),
    Etapa("roster", RosterOut, roster.executar),
]
NOMES_ETAPAS = [e.nome for e in ETAPAS]


def _versao(pacote: str) -> str:
    try:
        return version(pacote)
    except PackageNotFoundError:
        return "não instalado"


def coletar_versoes() -> dict[str, str]:
    return {p: _versao(p) for p in ("nfl-vision", "ultralytics", "torch", "paddleocr", "nflreadpy")}


def _runner() -> Runner:
    return Runner(ETAPAS, paths.runs_dir())


def _run_dir(analise_id: str) -> Path:
    run_dir = paths.runs_dir() / analise_id
    if not run_dir.exists():
        raise FileNotFoundError(f"análise '{analise_id}' não encontrada em {paths.runs_dir()}")
    return run_dir


def _registrar_pesos(estado: Estado) -> None:
    """Hash dos pesos do detector no manifest (SDD §4)."""
    sha = estado.saidas["detect"].pesos_sha256

    def aplicar(manifest: dict) -> None:
        manifest.setdefault("versoes", {})["detector_pesos_sha256"] = sha

    atualizar_manifest(estado.run_dir, aplicar)


def finalizar(estado: Estado) -> Analise:
    _registrar_pesos(estado)
    analise = montar(estado)
    gravar_texto(estado.run_dir / "analise.json", analise.model_dump_json(indent=2))

    times_df = teams.carregar_times(paths.cache_dir())
    cores = {t: hex_para_bgr(teams.cores(t, times_df)[0]) for t in estado.contexto.times}
    caixas = {d.det_id: d.bbox for d in estado.saidas["detect"].deteccoes}
    anotada = desenhar(estado.imagem(), analise.jogadores, caixas, cores)
    cv2.imencode(".png", anotada)[1].tofile(str(estado.run_dir / "anotada.png"))
    return analise


def analisar(imagem: Path, contexto: Contexto, config: Config | None = None) -> tuple[Path, Analise]:
    runner = _runner()
    run_dir = runner.nova_analise(imagem, contexto, config or Config(), coletar_versoes())
    return run_dir, finalizar(runner.executar(run_dir))


def reprocessar(analise_id: str, a_partir_de: str) -> tuple[Path, Analise]:
    run_dir = _run_dir(analise_id)
    return run_dir, finalizar(_runner().executar(run_dir, a_partir_de))


def corrigir(analise_id: str, det_id: int, time: str | None = None,
             numero: int | None = None) -> Analise:
    if time is None and numero is None:
        raise ValueError("informe --time e/ou --numero")
    run_dir = _run_dir(analise_id)
    analise = Analise.model_validate_json((run_dir / "analise.json").read_text("utf-8"))
    if det_id not in {j.track_id for j in analise.jogadores}:
        raise ValueError(f"det {det_id} não é um jogador desta análise")
    if time is not None:
        time = teams.normalizar(time)
        if time not in analise.contexto.times:
            raise ValueError(f"time deve ser um de {', '.join(analise.contexto.times)}")

    arquivo = run_dir / "corrections.json"
    lista = json.loads(arquivo.read_text("utf-8")) if arquivo.exists() else []
    lista.append(Correcao(
        det_id=det_id, time=time, numero=numero,
        timestamp=datetime.now(timezone.utc).isoformat(),
    ).model_dump(mode="json"))
    gravar_json(arquivo, lista)
    return reprocessar(analise_id, "roster")[1]
