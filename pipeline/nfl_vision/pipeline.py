"""Pipeline concreto: etapas na ordem, análise nova, reprocessamento e correção."""

import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable

import cv2

from nfl_vision import paths, teams
from nfl_vision.config import Config
from nfl_vision.montagem import montar
from nfl_vision.render import cores_de_exibicao, desenhar
from nfl_vision.runner import (
    Estado, Etapa, EtapaFalhou, Runner, atualizar_manifest, gravar_json, gravar_texto, ler_manifest,
)
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
    cores = cores_de_exibicao({t: teams.cores(t, times_df) for t in estado.contexto.times})
    caixas = {d.det_id: d.bbox for d in estado.saidas["detect"].deteccoes}
    anotada = desenhar(estado.imagem(), analise.jogadores, caixas, cores)
    ok, buf = cv2.imencode(".png", anotada)
    if not ok:
        raise RuntimeError("falha ao gerar anotada.png")
    buf.tofile(str(estado.run_dir / "anotada.png"))
    return analise


def analisar(imagem: Path, contexto: Contexto, config: Config | None = None) -> tuple[Path, Analise]:
    runner = _runner()
    run_dir = runner.nova_analise(imagem, contexto, config or Config(), coletar_versoes())
    return run_dir, finalizar(runner.executar(run_dir))


SAIDAS_FINAIS = ("analise.json", "anotada.png")


# Refazer a detecção renumera os det_id: correções antigas apontariam para outra pessoa.
ETAPAS_QUE_RENUMERAM = ("ingest", "detect")


def _arquivar_correcoes(run_dir: Path) -> Path | None:
    arquivo = run_dir / "corrections.json"
    if not arquivo.exists():
        return None
    carimbo = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destino = run_dir / f"corrections.{carimbo}.bak.json"
    arquivo.rename(destino)
    return destino


def reprocessar(analise_id: str, a_partir_de: str,
                ao_arquivar: Callable[[Path], None] | None = None) -> tuple[Path, Analise]:
    """Refaz a partir de `a_partir_de`. `ao_arquivar` recebe o caminho das correções arquivadas."""
    if a_partir_de not in NOMES_ETAPAS:
        raise ValueError(f"etapa desconhecida '{a_partir_de}'; use uma de: {', '.join(NOMES_ETAPAS)}")
    run_dir = _run_dir(analise_id)
    for nome in SAIDAS_FINAIS:
        (run_dir / nome).unlink(missing_ok=True)
    if a_partir_de in ETAPAS_QUE_RENUMERAM:
        arquivado = _arquivar_correcoes(run_dir)
        if arquivado is not None and ao_arquivar is not None:
            ao_arquivar(arquivado)
    return run_dir, finalizar(_runner().executar(run_dir, a_partir_de))


def _exigir_analise_completa(run_dir: Path) -> None:
    etapas = ler_manifest(run_dir).get("etapas", {})
    pendente = next(
        (n for n in NOMES_ETAPAS if etapas.get(n, {}).get("status") != "ok"), None)
    if pendente is None and not (run_dir / "analise.json").exists():
        pendente = NOMES_ETAPAS[-1]
    if pendente is not None:
        raise ValueError(
            f"análise incompleta; rode nfl-vision analyze --run {run_dir.name} "
            f"--from {pendente} antes")


def corrigir(analise_id: str, det_id: int, time: str | None = None,
             numero: int | None = None) -> Analise:
    if time is None and numero is None:
        raise ValueError("informe --time e/ou --numero")
    run_dir = _run_dir(analise_id)
    _exigir_analise_completa(run_dir)
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
    try:
        return reprocessar(analise_id, "roster")[1]
    except EtapaFalhou as exc:
        raise EtapaFalhou(
            exc.etapa,
            f"{exc.mensagem} (a correção foi salva e será aplicada no próximo reprocessamento)",
            exc.analise_id,
        ) from exc
