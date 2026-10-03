"""Comando `nfl-vision`."""

from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Tuple

import typer
from dotenv import load_dotenv
from PIL import Image
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from nfl_vision import paths, pipeline, teams
from nfl_vision.runner import EtapaFalhou, gravar_json
from nfl_vision.schemas import Analise, Contexto
from nfl_vision.stages import ingest

app = typer.Typer(help="Identificação de jogadores da NFL em fotos.", no_args_is_help=True)
eval_app = typer.Typer(help="Avaliação de detecção e de leitura de número.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")
console = Console()


@app.callback()
def _inicio() -> None:
    load_dotenv()


def _validar_contexto(times: Tuple[str, str], temporada: int, semana: int) -> Contexto:
    if temporada < 2002:
        raise typer.BadParameter("temporada deve ser 2002 ou posterior", param_hint="--temporada")
    if not 1 <= semana <= 22:
        raise typer.BadParameter("semana deve estar entre 1 e 22", param_hint="--semana")
    try:
        times_df = teams.carregar_times(paths.cache_dir())
        a, b = (teams.validar(t, times_df) for t in times)
    except (teams.TimeDesconhecido, teams.TimesIndisponiveis) as exc:
        raise typer.BadParameter(str(exc), param_hint="--times") from exc
    if a == b:
        raise typer.BadParameter("os dois times precisam ser diferentes", param_hint="--times")
    return Contexto(temporada=temporada, semana=semana, times=(a, b))


def _validar_imagem(foto: Path) -> None:
    try:
        with Image.open(foto) as img:
            img.verify()
    except Exception as exc:
        raise typer.BadParameter(f"imagem inválida ou corrompida: {foto}", param_hint="FOTO") from exc


def _imprimir(analise: Analise, run_dir: Path) -> None:
    tabela = Table(title=f"Análise {analise.analise_id}")
    for coluna in ("det", "time", "nº", "conf.", "pos.", "nome"):
        tabela.add_column(coluna)
    for j in analise.jogadores:
        tabela.add_row(
            str(j.track_id), j.time or "?", "?" if j.numero is None else str(j.numero),
            f"{j.confianca_numero:.2f}", j.posicao or "", j.nome or "desconhecido",
        )
    console.print(tabela)
    console.print(f"Artefatos: {run_dir}")


@contextmanager
def _tratando_falha_de_etapa():
    """Erro de etapa vira mensagem com o comando para retomar; Ctrl+C sai com 130."""
    try:
        yield
    except EtapaFalhou as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        console.print(
            f"Depois de resolver, rode: nfl-vision analyze --run {exc.analise_id} --from {exc.etapa}")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("[yellow]interrompido[/yellow]")
        raise typer.Exit(130)


def _avisar_arquivamento(caminho: Path) -> None:
    console.print(f"[yellow]correções anteriores arquivadas em {escape(str(caminho))}[/yellow]")


@app.command()
def analyze(
    foto: Optional[Path] = typer.Argument(None, help="Foto JPG ou PNG"),
    times: Tuple[str, str] = typer.Option((None, None), "--times", help="Siglas dos dois times"),
    temporada: Optional[int] = typer.Option(None, "--temporada"),
    semana: Optional[int] = typer.Option(None, "--semana"),
    run: Optional[str] = typer.Option(None, "--run", help="Reprocessar uma análise existente"),
    a_partir_de: Optional[str] = typer.Option(None, "--from", help="Etapa inicial do reprocessamento"),
) -> None:
    """Analisa uma foto ou reprocessa uma análise a partir de uma etapa."""
    with _tratando_falha_de_etapa():
        run_dir, analise = _analisar_ou_reprocessar(foto, times, temporada, semana, run, a_partir_de)
    _imprimir(analise, run_dir)


def _analisar_ou_reprocessar(foto, times, temporada, semana, run, a_partir_de):
    if a_partir_de is not None and not run:
        raise typer.BadParameter("--from exige --run <id>", param_hint="--from")
    if run and foto is not None:
        raise typer.BadParameter("não informe a foto junto com --run", param_hint="FOTO")
    if run and a_partir_de is None:
        raise typer.BadParameter("--run exige --from <etapa>", param_hint="--run")
    if run:
        if a_partir_de not in pipeline.NOMES_ETAPAS:
            raise typer.BadParameter(
                f"use uma etapa: {', '.join(pipeline.NOMES_ETAPAS)}", param_hint="--from")
        try:
            return pipeline.reprocessar(run, a_partir_de, ao_arquivar=_avisar_arquivamento)
        except FileNotFoundError as exc:
            raise typer.BadParameter(str(exc), param_hint="--run") from exc
    if foto is None or None in times or temporada is None or semana is None:
        raise typer.BadParameter("informe a foto, --times, --temporada e --semana")
    if not foto.exists():
        raise typer.BadParameter(f"arquivo não encontrado: {foto}", param_hint="FOTO")
    try:
        ingest.validar_formato(foto)
    except ingest.FormatoNaoSuportado as exc:
        raise typer.BadParameter(str(exc), param_hint="FOTO") from exc
    _validar_imagem(foto)
    contexto = _validar_contexto(times, temporada, semana)
    return pipeline.analisar(foto, contexto)


@app.command()
def correct(
    analise_id: str = typer.Argument(..., help="ID da análise, ex.: 2026-10-03-001"),
    det: int = typer.Option(..., "--det", help="det_id (track_id) do jogador"),
    time: Optional[str] = typer.Option(None, "--time"),
    numero: Optional[int] = typer.Option(None, "--numero", min=0, max=99),
) -> None:
    """Corrige o time e/ou o número de um jogador e refaz a consulta ao roster."""
    with _tratando_falha_de_etapa():
        try:
            analise = pipeline.corrigir(analise_id, det, time, numero)
        except (ValueError, FileNotFoundError) as exc:
            raise typer.BadParameter(str(exc)) from exc
    _imprimir(analise, paths.runs_dir() / analise_id)


BENCHMARKS = ("rfdetr", "roboflow-nfl")


def _salvar_avaliacao(nome: str, resultados) -> Path:
    from datetime import datetime

    destino = paths.avaliacoes_dir()
    destino.mkdir(parents=True, exist_ok=True)
    arquivo = destino / f"{nome}-{datetime.now():%Y%m%d-%H%M%S}.json"
    gravar_json(arquivo, resultados)
    return arquivo


@eval_app.command("baixar")
def eval_baixar(
    workspace: str = typer.Option(..., "--workspace"),
    projeto: str = typer.Option(..., "--projeto"),
    versao: int = typer.Option(..., "--versao"),
    formato: str = typer.Option("yolov11", "--formato", help="yolov11 (detecção) ou folder (recortes)"),
) -> None:
    """Baixa uma versão de dataset do Roboflow para data/datasets/."""
    from nfl_vision.eval.datasets import baixar

    console.print(f"Dataset em: {baixar(workspace, projeto, versao, formato, paths.datasets_dir())}")


@eval_app.command("detect")
def eval_detect_cmd(
    dataset: Path = typer.Option(..., "--dataset", help="Pasta do dataset em formato YOLO"),
    split: str = typer.Option("test", "--split"),
    benchmark: List[str] = typer.Option([], "--benchmark", help="rfdetr e/ou roboflow-nfl"),
    modelo_roboflow: str = typer.Option("nfl-player-model/4", "--modelo-roboflow"),
) -> None:
    """mAP@0.5 de jogador no split de teste, com benchmarks opcionais."""
    from nfl_vision.config import Config
    from nfl_vision.eval import detect as avaliacao
    from nfl_vision.eval import preditores
    from nfl_vision.eval.datasets import carregar_yolo

    invalidos = [b for b in benchmark if b not in BENCHMARKS]
    if invalidos:
        raise typer.BadParameter(f"use {' ou '.join(BENCHMARKS)}", param_hint="--benchmark")

    lista = [preditores.PreditorNosso(Config())]
    for b in benchmark:
        if b == "rfdetr":
            lista.append(preditores.PreditorRFDETR())
        else:
            lista.append(preditores.PreditorRoboflowNFL(modelo_roboflow))

    amostras = carregar_yolo(dataset, split)
    resultados = [avaliacao.avaliar(p, amostras) for p in lista]

    tabela = Table(title=f"Detecção — {dataset.name} ({split}, {len(amostras)} imagens)")
    for coluna in ("preditor", "mAP@0.5", "árbitros como jogador"):
        tabela.add_column(coluna)
    for r in resultados:
        arb = "—" if r["arbitros_como_jogador"] is None else f"{r['arbitros_como_jogador']:.2%}"
        tabela.add_row(r["preditor"], f"{r['map50']:.3f}", arb)
    console.print(tabela)
    console.print(f"Resultados: {_salvar_avaliacao('detect', resultados)}")


@eval_app.command("jersey")
def eval_jersey_cmd(
    dataset: Path = typer.Option(..., "--dataset", help="Pasta no formato <split>/<número>/<imagem>"),
    split: str = typer.Option("test", "--split"),
) -> None:
    """Acurácia do OCR em recortes de números legíveis."""
    from nfl_vision.config import Config
    from nfl_vision.eval import jersey as avaliacao
    from nfl_vision.eval.datasets import carregar_pastas
    from nfl_vision.stages.jersey import leitor_padrao

    cfg = Config()
    r = avaliacao.avaliar(leitor_padrao(cfg.ocr_device), carregar_pastas(dataset, split),
                          cfg.limiar_numero, cfg.numero_altura_min)
    console.print(r)
    console.print(f"Resultados: {_salvar_avaliacao('jersey', r)}")
