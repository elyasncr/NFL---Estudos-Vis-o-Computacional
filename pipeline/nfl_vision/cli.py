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
from nfl_vision.config import Config
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


BENCHMARKS = ("yolo-bruto", "rfdetr", "roboflow-nfl")
DESCRICAO_ARBITROS = "árbitros cobertos por caixa de jogador (IoU ≥ 0,5)"
AJUDA_SPLIT = "Split avaliado (exports do Roboflow usam 'valid', não 'val')"
SEM_EXTRAS = "dependências de avaliação ausentes ({}); rode: uv sync --extra ocr --extra eval"


def _arquivo_avaliacao(nome: str) -> Path:
    from datetime import datetime

    destino = paths.avaliacoes_dir()
    destino.mkdir(parents=True, exist_ok=True)
    return destino / f"{nome}-{datetime.now():%Y%m%d-%H%M%S}.json"


def _pasta_de_split_ausente(dataset: Path, pasta: Path) -> str:
    from nfl_vision.eval.datasets import splits_existentes

    existentes = ", ".join(splits_existentes(dataset)) or "nenhuma"
    return (f"pasta {pasta} não encontrada; pastas em {dataset}: {existentes} "
            "(exports do Roboflow usam 'valid', não 'val')")


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


def _carregar_deteccao(dataset: Path, split: str, classe_alvo: str):
    from nfl_vision.eval.datasets import carregar_yolo, nomes_das_classes

    try:
        nomes = nomes_das_classes(dataset)
        if classe_alvo not in nomes:
            raise typer.BadParameter(
                f"o dataset não tem a classe '{classe_alvo}'; classes: {', '.join(map(str, nomes))}",
                param_hint="--dataset")
        amostras = carregar_yolo(dataset, split)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--dataset") from exc
    if not amostras:
        raise typer.BadParameter(f"nenhuma imagem em {dataset / split / 'images'}", param_hint="--split")
    return amostras


def _criar_preditores(benchmark: List[str], cfg: Config, conf: float, modelo_roboflow: str) -> list:
    from nfl_vision.eval import preditores

    try:
        lista = [preditores.PreditorNosso(cfg)]
        for b in benchmark:
            if b == "yolo-bruto":
                lista.append(preditores.PreditorYoloBruto(cfg))
            elif b == "rfdetr":
                lista.append(preditores.PreditorRFDETR(conf))
            else:
                lista.append(preditores.PreditorRoboflowNFL(modelo_roboflow, conf))
    except ImportError as exc:
        raise typer.BadParameter(SEM_EXTRAS.format(exc), param_hint="--benchmark") from exc
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--benchmark") from exc
    return lista


@eval_app.command("detect")
def eval_detect_cmd(
    dataset: Path = typer.Option(..., "--dataset", help="Pasta do dataset em formato YOLO"),
    split: str = typer.Option("test", "--split", help=AJUDA_SPLIT),
    benchmark: List[str] = typer.Option(
        [], "--benchmark", help="yolo-bruto, rfdetr e/ou roboflow-nfl (repetível)"),
    modelo_roboflow: str = typer.Option("nfl-player-model/4", "--modelo-roboflow"),
    conf: float = typer.Option(
        Config().detector_conf, "--conf", min=0.0, max=1.0,
        help="Confiança mínima, a mesma para todos os preditores"),
) -> None:
    """mAP@0.5 de jogador num split do dataset, com benchmarks opcionais."""
    from nfl_vision.eval import detect as avaliacao

    invalidos = [b for b in benchmark if b not in BENCHMARKS]
    if invalidos:
        raise typer.BadParameter(
            f"use {', '.join(BENCHMARKS[:-1])} ou {BENCHMARKS[-1]}", param_hint="--benchmark")

    cfg = Config(detector_conf=conf)
    amostras = _carregar_deteccao(dataset, split, "player")
    lista = _criar_preditores(benchmark, cfg, conf, modelo_roboflow)

    cabecalho = {
        "dataset": str(dataset), "split": split, "conf": conf,
        "config": cfg.model_dump(mode="json"), "versao": pipeline._versao("nfl-vision"),
        "metricas": {"map50": "mAP@0.5 da classe player",
                     "arbitros_como_jogador": DESCRICAO_ARBITROS},
    }
    if "roboflow-nfl" in benchmark:
        cabecalho["modelo_roboflow"] = modelo_roboflow
    arquivo = _arquivo_avaliacao("detect")
    resultados = []
    for p in lista:
        try:
            r = avaliacao.avaliar(p, amostras)
        except Exception as exc:  # um preditor quebrado não invalida os outros
            r = {"preditor": p.nome, "pos_processamento": p.pos_processamento,
                 "erro": f"{type(exc).__name__}: {exc}"}
            console.print(f"[red]{escape(p.nome)}: {escape(r['erro'])}[/red]")
        resultados.append(r)
        gravar_json(arquivo, {**cabecalho, "resultados": resultados})  # parcial a cada preditor

    tabela = Table(title=f"Detecção — {dataset.name} ({split}, {len(amostras)} imagens)")
    for coluna in ("preditor", "pós-processamento", "mAP@0.5", DESCRICAO_ARBITROS):
        tabela.add_column(coluna)
    for r in resultados:
        if "erro" in r:
            tabela.add_row(r["preditor"], r["pos_processamento"], "erro", "—")
            continue
        arb = "—" if r["arbitros_como_jogador"] is None else f"{r['arbitros_como_jogador']:.2%}"
        tabela.add_row(r["preditor"], r["pos_processamento"], f"{r['map50']:.3f}", arb)
    console.print(tabela)
    console.print(f"Resultados: {arquivo}")
    if any("erro" in r for r in resultados):
        raise typer.Exit(1)


@eval_app.command("jersey")
def eval_jersey_cmd(
    dataset: Path = typer.Option(..., "--dataset", help="Pasta no formato <split>/<número>/<imagem>"),
    split: str = typer.Option("test", "--split", help=AJUDA_SPLIT),
) -> None:
    """Acurácia do OCR em recortes de números legíveis."""
    from nfl_vision.eval import jersey as avaliacao
    from nfl_vision.eval.datasets import carregar_pastas
    from nfl_vision.stages.jersey import leitor_padrao

    pasta = dataset / split
    if not pasta.is_dir():
        raise typer.BadParameter(_pasta_de_split_ausente(dataset, pasta), param_hint="--split")
    amostras = carregar_pastas(dataset, split)
    if not amostras:
        raise typer.BadParameter(f"nenhuma imagem em {pasta}/<número>/", param_hint="--dataset")

    cfg = Config()
    try:
        leitor = leitor_padrao(cfg.ocr_device)
    except ImportError as exc:
        raise typer.BadParameter(SEM_EXTRAS.format(exc)) from exc
    r = avaliacao.avaliar(leitor, amostras, cfg.limiar_numero, cfg.numero_altura_min)
    console.print(r)
    arquivo = _arquivo_avaliacao("jersey")
    gravar_json(arquivo, {
        "dataset": str(dataset), "split": split, "limiar": cfg.limiar_numero,
        "altura_min": cfg.numero_altura_min, "versao": pipeline._versao("nfl-vision"), "resultado": r,
    })
    console.print(f"Resultados: {arquivo}")
