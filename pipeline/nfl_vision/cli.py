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
treino_app = typer.Typer(help="Preparação do dataset e ajuste fino do detector de jogadores.",
                         no_args_is_help=True)
app.add_typer(treino_app, name="treino")
console = Console()

DETECTORES = ("rfdetr", "yolo")
AJUDA_DETECTOR_TIPO = "Detector: rfdetr (padrão) ou yolo"


def _validar_detector(detector_tipo: Optional[str], pesos: Optional[Path], opcao_pesos: str) -> None:
    """Tipo conhecido; pesos (.pt) são do YOLO e não combinam com rfdetr."""
    if detector_tipo is not None and detector_tipo not in DETECTORES:
        raise typer.BadParameter("use rfdetr ou yolo", param_hint="--detector-tipo")
    if pesos is not None and detector_tipo == "rfdetr":
        raise typer.BadParameter(
            f"{opcao_pesos} recebe pesos YOLO (.pt); não combine com --detector-tipo rfdetr",
            param_hint=opcao_pesos)


def _ajustes_do_detector(detector_tipo: Optional[str], pesos: Optional[Path]) -> dict:
    """Campos da Config para o detector escolhido; pesos .pt implicam yolo."""
    ajustes = {}
    if detector_tipo is not None:
        ajustes["detector_tipo"] = detector_tipo
    if pesos is not None:
        # caminho absoluto: o reprocessamento (--run) pode rodar de outro diretório
        ajustes.update(detector_tipo="yolo", detector_pesos=str(pesos.resolve()))
    return ajustes


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
def _tratando_falha_de_etapa(analise_id: Optional[str] = None):
    """Erro de etapa vira mensagem com o comando para retomar; Ctrl+C sai com 130.

    `analise_id`, quando conhecido (reprocessamento ou correção), também cobre
    falhas fora de uma etapa (ex.: `pipeline.finalizar`, depois que o runner já
    terminou): o erro não vem embrulhado em `EtapaFalhou`, mas a análise pode
    ser retomada a partir de `roster` (a última etapa do runner).
    """
    try:
        yield
    except EtapaFalhou as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        console.print(
            f"Depois de resolver, rode: nfl-vision analyze --run {exc.analise_id} --from {exc.etapa}")
        raise typer.Exit(1)
    except (RuntimeError, teams.TimesIndisponiveis) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        if analise_id is not None:
            console.print(
                f"Depois de resolver, rode: nfl-vision analyze --run {analise_id} --from roster")
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
    detector_tipo: Optional[str] = typer.Option(None, "--detector-tipo", help=AJUDA_DETECTOR_TIPO),
    detector: Optional[Path] = typer.Option(
        None, "--detector",
        help="Pesos YOLO (.pt) para esta análise; implica --detector-tipo yolo"),
) -> None:
    """Analisa uma foto ou reprocessa uma análise a partir de uma etapa."""
    with _tratando_falha_de_etapa(run):
        run_dir, analise = _analisar_ou_reprocessar(
            foto, times, temporada, semana, run, a_partir_de, detector, detector_tipo)
    _imprimir(analise, run_dir)


def _analisar_ou_reprocessar(foto, times, temporada, semana, run, a_partir_de, detector=None,
                             detector_tipo=None):
    if a_partir_de is not None and not run:
        raise typer.BadParameter("--from exige --run <id>", param_hint="--from")
    if run and foto is not None:
        raise typer.BadParameter("não informe a foto junto com --run", param_hint="FOTO")
    if run and a_partir_de is None:
        raise typer.BadParameter("--run exige --from <etapa>", param_hint="--run")
    if run and detector is not None:
        raise typer.BadParameter(
            "--detector só vale para análise nova; o reprocessamento usa a config gravada",
            param_hint="--detector")
    if run and detector_tipo is not None:
        raise typer.BadParameter(
            "--detector-tipo só vale para análise nova; o reprocessamento usa a config gravada",
            param_hint="--detector-tipo")
    _validar_detector(detector_tipo, detector, "--detector")
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
    if detector is not None and not detector.is_file():
        raise typer.BadParameter(f"pesos não encontrados: {detector}", param_hint="--detector")
    contexto = _validar_contexto(times, temporada, semana)
    ajustes = _ajustes_do_detector(detector_tipo, detector)
    config = Config(**ajustes) if ajustes else None
    return pipeline.analisar(foto, contexto, config)


@app.command()
def correct(
    analise_id: str = typer.Argument(..., help="ID da análise, ex.: 2026-10-03-001"),
    det: int = typer.Option(..., "--det", help="det_id (track_id) do jogador"),
    time: Optional[str] = typer.Option(None, "--time"),
    numero: Optional[int] = typer.Option(None, "--numero", min=0, max=99),
) -> None:
    """Corrige o time e/ou o número de um jogador e refaz a consulta ao roster."""
    with _tratando_falha_de_etapa(analise_id):
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

    try:
        destino = baixar(workspace, projeto, versao, formato, paths.datasets_dir())
    except RuntimeError as exc:  # chave ausente (chave_roboflow)
        raise typer.BadParameter(str(exc), param_hint="--workspace") from exc
    except Exception as exc:  # erro de rede/download do Roboflow
        raise typer.BadParameter(
            f"falha ao baixar o dataset: {type(exc).__name__}: {exc}", param_hint="--workspace"
        ) from exc
    console.print(f"Dataset em: {destino}")


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


def _criar_preditores(benchmark: List[str], cfg: Config, conf: Optional[float],
                      modelo_roboflow: str) -> list:
    from nfl_vision.eval import preditores

    try:
        lista = [preditores.PreditorNosso(cfg)]
        for b in benchmark:
            if b == "yolo-bruto":
                lista.append(preditores.PreditorYoloBruto(cfg))
            elif b == "rfdetr":
                lista.append(preditores.PreditorRFDETR(cfg))
            elif conf is None:  # sem --conf: usa o padrão do próprio preditor
                lista.append(preditores.PreditorRoboflowNFL(modelo_roboflow))
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
    conf: Optional[float] = typer.Option(
        None, "--conf", min=0.0, max=1.0,
        help="Confiança mínima para todos os preditores; padrão: a de cada um "
             "(rfdetr 0,4; yolo 0,25)"),
    pesos: Optional[Path] = typer.Option(
        None, "--pesos",
        help="Pesos YOLO (.pt) para os preditores nosso e yolo-bruto; implica --detector-tipo yolo"),
    detector_tipo: Optional[str] = typer.Option(
        None, "--detector-tipo", help=f"{AJUDA_DETECTOR_TIPO} para o preditor nosso"),
    resolucao: Optional[int] = typer.Option(
        None, "--resolucao",
        help="Lado de entrada do RF-DETR (múltiplo de 56) para nosso e rfdetr; padrão: config"),
) -> None:
    """mAP@0.5 de jogador num split do dataset, com benchmarks opcionais."""
    from nfl_vision.eval import detect as avaliacao

    invalidos = [b for b in benchmark if b not in BENCHMARKS]
    if invalidos:
        raise typer.BadParameter(
            f"use {', '.join(BENCHMARKS[:-1])} ou {BENCHMARKS[-1]}", param_hint="--benchmark")

    _validar_detector(detector_tipo, pesos, "--pesos")
    if resolucao is not None and (resolucao <= 0 or resolucao % 56):
        raise typer.BadParameter("--resolucao deve ser um múltiplo positivo de 56",
                                 param_hint="--resolucao")
    if pesos is not None and not pesos.is_file():
        raise typer.BadParameter(f"pesos não encontrados: {pesos}", param_hint="--pesos")
    ajustes = _ajustes_do_detector(detector_tipo, pesos)
    if resolucao is not None:
        ajustes["detector_resolucao"] = resolucao
    cfg = Config(detector_conf=conf, **ajustes)
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
    if pesos is not None:
        from nfl_vision.stages.detect import sha256_pesos

        cabecalho["pesos_sha256"] = sha256_pesos(cfg.detector_pesos)
    arquivo = _arquivo_avaliacao("detect")
    resultados = []
    for p in lista:
        try:
            r = avaliacao.avaliar(p, amostras)
        except Exception as exc:  # um preditor quebrado não invalida os outros
            r = {"preditor": p.nome, "pos_processamento": p.pos_processamento,
                 "erro": f"{type(exc).__name__}: {exc}"}
            console.print(f"[red]{escape(p.nome)}: {escape(r['erro'])}[/red]")
        r["conf"] = p.conf  # limiar efetivamente usado por este preditor
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


def _decisoes_externas(aprovar: List[str], rejeitar: List[str]) -> dict[str, tuple[bool, str]]:
    """{fonte: (aprovada, motivo)} a partir de valores 'nome:motivo'."""
    from nfl_vision.treino.fontes import EXTERNAS

    validas = [f.nome for f in EXTERNAS]
    decisoes: dict[str, tuple[bool, str]] = {}
    for opcao, aprovada, valores in (("--aprovar", True, aprovar), ("--rejeitar", False, rejeitar)):
        for valor in valores:
            nome, _, motivo = valor.partition(":")
            nome, motivo = nome.strip(), motivo.strip()
            if nome not in validas:
                raise typer.BadParameter(
                    f"fonte desconhecida '{nome}'; use: {', '.join(validas)}", param_hint=opcao)
            if not motivo:
                raise typer.BadParameter(f"informe o motivo: '{nome}:<motivo>'", param_hint=opcao)
            if nome in decisoes:
                raise typer.BadParameter(f"fonte '{nome}' decidida mais de uma vez", param_hint=opcao)
            decisoes[nome] = (aprovada, motivo)
    return decisoes


@treino_app.command("preparar")
def treino_preparar(
    saida: Optional[Path] = typer.Option(
        None, "--saida", help="Pasta do dataset; padrão: data/datasets/treino-player-v1"),
    so_triagem: bool = typer.Option(
        False, "--so-triagem", help="Só baixa as fontes externas e gera os painéis de triagem"),
    aprovar: List[str] = typer.Option(
        [], "--aprovar", help="Fonte externa aprovada, 'nome:motivo' (repetível)"),
    rejeitar: List[str] = typer.Option(
        [], "--rejeitar", help="Fonte externa rejeitada, 'nome:motivo' (repetível)"),
    semente: int = typer.Option(0, "--semente", help="Semente do sorteio de validação dos externos"),
) -> None:
    """Prepara o dataset de player: triagem das fontes externas (--so-triagem) e depois a montagem."""
    from nfl_vision.treino import fontes, preparar

    datasets = paths.datasets_dir()
    saida = saida or datasets / preparar.PASTA_PADRAO
    erros = (fontes.FonteIndisponivel, FileNotFoundError, FileExistsError, ValueError)

    if so_triagem:
        if aprovar or rejeitar:
            raise typer.BadParameter("--so-triagem não aceita --aprovar nem --rejeitar",
                                     param_hint="--so-triagem")
        try:
            pares = [(f, fontes.obter(f, datasets)) for f in fontes.EXTERNAS]
            resumo = preparar.triagem(pares, saida, semente)
        except erros as exc:
            raise typer.BadParameter(str(exc)) from exc
        tabela = Table(title="Triagem das fontes externas")
        for coluna in ("fonte", "imagens", "painel"):
            tabela.add_column(coluna)
        for r in resumo:
            tabela.add_row(r["fonte"], str(r["imagens"]), escape(str(r["painel"])))
        console.print(tabela)
        console.print("Veja os painéis e decida cada fonte: nfl-vision treino preparar "
                      "--aprovar <fonte>:<motivo> --rejeitar <fonte>:<motivo>")
        return

    decididas = _decisoes_externas(aprovar, rejeitar)
    faltam = [f.nome for f in fontes.EXTERNAS if f.nome not in decididas]
    if faltam:
        raise typer.BadParameter(
            f"decida todas as fontes externas (faltam: {', '.join(faltam)}); "
            "gere os painéis antes com --so-triagem", param_hint="--aprovar")
    try:
        decisoes = [preparar.Decisao(fontes.BASE, True, preparar.MOTIVO_BASE,
                                     fontes.obter(fontes.BASE, datasets))]
        for f in fontes.EXTERNAS:
            aprovada, motivo = decididas[f.nome]
            decisoes.append(preparar.Decisao(
                f, aprovada, motivo, fontes.obter(f, datasets) if aprovada else None))
        manifest = preparar.construir(saida, decisoes, semente)
    except erros as exc:
        raise typer.BadParameter(str(exc)) from exc

    tabela = Table(title=f"Dataset {saida.name}")
    for coluna in ("split", "fonte", "imagens", "caixas"):
        tabela.add_column(coluna)
    for split, por_fonte in manifest["contagens"].items():
        for nome, c in por_fonte.items():
            tabela.add_row(split, nome, str(c["imagens"]), str(c["caixas"]))
    console.print(tabela)
    console.print(f"Dataset em: {escape(str(saida))}")


@treino_app.command("rodar")
def treino_rodar(
    nome: str = typer.Option(..., "--nome", help="Nome do treino (pasta em data/treinos/)"),
    dataset: Optional[Path] = typer.Option(
        None, "--dataset", help="Pasta gerada por `treino preparar`"),
    epocas: Optional[int] = typer.Option(
        None, "--epocas", min=1, help="Padrão: 100 (com parada antecipada, patience 20)"),
    imgsz: Optional[int] = typer.Option(None, "--imgsz", min=32, help="Padrão: 1280"),
    batch: Optional[int] = typer.Option(
        None, "--batch", min=1, help="Imagens por lote. Padrão: 4 (8 transborda 16 GB com imgsz 1280)"),
    retomar: bool = typer.Option(
        False, "--retomar", help="Continua do weights/last.pt do treino --nome"),
) -> None:
    """Ajuste fino do YOLO11m para player (Ultralytics, GPU local)."""
    from nfl_vision.treino import rodar

    if retomar and (dataset is not None or epocas is not None or imgsz is not None
                    or batch is not None):
        raise typer.BadParameter(
            "--retomar usa os parâmetros do treino original; "
            "não informe --dataset, --epocas, --imgsz nem --batch",
            param_hint="--retomar")
    if not retomar and dataset is None:
        raise typer.BadParameter("informe --dataset (ou --retomar para continuar um treino)",
                                 param_hint="--dataset")
    destino = paths.treinos_dir()
    try:
        if retomar:
            pasta = rodar.retomar(nome, destino)
        else:
            ajustes = {k: v for k, v in (("epochs", epocas), ("imgsz", imgsz), ("batch", batch))
                       if v is not None}
            pasta = rodar.treinar(dataset, nome, destino, parametros=ajustes)
    except rodar.EntradaInvalida as exc:
        raise typer.BadParameter(str(exc)) from exc
    except KeyboardInterrupt:
        console.print("[yellow]interrompido; para continuar: "
                      f"nfl-vision treino rodar --nome {escape(nome)} --retomar[/yellow]")
        raise typer.Exit(130)
    except Exception as exc:  # erro do Ultralytics/CUDA: mensagem curta; o manifest guarda o status
        console.print(f"[red]treino falhou: {escape(type(exc).__name__)}: {escape(str(exc))}[/red]")
        console.print("Se houver weights/last.pt, continue com: "
                      f"nfl-vision treino rodar --nome {escape(nome)} --retomar")
        raise typer.Exit(1)
    console.print(f"Pesos: {escape(str(pasta / 'weights' / 'best.pt'))}")
    console.print(f"Manifest: {escape(str(pasta / 'manifest.json'))}")
