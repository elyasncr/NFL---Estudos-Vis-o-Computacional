"""Avaliação do time (cor do tronco) contra um gabarito rotulado à mão."""

import json
from dataclasses import dataclass
from pathlib import Path

from nfl_vision.config import Config
from nfl_vision.eval.metricas import iou
from nfl_vision.schemas import BBox, Deteccao
from nfl_vision.stages.detect import aplicar_filtros, detectar_pessoas
from nfl_vision.stages.ingest import carregar_imagem
from nfl_vision.stages.team import classificar

IOU_MINIMO = 0.5


@dataclass
class CaixaGabarito:
    imagem: str
    bbox: BBox
    time: str | None


@dataclass
class Gabarito:
    times: tuple[str, str]
    caixas: list[CaixaGabarito]


def _caixa_do_json(caminho: Path, i: int, c) -> CaixaGabarito:
    if not isinstance(c, dict):
        raise ValueError(f"{caminho}: caixa {i}: deve ser um objeto JSON")
    if "imagem" not in c:
        raise ValueError(f"{caminho}: caixa {i}: falta 'imagem'")
    if "bbox" not in c:
        raise ValueError(f"{caminho}: caixa {i}: falta 'bbox'")
    bbox = c["bbox"]
    if not (isinstance(bbox, list) and len(bbox) == 4):
        raise ValueError(f"{caminho}: caixa {i}: 'bbox' deve ter 4 valores [x1, y1, x2, y2]")
    try:
        bbox = tuple(float(v) for v in bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{caminho}: caixa {i}: 'bbox' deve ter 4 números") from exc
    return CaixaGabarito(imagem=c["imagem"], bbox=bbox, time=c.get("time"))


def carregar_gabarito(caminho: Path) -> Gabarito:
    """Lê o gabarito rotulado à mão (ver data/avaliacoes/time-gabarito-cin-cle.json):
    `{"times": [A, B], "caixas": [{"imagem", "bbox", "time": A|B|null}, ...]}`."""
    try:
        dados = json.loads(caminho.read_text("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{caminho}: JSON inválido: {exc}") from exc
    if not isinstance(dados, dict):
        raise ValueError(f"{caminho}: esperado um objeto JSON")
    times = dados.get("times")
    if not (isinstance(times, list) and len(times) == 2 and all(isinstance(t, str) for t in times)):
        raise ValueError(f"{caminho}: 'times' deve ser uma lista com as 2 siglas dos times")
    caixas_json = dados.get("caixas")
    if not isinstance(caixas_json, list) or not caixas_json:
        raise ValueError(f"{caminho}: 'caixas' deve ser uma lista não vazia")
    caixas = [_caixa_do_json(caminho, i, c) for i, c in enumerate(caixas_json)]
    for i, caixa in enumerate(caixas):
        if caixa.time is not None and caixa.time not in times:
            raise ValueError(
                f"{caminho}: caixa {i}: time '{caixa.time}' não está em {times}")
    return Gabarito(times=(times[0], times[1]), caixas=caixas)


def _casar_um_a_um(caixas: list[CaixaGabarito], deteccoes: list[Deteccao]) -> dict[int, Deteccao]:
    """Casamento um-a-um: todo par (caixa, detecção não descartada) com IoU >= 0,5, ordenado
    por IoU decrescente, atribuído greedily — cada detecção e cada caixa usada no máximo uma
    vez. Caixa sem par vira `sem_deteccao` em `avaliar`."""
    pares = []
    for i, caixa in enumerate(caixas):
        for d in deteccoes:
            if d.descartado:
                continue
            val = iou(caixa.bbox, d.bbox)
            if val >= IOU_MINIMO:
                pares.append((val, i, d))
    pares.sort(key=lambda p: p[0], reverse=True)

    casadas: dict[int, Deteccao] = {}
    usados: set[int] = set()
    for _, i, d in pares:
        if i in casadas or d.det_id in usados:
            continue
        casadas[i] = d
        usados.add(d.det_id)
    return casadas


def avaliar(gabarito: Gabarito, raiz_imagens: Path, paletas: dict, cfg: Config) -> dict:
    """Roda detecção + filtros + `team.classificar` em cada imagem do gabarito e casa as
    caixas rotuladas (`time` != null) com as detecções não descartadas, um-a-um (IoU >= 0,5,
    maior IoU primeiro; ver `_casar_um_a_um`).

    `acuracia` = acertos / (acertos + erros) [com time atribuído]. `cobertura` = (acertos +
    erros) / (acertos + erros + nulos) [caixas casadas]. `sem_deteccao` fica fora de
    `cobertura`: é caixa rotulada sem nenhuma detecção correspondente. `rotulados` é o total de
    caixas rotuladas (`time` != null) no gabarito. `cobertura_total` = (acertos + erros) /
    `rotulados`: fração de ponta a ponta, incluindo quem ficou sem detecção casada.
    """
    por_imagem: dict[str, list[CaixaGabarito]] = {}
    for c in gabarito.caixas:
        if c.time is not None:
            por_imagem.setdefault(c.imagem, []).append(c)
    rotulados = sum(len(caixas) for caixas in por_imagem.values())

    acertos = erros = nulos = sem_deteccao = 0
    for nome, caixas in por_imagem.items():
        img = carregar_imagem(raiz_imagens / nome)
        deteccoes = [d for d in aplicar_filtros(detectar_pessoas(img, cfg), img, cfg)
                     if not d.descartado]
        time_por_det = {t.det_id: t.time for t in classificar(img, deteccoes, paletas, cfg)}
        casadas = _casar_um_a_um(caixas, deteccoes)
        for i, caixa in enumerate(caixas):
            det = casadas.get(i)
            if det is None:
                sem_deteccao += 1
                continue
            previsto = time_por_det[det.det_id]
            if previsto is None:
                nulos += 1
            elif previsto == caixa.time:
                acertos += 1
            else:
                erros += 1

    com_time = acertos + erros
    combinados = com_time + nulos
    return {
        "acuracia": acertos / com_time if com_time else None,
        "cobertura": com_time / combinados if combinados else None,
        "cobertura_total": com_time / rotulados if rotulados else None,
        "acertos": acertos,
        "erros": erros,
        "nulos": nulos,
        "sem_deteccao": sem_deteccao,
        "rotulados": rotulados,
    }
