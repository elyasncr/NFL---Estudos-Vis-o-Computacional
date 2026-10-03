"""Etapa 3: time de cada jogador pela cor do tronco; detecção de árbitro."""

import math

import numpy as np
from sklearn.cluster import KMeans

from nfl_vision import paths, teams
from nfl_vision.config import Config
from nfl_vision.cores import bgr_para_lab, delta_e, hex_para_lab, mascara_gramado
from nfl_vision.geometria import caixa_inteira
from nfl_vision.schemas import Deteccao, TeamOut, TimeDet

MIN_PIXELS = 50
MAX_PIXELS_KMEANS = 3000


def recorte_tronco(img: np.ndarray, bbox) -> np.ndarray:
    """Tronco: 10–50% da altura e 20–80% da largura da caixa."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    xa, ya, xb, yb = caixa_inteira(
        (x1 + 0.20 * w, y1 + 0.10 * h, x1 + 0.80 * w, y1 + 0.50 * h), img.shape)
    return img[ya:yb, xa:xb]


def pixels_uteis(recorte: np.ndarray, cfg: Config) -> np.ndarray:
    """Pixels LAB do recorte, sem o gramado.

    Se a máscara cobre a maior parte do recorte, a camisa é verde: nada é removido.
    """
    if recorte.size == 0:
        return np.empty((0, 3))
    gramado = mascara_gramado(recorte, cfg.gramado_hsv_min, cfg.gramado_hsv_max)
    if gramado.mean() > cfg.mascara_gramado_max_tronco:
        return bgr_para_lab(recorte.reshape(-1, 3))
    return bgr_para_lab(recorte[~gramado])


def _rotulos_colunas(recorte_bgr: np.ndarray) -> np.ndarray:
    """Por coluna: 1 = escura (L < 30), 2 = clara (L > 80), 0 = mista; limiar de 70%."""
    luz = bgr_para_lab(recorte_bgr.reshape(-1, 3))[:, 0].reshape(recorte_bgr.shape[:2])
    escura = (luz < 30).mean(axis=0) >= 0.7
    clara = (luz > 80).mean(axis=0) >= 0.7
    return np.where(escura, 1, np.where(clara, 2, 0))


def eh_arbitro(recorte_bgr: np.ndarray) -> bool:
    """Listras verticais pretas e brancas: colunas puras que se alternam."""
    if recorte_bgr.size == 0:
        return False
    rotulos = _rotulos_colunas(recorte_bgr)
    puras = rotulos[rotulos > 0]
    if len(puras) < 0.7 * len(rotulos):
        return False
    if (rotulos == 1).mean() < 0.25 or (rotulos == 2).mean() < 0.25:
        return False
    return int((puras[1:] != puras[:-1]).sum()) >= 4


def _distintos(pontos: np.ndarray) -> int:
    return len(np.unique(pontos.round(2), axis=0))


def amostrar(lab_px: np.ndarray, n: int = MAX_PIXELS_KMEANS) -> np.ndarray:
    if len(lab_px) <= n:
        return lab_px
    return lab_px[np.random.default_rng(0).choice(len(lab_px), n, replace=False)]


def cor_dominante(lab_px: np.ndarray) -> np.ndarray:
    lab_px = amostrar(lab_px)
    k = min(3, _distintos(lab_px))
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(lab_px)
    return km.cluster_centers_[np.bincount(km.labels_).argmax()]


def eh_branco(lab) -> bool:
    return bool(lab[0] > 85 and np.hypot(lab[1], lab[2]) < 10)


def tamanho_minimo_grupo(n: int) -> int:
    return max(2, math.ceil(0.15 * n))


def agrupar(cores: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Rótulo de grupo por cor e centros dos grupos (1 ou 2)."""
    unico = (np.zeros(len(cores), int), cores.mean(axis=0, keepdims=True))
    if _distintos(cores) < 2:
        return unico
    km = KMeans(n_clusters=2, n_init=4, random_state=0).fit(cores)
    centros = km.cluster_centers_
    if delta_e(centros[0], centros[1]) < cfg.delta_e_grupo_unico:
        return unico
    contagem = np.bincount(km.labels_, minlength=2)
    if contagem.min() < tamanho_minimo_grupo(len(cores)):
        # grupo pequeno demais é outlier: um só grupo, centrado na maioria
        return np.zeros(len(cores), int), centros[[contagem.argmax()]]
    return km.labels_.astype(int), centros


def _custo(centro, paleta) -> float:
    return min(delta_e(centro, p) for p in paleta)


def mapear_grupos(centros: np.ndarray, paletas: dict[str, list[np.ndarray]]) -> dict[int, str | None]:
    a, b = list(paletas)

    def mais_proximo(centro):
        return min((a, b), key=lambda t: _custo(centro, paletas[t]))

    if len(centros) == 1:
        return {0: None if eh_branco(centros[0]) else mais_proximo(centros[0])}

    brancos = [eh_branco(c) for c in centros]
    if brancos[0] != brancos[1]:
        colorido = 1 if brancos[0] else 0
        time = mais_proximo(centros[colorido])
        return {colorido: time, 1 - colorido: b if time == a else a}

    direto = _custo(centros[0], paletas[a]) + _custo(centros[1], paletas[b])
    cruzado = _custo(centros[0], paletas[b]) + _custo(centros[1], paletas[a])
    return {0: a, 1: b} if direto <= cruzado else {0: b, 1: a}


def confianca(cor, centros: np.ndarray, rotulo: int) -> float:
    d_proprio = delta_e(cor, centros[rotulo])
    if len(centros) == 1:
        return float(np.clip(1 - d_proprio / 50, 0, 1))
    d_outro = delta_e(cor, centros[1 - rotulo])
    total = d_proprio + d_outro
    return 0.5 if total == 0 else float(d_outro / total)


def classificar(img: np.ndarray, deteccoes: list[Deteccao],
                paletas: dict[str, list[np.ndarray]], cfg: Config) -> list[TimeDet]:
    itens: dict[int, TimeDet] = {}
    candidatos: list[tuple[int, np.ndarray]] = []
    for d in deteccoes:
        if d.descartado:
            continue
        recorte = recorte_tronco(img, d.bbox)
        px = pixels_uteis(recorte, cfg)
        if len(px) < MIN_PIXELS:
            itens[d.det_id] = TimeDet(det_id=d.det_id, time=None, confianca=0.0)
        elif eh_arbitro(recorte):
            itens[d.det_id] = TimeDet(det_id=d.det_id, time=None, confianca=1.0, arbitro=True)
        else:
            candidatos.append((d.det_id, cor_dominante(px)))

    if candidatos:
        rotulos, centros = agrupar(np.array([c for _, c in candidatos]), cfg)
        mapa = mapear_grupos(centros, paletas)
        for (det_id, cor), rotulo in zip(candidatos, rotulos):
            conf = confianca(cor, centros, int(rotulo))
            itens[det_id] = TimeDet(
                det_id=det_id,
                time=mapa[int(rotulo)] if conf >= cfg.limiar_time else None,
                confianca=round(conf, 4),
                cor_lab=tuple(float(v) for v in cor),
            )
    return [itens[k] for k in sorted(itens)]


def executar(estado) -> TeamOut:
    times_df = teams.carregar_times(paths.cache_dir())
    paletas = {t: [hex_para_lab(c) for c in teams.cores(t, times_df)]
               for t in estado.contexto.times}
    deteccoes = estado.saidas["detect"].deteccoes
    return TeamOut(itens=classificar(estado.imagem(), deteccoes, paletas, estado.config))
