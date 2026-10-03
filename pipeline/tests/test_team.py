import cv2
import numpy as np

from nfl_vision.config import Config
from nfl_vision.cores import bgr_para_lab, hex_para_lab
from nfl_vision.schemas import Deteccao
from nfl_vision.stages.team import (
    agrupar, classificar, eh_arbitro, eh_branco, mapear_grupos,
)
from sintetico import AZUL_BUF, VERMELHO_KC, arbitro, campo, jogador

CFG = Config()
PALETAS = {
    "KC": [hex_para_lab("#E31837"), hex_para_lab("#FFB612")],
    "BUF": [hex_para_lab("#00338D"), hex_para_lab("#C60C30")],
}
VERMELHO = hex_para_lab("#E31837")
AZUL = hex_para_lab("#00338D")
BRANCO = hex_para_lab("#FFFFFF")


def _listrado(h=48, w=36, largura=4):
    img = np.zeros((h, w, 3), np.uint8)
    for x in range(w):
        if (x // largura) % 2:
            img[:, x] = 255
    return img


def test_listras_pretas_e_brancas_sao_arbitro():
    assert eh_arbitro(_listrado())
    assert not eh_arbitro(np.full((48, 36, 3), VERMELHO_KC, np.uint8))


def test_camisa_branca_com_numero_preto_nao_e_arbitro():
    img = np.full((48, 36, 3), 255, np.uint8)
    img[10:38, 7:29] = 0  # bloco do número: ~36% da área
    assert not eh_arbitro(img)


def test_camisa_preta_com_numero_branco_nao_e_arbitro():
    img = np.zeros((60, 60, 3), np.uint8)
    cv2.putText(img, "88", (2, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 6)
    assert not eh_arbitro(img)


def test_camisa_lisa_nao_e_arbitro():
    assert not eh_arbitro(np.full((48, 36, 3), VERMELHO_KC, np.uint8))


def test_eh_branco():
    assert eh_branco(BRANCO)
    assert not eh_branco(VERMELHO)


def test_agrupar_dois_grupos_e_grupo_unico():
    rotulos, centros = agrupar(np.array([VERMELHO, VERMELHO, AZUL, AZUL]), CFG)
    assert len(centros) == 2
    assert rotulos[0] == rotulos[1] != rotulos[2] == rotulos[3]

    rotulos, centros = agrupar(np.array([VERMELHO, VERMELHO + 1, VERMELHO - 1]), CFG)
    assert len(centros) == 1
    assert set(rotulos) == {0}


def test_mapear_grupos_por_paleta():
    assert mapear_grupos(np.array([AZUL, VERMELHO]), PALETAS) == {0: "BUF", 1: "KC"}


def test_grupo_branco_fica_com_o_outro_time():
    assert mapear_grupos(np.array([BRANCO, AZUL]), PALETAS) == {1: "BUF", 0: "KC"}


def test_grupo_unico_branco_e_desconhecido():
    assert mapear_grupos(np.array([BRANCO]), PALETAS) == {0: None}
    assert mapear_grupos(np.array([AZUL]), PALETAS) == {0: "BUF"}


def test_classificar_imagem_sintetica():
    img = campo()
    caixas = [
        jogador(img, 100, 300, VERMELHO_KC),
        jogador(img, 200, 300, VERMELHO_KC),
        jogador(img, 400, 300, AZUL_BUF),
        jogador(img, 500, 300, AZUL_BUF),
        arbitro(img, 650, 300),
    ]
    dets = [Deteccao(det_id=i, bbox=b, confianca=0.9) for i, b in enumerate(caixas)]
    dets.append(Deteccao(det_id=5, bbox=(0, 0, 10, 10), confianca=0.9, descartado=True,
                         motivo_descarte="fora_de_campo"))

    itens = {t.det_id: t for t in classificar(img, dets, PALETAS, CFG)}

    assert [itens[i].time for i in range(4)] == ["KC", "KC", "BUF", "BUF"]
    assert all(itens[i].confianca >= 0.99 for i in range(4))
    assert itens[4].arbitro and itens[4].time is None
    assert 5 not in itens


def test_camisa_verde_nao_e_removida_como_gramado():
    from sintetico import VERDE_GB

    paletas = {"GB": [hex_para_lab("#203731"), hex_para_lab("#FFB612")], "KC": PALETAS["KC"]}
    img = campo()
    caixas = [
        jogador(img, 100, 300, VERDE_GB),
        jogador(img, 200, 300, VERDE_GB),
        jogador(img, 400, 300, VERMELHO_KC),
        jogador(img, 500, 300, VERMELHO_KC),
    ]
    dets = [Deteccao(det_id=i, bbox=b, confianca=0.9) for i, b in enumerate(caixas)]

    itens = classificar(img, dets, paletas, CFG)

    assert [t.time for t in itens] == ["GB", "GB", "KC", "KC"]


def test_pixels_uteis_remove_gramado_so_quando_minoria():
    from sintetico import VERDE, VERDE_GB

    from nfl_vision.stages.team import pixels_uteis

    recorte = np.zeros((40, 40, 3), np.uint8)
    recorte[:, :] = VERMELHO_KC
    recorte[:, :12] = VERDE  # 30% de gramado nas laterais
    assert len(pixels_uteis(recorte, CFG)) == 40 * 28

    camisa_verde = np.full((40, 40, 3), VERDE_GB, np.uint8)
    assert len(pixels_uteis(camisa_verde, CFG)) == 40 * 40


MARINHO = hex_para_lab("#0B162A")


def test_outlier_sozinho_nao_forma_grupo():
    rotulos, centros = agrupar(np.array([VERMELHO] * 5 + [MARINHO]), CFG)
    assert len(centros) == 1
    assert set(rotulos) == {0}


def test_outlier_fica_sem_time_e_vermelhos_ficam_com_kc():
    img = campo()
    caixas = [jogador(img, 20 + 110 * i, 300, VERMELHO_KC) for i in range(5)]
    caixas.append(jogador(img, 600, 300, (42, 22, 11)))  # #0B162A em BGR
    dets = [Deteccao(det_id=i, bbox=b, confianca=0.9) for i, b in enumerate(caixas)]

    itens = classificar(img, dets, PALETAS, CFG)

    assert [t.time for t in itens[:5]] == ["KC"] * 5
    assert all(t.confianca >= 0.8 for t in itens[:5])
    assert itens[5].confianca < 0.60 and itens[5].time is None


def test_cor_dominante_amostra_muitos_pixels(monkeypatch):
    from nfl_vision.cores import delta_e
    from nfl_vision.stages import team

    px = np.array([VERMELHO] * 70_000 + [AZUL] * 30_000)
    tamanhos = []
    fit_original = team.KMeans.fit

    def fit_espiao(self, x, *a, **k):
        tamanhos.append(len(x))
        return fit_original(self, x, *a, **k)

    monkeypatch.setattr(team.KMeans, "fit", fit_espiao)

    assert delta_e(team.cor_dominante(px), VERMELHO) < 1
    assert tamanhos == [3000]
