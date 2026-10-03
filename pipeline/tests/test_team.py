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


def test_listras_pretas_e_brancas_sao_arbitro():
    listrado = np.array([[0, 0, 0], [255, 255, 255]] * 50, np.uint8)
    liso = np.array([[55, 24, 227]] * 100, np.uint8)
    assert eh_arbitro(bgr_para_lab(listrado))
    assert not eh_arbitro(bgr_para_lab(liso))


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
