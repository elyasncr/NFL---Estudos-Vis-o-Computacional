import cv2
import numpy as np

from nfl_vision.render import COR_DESCONHECIDO, desenhar, rotulo
from nfl_vision.schemas import Jogador


def _jogador(track_id, time, numero, posicao=None):
    return Jogador(track_id=track_id, time=time, numero=numero, confianca_numero=0.9,
                   posicao=posicao, nome=None, frames_visiveis=[0],
                   corrigido_pelo_usuario=False)


def test_rotulos():
    assert rotulo(_jogador(0, "KC", 87, "TE")) == "KC 87 TE"
    assert rotulo(_jogador(1, "KC", None)) == "KC ?"
    assert rotulo(_jogador(2, None, None)) == "? ?"


def test_desenha_na_cor_do_time_e_desconhecido_em_cinza():
    img = np.zeros((300, 300, 3), np.uint8)
    jogadores = [_jogador(0, "KC", 87, "TE"), _jogador(1, "KC", None)]
    caixas = {0: (50.0, 100.0, 110.0, 220.0), 1: (180.0, 100.0, 240.0, 220.0)}

    saida = desenhar(img, jogadores, caixas, {"KC": (55, 24, 227)})

    assert saida.shape == img.shape
    assert tuple(saida[200, 50]) == (55, 24, 227)       # borda esquerda do KC 87
    assert tuple(saida[200, 180]) == COR_DESCONHECIDO   # borda do desconhecido
    assert not img.any()                                # original intacta


def _faixa_do_rotulo(img, y1, texto):
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    topo = max(y1 - th - 6, 0)
    return img[topo:topo + th + 6], tw


def test_rotulo_na_borda_direita_fica_dentro_da_imagem():
    img = np.zeros((300, 300, 3), np.uint8)
    cor = (55, 24, 227)
    caixas = {0: (250.0, 100.0, 300.0, 220.0)}

    saida = desenhar(img, [_jogador(0, "KC", 87, "TE")], caixas, {"KC": cor})

    faixa, tw = _faixa_do_rotulo(saida, 100, "KC 87 TE")
    colunas = np.where((faixa == cor).all(axis=2).any(axis=0))[0]
    assert colunas.min() < 250                     # rótulo deslocado para a esquerda
    assert colunas.min() == 300 - tw - 4           # inteiro: começa onde cabe
    assert colunas.max() == 299


def test_texto_preto_em_cor_clara_e_branco_em_cor_escura():
    img = np.zeros((300, 300, 3), np.uint8)
    clara, escura = (200, 230, 255), (55, 24, 227)
    caixas = {0: (20.0, 100.0, 80.0, 220.0), 1: (150.0, 100.0, 210.0, 220.0)}
    jogadores = [_jogador(0, "AAA", 1), _jogador(1, "BBB", 2)]

    saida = desenhar(img, jogadores, caixas, {"AAA": clara, "BBB": escura})

    faixa, tw = _faixa_do_rotulo(saida, 100, "AAA 1")
    rot_claro = faixa[:, 20:20 + tw + 4]
    rot_escuro = faixa[:, 150:150 + tw + 4]
    assert (rot_claro < 40).all(axis=2).any()
    assert not (rot_claro > 250).all(axis=2).any()
    assert (rot_escuro > 250).all(axis=2).any()
