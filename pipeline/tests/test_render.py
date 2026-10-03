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
