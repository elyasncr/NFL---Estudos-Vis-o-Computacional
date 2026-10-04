import cv2
import numpy as np

from nfl_vision.render import COR_DESCONHECIDO, COR_RESERVA, cores_de_exibicao, desenhar, rotulo
from nfl_vision.schemas import Jogador


def _jogador(track_id, time, numero, posicao=None):
    return Jogador(track_id=track_id, time=time, numero=numero, confianca_numero=0.9,
                   posicao=posicao, nome=None, frames_visiveis=[0],
                   corrigido_pelo_usuario=False)


def test_rotulos():
    assert rotulo(_jogador(0, "KC", 87, "TE")) == "KC 87 TE"
    assert rotulo(_jogador(1, "KC", None)) == "KC ?"
    assert rotulo(_jogador(2, None, None)) == "? ?"


def test_cor_do_time_mesmo_sem_numero_e_cinza_so_sem_time():
    img = np.full((300, 400, 3), (63, 107, 46), np.uint8)  # gramado
    jogadores = [_jogador(0, "KC", 87, "TE"), _jogador(1, "KC", None), _jogador(2, None, None)]
    caixas = {0: (50.0, 100.0, 110.0, 220.0), 1: (180.0, 100.0, 240.0, 220.0),
              2: (300.0, 100.0, 360.0, 220.0)}

    saida = desenhar(img, jogadores, caixas, {"KC": (55, 24, 227)})

    assert saida.shape == img.shape
    assert tuple(saida[200, 50]) == (55, 24, 227)       # KC 87
    assert tuple(saida[200, 180]) == (55, 24, 227)      # KC sem número: ainda na cor do time
    assert tuple(saida[200, 300]) == COR_DESCONHECIDO   # sem time: cinza
    assert (img == (63, 107, 46)).all()                 # original intacta


def test_caixa_tem_contorno_escuro_e_espessura_proporcional():
    img = np.full((1000, 1800, 3), (63, 107, 46), np.uint8)
    saida = desenhar(img, [_jogador(0, "KC", 87)], {0: (500.0, 400.0, 600.0, 700.0)},
                     {"KC": (55, 24, 227)})

    linha = saida[600, 490:515]
    coloridos = (linha == (55, 24, 227)).all(axis=1).sum()
    escuros = (linha.max(axis=1) < 40).sum()
    assert coloridos >= 3            # mais grossa que 2 px numa imagem grande
    assert escuros >= 2              # contorno escuro dos dois lados da cor


def test_cores_de_exibicao_contrastam_com_o_gramado_e_entre_si():
    paletas = {"CLE": ["#FF3C00", "#311D00"], "CIN": ["#FB4F14", "#000000"]}
    cores = cores_de_exibicao(paletas)
    assert cores["CLE"] == (0, 60, 255)      # laranja (o marrom some no gramado)
    assert cores["CIN"] == COR_RESERVA       # laranja igual ao do CLE e preto: reserva


def test_cores_de_exibicao_usa_a_segunda_cor_quando_a_primeira_some():
    cores = cores_de_exibicao({"PIT": ["#000000", "#FFB612"], "KC": ["#E31837", "#FFB612"]})
    assert cores["PIT"] == (18, 182, 255)    # amarelo, não preto
    assert cores["KC"] == (55, 24, 227)      # vermelho, diferente do amarelo do PIT


def _faixa_do_rotulo(img, y1, texto):
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    topo = max(y1 - th - 4, 0)  # imagens pequenas: padding de 2 px em cada lado
    return img[topo:topo + th + 4], tw


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
