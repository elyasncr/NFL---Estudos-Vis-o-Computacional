import cv2
import numpy as np
import pytest

from nfl_vision.config import Config
from nfl_vision.schemas import Deteccao, TimeDet
from nfl_vision.stages.jersey import (
    Leitura, escolher_numero, juntar_digitos, ler_numeros, recorte_numero,
)


def L(texto, score, caixa=None):
    return Leitura(texto, score, caixa)


def test_prefere_numero_completo_a_digito_contido():
    leituras = [L("KC", 0.99), L("87", 0.91), L("8", 0.95), L("187", 0.97)]
    assert escolher_numero(leituras, 0.60) == (87, 0.91, "87")


def test_digito_bem_mais_confiante_vence():
    assert escolher_numero([L("87", 0.70), L("8", 0.95)], 0.60) == (8, 0.95, "8")


def test_digito_nao_contido_vence_pela_confianca():
    assert escolher_numero([L("87", 0.91), L("5", 0.95)], 0.60) == (5, 0.95, "5")


def test_junta_digitos_vizinhos():
    leituras = [L("8", 0.95, (10, 10, 30, 50)), L("7", 0.93, (34, 10, 54, 50))]
    assert escolher_numero(leituras, 0.60) == (87, 0.93, "87")


def test_junta_digitos_ordena_por_x():
    juntas = juntar_digitos([L("7", 0.93, (34, 10, 54, 50)), L("8", 0.95, (10, 10, 30, 50))])
    assert L("87", 0.93, (10, 10, 54, 50)) in juntas
    assert len(juntas) == 3


def test_nao_junta_digitos_distantes():
    leituras = [L("8", 0.95, (10, 10, 30, 50)), L("7", 0.93, (100, 10, 120, 50))]
    assert [l.texto for l in juntar_digitos(leituras)] == ["8", "7"]
    assert escolher_numero(leituras, 0.60)[0] == 8


def test_nao_junta_linhas_ou_alturas_diferentes():
    outra_linha = [L("8", 0.95, (10, 10, 30, 50)), L("7", 0.93, (34, 40, 54, 80))]
    outra_altura = [L("8", 0.95, (10, 10, 30, 50)), L("7", 0.93, (34, 10, 54, 25))]
    assert len(juntar_digitos(outra_linha)) == 2
    assert len(juntar_digitos(outra_altura)) == 2


def test_normaliza_texto():
    assert escolher_numero([L("#87", 0.9)], 0.60) == (87, 0.9, "87")
    assert escolher_numero([L("8 7", 0.9)], 0.60) == (87, 0.9, "87")
    assert escolher_numero([L("-87.", 0.9)], 0.60) == (87, 0.9, "87")


def test_zero_a_esquerda_e_invalido():
    assert escolher_numero([L("07", 0.9)], 0.60)[0] is None
    assert escolher_numero([L("00", 0.9)], 0.60)[0] is None
    assert escolher_numero([L("0", 0.9)], 0.60) == (0, 0.9, "0")


def test_abaixo_do_limiar_vira_desconhecido():
    assert escolher_numero([L("87", 0.40)], 0.60) == (None, 0.40, "87")


def test_sem_texto_valido():
    assert escolher_numero([L("KC", 0.99)], 0.60) == (None, 0.0, "KC")
    assert escolher_numero([], 0.60) == (None, 0.0, None)


def test_recorte_e_ampliado():
    img = np.zeros((200, 200, 3), np.uint8)
    recorte = recorte_numero(img, (0, 0, 40, 80), 128)
    assert recorte.shape[0] >= 128


class LeitorFalso:
    def __init__(self, respostas):
        self.respostas = list(respostas)

    def ler(self, img):
        return self.respostas.pop(0)


def test_ignora_arbitros_e_descartados():
    img = np.zeros((300, 300, 3), np.uint8)
    dets = [
        Deteccao(det_id=0, bbox=(10, 10, 70, 130), confianca=0.9),
        Deteccao(det_id=1, bbox=(100, 10, 160, 130), confianca=0.9),
        Deteccao(det_id=2, bbox=(200, 10, 260, 130), confianca=0.9, descartado=True,
                 motivo_descarte="pequeno"),
    ]
    times = [TimeDet(det_id=0, time="KC", confianca=1.0),
             TimeDet(det_id=1, time=None, confianca=1.0, arbitro=True)]

    itens = ler_numeros(img, dets, times, LeitorFalso([[L("87", 0.9)]]), Config())

    assert [(n.det_id, n.numero) for n in itens] == [(0, 87)]


@pytest.mark.model
def test_paddle_le_numero_renderizado():
    from nfl_vision.stages.jersey import leitor_padrao

    img = np.full((200, 300, 3), 255, np.uint8)
    cv2.putText(img, "87", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 12)

    leituras = leitor_padrao("cpu").ler(img)
    numero, conf, _ = escolher_numero(leituras, 0.60)

    assert all(isinstance(l, Leitura) and l.caixa is not None for l in leituras)

    assert numero == 87
    assert conf >= 0.60
