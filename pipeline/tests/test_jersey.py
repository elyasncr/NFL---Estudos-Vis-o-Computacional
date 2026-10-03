import cv2
import numpy as np
import pytest

from nfl_vision.config import Config
from nfl_vision.schemas import Deteccao, TimeDet
from nfl_vision.stages.jersey import escolher_numero, ler_numeros, recorte_numero


def test_escolhe_maior_confianca_entre_textos_validos():
    leituras = [("KC", 0.99), ("87", 0.91), ("8", 0.95), ("187", 0.97)]
    assert escolher_numero(leituras, 0.60) == (8, 0.95, "8")


def test_abaixo_do_limiar_vira_desconhecido():
    assert escolher_numero([("87", 0.40)], 0.60) == (None, 0.40, "87")


def test_sem_texto_valido():
    assert escolher_numero([("KC", 0.99)], 0.60) == (None, 0.0, "KC")
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

    itens = ler_numeros(img, dets, times, LeitorFalso([[("87", 0.9)]]), Config())

    assert [(n.det_id, n.numero) for n in itens] == [(0, 87)]


@pytest.mark.model
def test_paddle_le_numero_renderizado():
    from nfl_vision.stages.jersey import leitor_padrao

    img = np.full((200, 300, 3), 255, np.uint8)
    cv2.putText(img, "87", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 12)

    numero, conf, _ = escolher_numero(leitor_padrao("cpu").ler(img), 0.60)

    assert numero == 87
    assert conf >= 0.60
