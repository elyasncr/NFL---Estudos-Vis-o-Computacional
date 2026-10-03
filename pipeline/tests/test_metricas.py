import pytest

from nfl_vision.eval.metricas import average_precision, fracao_casada, iou

A = (0.0, 0.0, 10.0, 10.0)
B = (5.0, 0.0, 15.0, 10.0)
LONGE = (100.0, 100.0, 110.0, 110.0)


def test_iou():
    assert iou(A, A) == 1.0
    assert iou(A, B) == pytest.approx(50 / 150)
    assert iou(A, LONGE) == 0.0


def test_ap_perfeito():
    gts = {0: [A], 1: [LONGE]}
    preds = [(0, 0.9, A), (1, 0.8, LONGE)]
    assert average_precision(preds, gts) == pytest.approx(1.0)


def test_ap_com_falso_positivo_mais_confiante():
    gts = {0: [A]}
    preds = [(0, 0.9, LONGE), (0, 0.8, A)]
    assert average_precision(preds, gts) == pytest.approx(0.5)


def test_ap_duplicata_conta_como_falso_positivo():
    gts = {0: [A]}
    preds = [(0, 0.9, A), (0, 0.8, A)]
    assert average_precision(preds, gts) == pytest.approx(1.0)


def test_ap_sem_predicoes_ou_sem_gt():
    assert average_precision([], {0: [A]}) == 0.0
    assert average_precision([(0, 0.9, A)], {0: []}) == 0.0


def test_fracao_casada():
    alvos = {0: [A, LONGE]}
    assert fracao_casada(alvos, {0: [A]}) == 0.5
    assert fracao_casada({0: []}, {0: [A]}) == 0.0
