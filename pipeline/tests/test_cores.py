import numpy as np
import pytest

from nfl_vision import paths
from nfl_vision.config import Config
from nfl_vision.cores import (
    bgr_para_lab, delta_e, fracao_gramado, hex_para_bgr, hex_para_lab,
)

CFG = Config()


def _imagem(cor, tamanho=(20, 20)):
    return np.full((*tamanho, 3), cor, np.uint8)


def test_fracao_gramado_verde_e_vermelho():
    assert fracao_gramado(_imagem((40, 140, 40)), CFG.gramado_hsv_min, CFG.gramado_hsv_max) == 1.0
    assert fracao_gramado(_imagem((55, 24, 227)), CFG.gramado_hsv_min, CFG.gramado_hsv_max) == 0.0


def test_fracao_gramado_vazio_e_zero():
    vazio = np.empty((0, 0, 3), np.uint8)
    assert fracao_gramado(vazio, CFG.gramado_hsv_min, CFG.gramado_hsv_max) == 0.0


def test_conversoes_lab():
    branco = bgr_para_lab(np.array([[255, 255, 255]], np.uint8))[0]
    assert branco[0] == pytest.approx(100, abs=0.5)
    assert hex_para_lab("#FFFFFF")[0] == pytest.approx(100, abs=0.5)
    assert delta_e(hex_para_lab("#E31837"), hex_para_lab("#E31837")) == pytest.approx(0)
    assert delta_e(hex_para_lab("#E31837"), hex_para_lab("#00338D")) > 30


def test_hex_para_bgr():
    assert hex_para_bgr("#E31837") == (55, 24, 227)


def test_dados_dir_respeita_variavel(monkeypatch, tmp_path):
    monkeypatch.setenv("NFL_VISION_DATA", str(tmp_path))
    assert paths.runs_dir() == tmp_path / "runs"
    assert paths.cache_dir() == tmp_path / "cache"
