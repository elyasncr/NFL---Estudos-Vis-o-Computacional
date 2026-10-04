import pytest
from pydantic import ValidationError

from nfl_vision.config import Config


def test_padroes_do_detector():
    cfg = Config()
    assert cfg.detector_tipo == "rfdetr"
    assert cfg.detector_modelo_rfdetr == "base"
    assert cfg.detector_resolucao % 56 == 0
    assert 0.0 < cfg.detector_conf < 1.0
    assert cfg.detector_pesos == "yolo11m.pt" and cfg.detector_imgsz == 1280  # YOLO continua


def test_manifest_antigo_sem_campos_novos_carrega_com_padroes():
    antigo = {"detector_pesos": "yolo11m.pt", "detector_imgsz": 1280, "detector_conf": 0.25,
              "device": "cuda:0", "campo_removido_no_futuro": 1}
    cfg = Config.model_validate(antigo)
    assert cfg.detector_tipo == "rfdetr"
    assert cfg.detector_resolucao == Config().detector_resolucao
    assert cfg.detector_conf == 0.25  # o que estava gravado vale


@pytest.mark.parametrize("campos", [
    {"detector_tipo": "detr"},
    {"detector_resolucao": 900},
    {"detector_resolucao": 0},
])
def test_valores_invalidos(campos):
    with pytest.raises(ValidationError):
        Config(**campos)


def test_resolucao_multiplo_de_56_aceita():
    assert Config(detector_resolucao=560).detector_resolucao == 560
