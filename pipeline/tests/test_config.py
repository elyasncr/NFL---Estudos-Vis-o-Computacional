import pytest
from pydantic import ValidationError

from nfl_vision.config import Config


def test_padroes_do_detector():
    cfg = Config()
    assert cfg.detector_tipo == "rfdetr"
    assert cfg.detector_modelo_rfdetr == "base"
    assert cfg.detector_resolucao % 56 == 0
    assert cfg.detector_conf is None  # usa o padrão do detector escolhido; ver limiar()
    assert cfg.detector_pesos == "yolo11m.pt" and cfg.detector_imgsz == 1280  # YOLO continua


def test_limiar_usa_o_padrao_medido_de_cada_detector():
    assert Config().limiar() == 0.4  # RF-DETR: maior F1 no split test (medido)
    assert Config(detector_tipo="yolo").limiar() == 0.25  # YOLO: padrão de antes do RF-DETR


def test_limiar_respeita_detector_conf_quando_definido():
    assert Config(detector_conf=0.6).limiar() == 0.6
    assert Config(detector_tipo="yolo", detector_conf=0.6).limiar() == 0.6


def test_manifest_antigo_sem_campos_novos_carrega_com_padroes():
    """Manifest de antes do RF-DETR: sem detector_tipo, só existia YOLO."""
    antigo = {"detector_pesos": "yolo11m.pt", "detector_imgsz": 1280, "detector_conf": 0.25,
              "device": "cuda:0", "campo_removido_no_futuro": 1}
    cfg = Config.model_validate(antigo)
    assert cfg.detector_tipo == "yolo"
    assert cfg.detector_resolucao == Config().detector_resolucao
    assert cfg.detector_conf == 0.25  # o que estava gravado vale


def test_config_vazia_continua_rfdetr():
    assert Config().detector_tipo == "rfdetr"
    assert Config.model_validate({}).detector_tipo == "rfdetr"


def test_so_detector_conf_sem_cara_de_manifest_antigo_continua_rfdetr():
    # não tem detector_pesos (nem outra chave típica de manifest gravado): não é tratado como legado
    assert Config(detector_conf=0.3).detector_tipo == "rfdetr"


@pytest.mark.parametrize("campos", [
    {"detector_tipo": "detr"},
    {"detector_resolucao": 900},
    {"detector_resolucao": 0},
    {"detector_modelo_rfdetr": "large"},
])
def test_valores_invalidos(campos):
    with pytest.raises(ValidationError):
        Config(**campos)


def test_resolucao_multiplo_de_56_aceita():
    assert Config(detector_resolucao=560).detector_resolucao == 560


def test_variante_do_rfdetr_invalida_lista_as_suportadas():
    with pytest.raises(ValidationError, match="use uma de: base"):
        Config(detector_modelo_rfdetr="large")


def test_detectores_vem_do_literal_de_detector_tipo():
    from nfl_vision.config import DETECTORES

    assert DETECTORES == ("rfdetr", "yolo")


def test_detectores_tem_uma_unica_fonte_para_cli_e_detect():
    from nfl_vision import cli
    from nfl_vision.config import DETECTORES
    from nfl_vision.stages import detect

    assert cli.DETECTORES is DETECTORES
    assert detect.DETECTORES is DETECTORES
