"""Parâmetros do pipeline. A configuração usada é gravada em cada análise."""

from pydantic import BaseModel


class Config(BaseModel):
    detector_pesos: str = "yolo11m.pt"
    detector_imgsz: int = 1280
    detector_conf: float = 0.25
    device: str = "cuda:0"

    filtro_altura_rel: float = 0.4
    campo_area_min: float = 0.05
    campo_margem_rel: float = 0.02
    campo_close_altura_rel: float = 0.5
    gramado_hsv_min: tuple[int, int, int] = (35, 40, 40)
    gramado_hsv_max: tuple[int, int, int] = (85, 255, 255)
    mascara_gramado_max_tronco: float = 0.6

    limiar_time: float = 0.60
    delta_e_grupo_unico: float = 15.0

    limiar_numero: float = 0.60
    numero_altura_min: int = 128
    ocr_device: str = "cpu"
