"""Parâmetros do pipeline. A configuração usada é gravada em cada análise."""

from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# variante do RF-DETR -> classe do pacote `rfdetr`. Só "base" é suportada por ora; a regra
# de `detector_resolucao` (múltiplo de 56) é específica do RFDETRBase.
VARIANTES_RFDETR = {"base": "RFDETRBase"}


class Config(BaseModel):
    # Campos desconhecidos (removidos numa versão futura) não quebram a leitura
    # de um manifest antigo; campos novos ausentes usam o valor padrão.
    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _detector_tipo_legado(cls, dados: Any) -> Any:
        """Manifest de antes do RF-DETR (sem detector_tipo): só existia YOLO.

        Só se aplica a um dict que parece um manifest gravado (tem outra chave típica
        de config, ex. detector_pesos) e sem detector_tipo; `Config()` sem argumentos
        (dict vazio) continua caindo no padrão da classe (rfdetr).
        """
        if (isinstance(dados, dict) and dados and "detector_tipo" not in dados
                and "detector_pesos" in dados):
            dados = {**dados, "detector_tipo": "yolo"}
        return dados

    # "rfdetr" (padrão) ou "yolo"
    detector_tipo: Literal["rfdetr", "yolo"] = "rfdetr"
    detector_modelo_rfdetr: str = "base"  # variante do RF-DETR; ver VARIANTES_RFDETR
    # medida em data/avaliacoes/medicao-rfdetr-resolucao.json (split test, CIN×CLE)
    # múltiplo de 56: regra do RFDETRBase (única variante suportada; ver VARIANTES_RFDETR)
    detector_resolucao: int = 1120  # lado de entrada do RF-DETR; múltiplo de 56 (medido)
    # limiar de confiança do detector. None: usa o padrão do detector_tipo (ver limiar()),
    # que é o de maior F1 medido para cada um; definido aqui, vale para qualquer detector_tipo.
    detector_conf: float | None = None
    # só YOLO
    detector_pesos: str = "yolo11m.pt"
    detector_imgsz: int = 1280
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

    @field_validator("detector_resolucao")
    @classmethod
    def _multiplo_de_56(cls, valor: int) -> int:
        if valor <= 0 or valor % 56:
            raise ValueError("detector_resolucao deve ser um múltiplo positivo de 56")
        return valor

    @field_validator("detector_modelo_rfdetr")
    @classmethod
    def _variante_rfdetr_conhecida(cls, valor: str) -> str:
        if valor not in VARIANTES_RFDETR:
            raise ValueError(
                f"variante do RF-DETR desconhecida: '{valor}'; use uma de: "
                f"{', '.join(VARIANTES_RFDETR)}")
        return valor

    def limiar(self) -> float:
        """Confiança mínima do detector: `detector_conf`, se definido; senão o padrão medido
        de `detector_tipo` (data/avaliacoes/medicao-rfdetr-resolucao.json): 0.4 para o
        RF-DETR (maior F1 no split test) e 0.25 para o YOLO (padrão de antes do RF-DETR)."""
        if self.detector_conf is not None:
            return self.detector_conf
        return 0.4 if self.detector_tipo == "rfdetr" else 0.25


# única fonte dos detectores aceitos: o Literal do campo detector_tipo (cli.py e
# stages/detect.py importam daqui, em vez de repetir a tupla)
DETECTORES: tuple[str, ...] = get_args(Config.model_fields["detector_tipo"].annotation)
