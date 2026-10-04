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

    # recorte do tronco (fração da caixa): medido em data/avaliacoes/medicao-time-recorte.json
    # (CIN x CLE, split test, 159 caixas rotuladas, limiar_time 0,6 nessa medição) — 20-55%/25-75%
    # tem a maior soma de acurácia entre RF-DETR e YOLO (0,817 + 0,869); isoladamente, 15-45%/25-75%
    # é melhor só para o RF-DETR (0,836 contra 0,817). Colunas mais estreitas evitam braço e gramado
    # nas caixas mais justas do RF-DETR. Número de dentro da amostra (mesmo gabarito e um só jogo);
    # ver spec §6 (team, passo 1) e §9 (eval time) para a ressalva completa.
    tronco_altura: tuple[float, float] = (0.20, 0.55)
    tronco_largura: tuple[float, float] = (0.25, 0.75)
    # medido em data/avaliacoes/time-20261004-161852.json (RF-DETR) e time-20261004-162006.json
    # (YOLO), mesmo gabarito acima: 0,9 dá acurácia 0,943 (RF-DETR) e 0,951 (YOLO), cobertura 0,791
    # e 0,836 das caixas casadas — mas, de ponta a ponta, só 87/159 e 102/159 dos jogadores
    # rotulados chegam a um time atribuído (0,55 e 0,64; campo cobertura_total de eval time). Meta
    # do SDD (acurácia >= 0,95) NÃO está demonstrada: precisão antes de cobertura, mas os números
    # são de dentro da amostra (mesmo gabarito de 159 caixas usado para escolher geometria e
    # limiar, um só jogo; IC95% de Wilson ~[0,87; 0,98] e ~[0,89; 0,98]) — falta confirmar noutro
    # jogo sem reajustar. Com grupo único (só um time visível no recorte), a confiança
    # 1 - ΔE/50 raramente cruza 0,9: ver spec §6 (team, passo 7) e §9 (eval time).
    limiar_time: float = 0.90
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

    @field_validator("tronco_altura", "tronco_largura")
    @classmethod
    def _fracao_valida(cls, valor: tuple[float, float]) -> tuple[float, float]:
        a, b = valor
        if not 0 <= a < b <= 1:
            raise ValueError("a fração do tronco deve ser 0 <= início < fim <= 1")
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
