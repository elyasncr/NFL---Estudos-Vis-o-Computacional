"""Contratos de dados entre as etapas do pipeline e a saída final."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 em pixels
Lab = tuple[float, float, float]


class Contexto(BaseModel):
    temporada: int = Field(ge=2002)
    semana: int = Field(ge=1, le=22)
    times: tuple[str, str]

    @model_validator(mode="after")
    def _times_diferentes(self) -> "Contexto":
        if self.times[0] == self.times[1]:
            raise ValueError("os dois times do contexto devem ser diferentes")
        return self


class IngestOut(BaseModel):
    caminho: str
    largura: int
    altura: int
    sha256: str


class Deteccao(BaseModel):
    det_id: int
    bbox: BBox
    confianca: float
    classe: Literal["pessoa"] = "pessoa"
    descartado: bool = False
    motivo_descarte: Literal["fora_de_campo", "pequeno"] | None = None


class DetectOut(BaseModel):
    deteccoes: list[Deteccao]
    pesos_sha256: str | None = None


class TimeDet(BaseModel):
    det_id: int
    time: str | None
    confianca: float
    cor_lab: Lab | None = None
    arbitro: bool = False


class TeamOut(BaseModel):
    itens: list[TimeDet]


class NumeroDet(BaseModel):
    det_id: int
    numero: int | None = Field(default=None, ge=0, le=99)
    confianca: float
    texto_bruto: str | None = None


class JerseyOut(BaseModel):
    itens: list[NumeroDet]


MotivoRoster = Literal["ok", "sem_numero", "sem_time", "numero_fora_do_roster", "ambiguo"]


class RosterDet(BaseModel):
    det_id: int
    posicao: str | None = None
    nome: str | None = None
    gsis_id: str | None = None
    motivo: MotivoRoster


class RosterOut(BaseModel):
    itens: list[RosterDet]


class Correcao(BaseModel):
    det_id: int
    time: str | None = None
    numero: int | None = Field(default=None, ge=0, le=99)
    timestamp: str


class Jogador(BaseModel):
    track_id: int
    time: str | None
    numero: Annotated[int, Field(ge=0, le=99)] | None
    confianca_numero: float
    posicao: str | None
    nome: str | None
    frames_visiveis: list[int]
    corrigido_pelo_usuario: bool


class Analise(BaseModel):
    analise_id: str
    midia: dict[str, str | int | float]
    contexto: Contexto
    modelos: dict[str, str]
    jogadores: list[Jogador]
