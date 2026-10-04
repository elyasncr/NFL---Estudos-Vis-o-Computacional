"""Fontes do dataset de treino do detector e o mapeamento das suas classes para `player`."""

import re
from dataclasses import dataclass
from pathlib import Path

from nfl_vision.eval.datasets import baixar

FORMATO = "yolov11"


class FonteIndisponivel(RuntimeError):
    """Download falhou, sem acesso ou sem ROBOFLOW_API_KEY."""


class MapeamentoInvalido(ValueError):
    """Classe declarada no mapeamento não existe no data.yaml da fonte."""


@dataclass(frozen=True)
class Fonte:
    nome: str                        # identificador na CLI e prefixo dos arquivos
    workspace: str
    projeto: str
    versao: int
    classes_player: tuple[str, ...]  # classes da fonte que viram `player`; as demais são descartadas
    externa: bool = True

    def pasta(self, datasets_dir: Path) -> Path:
        """Pasta do download (o mesmo nome que `eval.datasets.baixar` usa)."""
        return datasets_dir / f"{self.projeto}-v{self.versao}-{FORMATO}"


BASE = Fonte("base", "elyas-carvalho", "nfl-player-model-ymsui", 1, ("player",), externa=False)
EXTERNAS = (
    Fonte("pitchcamera", "pitchcamera", "football-player-referee", 7, ("players",)),
    Fonte("fhtw", "fh-technikum-wien-m15r2", "american-football-player-detection", 3,
          ("american-football-players",)),
    Fonte("evzn", "evzn", "american-football-analyst", 4, ("football-players",)),
)
POR_NOME = {f.nome: f for f in (BASE, *EXTERNAS)}

# Base: 6 clipes (3 jogadas x 2 câmeras). Separar por quadro vazaria quadros vizinhos.
PREFIXO_TESTE = "cin_cle_"
CLIPE_VALID = "tb_atl_wk1_penix_pass_all22"
_CLIPE = re.compile(r"^(?P<clipe>.+)_\d{5}_jpg\.rf\.")


def clipe_base(nome_arquivo: str) -> str:
    m = _CLIPE.match(nome_arquivo)
    if m is None:
        raise ValueError(
            f"nome fora do padrão <clipe>_<5 dígitos>_jpg.rf.<hash>: {nome_arquivo}")
    return m["clipe"]


def split_base(nome_arquivo: str) -> str:
    clipe = clipe_base(nome_arquivo)
    if clipe.startswith(PREFIXO_TESTE):
        return "test"
    if clipe == CLIPE_VALID:
        return "valid"
    return "train"


def _descricao(fonte: Fonte) -> str:
    return f"fonte '{fonte.nome}' ({fonte.workspace}/{fonte.projeto} v{fonte.versao})"


def validar_mapeamento(fonte: Fonte, nomes: list) -> None:
    disponiveis = [str(n) for n in nomes]
    faltando = [c for c in fonte.classes_player if c not in disponiveis]
    if faltando:
        raise MapeamentoInvalido(
            f"{_descricao(fonte)}: classe(s) {', '.join(faltando)} não existe(m) no data.yaml; "
            f"classes disponíveis: {', '.join(disponiveis)}")


def obter(fonte: Fonte, datasets_dir: Path) -> Path:
    """Pasta da fonte em formato YOLO, baixando do Roboflow se ainda não existir."""
    try:
        return baixar(fonte.workspace, fonte.projeto, fonte.versao, FORMATO, datasets_dir)
    except Exception as exc:
        raise FonteIndisponivel(f"{_descricao(fonte)}: {type(exc).__name__}: {exc}") from exc
