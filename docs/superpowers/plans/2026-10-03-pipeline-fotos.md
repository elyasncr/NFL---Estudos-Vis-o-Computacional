# Pipeline de identificação em fotos — Plano de implementação

> Nota: plano histórico; a spec (`docs/superpowers/specs/2026-10-03-pipeline-fotos-design.md`) e o código o substituem onde houver diferença (ex.: filtro de fora de campo, OCR, roster).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** CLI `nfl-vision` que recebe uma foto de partida da NFL e devolve cada jogador com time, número, posição e nome (`analise.json` + imagem anotada), com avaliação de detecção e de OCR.

**Architecture:** Pacote Python `nfl_vision` com cinco etapas puras (`ingest`, `detect`, `team`, `jersey`, `roster`) orquestradas por um runner genérico que grava a saída de cada etapa em `data/runs/<analise_id>/<etapa>.json` e permite retomar com `--from`. Modelos pesados (YOLO, PaddleOCR) ficam atrás de funções finas que os testes rápidos substituem por falsos; a lógica de visão (filtros, cores, agrupamento, escolha de número, roster) é testada com imagens sintéticas e fixtures.

**Tech Stack:** Python 3.12, uv, PyTorch cu128 (RTX 5070 Ti / Blackwell), Ultralytics YOLO11, OpenCV, scikit-learn, scikit-image, PaddleOCR 3.x, nflreadpy (Polars), Pydantic 2, Typer + Rich, pytest. Avaliação: roboflow, rfdetr, inference-sdk.

**Spec:** `docs/superpowers/specs/2026-10-03-pipeline-fotos-design.md`

## Convenções

- Todos os comandos rodam a partir de `pipeline/` (`cd pipeline`), num terminal Git Bash ou PowerShell.
- Testes rápidos: `uv run pytest`. Testes com modelos reais: `uv run pytest -m model`.
- Commits **sem** linhas `Co-Authored-By` ou menções ao Claude.
- Textos de interface, mensagens de erro e nomes de domínio em português, como no SDD.

## Mapa de arquivos

```
pipeline/
  pyproject.toml
  nfl_vision/
    __init__.py
    schemas.py        # contratos Pydantic entre etapas e saída final
    config.py         # parâmetros (limiares, modelos) gravados no manifest
    paths.py          # data/, runs/, cache/ (sobrescrevível por NFL_VISION_DATA)
    cores.py          # máscara de gramado, conversões LAB/hex/BGR, ΔE
    teams.py          # siglas, validação e cores oficiais (nflverse)
    runner.py         # Etapa, Estado, Runner, EtapaFalhou
    render.py         # desenho das caixas e rótulos
    montagem.py       # monta o analise.json a partir das saídas
    pipeline.py       # ETAPAS concretas, analisar / reprocessar / corrigir
    cli.py            # comandos analyze, correct, eval
    stages/
      __init__.py
      ingest.py  detect.py  team.py  jersey.py  roster.py
    eval/
      __init__.py
      metricas.py     # IoU, AP@0.5, fração casada
      datasets.py     # download Roboflow, leitores YOLO e por pastas
      preditores.py   # nosso detector, RF-DETR, modelo NFL do Roboflow
      detect.py       # avaliação de detecção
      jersey.py       # avaliação de OCR
  tests/
    conftest.py       # fixture `dados` (diretório temporário + caches)
    sintetico.py      # imagens sintéticas de campo, jogadores e árbitro
    test_*.py
```

---

### Task 1: Setup do ambiente e do pacote

**Files:**
- Create: `pipeline/pyproject.toml`
- Create: `pipeline/nfl_vision/__init__.py`
- Create: `pipeline/nfl_vision/stages/__init__.py`
- Create: `pipeline/nfl_vision/eval/__init__.py`
- Create: `pipeline/tests/test_ambiente.py`
- Modify: `docs/superpowers/specs/2026-10-03-pipeline-fotos-design.md` (§3 e §5)

- [ ] **Step 1: Instalar o uv (pedir confirmação ao usuário antes)**

uv não está instalado nesta máquina. Com a aprovação do usuário, no PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Abrir um novo terminal e conferir: `uv --version` → `uv 0.x.y`.

- [ ] **Step 2: Criar `pipeline/pyproject.toml`**

```toml
[project]
name = "nfl-vision"
version = "0.1.0"
description = "Identificação de jogadores da NFL em fotos"
requires-python = ">=3.12,<3.13"
dependencies = [
  "numpy>=1.26",
  "opencv-python>=4.10",
  "pillow>=10.4",
  "pydantic>=2.8",
  "pyyaml>=6.0",
  "scikit-learn>=1.5",
  "scikit-image>=0.24",
  "polars>=1.0",
  "nflreadpy>=0.1",
  "typer>=0.12",
  "rich>=13",
  "python-dotenv>=1.0",
  "torch>=2.7",
  "torchvision>=0.22",
  "ultralytics>=8.3",
]

[project.optional-dependencies]
ocr = ["paddleocr>=3.0", "paddlepaddle>=3.0"]
eval = ["roboflow>=1.1", "rfdetr>=1.2", "inference-sdk>=0.50"]

[project.scripts]
nfl-vision = "nfl_vision.cli:app"

[dependency-groups]
dev = ["pytest>=8"]

[tool.uv.sources]
torch = [{ index = "pytorch-cu128" }]
torchvision = [{ index = "pytorch-cu128" }]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["nfl_vision"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["model: testes que carregam modelos reais (lentos, precisam de GPU/rede)"]
addopts = "-m 'not model'"
```

- [ ] **Step 3: Criar os pacotes vazios**

`pipeline/nfl_vision/__init__.py`:

```python
"""Identificação de jogadores da NFL em fotos."""
```

`pipeline/nfl_vision/stages/__init__.py`:

```python
"""Etapas do pipeline: ingest, detect, team, jersey, roster."""
```

`pipeline/nfl_vision/eval/__init__.py`:

```python
"""Avaliação de detecção e de leitura de número."""
```

- [ ] **Step 4: Escrever o teste de ambiente (marcado `model`)**

`pipeline/tests/test_ambiente.py`:

```python
import pytest


@pytest.mark.model
def test_gpu_blackwell_disponivel():
    import torch

    assert torch.cuda.is_available(), "PyTorch não enxerga a GPU; confira o índice cu128"
    assert torch.cuda.get_device_capability(0) >= (12, 0)
    x = torch.ones(1024, device="cuda")
    assert float((x * 2).sum()) == 2048.0
```

- [ ] **Step 5: Instalar e rodar**

```bash
cd pipeline
uv sync
uv run pytest -m model tests/test_ambiente.py -v
```

Expected: `1 passed`. Se falhar com "no kernel image is available", o torch instalado não é cu128: conferir `uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"` → deve terminar em `+cu128` e `12.8`.

- [ ] **Step 6: Ajustar a spec à estrutura real**

Em `docs/superpowers/specs/2026-10-03-pipeline-fotos-design.md`, §3, substituir o bloco de estrutura por:

```
nfl-vision/                 (raiz do repositório)
  pipeline/
    pyproject.toml
    nfl_vision/
      schemas.py  config.py  paths.py  cores.py  teams.py
      runner.py  render.py  montagem.py  pipeline.py  cli.py
      stages/  ingest.py  detect.py  team.py  jersey.py  roster.py
      eval/    metricas.py  datasets.py  preditores.py  detect.py  jersey.py
    tests/                  # testes do pipeline (rápidos e @model)
  notebooks/                # exploração e avaliação; importam o pacote
  data/                     # ignorado pelo git
    runs/<analise_id>/
    cache/
    datasets/
  docs/
```

E na §5 acrescentar a linha `nfl-vision eval baixar --workspace <ws> --projeto <slug> --versao <n> --formato <yolov11|folder>` ao bloco de comandos.

- [ ] **Step 7: Commit**

```bash
cd ..
git add pipeline/pyproject.toml pipeline/uv.lock pipeline/nfl_vision pipeline/tests docs/superpowers/specs
git commit -m "chore: setup do pacote nfl_vision com uv e PyTorch cu128"
```

---

### Task 2: Contratos de dados (`schemas.py`)

**Files:**
- Create: `pipeline/nfl_vision/schemas.py`
- Test: `pipeline/tests/test_schemas.py`

- [ ] **Step 1: Escrever o teste**

`pipeline/tests/test_schemas.py`:

```python
import json

import pytest
from pydantic import ValidationError

from nfl_vision.schemas import Analise, Contexto, Jogador, NumeroDet


def test_analise_serializa_no_formato_do_sdd():
    analise = Analise(
        analise_id="2026-10-03-001",
        midia={"tipo": "foto", "largura": 1280, "altura": 720},
        contexto=Contexto(temporada=2025, semana=11, times=("KC", "BUF")),
        modelos={"detector": "yolo11m", "ocr": "paddleocr"},
        jogadores=[
            Jogador(
                track_id=7, time="KC", numero=87, confianca_numero=0.91,
                posicao="TE", nome="Fulano", frames_visiveis=[0],
                corrigido_pelo_usuario=False,
            )
        ],
    )
    dados = json.loads(analise.model_dump_json())
    assert dados["contexto"] == {"temporada": 2025, "semana": 11, "times": ["KC", "BUF"]}
    assert set(dados["jogadores"][0]) == {
        "track_id", "time", "numero", "confianca_numero", "posicao", "nome",
        "frames_visiveis", "corrigido_pelo_usuario",
    }
    assert Analise.model_validate_json(analise.model_dump_json()) == analise


def test_numero_fora_de_0_a_99_e_rejeitado():
    with pytest.raises(ValidationError):
        NumeroDet(det_id=1, numero=100, confianca=0.9)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'nfl_vision.schemas'`

- [ ] **Step 3: Implementar**

`pipeline/nfl_vision/schemas.py`:

```python
"""Contratos de dados entre as etapas do pipeline e a saída final."""

from typing import Literal

from pydantic import BaseModel, Field

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 em pixels
Lab = tuple[float, float, float]


class Contexto(BaseModel):
    temporada: int
    semana: int
    times: tuple[str, str]


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
    numero: int | None
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
```

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add pipeline/nfl_vision/schemas.py pipeline/tests/test_schemas.py
git commit -m "feat: contratos de dados entre etapas"
```

---

### Task 3: Configuração, caminhos e cores (`config.py`, `paths.py`, `cores.py`)

**Files:**
- Create: `pipeline/nfl_vision/config.py`
- Create: `pipeline/nfl_vision/paths.py`
- Create: `pipeline/nfl_vision/cores.py`
- Test: `pipeline/tests/test_cores.py`

- [ ] **Step 1: Escrever o teste**

`pipeline/tests/test_cores.py`:

```python
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
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_cores.py -v`
Expected: FAIL com `ImportError` (módulos não existem)

- [ ] **Step 3: Implementar `config.py`**

```python
"""Parâmetros do pipeline. A configuração usada é gravada em cada análise."""

from pydantic import BaseModel


class Config(BaseModel):
    detector_pesos: str = "yolo11m.pt"
    detector_imgsz: int = 1280
    detector_conf: float = 0.25
    device: str = "cuda:0"

    filtro_altura_rel: float = 0.4
    filtro_gramado_min: float = 0.3
    gramado_hsv_min: tuple[int, int, int] = (35, 40, 40)
    gramado_hsv_max: tuple[int, int, int] = (85, 255, 255)

    limiar_time: float = 0.60
    delta_e_grupo_unico: float = 15.0

    limiar_numero: float = 0.60
    numero_altura_min: int = 128
    ocr_device: str = "cpu"
```

- [ ] **Step 4: Implementar `paths.py`**

```python
"""Localização dos dados. NFL_VISION_DATA sobrescreve o padrão <repo>/data."""

import os
from pathlib import Path

_PADRAO = Path(__file__).resolve().parents[2] / "data"


def dados_dir() -> Path:
    return Path(os.environ.get("NFL_VISION_DATA", _PADRAO))


def runs_dir() -> Path:
    return dados_dir() / "runs"


def cache_dir() -> Path:
    return dados_dir() / "cache"


def datasets_dir() -> Path:
    return dados_dir() / "datasets"


def avaliacoes_dir() -> Path:
    return dados_dir() / "avaliacoes"
```

- [ ] **Step 5: Implementar `cores.py`**

```python
"""Operações de cor: máscara de gramado, LAB, ΔE (CIEDE2000)."""

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab


def mascara_gramado(bgr: np.ndarray, hsv_min, hsv_max) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(hsv_min, np.uint8), np.array(hsv_max, np.uint8)) > 0


def fracao_gramado(bgr: np.ndarray, hsv_min, hsv_max) -> float:
    if bgr.size == 0:
        return 0.0
    return float(mascara_gramado(bgr, hsv_min, hsv_max).mean())


def bgr_para_lab(pixels_bgr: np.ndarray) -> np.ndarray:
    """(N, 3) uint8 BGR -> (N, 3) float LAB, com L entre 0 e 100."""
    rgb = pixels_bgr[:, ::-1].astype(np.float64) / 255.0
    return rgb2lab(rgb.reshape(-1, 1, 3)).reshape(-1, 3)


def hex_para_lab(cor_hex: str) -> np.ndarray:
    h = cor_hex.lstrip("#")
    rgb = np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float64) / 255.0
    return rgb2lab(rgb.reshape(1, 1, 3)).reshape(3)


def hex_para_bgr(cor_hex: str) -> tuple[int, int, int]:
    h = cor_hex.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def delta_e(lab1, lab2) -> float:
    return float(deltaE_ciede2000(np.asarray(lab1, float), np.asarray(lab2, float)))
```

- [ ] **Step 6: Rodar e ver passar**

Run: `uv run pytest tests/test_cores.py -v`
Expected: `5 passed`

- [ ] **Step 7: Commit**

```bash
git add pipeline/nfl_vision/config.py pipeline/nfl_vision/paths.py pipeline/nfl_vision/cores.py pipeline/tests/test_cores.py
git commit -m "feat: configuração, caminhos de dados e utilitários de cor"
```

---

### Task 4: Runner genérico (`runner.py`)

**Files:**
- Create: `pipeline/nfl_vision/runner.py`
- Test: `pipeline/tests/test_runner.py`

- [ ] **Step 1: Escrever o teste com etapas falsas**

`pipeline/tests/test_runner.py`:

```python
import json
from datetime import date

import pytest
from pydantic import BaseModel

from nfl_vision.config import Config
from nfl_vision.runner import Etapa, EtapaFalhou, Runner, ler_manifest, proximo_id
from nfl_vision.schemas import Contexto

CTX = Contexto(temporada=2025, semana=11, times=("KC", "BUF"))


class Numero(BaseModel):
    valor: int


def _etapas(chamadas, falhar_b=False):
    def a(estado):
        chamadas.append("a")
        return Numero(valor=1)

    def b(estado):
        chamadas.append("b")
        if falhar_b:
            raise RuntimeError("quebrou")
        return Numero(valor=estado.saidas["a"].valor + 1)

    return [Etapa("a", Numero, a), Etapa("b", Numero, b)]


@pytest.fixture
def foto(tmp_path):
    p = tmp_path / "foto.JPG"
    p.write_bytes(b"conteudo")
    return p


def test_executa_todas_e_grava_artefatos(tmp_path, foto):
    chamadas = []
    runner = Runner(_etapas(chamadas), tmp_path / "runs")
    run_dir = runner.nova_analise(foto, CTX, Config(), {"pacote": "0.1"})
    estado = runner.executar(run_dir)

    assert chamadas == ["a", "b"]
    assert estado.saidas["b"].valor == 2
    assert (run_dir / "input.jpg").read_bytes() == b"conteudo"
    assert json.loads((run_dir / "b.json").read_text())["valor"] == 2
    manifest = ler_manifest(run_dir)
    assert manifest["etapas"]["a"]["status"] == "ok"
    assert manifest["config"]["detector_pesos"] == "yolo11m.pt"
    assert manifest["versoes"] == {"pacote": "0.1"}


def test_retoma_a_partir_de_uma_etapa(tmp_path, foto):
    chamadas = []
    runner = Runner(_etapas(chamadas), tmp_path / "runs")
    run_dir = runner.nova_analise(foto, CTX, Config(), {})
    runner.executar(run_dir)
    chamadas.clear()

    estado = runner.executar(run_dir, a_partir_de="b")

    assert chamadas == ["b"]
    assert estado.saidas["a"].valor == 1


def test_falha_registra_erro_e_preserva_anteriores(tmp_path, foto):
    runner = Runner(_etapas([], falhar_b=True), tmp_path / "runs")
    run_dir = runner.nova_analise(foto, CTX, Config(), {})

    with pytest.raises(EtapaFalhou) as erro:
        runner.executar(run_dir)

    assert erro.value.etapa == "b"
    assert erro.value.analise_id == run_dir.name
    assert (run_dir / "a.json").exists()
    manifest = ler_manifest(run_dir)
    assert manifest["etapas"]["b"]["status"] == "erro"
    assert "quebrou" in manifest["etapas"]["b"]["mensagem"]


def test_etapa_desconhecida(tmp_path, foto):
    runner = Runner(_etapas([]), tmp_path / "runs")
    run_dir = runner.nova_analise(foto, CTX, Config(), {})
    with pytest.raises(ValueError, match="etapa desconhecida"):
        runner.executar(run_dir, a_partir_de="zzz")


def test_proximo_id_sequencial(tmp_path):
    runs = tmp_path / "runs"
    hoje = date(2026, 10, 3)
    assert proximo_id(runs, hoje) == "2026-10-03-001"
    (runs / "2026-10-03-001").mkdir(parents=True)
    (runs / "2026-10-03-007").mkdir()
    assert proximo_id(runs, hoje) == "2026-10-03-008"


def test_le_correcoes(tmp_path, foto):
    runner = Runner(_etapas([]), tmp_path / "runs")
    run_dir = runner.nova_analise(foto, CTX, Config(), {})
    (run_dir / "corrections.json").write_text(
        json.dumps([{"det_id": 3, "numero": 87, "timestamp": "t"}]), encoding="utf-8"
    )
    estado = runner.carregar_estado(run_dir)
    assert estado.correcoes()[0].numero == 87
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_runner.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'nfl_vision.runner'`

- [ ] **Step 3: Implementar**

`pipeline/nfl_vision/runner.py`:

```python
"""Executa etapas em sequência, gravando a saída de cada uma em disco."""

import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
from pydantic import BaseModel

from nfl_vision.config import Config
from nfl_vision.schemas import Contexto, Correcao


class EtapaFalhou(Exception):
    def __init__(self, etapa: str, mensagem: str, analise_id: str | None = None):
        super().__init__(f"etapa '{etapa}' falhou: {mensagem}")
        self.etapa = etapa
        self.mensagem = mensagem
        self.analise_id = analise_id


@dataclass
class Estado:
    run_dir: Path
    contexto: Contexto
    config: Config
    saidas: dict[str, BaseModel] = field(default_factory=dict)
    _imagem: np.ndarray | None = None

    @property
    def caminho_imagem(self) -> Path:
        return next(self.run_dir.glob("input.*"))

    def imagem(self) -> np.ndarray:
        if self._imagem is None:
            from nfl_vision.stages.ingest import carregar_imagem

            self._imagem = carregar_imagem(self.caminho_imagem)
        return self._imagem

    def correcoes(self) -> list[Correcao]:
        arquivo = self.run_dir / "corrections.json"
        if not arquivo.exists():
            return []
        return [Correcao.model_validate(c) for c in json.loads(arquivo.read_text("utf-8"))]


@dataclass(frozen=True)
class Etapa:
    nome: str
    saida: type[BaseModel]
    executar: Callable[[Estado], BaseModel]


def proximo_id(runs_dir: Path, hoje: date | None = None) -> str:
    prefixo = (hoje or date.today()).isoformat()
    existentes = list(runs_dir.glob(f"{prefixo}-*")) if runs_dir.exists() else []
    n = max((int(p.name.rsplit("-", 1)[1]) for p in existentes), default=0) + 1
    return f"{prefixo}-{n:03d}"


def ler_manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text("utf-8"))


def _gravar_json(caminho: Path, dados) -> None:
    caminho.write_text(json.dumps(dados, indent=2, ensure_ascii=False), encoding="utf-8")


class Runner:
    def __init__(self, etapas: list[Etapa], runs_dir: Path):
        self.etapas = etapas
        self.runs_dir = runs_dir

    def nova_analise(self, imagem: Path, contexto: Contexto, config: Config,
                     versoes: dict[str, str]) -> Path:
        run_dir = self.runs_dir / proximo_id(self.runs_dir)
        run_dir.mkdir(parents=True)
        shutil.copy2(imagem, run_dir / f"input{imagem.suffix.lower()}")
        _gravar_json(run_dir / "manifest.json", {
            "analise_id": run_dir.name,
            "contexto": contexto.model_dump(mode="json"),
            "config": config.model_dump(mode="json"),
            "versoes": versoes,
            "etapas": {},
        })
        return run_dir

    def carregar_estado(self, run_dir: Path) -> Estado:
        manifest = ler_manifest(run_dir)
        return Estado(
            run_dir=run_dir,
            contexto=Contexto.model_validate(manifest["contexto"]),
            config=Config.model_validate(manifest["config"]),
        )

    def executar(self, run_dir: Path, a_partir_de: str | None = None) -> Estado:
        nomes = [e.nome for e in self.etapas]
        if a_partir_de is not None and a_partir_de not in nomes:
            raise ValueError(f"etapa desconhecida: {a_partir_de}. Opções: {', '.join(nomes)}")
        inicio = nomes.index(a_partir_de) if a_partir_de else 0
        estado = self.carregar_estado(run_dir)
        manifest = ler_manifest(run_dir)

        for i, etapa in enumerate(self.etapas):
            arquivo = run_dir / f"{etapa.nome}.json"
            if i < inicio:
                if not arquivo.exists():
                    raise EtapaFalhou(etapa.nome, "artefato ausente", run_dir.name)
                estado.saidas[etapa.nome] = etapa.saida.model_validate_json(arquivo.read_text("utf-8"))
                continue

            t0 = time.perf_counter()
            try:
                saida = etapa.executar(estado)
            except Exception as exc:
                manifest["etapas"][etapa.nome] = {
                    "status": "erro", "mensagem": str(exc),
                    "duracao_s": round(time.perf_counter() - t0, 3),
                }
                _gravar_json(run_dir / "manifest.json", manifest)
                raise EtapaFalhou(etapa.nome, str(exc), run_dir.name) from exc

            arquivo.write_text(saida.model_dump_json(indent=2), encoding="utf-8")
            estado.saidas[etapa.nome] = saida
            manifest["etapas"][etapa.nome] = {
                "status": "ok", "mensagem": None,
                "duracao_s": round(time.perf_counter() - t0, 3),
            }
            _gravar_json(run_dir / "manifest.json", manifest)

        return estado
```

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_runner.py -v`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add pipeline/nfl_vision/runner.py pipeline/tests/test_runner.py
git commit -m "feat: runner com artefatos por etapa, manifest e retomada"
```

---

### Task 5: Etapa `ingest`

**Files:**
- Create: `pipeline/nfl_vision/stages/ingest.py`
- Test: `pipeline/tests/test_ingest.py`

- [ ] **Step 1: Escrever o teste**

`pipeline/tests/test_ingest.py`:

```python
import pytest
from PIL import Image

from nfl_vision.stages.ingest import FormatoNaoSuportado, carregar_imagem, validar_formato


def test_aplica_orientacao_exif(tmp_path):
    caminho = tmp_path / "celular.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # girar 90° no sentido horário ao exibir
    Image.new("RGB", (40, 20), (255, 0, 0)).save(caminho, exif=exif)

    img = carregar_imagem(caminho)

    assert img.shape == (40, 20, 3)  # altura 40, largura 20 após girar
    assert tuple(img[0, 0]) == pytest.approx((0, 0, 254), abs=3)  # BGR


def test_png_aceito(tmp_path):
    caminho = tmp_path / "a.PNG"
    Image.new("RGB", (10, 8)).save(caminho)
    assert carregar_imagem(caminho).shape == (8, 10, 3)


@pytest.mark.parametrize("nome", ["video.mp4", "foto.gif", "doc.pdf"])
def test_formatos_recusados(tmp_path, nome):
    with pytest.raises(FormatoNaoSuportado, match="use JPG ou PNG"):
        validar_formato(tmp_path / nome)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: FAIL com `ModuleNotFoundError`

- [ ] **Step 3: Implementar**

`pipeline/nfl_vision/stages/ingest.py`:

```python
"""Etapa 1: leitura da foto com orientação EXIF e hash."""

import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from nfl_vision.schemas import IngestOut

FORMATOS = {".jpg", ".jpeg", ".png"}


class FormatoNaoSuportado(ValueError):
    pass


def validar_formato(caminho: Path) -> None:
    if caminho.suffix.lower() not in FORMATOS:
        raise FormatoNaoSuportado(
            f"formato '{caminho.suffix}' não suportado; use JPG ou PNG (vídeo ainda não é aceito)"
        )


def carregar_imagem(caminho: Path) -> np.ndarray:
    validar_formato(caminho)
    with Image.open(caminho) as img:
        rgb = np.asarray(ImageOps.exif_transpose(img).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def executar(estado) -> IngestOut:
    caminho = estado.caminho_imagem
    img = estado.imagem()
    return IngestOut(
        caminho=str(caminho),
        largura=img.shape[1],
        altura=img.shape[0],
        sha256=hashlib.sha256(caminho.read_bytes()).hexdigest(),
    )
```

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add pipeline/nfl_vision/stages/ingest.py pipeline/tests/test_ingest.py
git commit -m "feat: etapa ingest com orientação EXIF"
```

---

### Task 6: Times e cores oficiais (`teams.py`) + fixture de dados

**Files:**
- Create: `pipeline/nfl_vision/teams.py`
- Create: `pipeline/tests/conftest.py`
- Test: `pipeline/tests/test_teams.py`

- [ ] **Step 1: Conferir as colunas reais do nflverse**

```bash
uv run python -c "import nflreadpy as nfl; df = nfl.load_teams(); print(df.columns); print(df.filter(df['team_abbr'].is_in(['KC','LA','LAR'])).select('team_abbr','team_color','team_color2'))"
```

Expected: colunas incluem `team_abbr`, `team_color`, `team_color2`; os Rams aparecem como `LA`. Se algum nome de coluna for diferente, usar o nome real em `teams.py` e no fixture abaixo.

- [ ] **Step 2: Criar o fixture compartilhado**

`pipeline/tests/conftest.py`:

```python
import polars as pl
import pytest

TIMES = pl.DataFrame({
    "team_abbr": ["KC", "BUF", "LA"],
    "team_color": ["#E31837", "#00338D", "#003594"],
    "team_color2": ["#FFB612", "#C60C30", "#FFA300"],
})


@pytest.fixture
def dados(tmp_path, monkeypatch):
    """Diretório de dados temporário com caches prontos (sem rede)."""
    monkeypatch.setenv("NFL_VISION_DATA", str(tmp_path))
    cache = tmp_path / "cache"
    cache.mkdir()
    TIMES.write_parquet(cache / "teams.parquet")
    return tmp_path
```

- [ ] **Step 3: Escrever o teste**

`pipeline/tests/test_teams.py`:

```python
import pytest

from nfl_vision import paths, teams


def test_normalizar_aliases():
    assert teams.normalizar(" lar ") == "LA"
    assert teams.normalizar("JAC") == "JAX"
    assert teams.normalizar("kc") == "KC"


def test_validar_e_cores_usando_cache(dados):
    df = teams.carregar_times(paths.cache_dir())
    assert teams.validar("LAR", df) == "LA"
    assert teams.cores("KC", df) == ["#E31837", "#FFB612"]
    with pytest.raises(teams.TimeDesconhecido, match="XYZ"):
        teams.validar("XYZ", df)
```

- [ ] **Step 4: Rodar e ver falhar**

Run: `uv run pytest tests/test_teams.py -v`
Expected: FAIL com `ImportError`

- [ ] **Step 5: Implementar**

`pipeline/nfl_vision/teams.py`:

```python
"""Siglas e cores oficiais dos times (nflverse), com cache em parquet."""

from pathlib import Path

import polars as pl

ALIASES = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "LVR": "LV"}


class TimeDesconhecido(ValueError):
    pass


def normalizar(sigla: str) -> str:
    s = sigla.strip().upper()
    return ALIASES.get(s, s)


def carregar_times(cache_dir: Path) -> pl.DataFrame:
    arquivo = cache_dir / "teams.parquet"
    if arquivo.exists():
        return pl.read_parquet(arquivo)
    import nflreadpy as nfl

    df = nfl.load_teams().select("team_abbr", "team_color", "team_color2")
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(arquivo)
    return df


def validar(sigla: str, times_df: pl.DataFrame) -> str:
    s = normalizar(sigla)
    if s not in set(times_df["team_abbr"].to_list()):
        raise TimeDesconhecido(f"time '{sigla}' não existe no nflverse")
    return s


def cores(sigla: str, times_df: pl.DataFrame) -> list[str]:
    linha = times_df.filter(pl.col("team_abbr") == normalizar(sigla))
    if linha.height == 0:
        raise TimeDesconhecido(f"time '{sigla}' não existe no nflverse")
    return [c for c in (linha["team_color"][0], linha["team_color2"][0]) if c]
```

- [ ] **Step 6: Rodar e ver passar**

Run: `uv run pytest tests/test_teams.py -v`
Expected: `2 passed`

- [ ] **Step 7: Commit**

```bash
git add pipeline/nfl_vision/teams.py pipeline/tests/conftest.py pipeline/tests/test_teams.py
git commit -m "feat: siglas e cores oficiais dos times com cache"
```

---

### Task 7: Etapa `detect` (YOLO + filtros de fora de campo)

**Files:**
- Create: `pipeline/nfl_vision/stages/detect.py`
- Create: `pipeline/tests/sintetico.py`
- Test: `pipeline/tests/test_detect.py`

- [ ] **Step 1: Criar o gerador de imagens sintéticas**

`pipeline/tests/sintetico.py`:

```python
"""Imagens sintéticas: campo verde, arquibancada cinza, jogadores e árbitro."""

import numpy as np

VERDE = (40, 140, 40)
CINZA = (128, 128, 128)
VERMELHO_KC = (55, 24, 227)   # #E31837 em BGR
AZUL_BUF = (141, 51, 0)       # #00338D em BGR


def campo(largura=800, altura=600):
    img = np.full((altura, largura, 3), VERDE, np.uint8)
    img[:150] = CINZA  # arquibancada
    return img


def jogador(img, x, y, cor, w=60, h=120):
    img[y:y + h, x:x + w] = cor
    return (float(x), float(y), float(x + w), float(y + h))


def arbitro(img, x, y, w=60, h=120):
    for i in range(w):
        img[y:y + h, x + i] = (0, 0, 0) if (i // 4) % 2 == 0 else (255, 255, 255)
    return (float(x), float(y), float(x + w), float(y + h))
```

- [ ] **Step 2: Escrever o teste**

`pipeline/tests/test_detect.py`:

```python
import pytest

from nfl_vision.config import Config
from nfl_vision.schemas import Deteccao
from nfl_vision.stages.detect import aplicar_filtros
from sintetico import VERMELHO_KC, campo, jogador


def _det(i, bbox):
    return Deteccao(det_id=i, bbox=bbox, confianca=0.9)


def test_filtros_de_fora_de_campo_e_tamanho():
    img = campo()
    em_campo = jogador(img, 100, 300, VERMELHO_KC)
    na_arquibancada = jogador(img, 300, 10, VERMELHO_KC)
    pequeno = (500.0, 400.0, 510.0, 420.0)
    # caixa até a borda inferior; camisa desenhada só até y=580, pés sobre o verde
    na_borda = jogador(img, 600, 480, VERMELHO_KC, h=100)
    na_borda = (na_borda[0], na_borda[1], na_borda[2], 600.0)

    saida = aplicar_filtros(
        [_det(0, em_campo), _det(1, na_arquibancada), _det(2, pequeno), _det(3, na_borda)],
        img, Config(),
    )

    assert [(d.descartado, d.motivo_descarte) for d in saida] == [
        (False, None), (True, "fora_de_campo"), (True, "pequeno"), (False, None),
    ]


def test_sem_deteccoes():
    assert aplicar_filtros([], campo(), Config()) == []


@pytest.mark.model
def test_yolo_detecta_pessoas_em_imagem_real():
    from ultralytics.utils import ASSETS

    from nfl_vision.stages.detect import detectar_pessoas
    from nfl_vision.stages.ingest import carregar_imagem

    dets = detectar_pessoas(carregar_imagem(ASSETS / "bus.jpg"), Config())
    assert len(dets) >= 3
    assert all(d.confianca >= 0.25 for d in dets)
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `uv run pytest tests/test_detect.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'nfl_vision.stages.detect'`

- [ ] **Step 4: Implementar**

`pipeline/nfl_vision/stages/detect.py`:

```python
"""Etapa 2: detecção de pessoas (YOLO) e descarte de quem está fora de campo."""

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

from nfl_vision.config import Config
from nfl_vision.cores import fracao_gramado
from nfl_vision.schemas import Deteccao, DetectOut


def faixa_dos_pes(img: np.ndarray, bbox) -> np.ndarray:
    """Faixa logo abaixo da caixa (10% da altura); na borda, a faixa interna inferior."""
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, x2 = max(x1, 0), min(x2, img.shape[1])
    altura = max(1, int(round((y2 - y1) * 0.10)))
    if y2 + altura <= img.shape[0]:
        return img[y2:y2 + altura, x1:x2]
    return img[max(y2 - altura, 0):y2, x1:x2]


def aplicar_filtros(deteccoes: list[Deteccao], img: np.ndarray, cfg: Config) -> list[Deteccao]:
    if not deteccoes:
        return []
    alturas = [d.bbox[3] - d.bbox[1] for d in deteccoes]
    mediana = float(np.median(alturas))
    saida = []
    for d, h in zip(deteccoes, alturas):
        motivo = None
        if h < cfg.filtro_altura_rel * mediana:
            motivo = "pequeno"
        elif fracao_gramado(faixa_dos_pes(img, d.bbox), cfg.gramado_hsv_min,
                            cfg.gramado_hsv_max) < cfg.filtro_gramado_min:
            motivo = "fora_de_campo"
        saida.append(d.model_copy(update={"descartado": motivo is not None,
                                          "motivo_descarte": motivo}))
    return saida


@lru_cache(maxsize=2)
def _modelo(pesos: str):
    from ultralytics import YOLO

    return YOLO(pesos)


def resolver_device(device: str) -> str:
    if device.startswith("cuda"):
        import torch

        return device if torch.cuda.is_available() else "cpu"
    return device


def detectar_pessoas(img: np.ndarray, cfg: Config) -> list[Deteccao]:
    resultado = _modelo(cfg.detector_pesos).predict(
        img, imgsz=cfg.detector_imgsz, conf=cfg.detector_conf, classes=[0],
        device=resolver_device(cfg.device), verbose=False,
    )[0]
    caixas = resultado.boxes.xyxy.cpu().numpy()
    confs = resultado.boxes.conf.cpu().numpy()
    return [
        Deteccao(det_id=i, bbox=tuple(float(v) for v in caixa), confianca=float(conf))
        for i, (caixa, conf) in enumerate(zip(caixas, confs))
    ]


def sha256_pesos(pesos: str) -> str | None:
    caminho = Path(pesos)
    return hashlib.sha256(caminho.read_bytes()).hexdigest() if caminho.exists() else None


def executar(estado) -> DetectOut:
    img = estado.imagem()
    cfg = estado.config
    return DetectOut(
        deteccoes=aplicar_filtros(detectar_pessoas(img, cfg), img, cfg),
        pesos_sha256=sha256_pesos(cfg.detector_pesos),
    )
```

- [ ] **Step 5: Rodar os testes rápidos e o de modelo**

```bash
uv run pytest tests/test_detect.py -v
uv run pytest -m model tests/test_detect.py -v
```

Expected: `2 passed` e depois `1 passed` (o primeiro uso baixa `yolo11m.pt` para `pipeline/`; o arquivo é ignorado pelo git via `*.pt`).

- [ ] **Step 6: Commit**

```bash
git add pipeline/nfl_vision/stages/detect.py pipeline/tests/sintetico.py pipeline/tests/test_detect.py
git commit -m "feat: etapa detect com YOLO11 e filtros de fora de campo"
```

---

### Task 8: Etapa `team` (cor do tronco, árbitro, agrupamento e mapeamento)

**Files:**
- Create: `pipeline/nfl_vision/stages/team.py`
- Test: `pipeline/tests/test_team.py`

- [ ] **Step 1: Escrever o teste**

`pipeline/tests/test_team.py`:

```python
import numpy as np

from nfl_vision.config import Config
from nfl_vision.cores import bgr_para_lab, hex_para_lab
from nfl_vision.schemas import Deteccao
from nfl_vision.stages.team import (
    agrupar, classificar, eh_arbitro, eh_branco, mapear_grupos,
)
from sintetico import AZUL_BUF, VERMELHO_KC, arbitro, campo, jogador

CFG = Config()
PALETAS = {
    "KC": [hex_para_lab("#E31837"), hex_para_lab("#FFB612")],
    "BUF": [hex_para_lab("#00338D"), hex_para_lab("#C60C30")],
}
VERMELHO = hex_para_lab("#E31837")
AZUL = hex_para_lab("#00338D")
BRANCO = hex_para_lab("#FFFFFF")


def test_listras_pretas_e_brancas_sao_arbitro():
    listrado = np.array([[0, 0, 0], [255, 255, 255]] * 50, np.uint8)
    liso = np.array([[55, 24, 227]] * 100, np.uint8)
    assert eh_arbitro(bgr_para_lab(listrado))
    assert not eh_arbitro(bgr_para_lab(liso))


def test_eh_branco():
    assert eh_branco(BRANCO)
    assert not eh_branco(VERMELHO)


def test_agrupar_dois_grupos_e_grupo_unico():
    rotulos, centros = agrupar(np.array([VERMELHO, VERMELHO, AZUL, AZUL]), CFG)
    assert len(centros) == 2
    assert rotulos[0] == rotulos[1] != rotulos[2] == rotulos[3]

    rotulos, centros = agrupar(np.array([VERMELHO, VERMELHO + 1, VERMELHO - 1]), CFG)
    assert len(centros) == 1
    assert set(rotulos) == {0}


def test_mapear_grupos_por_paleta():
    assert mapear_grupos(np.array([AZUL, VERMELHO]), PALETAS) == {0: "BUF", 1: "KC"}


def test_grupo_branco_fica_com_o_outro_time():
    assert mapear_grupos(np.array([BRANCO, AZUL]), PALETAS) == {1: "BUF", 0: "KC"}


def test_grupo_unico_branco_e_desconhecido():
    assert mapear_grupos(np.array([BRANCO]), PALETAS) == {0: None}
    assert mapear_grupos(np.array([AZUL]), PALETAS) == {0: "BUF"}


def test_classificar_imagem_sintetica():
    img = campo()
    caixas = [
        jogador(img, 100, 300, VERMELHO_KC),
        jogador(img, 200, 300, VERMELHO_KC),
        jogador(img, 400, 300, AZUL_BUF),
        jogador(img, 500, 300, AZUL_BUF),
        arbitro(img, 650, 300),
    ]
    dets = [Deteccao(det_id=i, bbox=b, confianca=0.9) for i, b in enumerate(caixas)]
    dets.append(Deteccao(det_id=5, bbox=(0, 0, 10, 10), confianca=0.9, descartado=True,
                         motivo_descarte="fora_de_campo"))

    itens = {t.det_id: t for t in classificar(img, dets, PALETAS, CFG)}

    assert [itens[i].time for i in range(4)] == ["KC", "KC", "BUF", "BUF"]
    assert all(itens[i].confianca >= 0.99 for i in range(4))
    assert itens[4].arbitro and itens[4].time is None
    assert 5 not in itens
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_team.py -v`
Expected: FAIL com `ModuleNotFoundError`

- [ ] **Step 3: Implementar**

`pipeline/nfl_vision/stages/team.py`:

```python
"""Etapa 3: time de cada jogador pela cor do tronco; detecção de árbitro."""

import numpy as np
from sklearn.cluster import KMeans

from nfl_vision import paths, teams
from nfl_vision.config import Config
from nfl_vision.cores import bgr_para_lab, delta_e, hex_para_lab, mascara_gramado
from nfl_vision.schemas import Deteccao, TeamOut, TimeDet

MIN_PIXELS = 50


def recorte_tronco(img: np.ndarray, bbox) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    ya, yb = max(int(y1 + 0.10 * h), 0), max(int(y1 + 0.50 * h), 0)
    xa, xb = max(int(x1 + 0.20 * w), 0), max(int(x1 + 0.80 * w), 0)
    return img[ya:yb, xa:xb]


def pixels_uteis(recorte: np.ndarray, cfg: Config) -> np.ndarray:
    """Pixels LAB do recorte, sem o gramado."""
    if recorte.size == 0:
        return np.empty((0, 3))
    fora = ~mascara_gramado(recorte, cfg.gramado_hsv_min, cfg.gramado_hsv_max)
    return bgr_para_lab(recorte[fora])


def eh_arbitro(lab_px: np.ndarray) -> bool:
    luz = lab_px[:, 0]
    return bool((luz < 30).mean() >= 0.25 and (luz > 80).mean() >= 0.25)


def _distintos(pontos: np.ndarray) -> int:
    return len(np.unique(pontos.round(2), axis=0))


def cor_dominante(lab_px: np.ndarray) -> np.ndarray:
    k = min(3, _distintos(lab_px))
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(lab_px)
    return km.cluster_centers_[np.bincount(km.labels_).argmax()]


def eh_branco(lab) -> bool:
    return bool(lab[0] > 85 and np.hypot(lab[1], lab[2]) < 10)


def agrupar(cores: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Rótulo de grupo por cor e centros dos grupos (1 ou 2)."""
    unico = (np.zeros(len(cores), int), cores.mean(axis=0, keepdims=True))
    if _distintos(cores) < 2:
        return unico
    km = KMeans(n_clusters=2, n_init=4, random_state=0).fit(cores)
    centros = km.cluster_centers_
    if delta_e(centros[0], centros[1]) < cfg.delta_e_grupo_unico:
        return unico
    return km.labels_.astype(int), centros


def _custo(centro, paleta) -> float:
    return min(delta_e(centro, p) for p in paleta)


def mapear_grupos(centros: np.ndarray, paletas: dict[str, list[np.ndarray]]) -> dict[int, str | None]:
    a, b = list(paletas)

    def mais_proximo(centro):
        return min((a, b), key=lambda t: _custo(centro, paletas[t]))

    if len(centros) == 1:
        return {0: None if eh_branco(centros[0]) else mais_proximo(centros[0])}

    brancos = [eh_branco(c) for c in centros]
    if brancos[0] != brancos[1]:
        colorido = 1 if brancos[0] else 0
        time = mais_proximo(centros[colorido])
        return {colorido: time, 1 - colorido: b if time == a else a}

    direto = _custo(centros[0], paletas[a]) + _custo(centros[1], paletas[b])
    cruzado = _custo(centros[0], paletas[b]) + _custo(centros[1], paletas[a])
    return {0: a, 1: b} if direto <= cruzado else {0: b, 1: a}


def confianca(cor, centros: np.ndarray, rotulo: int) -> float:
    d_proprio = delta_e(cor, centros[rotulo])
    if len(centros) == 1:
        return float(np.clip(1 - d_proprio / 50, 0, 1))
    d_outro = delta_e(cor, centros[1 - rotulo])
    total = d_proprio + d_outro
    return 0.5 if total == 0 else float(d_outro / total)


def classificar(img: np.ndarray, deteccoes: list[Deteccao],
                paletas: dict[str, list[np.ndarray]], cfg: Config) -> list[TimeDet]:
    itens: dict[int, TimeDet] = {}
    candidatos: list[tuple[int, np.ndarray]] = []
    for d in deteccoes:
        if d.descartado:
            continue
        px = pixels_uteis(recorte_tronco(img, d.bbox), cfg)
        if len(px) < MIN_PIXELS:
            itens[d.det_id] = TimeDet(det_id=d.det_id, time=None, confianca=0.0)
        elif eh_arbitro(px):
            itens[d.det_id] = TimeDet(det_id=d.det_id, time=None, confianca=1.0, arbitro=True)
        else:
            candidatos.append((d.det_id, cor_dominante(px)))

    if candidatos:
        rotulos, centros = agrupar(np.array([c for _, c in candidatos]), cfg)
        mapa = mapear_grupos(centros, paletas)
        for (det_id, cor), rotulo in zip(candidatos, rotulos):
            conf = confianca(cor, centros, int(rotulo))
            itens[det_id] = TimeDet(
                det_id=det_id,
                time=mapa[int(rotulo)] if conf >= cfg.limiar_time else None,
                confianca=round(conf, 4),
                cor_lab=tuple(float(v) for v in cor),
            )
    return [itens[k] for k in sorted(itens)]


def executar(estado) -> TeamOut:
    times_df = teams.carregar_times(paths.cache_dir())
    paletas = {t: [hex_para_lab(c) for c in teams.cores(t, times_df)]
               for t in estado.contexto.times}
    deteccoes = estado.saidas["detect"].deteccoes
    return TeamOut(itens=classificar(estado.imagem(), deteccoes, paletas, estado.config))
```

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_team.py -v`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add pipeline/nfl_vision/stages/team.py pipeline/tests/test_team.py
git commit -m "feat: etapa team com agrupamento por cor, árbitro e mapeamento por paleta"
```

---

### Task 9: Etapa `jersey` (OCR do número)

**Files:**
- Create: `pipeline/nfl_vision/stages/jersey.py`
- Test: `pipeline/tests/test_jersey.py`

- [ ] **Step 1: Instalar o extra de OCR (pedir confirmação: dependência nativa pesada)**

```bash
uv sync --extra ocr
uv run python -c "import paddle, cv2; print(paddle.__version__, cv2.__version__); paddle.utils.run_check()"
```

Expected: versão do Paddle 3.x e mensagem `PaddlePaddle is installed successfully!`. O padrão é OCR em CPU (`Config.ocr_device = "cpu"`). Se `import cv2` quebrar depois disso (conflito entre `opencv-python` e `opencv-contrib-python`), trocar `opencv-python` por `opencv-contrib-python` nas dependências e rodar `uv sync --extra ocr` de novo.

- [ ] **Step 2: Escrever o teste**

`pipeline/tests/test_jersey.py`:

```python
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
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `uv run pytest tests/test_jersey.py -v`
Expected: FAIL com `ModuleNotFoundError`

- [ ] **Step 4: Implementar**

`pipeline/nfl_vision/stages/jersey.py`:

```python
"""Etapa 4: leitura do número da camisa com OCR."""

import re
from functools import lru_cache
from typing import Protocol

import cv2
import numpy as np

from nfl_vision.config import Config
from nfl_vision.schemas import Deteccao, JerseyOut, NumeroDet, TimeDet

PADRAO = re.compile(r"^\d{1,2}$")


class LeitorOCR(Protocol):
    def ler(self, img: np.ndarray) -> list[tuple[str, float]]: ...


def ampliar(img: np.ndarray, altura_min: int) -> np.ndarray:
    if img.size == 0 or img.shape[0] >= altura_min:
        return img
    fator = altura_min / img.shape[0]
    return cv2.resize(img, None, fx=fator, fy=fator, interpolation=cv2.INTER_CUBIC)


def recorte_numero(img: np.ndarray, bbox, altura_min: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    ya, yb = max(int(y1 + 0.15 * h), 0), max(int(y1 + 0.60 * h), 0)
    xa, xb = max(int(x1 + 0.10 * w), 0), max(int(x1 + 0.90 * w), 0)
    return ampliar(img[ya:yb, xa:xb], altura_min)


def escolher_numero(leituras: list[tuple[str, float]], limiar: float) -> tuple[int | None, float, str | None]:
    validas = [(t.strip(), s) for t, s in leituras if PADRAO.match(t.strip())]
    if not validas:
        bruto = max(leituras, key=lambda x: x[1])[0] if leituras else None
        return None, 0.0, bruto
    texto, score = max(validas, key=lambda x: x[1])
    if score < limiar:
        return None, float(score), texto
    return int(texto), float(score), texto


class PaddleLeitor:
    def __init__(self, device: str):
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=device,
        )

    def ler(self, img: np.ndarray) -> list[tuple[str, float]]:
        leituras = []
        for r in self._ocr.predict(img):
            leituras.extend(zip(r["rec_texts"], (float(s) for s in r["rec_scores"])))
        return leituras


@lru_cache(maxsize=2)
def leitor_padrao(device: str) -> LeitorOCR:
    return PaddleLeitor(device)


def ler_numeros(img: np.ndarray, deteccoes: list[Deteccao], times: list[TimeDet],
                leitor: LeitorOCR, cfg: Config) -> list[NumeroDet]:
    arbitros = {t.det_id for t in times if t.arbitro}
    itens = []
    for d in deteccoes:
        if d.descartado or d.det_id in arbitros:
            continue
        recorte = recorte_numero(img, d.bbox, cfg.numero_altura_min)
        leituras = leitor.ler(recorte) if recorte.size else []
        numero, conf, bruto = escolher_numero(leituras, cfg.limiar_numero)
        itens.append(NumeroDet(det_id=d.det_id, numero=numero, confianca=round(conf, 4),
                               texto_bruto=bruto))
    return itens


def executar(estado) -> JerseyOut:
    leitor = leitor_padrao(estado.config.ocr_device)
    return JerseyOut(itens=ler_numeros(
        estado.imagem(), estado.saidas["detect"].deteccoes,
        estado.saidas["team"].itens, leitor, estado.config,
    ))
```

- [ ] **Step 5: Rodar os testes rápidos e o de modelo**

```bash
uv run pytest tests/test_jersey.py -v
uv run pytest -m model tests/test_jersey.py -v
```

Expected: `5 passed` e `1 passed`. Se o teste de modelo falhar com `KeyError: 'rec_texts'`, a API de resultado do PaddleOCR mudou: inspecionar com `uv run python -c "from paddleocr import PaddleOCR; import numpy as np; r = PaddleOCR(device='cpu').predict(np.full((100,100,3),255,np.uint8)); print(type(r[0]), list(r[0].keys()))"` e ajustar `PaddleLeitor.ler`.

- [ ] **Step 6: Commit**

```bash
git add pipeline/nfl_vision/stages/jersey.py pipeline/tests/test_jersey.py pipeline/pyproject.toml pipeline/uv.lock
git commit -m "feat: etapa jersey com PaddleOCR e filtro de números 0-99"
```

---

### Task 10: Etapa `roster` (consulta e correções)

**Files:**
- Create: `pipeline/nfl_vision/stages/roster.py`
- Modify: `pipeline/tests/conftest.py`
- Test: `pipeline/tests/test_roster.py`

- [ ] **Step 1: Conferir as colunas reais do roster semanal**

```bash
uv run python -c "import nflreadpy as nfl; df = nfl.load_rosters_weekly(seasons=[2025]); print(df.columns); print(df['status'].unique().to_list()); print(df.select('season','week','team','jersey_number','position','full_name','gsis_id','status').head(3))"
```

Expected: as oito colunas usadas existem; `status` inclui `ACT`. Se algum nome diferir, ajustar `COLUNAS` em `roster.py` e o fixture.

- [ ] **Step 2: Acrescentar o roster ao fixture**

`pipeline/tests/conftest.py` (arquivo completo):

```python
import polars as pl
import pytest

TIMES = pl.DataFrame({
    "team_abbr": ["KC", "BUF", "LA"],
    "team_color": ["#E31837", "#00338D", "#003594"],
    "team_color2": ["#FFB612", "#C60C30", "#FFA300"],
})

ROSTER = pl.DataFrame({
    "season": [2025] * 7,
    "week": [11] * 7,
    "team": ["KC", "KC", "BUF", "BUF", "BUF", "KC", "KC"],
    "jersey_number": [87, 15, 17, 14, 14, 10, 10],
    "position": ["TE", "QB", "QB", "WR", "WR", "WR", "RB"],
    "full_name": ["Jogador KC 87", "Jogador KC 15", "Jogador BUF 17",
                  "Jogador BUF 14 Ativo", "Jogador BUF 14 Reserva",
                  "Jogador KC 10 A", "Jogador KC 10 B"],
    "gsis_id": ["00-1", "00-2", "00-3", "00-4", "00-5", "00-6", "00-7"],
    "status": ["ACT", "ACT", "ACT", "ACT", "RES", "ACT", "ACT"],
})


@pytest.fixture
def dados(tmp_path, monkeypatch):
    """Diretório de dados temporário com caches prontos (sem rede)."""
    monkeypatch.setenv("NFL_VISION_DATA", str(tmp_path))
    cache = tmp_path / "cache"
    (cache / "rosters").mkdir(parents=True)
    TIMES.write_parquet(cache / "teams.parquet")
    ROSTER.write_parquet(cache / "rosters" / "2025.parquet")
    return tmp_path
```

- [ ] **Step 3: Escrever o teste**

`pipeline/tests/test_roster.py`:

```python
import pytest

from nfl_vision import paths
from nfl_vision.schemas import Contexto, Correcao, NumeroDet, TimeDet
from nfl_vision.stages.roster import (
    RosterIndisponivel, aplicar_correcoes, buscar, carregar_roster, resolver,
)

CTX = Contexto(temporada=2025, semana=11, times=("KC", "BUF"))


@pytest.fixture
def df(dados):
    return carregar_roster(2025, paths.cache_dir())


def test_busca_simples(df):
    assert buscar(df, 2025, 11, "KC", 87) == ("TE", "Jogador KC 87", "00-1", "ok")


def test_duplicado_prefere_ativo(df):
    assert buscar(df, 2025, 11, "BUF", 14)[1] == "Jogador BUF 14 Ativo"


def test_duplicado_ativo_e_ambiguo(df):
    assert buscar(df, 2025, 11, "KC", 10) == (None, None, None, "ambiguo")


def test_numero_fora_do_roster(df):
    assert buscar(df, 2025, 11, "KC", 99)[3] == "numero_fora_do_roster"


def test_resolver_motivos(df):
    itens = resolver(df, CTX, {0: "KC", 1: None, 2: "BUF"}, {0: 87, 1: 15, 2: None})
    assert [(r.det_id, r.motivo) for r in itens] == [(0, "ok"), (1, "sem_time"), (2, "sem_numero")]


def test_correcoes_sobrescrevem_em_ordem():
    times = [TimeDet(det_id=0, time="KC", confianca=0.9),
             TimeDet(det_id=1, time=None, confianca=1.0, arbitro=True)]
    numeros = [NumeroDet(det_id=0, numero=None, confianca=0.0)]
    correcoes = [Correcao(det_id=0, numero=15, timestamp="t1"),
                 Correcao(det_id=0, time="buf", timestamp="t2")]

    time_por_det, num_por_det = aplicar_correcoes(times, numeros, correcoes)

    assert time_por_det == {0: "BUF"}
    assert num_por_det == {0: 15}


def test_sem_cache_e_sem_rede(tmp_path, monkeypatch):
    import nflreadpy

    def sem_rede(**kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(nflreadpy, "load_rosters_weekly", sem_rede)
    with pytest.raises(RosterIndisponivel, match="2024"):
        carregar_roster(2024, tmp_path)
```

- [ ] **Step 4: Rodar e ver falhar**

Run: `uv run pytest tests/test_roster.py -v`
Expected: FAIL com `ModuleNotFoundError`

- [ ] **Step 5: Implementar**

`pipeline/nfl_vision/stages/roster.py`:

```python
"""Etapa 5: posição e nome pelo roster semanal (temporada + semana + time + número)."""

from pathlib import Path
from typing import Iterable

import polars as pl

from nfl_vision import paths, teams
from nfl_vision.schemas import Contexto, Correcao, NumeroDet, RosterDet, RosterOut, TimeDet

COLUNAS = ["season", "week", "team", "jersey_number", "position", "full_name", "gsis_id", "status"]


class RosterIndisponivel(RuntimeError):
    pass


def carregar_roster(temporada: int, cache_dir: Path) -> pl.DataFrame:
    arquivo = cache_dir / "rosters" / f"{temporada}.parquet"
    if arquivo.exists():
        return pl.read_parquet(arquivo)
    try:
        import nflreadpy as nfl

        df = nfl.load_rosters_weekly(seasons=[temporada]).select(COLUNAS)
    except Exception as exc:
        raise RosterIndisponivel(
            f"roster de {temporada} indisponível (sem cache e sem rede?): {exc}"
        ) from exc
    df = df.with_columns(
        pl.col("jersey_number").cast(pl.Int64, strict=False),
        pl.col("team").map_elements(teams.normalizar, return_dtype=pl.String),
    )
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(arquivo)
    return df


def buscar(df: pl.DataFrame, temporada: int, semana: int, time: str,
           numero: int) -> tuple[str | None, str | None, str | None, str]:
    linhas = df.filter(
        (pl.col("season") == temporada) & (pl.col("week") == semana)
        & (pl.col("team") == time) & (pl.col("jersey_number") == numero)
    )
    if linhas.height == 0:
        return None, None, None, "numero_fora_do_roster"
    if linhas.height > 1:
        linhas = linhas.filter(pl.col("status") == "ACT")
        if linhas.height != 1:
            return None, None, None, "ambiguo"
    r = linhas.row(0, named=True)
    return r["position"], r["full_name"], r["gsis_id"], "ok"


def aplicar_correcoes(times: Iterable[TimeDet], numeros: Iterable[NumeroDet],
                      correcoes: list[Correcao]) -> tuple[dict[int, str | None], dict[int, int | None]]:
    time_por_det = {t.det_id: t.time for t in times if not t.arbitro}
    num_por_det = {n.det_id: n.numero for n in numeros}
    for c in correcoes:
        if c.time is not None:
            time_por_det[c.det_id] = teams.normalizar(c.time)
        if c.numero is not None:
            num_por_det[c.det_id] = c.numero
    return time_por_det, num_por_det


def resolver(df: pl.DataFrame, ctx: Contexto, time_por_det: dict[int, str | None],
             num_por_det: dict[int, int | None]) -> list[RosterDet]:
    itens = []
    for det_id in sorted(time_por_det):
        time, numero = time_por_det[det_id], num_por_det.get(det_id)
        if time is None:
            itens.append(RosterDet(det_id=det_id, motivo="sem_time"))
        elif numero is None:
            itens.append(RosterDet(det_id=det_id, motivo="sem_numero"))
        else:
            posicao, nome, gsis_id, motivo = buscar(df, ctx.temporada, ctx.semana, time, numero)
            itens.append(RosterDet(det_id=det_id, posicao=posicao, nome=nome,
                                   gsis_id=gsis_id, motivo=motivo))
    return itens


def executar(estado) -> RosterOut:
    df = carregar_roster(estado.contexto.temporada, paths.cache_dir())
    time_por_det, num_por_det = aplicar_correcoes(
        estado.saidas["team"].itens, estado.saidas["jersey"].itens, estado.correcoes()
    )
    return RosterOut(itens=resolver(df, estado.contexto, time_por_det, num_por_det))
```

- [ ] **Step 6: Rodar tudo e ver passar**

Run: `uv run pytest -v`
Expected: todos passam (incluindo `test_teams.py`, que agora usa o fixture com roster).

- [ ] **Step 7: Commit**

```bash
git add pipeline/nfl_vision/stages/roster.py pipeline/tests/conftest.py pipeline/tests/test_roster.py
git commit -m "feat: etapa roster com checagem cruzada e correções"
```

---

### Task 11: Imagem anotada (`render.py`)

**Files:**
- Create: `pipeline/nfl_vision/render.py`
- Test: `pipeline/tests/test_render.py`

- [ ] **Step 1: Escrever o teste**

`pipeline/tests/test_render.py`:

```python
import numpy as np

from nfl_vision.render import COR_DESCONHECIDO, desenhar, rotulo
from nfl_vision.schemas import Jogador


def _jogador(track_id, time, numero, posicao=None):
    return Jogador(track_id=track_id, time=time, numero=numero, confianca_numero=0.9,
                   posicao=posicao, nome=None, frames_visiveis=[0],
                   corrigido_pelo_usuario=False)


def test_rotulos():
    assert rotulo(_jogador(0, "KC", 87, "TE")) == "KC 87 TE"
    assert rotulo(_jogador(1, "KC", None)) == "KC ?"
    assert rotulo(_jogador(2, None, None)) == "? ?"


def test_desenha_na_cor_do_time_e_desconhecido_em_cinza():
    img = np.zeros((300, 300, 3), np.uint8)
    jogadores = [_jogador(0, "KC", 87, "TE"), _jogador(1, "KC", None)]
    caixas = {0: (50.0, 100.0, 110.0, 220.0), 1: (180.0, 100.0, 240.0, 220.0)}

    saida = desenhar(img, jogadores, caixas, {"KC": (55, 24, 227)})

    assert saida.shape == img.shape
    assert tuple(saida[200, 50]) == (55, 24, 227)       # borda esquerda do KC 87
    assert tuple(saida[200, 180]) == COR_DESCONHECIDO   # borda do desconhecido
    assert not img.any()                                # original intacta
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_render.py -v`
Expected: FAIL com `ModuleNotFoundError`

- [ ] **Step 3: Implementar**

`pipeline/nfl_vision/render.py`:

```python
"""Desenha caixas e rótulos sobre a foto."""

import cv2
import numpy as np

from nfl_vision.schemas import BBox, Jogador

COR_DESCONHECIDO = (128, 116, 107)  # #6B7480 em BGR (confidence/unknown)
FONTE = cv2.FONT_HERSHEY_SIMPLEX


def rotulo(j: Jogador) -> str:
    partes = [j.time or "?", "?" if j.numero is None else str(j.numero)]
    if j.posicao:
        partes.append(j.posicao)
    return " ".join(partes)


def desenhar(img: np.ndarray, jogadores: list[Jogador], caixas: dict[int, BBox],
             cores_times: dict[str, tuple[int, int, int]]) -> np.ndarray:
    saida = img.copy()
    for j in jogadores:
        x1, y1, x2, y2 = (int(round(v)) for v in caixas[j.track_id])
        conhecido = j.time is not None and j.numero is not None
        cor = cores_times.get(j.time, COR_DESCONHECIDO) if conhecido else COR_DESCONHECIDO
        cv2.rectangle(saida, (x1, y1), (x2, y2), cor, 2)
        texto = rotulo(j)
        (tw, th), _ = cv2.getTextSize(texto, FONTE, 0.5, 1)
        topo = max(y1 - th - 6, 0)
        cv2.rectangle(saida, (x1, topo), (x1 + tw + 4, topo + th + 6), cor, -1)
        cv2.putText(saida, texto, (x1 + 2, topo + th + 2), FONTE, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return saida
```

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_render.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add pipeline/nfl_vision/render.py pipeline/tests/test_render.py
git commit -m "feat: render da imagem anotada"
```

---

### Task 12: Montagem do resultado e orquestração (`montagem.py`, `pipeline.py`)

**Files:**
- Create: `pipeline/nfl_vision/montagem.py`
- Create: `pipeline/nfl_vision/pipeline.py`
- Modify: `pipeline/tests/conftest.py` (fixtures `foto_sintetica` e `modelos_falsos`)
- Test: `pipeline/tests/test_pipeline.py`

- [ ] **Step 1: Acrescentar fixtures ao final de `pipeline/tests/conftest.py`**

```python
@pytest.fixture
def foto_sintetica(tmp_path):
    """Foto com 2 KC, 2 BUF, 1 árbitro e 1 pessoa na arquibancada; caixas por det_id."""
    import cv2

    from sintetico import AZUL_BUF, VERMELHO_KC, arbitro, campo, jogador

    img = campo()
    caixas = [
        jogador(img, 100, 300, VERMELHO_KC),
        jogador(img, 200, 300, VERMELHO_KC),
        jogador(img, 400, 300, AZUL_BUF),
        jogador(img, 500, 300, AZUL_BUF),
        arbitro(img, 650, 300),
        jogador(img, 300, 10, VERMELHO_KC),
    ]
    caminho = tmp_path / "foto.png"
    cv2.imwrite(str(caminho), img)
    return caminho, caixas


class LeitorFalso:
    def __init__(self, respostas):
        self.respostas = list(respostas)

    def ler(self, img):
        return self.respostas.pop(0) if self.respostas else []


@pytest.fixture
def modelos_falsos(monkeypatch, foto_sintetica):
    """Substitui YOLO e PaddleOCR. Conta as chamadas ao detector."""
    from nfl_vision.schemas import Deteccao
    from nfl_vision.stages import detect, jersey

    _, caixas = foto_sintetica
    chamadas = {"detect": 0}

    def detectar(img, cfg):
        chamadas["detect"] += 1
        return [Deteccao(det_id=i, bbox=b, confianca=0.9) for i, b in enumerate(caixas)]

    leitor = LeitorFalso([[("87", 0.95)], [("15", 0.90)], [("17", 0.92)], [("3x", 0.99)]])
    monkeypatch.setattr(detect, "detectar_pessoas", detectar)
    monkeypatch.setattr(jersey, "leitor_padrao", lambda device: leitor)
    return chamadas
```

- [ ] **Step 2: Escrever o teste**

`pipeline/tests/test_pipeline.py`:

```python
import json

import pytest

from nfl_vision import pipeline
from nfl_vision.runner import ler_manifest
from nfl_vision.schemas import Contexto

CTX = Contexto(temporada=2025, semana=11, times=("KC", "BUF"))


def _por_id(analise):
    return {j.track_id: j for j in analise.jogadores}


def test_analise_completa(dados, foto_sintetica, modelos_falsos):
    run_dir, analise = pipeline.analisar(foto_sintetica[0], CTX)

    jogadores = _por_id(analise)
    assert sorted(jogadores) == [0, 1, 2, 3]  # sem árbitro (4) e arquibancada (5)
    assert (jogadores[0].time, jogadores[0].numero, jogadores[0].posicao, jogadores[0].nome) == (
        "KC", 87, "TE", "Jogador KC 87")
    assert jogadores[2].nome == "Jogador BUF 17"
    assert jogadores[3].time == "BUF" and jogadores[3].numero is None
    assert analise.midia == {"tipo": "foto", "largura": 800, "altura": 600}
    assert analise.modelos == {"detector": "yolo11m", "ocr": "paddleocr"}

    assert (run_dir / "anotada.png").exists()
    salvo = json.loads((run_dir / "analise.json").read_text("utf-8"))
    assert salvo["analise_id"] == run_dir.name
    assert all(e["status"] == "ok" for e in ler_manifest(run_dir)["etapas"].values())


def test_reprocessar_nao_roda_detector_de_novo(dados, foto_sintetica, modelos_falsos):
    run_dir, _ = pipeline.analisar(foto_sintetica[0], CTX)
    pipeline.reprocessar(run_dir.name, "roster")
    assert modelos_falsos["detect"] == 1


def test_corrigir_numero(dados, foto_sintetica, modelos_falsos):
    run_dir, _ = pipeline.analisar(foto_sintetica[0], CTX)

    analise = pipeline.corrigir(run_dir.name, det_id=3, numero=14)

    j = _por_id(analise)[3]
    assert (j.numero, j.nome, j.confianca_numero, j.corrigido_pelo_usuario) == (
        14, "Jogador BUF 14 Ativo", 1.0, True)
    assert json.loads((run_dir / "corrections.json").read_text("utf-8"))[0]["numero"] == 14


def test_corrigir_valida_entrada(dados, foto_sintetica, modelos_falsos):
    run_dir, _ = pipeline.analisar(foto_sintetica[0], CTX)
    with pytest.raises(ValueError, match="não é um jogador"):
        pipeline.corrigir(run_dir.name, det_id=4, numero=10)
    with pytest.raises(ValueError, match="time deve ser"):
        pipeline.corrigir(run_dir.name, det_id=0, time="LA")
    with pytest.raises(ValueError, match="informe"):
        pipeline.corrigir(run_dir.name, det_id=0)
    with pytest.raises(FileNotFoundError):
        pipeline.corrigir("2000-01-01-001", det_id=0, numero=1)
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL com `ImportError: cannot import name 'pipeline'`

- [ ] **Step 4: Implementar `montagem.py`**

```python
"""Monta o analise.json (formato do SDD) a partir das saídas das etapas."""

from pathlib import Path

from nfl_vision.schemas import Analise, Jogador
from nfl_vision.stages.roster import aplicar_correcoes


def montar(estado) -> Analise:
    ingest = estado.saidas["ingest"]
    numeros = {n.det_id: n for n in estado.saidas["jersey"].itens}
    roster = {r.det_id: r for r in estado.saidas["roster"].itens}
    correcoes = estado.correcoes()
    time_final, numero_final = aplicar_correcoes(
        estado.saidas["team"].itens, numeros.values(), correcoes
    )
    corrigidos = {c.det_id for c in correcoes}
    numero_corrigido = {c.det_id for c in correcoes if c.numero is not None}

    jogadores = []
    for det_id in sorted(time_final):
        leitura, r = numeros.get(det_id), roster.get(det_id)
        if det_id in numero_corrigido:
            conf = 1.0
        else:
            conf = leitura.confianca if leitura else 0.0
        jogadores.append(Jogador(
            track_id=det_id,
            time=time_final[det_id],
            numero=numero_final.get(det_id),
            confianca_numero=conf,
            posicao=r.posicao if r else None,
            nome=r.nome if r else None,
            frames_visiveis=[0],
            corrigido_pelo_usuario=det_id in corrigidos,
        ))

    return Analise(
        analise_id=estado.run_dir.name,
        midia={"tipo": "foto", "largura": ingest.largura, "altura": ingest.altura},
        contexto=estado.contexto,
        modelos={"detector": Path(estado.config.detector_pesos).stem, "ocr": "paddleocr"},
        jogadores=jogadores,
    )
```

- [ ] **Step 5: Implementar `pipeline.py`**

```python
"""Pipeline concreto: etapas na ordem, análise nova, reprocessamento e correção."""

import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cv2

from nfl_vision import paths, teams
from nfl_vision.config import Config
from nfl_vision.cores import hex_para_bgr
from nfl_vision.montagem import montar
from nfl_vision.render import desenhar
from nfl_vision.runner import Estado, Etapa, Runner
from nfl_vision.schemas import (
    Analise, Contexto, Correcao, DetectOut, IngestOut, JerseyOut, RosterOut, TeamOut,
)
from nfl_vision.stages import detect, ingest, jersey, roster, team

ETAPAS = [
    Etapa("ingest", IngestOut, ingest.executar),
    Etapa("detect", DetectOut, detect.executar),
    Etapa("team", TeamOut, team.executar),
    Etapa("jersey", JerseyOut, jersey.executar),
    Etapa("roster", RosterOut, roster.executar),
]
NOMES_ETAPAS = [e.nome for e in ETAPAS]


def _versao(pacote: str) -> str:
    try:
        return version(pacote)
    except PackageNotFoundError:
        return "não instalado"


def coletar_versoes() -> dict[str, str]:
    return {p: _versao(p) for p in ("nfl-vision", "ultralytics", "torch", "paddleocr", "nflreadpy")}


def _runner() -> Runner:
    return Runner(ETAPAS, paths.runs_dir())


def _run_dir(analise_id: str) -> Path:
    run_dir = paths.runs_dir() / analise_id
    if not run_dir.exists():
        raise FileNotFoundError(f"análise '{analise_id}' não encontrada em {paths.runs_dir()}")
    return run_dir


def finalizar(estado: Estado) -> Analise:
    analise = montar(estado)
    (estado.run_dir / "analise.json").write_text(analise.model_dump_json(indent=2), encoding="utf-8")

    times_df = teams.carregar_times(paths.cache_dir())
    cores = {t: hex_para_bgr(teams.cores(t, times_df)[0]) for t in estado.contexto.times}
    caixas = {d.det_id: d.bbox for d in estado.saidas["detect"].deteccoes}
    anotada = desenhar(estado.imagem(), analise.jogadores, caixas, cores)
    cv2.imencode(".png", anotada)[1].tofile(str(estado.run_dir / "anotada.png"))
    return analise


def analisar(imagem: Path, contexto: Contexto, config: Config | None = None) -> tuple[Path, Analise]:
    runner = _runner()
    run_dir = runner.nova_analise(imagem, contexto, config or Config(), coletar_versoes())
    return run_dir, finalizar(runner.executar(run_dir))


def reprocessar(analise_id: str, a_partir_de: str) -> tuple[Path, Analise]:
    run_dir = _run_dir(analise_id)
    return run_dir, finalizar(_runner().executar(run_dir, a_partir_de))


def corrigir(analise_id: str, det_id: int, time: str | None = None,
             numero: int | None = None) -> Analise:
    if time is None and numero is None:
        raise ValueError("informe --time e/ou --numero")
    run_dir = _run_dir(analise_id)
    analise = Analise.model_validate_json((run_dir / "analise.json").read_text("utf-8"))
    if det_id not in {j.track_id for j in analise.jogadores}:
        raise ValueError(f"det {det_id} não é um jogador desta análise")
    if time is not None:
        time = teams.normalizar(time)
        if time not in analise.contexto.times:
            raise ValueError(f"time deve ser um de {', '.join(analise.contexto.times)}")

    arquivo = run_dir / "corrections.json"
    lista = json.loads(arquivo.read_text("utf-8")) if arquivo.exists() else []
    lista.append(Correcao(
        det_id=det_id, time=time, numero=numero,
        timestamp=datetime.now(timezone.utc).isoformat(),
    ).model_dump(mode="json"))
    arquivo.write_text(json.dumps(lista, indent=2, ensure_ascii=False), encoding="utf-8")
    return reprocessar(analise_id, "roster")[1]
```

- [ ] **Step 6: Rodar e ver passar**

Run: `uv run pytest -v`
Expected: todos passam, incluindo os 4 de `test_pipeline.py`.

- [ ] **Step 7: Commit**

```bash
git add pipeline/nfl_vision/montagem.py pipeline/nfl_vision/pipeline.py pipeline/tests/conftest.py pipeline/tests/test_pipeline.py
git commit -m "feat: montagem do analise.json, imagem anotada, reprocessamento e correção"
```

---

### Task 13: CLI (`analyze`, `correct`)

**Files:**
- Create: `pipeline/nfl_vision/cli.py`
- Test: `pipeline/tests/test_cli.py`

- [ ] **Step 1: Escrever o teste**

`pipeline/tests/test_cli.py`:

```python
from typer.testing import CliRunner

from nfl_vision import paths
from nfl_vision.cli import app

runner = CliRunner()
BASE = ["--times", "KC", "BUF", "--temporada", "2025", "--semana", "11"]


def test_analyze_ok_e_correct(dados, foto_sintetica, modelos_falsos):
    r = runner.invoke(app, ["analyze", str(foto_sintetica[0]), *BASE])
    assert r.exit_code == 0, r.output
    run_dir = next(paths.runs_dir().iterdir())
    assert (run_dir / "analise.json").exists()

    r = runner.invoke(app, ["correct", run_dir.name, "--det", "3", "--numero", "14"])
    assert r.exit_code == 0, r.output
    assert (run_dir / "corrections.json").exists()


def test_analyze_reprocessa_com_from(dados, foto_sintetica, modelos_falsos):
    runner.invoke(app, ["analyze", str(foto_sintetica[0]), *BASE])
    run_id = next(paths.runs_dir().iterdir()).name
    r = runner.invoke(app, ["analyze", "--run", run_id, "--from", "jersey"])
    assert r.exit_code == 0, r.output


def test_validacoes(dados, tmp_path):
    video = tmp_path / "jogo.mp4"
    video.write_bytes(b"x")
    foto = tmp_path / "f.jpg"
    foto.write_bytes(b"x")

    casos = [
        (["analyze", str(video), *BASE], "use JPG ou PNG"),
        (["analyze", str(tmp_path / "nao.jpg"), *BASE], "não encontrado"),
        (["analyze", str(foto), "--times", "KC", "XYZ", "--temporada", "2025", "--semana", "11"], "XYZ"),
        (["analyze", str(foto), "--times", "KC", "kc", "--temporada", "2025", "--semana", "11"], "diferentes"),
        (["analyze", str(foto), "--times", "KC", "BUF", "--temporada", "2025", "--semana", "30"], "semana"),
        (["analyze", str(foto), "--times", "KC", "BUF", "--temporada", "1990", "--semana", "1"], "temporada"),
        (["analyze", "--run", "x", "--from", "zzz"], "--from"),
    ]
    for args, trecho in casos:
        r = runner.invoke(app, args)
        assert r.exit_code == 2, (args, r.output)
        assert trecho in r.output, (args, r.output)


def test_etapa_com_erro_sugere_from(dados, foto_sintetica, modelos_falsos, monkeypatch):
    from nfl_vision.stages import roster

    def quebrar(*a, **k):
        raise roster.RosterIndisponivel("sem rede")

    monkeypatch.setattr(roster, "carregar_roster", quebrar)
    r = runner.invoke(app, ["analyze", str(foto_sintetica[0]), *BASE])
    assert r.exit_code == 1
    assert "--from roster" in r.output
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'nfl_vision.cli'`

- [ ] **Step 3: Implementar**

`pipeline/nfl_vision/cli.py`:

```python
"""Comando `nfl-vision`."""

from pathlib import Path
from typing import List, Optional, Tuple

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from nfl_vision import paths, pipeline, teams
from nfl_vision.runner import EtapaFalhou
from nfl_vision.schemas import Analise, Contexto
from nfl_vision.stages import ingest

app = typer.Typer(help="Identificação de jogadores da NFL em fotos.", no_args_is_help=True)
eval_app = typer.Typer(help="Avaliação de detecção e de leitura de número.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")
console = Console()


@app.callback()
def _inicio() -> None:
    load_dotenv()


def _validar_contexto(times: Tuple[str, str], temporada: int, semana: int) -> Contexto:
    if temporada < 2002:
        raise typer.BadParameter("temporada deve ser 2002 ou posterior", param_hint="--temporada")
    if not 1 <= semana <= 22:
        raise typer.BadParameter("semana deve estar entre 1 e 22", param_hint="--semana")
    times_df = teams.carregar_times(paths.cache_dir())
    try:
        a, b = (teams.validar(t, times_df) for t in times)
    except teams.TimeDesconhecido as exc:
        raise typer.BadParameter(str(exc), param_hint="--times") from exc
    if a == b:
        raise typer.BadParameter("os dois times precisam ser diferentes", param_hint="--times")
    return Contexto(temporada=temporada, semana=semana, times=(a, b))


def _imprimir(analise: Analise, run_dir: Path) -> None:
    tabela = Table(title=f"Análise {analise.analise_id}")
    for coluna in ("det", "time", "nº", "conf.", "pos.", "nome"):
        tabela.add_column(coluna)
    for j in analise.jogadores:
        tabela.add_row(
            str(j.track_id), j.time or "?", "?" if j.numero is None else str(j.numero),
            f"{j.confianca_numero:.2f}", j.posicao or "", j.nome or "desconhecido",
        )
    console.print(tabela)
    console.print(f"Artefatos: {run_dir}")


@app.command()
def analyze(
    foto: Optional[Path] = typer.Argument(None, help="Foto JPG ou PNG"),
    times: Tuple[str, str] = typer.Option((None, None), "--times", help="Siglas dos dois times"),
    temporada: Optional[int] = typer.Option(None, "--temporada"),
    semana: Optional[int] = typer.Option(None, "--semana"),
    run: Optional[str] = typer.Option(None, "--run", help="Reprocessar uma análise existente"),
    a_partir_de: Optional[str] = typer.Option(None, "--from", help="Etapa inicial do reprocessamento"),
) -> None:
    """Analisa uma foto ou reprocessa uma análise a partir de uma etapa."""
    try:
        if run:
            if a_partir_de not in pipeline.NOMES_ETAPAS:
                raise typer.BadParameter(
                    f"use uma etapa: {', '.join(pipeline.NOMES_ETAPAS)}", param_hint="--from")
            run_dir, analise = pipeline.reprocessar(run, a_partir_de)
        else:
            if foto is None or None in times or temporada is None or semana is None:
                raise typer.BadParameter("informe a foto, --times, --temporada e --semana")
            if not foto.exists():
                raise typer.BadParameter(f"arquivo não encontrado: {foto}", param_hint="FOTO")
            try:
                ingest.validar_formato(foto)
            except ingest.FormatoNaoSuportado as exc:
                raise typer.BadParameter(str(exc), param_hint="FOTO") from exc
            contexto = _validar_contexto(times, temporada, semana)
            run_dir, analise = pipeline.analisar(foto, contexto)
    except EtapaFalhou as exc:
        console.print(f"[red]{exc}[/red]")
        console.print(f"Depois de resolver, rode: nfl-vision analyze --run {exc.analise_id} --from {exc.etapa}")
        raise typer.Exit(1)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc), param_hint="--run") from exc
    _imprimir(analise, run_dir)


@app.command()
def correct(
    analise_id: str = typer.Argument(..., help="ID da análise, ex.: 2026-10-03-001"),
    det: int = typer.Option(..., "--det", help="det_id (track_id) do jogador"),
    time: Optional[str] = typer.Option(None, "--time"),
    numero: Optional[int] = typer.Option(None, "--numero", min=0, max=99),
) -> None:
    """Corrige o time e/ou o número de um jogador e refaz a consulta ao roster."""
    try:
        analise = pipeline.corrigir(analise_id, det, time, numero)
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _imprimir(analise, paths.runs_dir() / analise_id)
```

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest -v`
Expected: todos passam. Se o Rich quebrar linhas nas mensagens de erro e algum `trecho in r.output` falhar, rodar os testes com `COLUMNS=200` definindo `monkeypatch.setenv("COLUMNS", "200")` no início de `test_validacoes`.

- [ ] **Step 5: Teste manual do help**

Run: `uv run nfl-vision --help` e `uv run nfl-vision analyze --help`
Expected: comandos `analyze`, `correct` e `eval` listados; opções em português.

- [ ] **Step 6: Commit**

```bash
git add pipeline/nfl_vision/cli.py pipeline/tests/test_cli.py
git commit -m "feat: CLI analyze e correct"
```

---

### Task 14: Métricas de avaliação (`eval/metricas.py`)

**Files:**
- Create: `pipeline/nfl_vision/eval/metricas.py`
- Test: `pipeline/tests/test_metricas.py`

- [ ] **Step 1: Escrever o teste**

`pipeline/tests/test_metricas.py`:

```python
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
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_metricas.py -v`
Expected: FAIL com `ModuleNotFoundError`

- [ ] **Step 3: Implementar**

`pipeline/nfl_vision/eval/metricas.py`:

```python
"""IoU, AP@0.5 (interpolação de todos os pontos) e fração de alvos casados."""

import numpy as np

from nfl_vision.schemas import BBox


def iou(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    uniao = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / uniao if uniao > 0 else 0.0


def average_precision(predicoes: list[tuple[int, float, BBox]], gts: dict[int, list[BBox]],
                      limiar_iou: float = 0.5) -> float:
    """predicoes: (id_imagem, confiança, caixa). gts: id_imagem -> caixas verdadeiras."""
    total = sum(len(v) for v in gts.values())
    if total == 0 or not predicoes:
        return 0.0
    usados = {k: [False] * len(v) for k, v in gts.items()}
    ordenadas = sorted(predicoes, key=lambda p: -p[1])
    tp = np.zeros(len(ordenadas))
    for i, (img_id, _, caixa) in enumerate(ordenadas):
        candidatos = gts.get(img_id, [])
        ious = [iou(caixa, g) for g in candidatos]
        j = int(np.argmax(ious)) if ious else -1
        if j >= 0 and ious[j] >= limiar_iou and not usados[img_id][j]:
            tp[i] = 1
            usados[img_id][j] = True
    tp_acum = np.cumsum(tp)
    recall = tp_acum / total
    precisao = tp_acum / np.arange(1, len(ordenadas) + 1)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precisao, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def fracao_casada(alvos: dict[int, list[BBox]], predicoes: dict[int, list[BBox]],
                  limiar_iou: float = 0.5) -> float:
    """Fração das caixas-alvo que têm alguma predição com IoU >= limiar."""
    total = casados = 0
    for img_id, caixas in alvos.items():
        for alvo in caixas:
            total += 1
            if any(iou(alvo, p) >= limiar_iou for p in predicoes.get(img_id, [])):
                casados += 1
    return casados / total if total else 0.0
```

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_metricas.py -v`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add pipeline/nfl_vision/eval/metricas.py pipeline/tests/test_metricas.py
git commit -m "feat: métricas IoU, AP@0.5 e fração casada"
```

---

### Task 15: Datasets de avaliação (`eval/datasets.py`)

**Files:**
- Create: `pipeline/nfl_vision/eval/datasets.py`
- Test: `pipeline/tests/test_datasets.py`

- [ ] **Step 1: Escrever o teste**

`pipeline/tests/test_datasets.py`:

```python
import pytest
from PIL import Image

from nfl_vision.eval.datasets import baixar, carregar_pastas, carregar_yolo


def test_carregar_yolo(tmp_path):
    (tmp_path / "data.yaml").write_text("names: ['ball', 'player', 'referee']\n", encoding="utf-8")
    imgs = tmp_path / "test" / "images"
    lbls = tmp_path / "test" / "labels"
    imgs.mkdir(parents=True)
    lbls.mkdir(parents=True)
    Image.new("RGB", (100, 50)).save(imgs / "a.jpg")
    Image.new("RGB", (100, 50)).save(imgs / "b.jpg")  # sem rótulos
    (lbls / "a.txt").write_text("1 0.5 0.5 0.2 0.4\n2 0.1 0.1 0.2 0.2\n0 0.5 0.5 0.1 0.1 0.2 0.2\n")

    amostras = carregar_yolo(tmp_path, "test")

    assert [a.imagem.name for a in amostras] == ["a.jpg", "b.jpg"]
    assert amostras[0].caixas["player"] == [pytest.approx((40.0, 15.0, 60.0, 35.0))]
    assert len(amostras[0].caixas["referee"]) == 1
    assert "ball" not in amostras[0].caixas  # linha de polígono ignorada
    assert amostras[1].caixas == {}


def test_carregar_pastas(tmp_path):
    for rotulo in ("87", "9"):
        (tmp_path / "test" / rotulo).mkdir(parents=True)
        Image.new("RGB", (10, 10)).save(tmp_path / "test" / rotulo / "x.jpg")
    assert [r for _, r in carregar_pastas(tmp_path, "test")] == ["87", "9"]


def test_baixar_exige_chave(tmp_path, monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ROBOFLOW_API_KEY"):
        baixar("ws", "proj", 1, "yolov11", tmp_path)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_datasets.py -v`
Expected: FAIL com `ModuleNotFoundError`

- [ ] **Step 3: Implementar**

`pipeline/nfl_vision/eval/datasets.py`:

```python
"""Download de datasets do Roboflow e leitura nos formatos YOLO e por pastas."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image

from nfl_vision.schemas import BBox

IMAGENS = {".jpg", ".jpeg", ".png"}


@dataclass
class AmostraDeteccao:
    imagem: Path
    caixas: dict[str, list[BBox]]  # classe -> caixas em pixels


def baixar(workspace: str, projeto: str, versao: int, formato: str, destino: Path) -> Path:
    chave = os.environ.get("ROBOFLOW_API_KEY")
    if not chave:
        raise RuntimeError("defina ROBOFLOW_API_KEY no .env (app.roboflow.com/settings/api)")
    alvo = destino / f"{projeto}-v{versao}-{formato}"
    if alvo.exists():
        return alvo
    from roboflow import Roboflow

    Roboflow(api_key=chave).workspace(workspace).project(projeto).version(versao).download(
        formato, location=str(alvo)
    )
    return alvo


def carregar_yolo(raiz: Path, split: str = "test") -> list[AmostraDeteccao]:
    nomes = yaml.safe_load((raiz / "data.yaml").read_text("utf-8"))["names"]
    if isinstance(nomes, dict):
        nomes = [nomes[k] for k in sorted(nomes)]
    amostras = []
    for caminho in sorted((raiz / split / "images").iterdir()):
        if caminho.suffix.lower() not in IMAGENS:
            continue
        with Image.open(caminho) as im:
            w, h = im.size
        caixas: dict[str, list[BBox]] = {}
        rotulos = raiz / split / "labels" / f"{caminho.stem}.txt"
        if rotulos.exists():
            for linha in rotulos.read_text().splitlines():
                partes = linha.split()
                if len(partes) != 5:  # polígonos e linhas vazias ficam de fora
                    continue
                classe = nomes[int(partes[0])]
                cx, cy, bw, bh = map(float, partes[1:])
                caixas.setdefault(classe, []).append(
                    ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
                )
        amostras.append(AmostraDeteccao(caminho, caixas))
    return amostras


def carregar_pastas(raiz: Path, split: str = "test") -> list[tuple[Path, str]]:
    """Formato de classificação: <split>/<rótulo>/<imagem>."""
    return [(p, p.parent.name) for p in sorted((raiz / split).glob("*/*"))
            if p.suffix.lower() in IMAGENS]
```

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_datasets.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add pipeline/nfl_vision/eval/datasets.py pipeline/tests/test_datasets.py
git commit -m "feat: download e leitura de datasets de avaliação"
```

---

### Task 16: Preditores, avaliação e comandos `eval`

**Files:**
- Create: `pipeline/nfl_vision/eval/preditores.py`
- Create: `pipeline/nfl_vision/eval/detect.py`
- Create: `pipeline/nfl_vision/eval/jersey.py`
- Modify: `pipeline/nfl_vision/cli.py` (comandos `eval baixar`, `eval detect`, `eval jersey`)
- Test: `pipeline/tests/test_avaliacao.py`

- [ ] **Step 1: Escrever o teste com preditor e leitor falsos**

`pipeline/tests/test_avaliacao.py`:

```python
from pathlib import Path

import pytest
from PIL import Image

from nfl_vision.eval import detect as eval_detect
from nfl_vision.eval import jersey as eval_jersey
from nfl_vision.eval.datasets import AmostraDeteccao

JOGADOR = (10.0, 10.0, 50.0, 90.0)
ARBITRO = (60.0, 10.0, 100.0, 90.0)


class PreditorFalso:
    nome = "falso"

    def __init__(self, caixas):
        self.caixas = caixas

    def prever(self, imagem):
        return [(0.9, c) for c in self.caixas]


def test_avaliar_deteccao():
    amostras = [AmostraDeteccao(Path("x.jpg"), {"player": [JOGADOR], "referee": [ARBITRO]})]

    so_jogador = eval_detect.avaliar(PreditorFalso([JOGADOR]), amostras)
    com_arbitro = eval_detect.avaliar(PreditorFalso([JOGADOR, ARBITRO]), amostras)

    assert so_jogador == {"preditor": "falso", "imagens": 1, "map50": 1.0, "arbitros_como_jogador": 0.0}
    assert com_arbitro["map50"] == 1.0  # árbitro não é GT de jogador, vira FP de menor rank
    assert com_arbitro["arbitros_como_jogador"] == 1.0


class LeitorFalso:
    def __init__(self, respostas):
        self.respostas = list(respostas)

    def ler(self, img):
        return self.respostas.pop(0)


def test_avaliar_ocr(tmp_path):
    amostras = []
    for i, rotulo in enumerate(["87", "9", "15", "-1"]):
        caminho = tmp_path / f"{i}.jpg"
        Image.new("RGB", (20, 20)).save(caminho)
        amostras.append((caminho, rotulo))
    leitor = LeitorFalso([[("87", 0.9)], [("8", 0.9)], [("15", 0.3)]])

    r = eval_jersey.avaliar(leitor, amostras, limiar=0.60)

    assert r == {
        "amostras": 3,                  # "-1" não é número legível
        "taxa_null": pytest.approx(1 / 3),
        "acuracia_entre_lidos": 0.5,
        "acuracia_geral": pytest.approx(1 / 3),
    }
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_avaliacao.py -v`
Expected: FAIL com `ImportError`

- [ ] **Step 3: Implementar `eval/preditores.py`**

```python
"""Preditores comparados na avaliação: o nosso, RF-DETR local e o modelo NFL do Roboflow."""

import os
from pathlib import Path
from typing import Protocol

from PIL import Image

from nfl_vision.config import Config
from nfl_vision.schemas import BBox
from nfl_vision.stages.detect import aplicar_filtros, detectar_pessoas
from nfl_vision.stages.ingest import carregar_imagem
from nfl_vision.stages.team import MIN_PIXELS, eh_arbitro, pixels_uteis, recorte_tronco

Predicao = tuple[float, BBox]


class Preditor(Protocol):
    nome: str

    def prever(self, imagem: Path) -> list[Predicao]: ...


class PreditorNosso:
    nome = "nosso (yolo11m + filtros + árbitro)"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def prever(self, imagem: Path) -> list[Predicao]:
        img = carregar_imagem(imagem)
        saida = []
        for d in aplicar_filtros(detectar_pessoas(img, self.cfg), img, self.cfg):
            if d.descartado:
                continue
            px = pixels_uteis(recorte_tronco(img, d.bbox), self.cfg)
            if len(px) >= MIN_PIXELS and eh_arbitro(px):
                continue
            saida.append((d.confianca, d.bbox))
        return saida


class PreditorRFDETR:
    nome = "rf-detr base (COCO, pessoa)"

    def __init__(self, conf: float = 0.25):
        from rfdetr import RFDETRBase
        from rfdetr.util.coco_classes import COCO_CLASSES

        self.modelo = RFDETRBase()
        self.conf = conf
        self.ids_pessoa = {i for i, nome in COCO_CLASSES.items() if nome == "person"}

    def prever(self, imagem: Path) -> list[Predicao]:
        with Image.open(imagem) as im:
            dets = self.modelo.predict(im.convert("RGB"), threshold=self.conf)
        return [
            (float(c), tuple(float(v) for v in caixa))
            for caixa, c, k in zip(dets.xyxy, dets.confidence, dets.class_id)
            if int(k) in self.ids_pessoa
        ]


class PreditorRoboflowNFL:
    def __init__(self, modelo_id: str, conf: float = 0.25):
        from inference_sdk import InferenceHTTPClient

        chave = os.environ.get("ROBOFLOW_API_KEY")
        if not chave:
            raise RuntimeError("defina ROBOFLOW_API_KEY no .env (app.roboflow.com/settings/api)")
        self.cliente = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=chave)
        self.modelo_id = modelo_id
        self.conf = conf
        self.nome = f"roboflow {modelo_id} (NFL, player)"

    def prever(self, imagem: Path) -> list[Predicao]:
        resposta = self.cliente.infer(str(imagem), model_id=self.modelo_id)
        saida = []
        for p in resposta.get("predictions", []):
            if p.get("class") != "player" or p["confidence"] < self.conf:
                continue
            x, y, w, h = p["x"], p["y"], p["width"], p["height"]
            saida.append((float(p["confidence"]), (x - w / 2, y - h / 2, x + w / 2, y + h / 2)))
        return saida
```

- [ ] **Step 4: Implementar `eval/detect.py`**

```python
"""Avaliação de detecção: mAP@0.5 de jogador e árbitros mantidos como jogador."""

from nfl_vision.eval.datasets import AmostraDeteccao
from nfl_vision.eval.metricas import average_precision, fracao_casada
from nfl_vision.eval.preditores import Preditor


def avaliar(preditor: Preditor, amostras: list[AmostraDeteccao],
            classe_alvo: str = "player", classe_arbitro: str = "referee") -> dict:
    predicoes, gts, arbitros, mantidas = [], {}, {}, {}
    for i, amostra in enumerate(amostras):
        gts[i] = amostra.caixas.get(classe_alvo, [])
        arbitros[i] = amostra.caixas.get(classe_arbitro, [])
        saida = preditor.prever(amostra.imagem)
        mantidas[i] = [caixa for _, caixa in saida]
        predicoes.extend((i, conf, caixa) for conf, caixa in saida)
    tem_arbitros = any(arbitros.values())
    return {
        "preditor": preditor.nome,
        "imagens": len(amostras),
        "map50": round(average_precision(predicoes, gts), 4),
        "arbitros_como_jogador": round(fracao_casada(arbitros, mantidas), 4) if tem_arbitros else None,
    }
```

- [ ] **Step 5: Implementar `eval/jersey.py`**

```python
"""Avaliação do OCR em recortes de números rotulados por pasta."""

from pathlib import Path

from nfl_vision.stages.ingest import carregar_imagem
from nfl_vision.stages.jersey import PADRAO, LeitorOCR, ampliar, escolher_numero


def avaliar(leitor: LeitorOCR, amostras: list[tuple[Path, str]], limiar: float,
            altura_min: int = 128) -> dict:
    legiveis = [(p, int(r)) for p, r in amostras if PADRAO.match(r)]
    acertos = nulos = 0
    for caminho, esperado in legiveis:
        img = ampliar(carregar_imagem(caminho), altura_min)
        numero, _, _ = escolher_numero(leitor.ler(img), limiar)
        if numero is None:
            nulos += 1
        elif numero == esperado:
            acertos += 1
    n = len(legiveis)
    lidos = n - nulos
    return {
        "amostras": n,
        "taxa_null": nulos / n if n else None,
        "acuracia_entre_lidos": acertos / lidos if lidos else None,
        "acuracia_geral": acertos / n if n else None,
    }
```

- [ ] **Step 6: Rodar e ver passar**

Run: `uv run pytest tests/test_avaliacao.py -v`
Expected: `2 passed`

- [ ] **Step 7: Acrescentar os comandos `eval` ao final de `pipeline/nfl_vision/cli.py`**

```python
def _salvar_avaliacao(nome: str, resultados) -> Path:
    import json
    from datetime import datetime

    destino = paths.avaliacoes_dir()
    destino.mkdir(parents=True, exist_ok=True)
    arquivo = destino / f"{nome}-{datetime.now():%Y%m%d-%H%M%S}.json"
    arquivo.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    return arquivo


@eval_app.command("baixar")
def eval_baixar(
    workspace: str = typer.Option(..., "--workspace"),
    projeto: str = typer.Option(..., "--projeto"),
    versao: int = typer.Option(..., "--versao"),
    formato: str = typer.Option("yolov11", "--formato", help="yolov11 (detecção) ou folder (recortes)"),
) -> None:
    """Baixa uma versão de dataset do Roboflow para data/datasets/."""
    from nfl_vision.eval.datasets import baixar

    console.print(f"Dataset em: {baixar(workspace, projeto, versao, formato, paths.datasets_dir())}")


@eval_app.command("detect")
def eval_detect_cmd(
    dataset: Path = typer.Option(..., "--dataset", help="Pasta do dataset em formato YOLO"),
    split: str = typer.Option("test", "--split"),
    benchmark: List[str] = typer.Option([], "--benchmark", help="rfdetr e/ou roboflow-nfl"),
    modelo_roboflow: str = typer.Option("nfl-player-model/4", "--modelo-roboflow"),
) -> None:
    """mAP@0.5 de jogador no split de teste, com benchmarks opcionais."""
    from nfl_vision.config import Config
    from nfl_vision.eval import detect as avaliacao
    from nfl_vision.eval import preditores
    from nfl_vision.eval.datasets import carregar_yolo

    lista = [preditores.PreditorNosso(Config())]
    for b in benchmark:
        if b == "rfdetr":
            lista.append(preditores.PreditorRFDETR())
        elif b == "roboflow-nfl":
            lista.append(preditores.PreditorRoboflowNFL(modelo_roboflow))
        else:
            raise typer.BadParameter("use rfdetr ou roboflow-nfl", param_hint="--benchmark")

    amostras = carregar_yolo(dataset, split)
    resultados = [avaliacao.avaliar(p, amostras) for p in lista]

    tabela = Table(title=f"Detecção — {dataset.name} ({split}, {len(amostras)} imagens)")
    for coluna in ("preditor", "mAP@0.5", "árbitros como jogador"):
        tabela.add_column(coluna)
    for r in resultados:
        arb = "—" if r["arbitros_como_jogador"] is None else f"{r['arbitros_como_jogador']:.2%}"
        tabela.add_row(r["preditor"], f"{r['map50']:.3f}", arb)
    console.print(tabela)
    console.print(f"Resultados: {_salvar_avaliacao('detect', resultados)}")


@eval_app.command("jersey")
def eval_jersey_cmd(
    dataset: Path = typer.Option(..., "--dataset", help="Pasta no formato <split>/<número>/<imagem>"),
    split: str = typer.Option("test", "--split"),
) -> None:
    """Acurácia do OCR em recortes de números legíveis."""
    from nfl_vision.config import Config
    from nfl_vision.eval import jersey as avaliacao
    from nfl_vision.eval.datasets import carregar_pastas
    from nfl_vision.stages.jersey import leitor_padrao

    cfg = Config()
    r = avaliacao.avaliar(leitor_padrao(cfg.ocr_device), carregar_pastas(dataset, split),
                          cfg.limiar_numero, cfg.numero_altura_min)
    console.print(r)
    console.print(f"Resultados: {_salvar_avaliacao('jersey', r)}")
```

- [ ] **Step 8: Instalar o extra de avaliação e conferir importações**

```bash
uv sync --extra ocr --extra eval
uv run python -c "from rfdetr import RFDETRBase; from rfdetr.util.coco_classes import COCO_CLASSES; print([k for k, v in COCO_CLASSES.items() if v == 'person'])"
uv run nfl-vision eval --help
uv run pytest -v
```

Expected: imprime o id da classe pessoa (ex.: `[1]`); help lista `baixar`, `detect`, `jersey`; todos os testes passam. Se o caminho `rfdetr.util.coco_classes` não existir na versão instalada, achar com `uv run python -c "import rfdetr, pkgutil; print([m.name for m in pkgutil.walk_packages(rfdetr.__path__, 'rfdetr.')])"` e ajustar o import em `PreditorRFDETR`. Se a instalação do `rfdetr` puxar outra versão do torch, conferir de novo `uv run pytest -m model tests/test_ambiente.py`.

- [ ] **Step 9: Commit**

```bash
git add pipeline/nfl_vision/eval pipeline/nfl_vision/cli.py pipeline/tests/test_avaliacao.py pipeline/pyproject.toml pipeline/uv.lock
git commit -m "feat: avaliação de detecção e OCR com benchmarks rfdetr e roboflow-nfl"
```

---

### Task 17: Linha de base, foto real e documentação

**Files:**
- Create: `docs/avaliacao/2026-10-linha-de-base.md`
- Create: `README.md`
- Create: `.env` (local, não versionado)

- [ ] **Step 1: Confirmar o fork e a versão (via MCP do Roboflow)**

Com `projects_get` no projeto do fork em `elyas-carvalho`, anotar o slug exato e a versão com splits `train/valid/test`. Se o fork não tiver versão gerada, gerar uma com `versions_generate` **sem** augmentation (pedir confirmação ao usuário). Com `models_list` no projeto original `nflplayerdetection-mjrl1/nfl-player-model`, escolher o modelo treinado mais recente para `--modelo-roboflow` (formato `nfl-player-model/<versão>`).

- [ ] **Step 2: Criar `.env` na raiz do repositório**

O usuário cola a própria chave (nunca no chat):

```
ROBOFLOW_API_KEY=<chave de app.roboflow.com/settings/api>
```

Conferir que `git status` não lista o `.env`.

- [ ] **Step 3: Baixar os datasets**

```bash
cd pipeline
uv run nfl-vision eval baixar --workspace elyas-carvalho --projeto <slug-do-fork> --versao <n> --formato yolov11
uv run nfl-vision eval baixar --workspace taiseis-workspace --projeto jersey-number-ijbaq --versao 1 --formato folder
```

Expected: as duas pastas em `data/datasets/`. Se o dataset de números não tiver split `test`, usar `--split valid` no passo seguinte e registrar isso.

- [ ] **Step 4: Rodar as avaliações**

```bash
uv run nfl-vision eval detect --dataset ../data/datasets/<pasta-do-fork> --benchmark rfdetr --benchmark roboflow-nfl --modelo-roboflow nfl-player-model/<n>
uv run nfl-vision eval jersey --dataset ../data/datasets/jersey-number-ijbaq-v1-folder
```

Expected: tabela com três linhas de detecção e o dicionário de OCR; JSONs salvos em `data/avaliacoes/`.

- [ ] **Step 5: Rodar todos os testes de modelo**

Run: `uv run pytest -m model -v`
Expected: `3 passed` (GPU, YOLO, PaddleOCR).

- [ ] **Step 6: Foto real de ponta a ponta**

Pedir ao usuário uma captura de uma partida conhecida (times, temporada, semana). Rodar:

```bash
uv run nfl-vision analyze <captura.jpg> --times <A> <B> --temporada <AAAA> --semana <N>
```

Abrir `data/runs/<id>/anotada.png` e conferir com o usuário: caixas sobre os jogadores, árbitros sem rótulo, rótulos coerentes. Testar `correct` num jogador errado ou desconhecido.

- [ ] **Step 7: Registrar a linha de base**

`docs/avaliacao/2026-10-linha-de-base.md`, preenchido com os números reais dos JSONs do Step 4:

```markdown
# Linha de base — pipeline em fotos

Data: <data da execução> · Commit: <hash curto>

## Detecção (split de teste de <slug-do-fork> v<n>, <k> imagens)

| Preditor | mAP@0.5 jogador | Árbitros mantidos como jogador |
| --- | --- | --- |
| nosso (yolo11m + filtros + árbitro) | <valor> | <valor> |
| rf-detr base (COCO, pessoa) | <valor> | <valor> |
| roboflow nfl-player-model/<n> | <valor> | <valor> |

Meta do SDD: ≥ 0,85.

## Número (jersey-number-ijbaq v1, <split>, <k> recortes legíveis; outro esporte)

| Métrica | Valor |
| --- | --- |
| Acurácia entre lidos | <valor> |
| Taxa de desconhecido | <valor> |
| Acurácia geral | <valor> |

Meta do SDD: ≥ 0,80 entre os legíveis.

## Observações

<o que a foto real mostrou; erros típicos; próximos ajustes de limiar>

Dataset de detecção: nflplayerdetection-mjrl1/nfl-player-model (CC BY 4.0).
```

- [ ] **Step 8: Criar `README.md`**

```markdown
# NFL — Visão computacional

Identifica jogadores da NFL em fotos de partidas: time, número, posição e nome.
Projeto de estudo; detalhes no [SDD](SDD%20—%20Visão%20computacional%20NFL.md) e na
[spec do pipeline em fotos](docs/superpowers/specs/2026-10-03-pipeline-fotos-design.md).

## Setup (Windows + GPU NVIDIA)

1. Instale o [uv](https://docs.astral.sh/uv/).
2. `cd pipeline && uv sync --extra ocr --extra eval`
3. Para avaliar com datasets do Roboflow, crie `.env` na raiz com `ROBOFLOW_API_KEY=...`.

## Uso

    uv run nfl-vision analyze foto.jpg --times KC BUF --temporada 2025 --semana 11
    uv run nfl-vision analyze --run 2026-10-03-001 --from jersey
    uv run nfl-vision correct 2026-10-03-001 --det 4 --numero 87
    uv run nfl-vision eval detect --dataset ../data/datasets/<pasta> --benchmark rfdetr

Os resultados ficam em `data/runs/<id>/`: `analise.json`, `anotada.png` e a saída de cada etapa.

## Testes

    uv run pytest            # rápidos, sem modelos
    uv run pytest -m model   # com GPU, YOLO e PaddleOCR

Fotos e vídeos da NFL são usados só para estudo pessoal e não são publicados.
```

- [ ] **Step 9: Commit**

```bash
cd ..
git add README.md docs/avaliacao
git commit -m "docs: linha de base de detecção e OCR, README"
git push origin main
```

---

## Self-review (feito)

- **Cobertura da spec:** §2 ambiente → T1; §4 contratos/artefatos/manifest/correções → T2, T4, T12; §5 CLI → T13, T16; §6 módulos → T5, T7, T8, T9, T10, T11; §7 erros → T4 (falha + retomada), T10 (sem rede), T13 (validações, sugestão de `--from`); §8 testes → todas as tasks; §9 avaliação e benchmarks → T14–T17; §10 critério de pronto → T17.
- **Fora do plano por depender do usuário:** captura de jogo conhecido (T17 Step 6) e teste golden ponta a ponta, que entra quando houver capturas rotuladas.
- **Consistência de nomes:** `Estado`, `Etapa`, `EtapaFalhou(etapa, mensagem, analise_id)`, `aplicar_correcoes`, `leitor_padrao`, `detectar_pessoas`, `NOMES_ETAPAS` usados igual em todas as tasks.
