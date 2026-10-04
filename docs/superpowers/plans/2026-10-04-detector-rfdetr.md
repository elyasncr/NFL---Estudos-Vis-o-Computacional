# RF-DETR como detector padrão — Plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tornar o RF-DETR base (COCO) o detector padrão da etapa `detect`, mantendo o YOLO como opção, com o mesmo código servindo o pipeline e a avaliação, e com resolução e limiar escolhidos por medição.

**Architecture:** `Config` ganha `detector_tipo`, `detector_modelo_rfdetr` e `detector_resolucao`. Em `stages/detect.py`, `detectar_pessoas(img, cfg)` continua sendo o ponto único de entrada e **despacha por dentro** (`_detectar_rfdetr` / `_detectar_yolo`); o modelo RF-DETR fica atrás de `_modelo_rfdetr(variante, resolucao, device)` (`lru_cache`), que os testes rápidos substituem por um falso. `caminho_pesos(cfg)` devolve o arquivo realmente carregado (`ckpt_path` no YOLO, `model_config.pretrain_weights` no RF-DETR, ou `None`). `rotulo_pesos` sai de `eval/preditores.py` para `stages/detect.py` (evita import circular) junto com o novo `rotulo_detector(cfg)` usado por `montagem` e pelos nomes dos preditores. Os preditores de avaliação passam a chamar `detectar_pessoas` com a config forçada para o tipo deles. CLI: `analyze --detector-tipo`, `eval detect --detector-tipo/--resolucao`; `--detector`/`--pesos` com um `.pt` implicam `yolo`.

**Decisão sobre a fixture `modelos_falsos`:** como o despacho acontece **dentro** de `detectar_pessoas`, o patch de `detect.detectar_pessoas` da fixture continua cobrindo os dois detectores. A fixture só ganha o patch de `detect._modelo_rfdetr` (um falso com `model_config.pretrain_weights`), para que o hash dos pesos no `executar` não carregue o RF-DETR real.

**Tech Stack:** Python 3.12, uv, rfdetr 1.11.1 (`RFDETRBase(resolution=..., device=...)`, `predict(PIL RGB, threshold=)` → `supervision.Detections`), Ultralytics, PyTorch 2.11 cu128, Typer + Rich, pytest.

**Spec:** `docs/superpowers/specs/2026-10-04-detector-rfdetr-design.md`

## Convenções

- Todos os comandos rodam a partir de `pipeline/`, num Git Bash com `export PATH="/c/Users/User/.local/bin:$PATH"` (onde está o `uv`).
- Testes rápidos: `uv run pytest`. Testes com modelos reais: `uv run pytest -m model`. Linha de base antes deste plano: `225 passed, 4 deselected`.
- Commits em português, **sem** linhas `Co-Authored-By` e sem menção a Claude/IA.
- Textos de interface, mensagens de erro e nomes de domínio em português.
- Erros de entrada do usuário viram `typer.BadParameter` (saída 2, sem traceback).
- **Padrões provisórios:** `detector_resolucao = 896` e `detector_conf = 0.3` em todas as tasks até a Task 5, que grava os valores medidos. Nenhum teste fixa esses números: os testes usam `Config().detector_resolucao` / `Config().detector_conf`.

## Mapa de arquivos

```
pipeline/
  pyproject.toml               # rfdetr sai do extra `eval` e vira dependência principal
  uv.lock                      # regenerado
  nfl_vision/
    config.py                  # + detector_tipo, detector_modelo_rfdetr, detector_resolucao (múltiplo de 56); detector_conf 0.3
    stages/detect.py           # + DETECTORES, rotulo_pesos (movida), rotulo_detector, _modelo_rfdetr, _classes_coco (movida),
                               #   _eh_pessoa, _detectar_rfdetr, _detectar_yolo; detectar_pessoas despacha; caminho_pesos(cfg)
    eval/preditores.py         # PreditorNosso segue detector_tipo; PreditorYoloBruto força yolo; PreditorRFDETR(cfg) usa detectar_pessoas
    montagem.py                # modelos["detector"] = rotulo_detector(cfg)
    pipeline.py                # coletar_versoes inclui rfdetr
    cli.py                     # analyze --detector-tipo; eval detect --detector-tipo, --resolucao; PreditorRFDETR(cfg)
  tests/
    conftest.py                # modelos_falsos também substitui detect._modelo_rfdetr
    test_config.py             # novo: padrões, validação, manifest antigo
    test_runner.py             # + manifest antigo sem os campos do RF-DETR
    test_detect.py             # + despacho, RF-DETR falso, hash, rótulos; @model RF-DETR em bus.jpg
    test_avaliacao.py          # preditores sobre detectar_pessoas; testes do RF-DETR movidos para test_detect
    test_pipeline.py           # modelos["detector"] do RF-DETR e do YOLO; versão do rfdetr
    test_cli.py                # --detector-tipo, --resolucao, combinações inválidas
```

---

### Task 1: Campos novos na `Config`

**Files:**
- Modify: `pipeline/nfl_vision/config.py`
- Create: `pipeline/tests/test_config.py`
- Modify: `pipeline/tests/test_runner.py` (novo teste no fim)

- [ ] **Step 1: Escrever os testes que falham**

Criar `pipeline/tests/test_config.py`:

```python
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
```

Acrescentar ao fim de `pipeline/tests/test_runner.py`:

```python
def test_manifest_antigo_sem_campos_do_rfdetr_ganha_padroes(tmp_path, foto):
    chamadas = []
    runner = Runner(_etapas(chamadas), tmp_path / "runs")
    run_dir = runner.nova_analise(foto, CTX, Config(), {})
    runner.executar(run_dir)

    manifest = ler_manifest(run_dir)
    antigo = {k: v for k, v in manifest["config"].items()
              if k not in ("detector_tipo", "detector_modelo_rfdetr", "detector_resolucao")}
    manifest["config"] = antigo
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    estado = runner.executar(run_dir, a_partir_de="b")

    assert estado.config.detector_tipo == "rfdetr"
    novo = ler_manifest(run_dir)
    assert novo["config"]["detector_tipo"] == "rfdetr"
    assert novo["config"]["detector_resolucao"] == Config().detector_resolucao
    assert novo["config_original"] == antigo
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_config.py tests/test_runner.py -q`
Expected: FAIL — `AttributeError: 'Config' object has no attribute 'detector_tipo'` (e `detector_resolucao`); os três casos de `test_valores_invalidos` falham com `DID NOT RAISE`, porque `extra="ignore"` ignora os campos ainda desconhecidos.

- [ ] **Step 3: Implementar**

Substituir `pipeline/nfl_vision/config.py` inteiro por:

```python
"""Parâmetros do pipeline. A configuração usada é gravada em cada análise."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class Config(BaseModel):
    # Campos desconhecidos (removidos numa versão futura) não quebram a leitura
    # de um manifest antigo; campos novos ausentes usam o valor padrão.
    model_config = ConfigDict(extra="ignore")

    # "rfdetr" (padrão) ou "yolo"; detector_conf vale para os dois
    detector_tipo: Literal["rfdetr", "yolo"] = "rfdetr"
    detector_modelo_rfdetr: str = "base"  # variante: classe RFDETR<Variante> do pacote rfdetr
    detector_resolucao: int = 896         # lado de entrada do RF-DETR; múltiplo de 56
    detector_conf: float = 0.3
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
```

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_config.py tests/test_runner.py -q`
Expected: PASS (todos).

Run: `uv run pytest -q`
Expected: todos passam (7 testes a mais que a linha de base). O padrão novo `detector_tipo="rfdetr"` ainda não tem efeito: `detectar_pessoas` só passa a despachar na Task 2.

- [ ] **Step 5: Commit**

```bash
git add nfl_vision/config.py tests/test_config.py tests/test_runner.py
git commit -m "feat: campos do RF-DETR na config (tipo, variante, resolução)"
```

---

### Task 2: Etapa `detect` com RF-DETR e despacho por tipo

**Files:**
- Modify: `pipeline/nfl_vision/stages/detect.py`
- Modify: `pipeline/nfl_vision/pipeline.py` (`coletar_versoes`)
- Modify: `pipeline/pyproject.toml`, `pipeline/uv.lock`
- Modify: `pipeline/tests/conftest.py` (fixture `modelos_falsos`)
- Modify: `pipeline/tests/test_detect.py`

- [ ] **Step 1: rfdetr vira dependência principal**

O detector padrão não pode depender de um extra opcional. Em `pipeline/pyproject.toml`, acrescentar `"rfdetr>=1.2",` ao fim da lista `dependencies` (depois de `"ultralytics>=8.3",`) e trocar a linha do extra `eval` por:

```toml
eval = ["roboflow>=1.1", "inference-sdk>=0.50"]
```

Run: `uv lock && uv sync --extra ocr --extra eval`
Expected: `Resolved ... packages` sem erro; `uv sync` não remove o `rfdetr` (`uv run python -c "import rfdetr; print(rfdetr.__version__)"` imprime `1.11.1`).

- [ ] **Step 2: Escrever os testes que falham**

Em `pipeline/tests/test_detect.py`:

1. Trocar o teste `@model` existente `test_yolo_detecta_pessoas_em_imagem_real` por:

```python
@pytest.mark.model
def test_yolo_detecta_pessoas_em_imagem_real():
    from ultralytics.utils import ASSETS

    from nfl_vision.stages.detect import detectar_pessoas
    from nfl_vision.stages.ingest import carregar_imagem

    cfg = Config(detector_tipo="yolo")
    dets = detectar_pessoas(carregar_imagem(ASSETS / "bus.jpg"), cfg)
    assert len(dets) >= 3
    assert all(d.confianca >= cfg.detector_conf for d in dets)


@pytest.mark.model
def test_rfdetr_detecta_pessoas_em_imagem_real():
    from ultralytics.utils import ASSETS

    from nfl_vision.stages.detect import caminho_pesos, detectar_pessoas, sha256_pesos
    from nfl_vision.stages.ingest import carregar_imagem

    cfg = Config()
    assert cfg.detector_tipo == "rfdetr"
    dets = detectar_pessoas(carregar_imagem(ASSETS / "bus.jpg"), cfg)
    assert len(dets) >= 3
    assert all(d.confianca >= cfg.detector_conf for d in dets)
    assert [d.det_id for d in dets] == list(range(len(dets)))
    caminho = caminho_pesos(cfg)
    assert caminho is not None and caminho.endswith(".pth")
    assert sha256_pesos(caminho) is not None
```

2. No teste `test_hash_dos_pesos_usa_o_caminho_carregado`, trocar `config=Config()` por `config=Config(detector_tipo="yolo")`.

3. Acrescentar ao fim do arquivo:

```python
def _dets_falsas(xyxy, confs, class_ids, nomes=None):
    from types import SimpleNamespace

    data = {} if nomes is None else {"class_name": np.array(nomes, dtype=object)}
    return SimpleNamespace(xyxy=np.array(xyxy, float).reshape(-1, 4), confidence=np.array(confs, float),
                           class_id=np.array(class_ids, int), data=data)


class ModeloRFDETRFalso:
    def __init__(self, dets, pesos=None):
        from types import SimpleNamespace

        self.dets = dets
        self.chamadas = []
        self.model_config = SimpleNamespace(pretrain_weights=pesos)

    def predict(self, imagem, threshold):
        self.chamadas.append((imagem, threshold))
        return self.dets


def _usar_rfdetr_falso(monkeypatch, modelo):
    from nfl_vision.stages import detect

    cargas = []

    def carregar(variante, resolucao, device):
        cargas.append((variante, resolucao, device))
        return modelo

    monkeypatch.setattr(detect, "_modelo_rfdetr", carregar)
    return cargas


def test_detectar_pessoas_despacha_pelo_tipo(monkeypatch):
    from nfl_vision.stages import detect

    monkeypatch.setattr(detect, "_detectar_rfdetr", lambda img, cfg: ["rfdetr"])
    monkeypatch.setattr(detect, "_detectar_yolo", lambda img, cfg: ["yolo"])
    img = campo()

    assert detect.detectar_pessoas(img, Config()) == ["rfdetr"]
    assert detect.detectar_pessoas(img, Config(detector_tipo="rfdetr")) == ["rfdetr"]
    assert detect.detectar_pessoas(img, Config(detector_tipo="yolo")) == ["yolo"]


def test_rfdetr_mantem_so_pessoa_converte_caixas_e_ordena(monkeypatch):
    from nfl_vision.stages import detect

    dets = _dets_falsas([[0, 0, 5, 5], [1, 1, 6, 6], [2, 2, 9, 9]], [0.6, 0.7, 0.9], [1, 18, 1],
                        ["person", "dog", "person"])
    modelo = ModeloRFDETRFalso(dets)
    cargas = _usar_rfdetr_falso(monkeypatch, modelo)
    img = np.zeros((10, 20, 3), np.uint8)
    img[..., 0] = 255  # azul em BGR
    cfg = Config(device="cpu", detector_resolucao=560, detector_conf=0.4)

    saida = detect.detectar_pessoas(img, cfg)

    assert [(d.det_id, d.bbox, d.confianca) for d in saida] == [
        (0, (2.0, 2.0, 9.0, 9.0), 0.9), (1, (0.0, 0.0, 5.0, 5.0), 0.6)]
    assert all(isinstance(v, float) for d in saida for v in d.bbox)
    assert cargas == [("base", 560, "cpu")]
    imagem, limiar = modelo.chamadas[0]
    assert limiar == 0.4
    assert imagem.mode == "RGB" and imagem.size == (20, 10)
    assert imagem.getpixel((0, 0)) == (0, 0, 255)  # BGR -> RGB


def test_rfdetr_sem_class_name_usa_ids_do_coco(monkeypatch):
    from nfl_vision.stages import detect

    monkeypatch.setattr(detect, "_classes_coco", lambda: {1: "person", 18: "dog"})
    _usar_rfdetr_falso(monkeypatch, ModeloRFDETRFalso(
        _dets_falsas([[0, 0, 5, 5], [1, 1, 6, 6]], [0.8, 0.7], [18, 1])))

    saida = detect.detectar_pessoas(np.zeros((10, 20, 3), np.uint8), Config(device="cpu"))

    assert [(d.bbox, d.confianca) for d in saida] == [((1.0, 1.0, 6.0, 6.0), 0.7)]


def test_rfdetr_sem_deteccoes(monkeypatch):
    from nfl_vision.stages import detect

    _usar_rfdetr_falso(monkeypatch, ModeloRFDETRFalso(_dets_falsas([], [], [], [])))
    assert detect.detectar_pessoas(np.zeros((10, 20, 3), np.uint8), Config(device="cpu")) == []


def test_classes_coco_fallback_para_rfdetr_antigo(monkeypatch):
    import sys
    from types import ModuleType

    from nfl_vision.stages import detect

    antigo = ModuleType("rfdetr.util.coco_classes")
    antigo.COCO_CLASSES = {1: "person", 2: "bicycle"}
    monkeypatch.setitem(sys.modules, "rfdetr.assets.coco_classes", None)  # import falha
    monkeypatch.setitem(sys.modules, "rfdetr.util.coco_classes", antigo)

    assert detect._classes_coco() == {1: "person", 2: "bicycle"}


def test_hash_dos_pesos_do_rfdetr_usa_o_arquivo_carregado(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace

    from nfl_vision.stages import detect

    arquivo = tmp_path / "models" / "rf-detr-base.pth"
    arquivo.parent.mkdir()
    arquivo.write_bytes(b"pesos rfdetr")
    _usar_rfdetr_falso(monkeypatch, ModeloRFDETRFalso(None, pesos=str(arquivo)))
    monkeypatch.setattr(detect, "detectar_pessoas", lambda img, cfg: [])
    estado = SimpleNamespace(imagem=lambda: campo(), config=Config(device="cpu"))

    saida = detect.executar(estado)

    assert saida.pesos_sha256 == hashlib.sha256(b"pesos rfdetr").hexdigest()


def test_hash_dos_pesos_do_rfdetr_desconhecido_vira_none(monkeypatch):
    from types import SimpleNamespace

    from nfl_vision.stages import detect

    monkeypatch.setattr(detect, "_modelo_rfdetr", lambda *a: SimpleNamespace())  # sem model_config
    assert detect.caminho_pesos(Config(device="cpu")) is None
    assert detect.sha256_pesos(None) is None


def test_variante_do_rfdetr_desconhecida():
    from nfl_vision.stages import detect

    with pytest.raises(ValueError, match="variante do RF-DETR desconhecida"):
        detect._modelo_rfdetr("inexistente", 560, "cpu")


def test_rotulos_do_detector():
    from nfl_vision.stages.detect import rotulo_detector, rotulo_pesos

    assert rotulo_detector(Config(detector_resolucao=560)) == "rfdetr-base@560"
    assert rotulo_detector(Config(detector_tipo="yolo")) == "yolo11m"
    ajustado = Config(detector_tipo="yolo", detector_pesos="/x/treinos/player-v1/weights/best.pt")
    assert rotulo_detector(ajustado) == "player-v1"
    assert rotulo_pesos("yolo11m.pt") == "yolo11m"
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `uv run pytest tests/test_detect.py -q`
Expected: FAIL — `AttributeError: ... has no attribute '_detectar_rfdetr'` / `'_modelo_rfdetr'` / `'_classes_coco'`, `ImportError: cannot import name 'rotulo_detector'`.

- [ ] **Step 4: Implementar em `stages/detect.py`**

Trocar a docstring do módulo e os imports do topo por:

```python
"""Etapa 2: detecção de pessoas (RF-DETR ou YOLO) e descarte de quem está fora de campo."""

import hashlib
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from nfl_vision.config import Config
from nfl_vision.cores import mascara_gramado
from nfl_vision.schemas import Deteccao, DetectOut


LADO_MAX_CAMPO = 640
DETECTORES = ("rfdetr", "yolo")
```

Substituir tudo a partir de `@lru_cache(maxsize=2)` / `def _modelo(pesos: str):` até o fim do arquivo por:

```python
def rotulo_pesos(pesos: str) -> str:
    """Nome curto dos pesos YOLO: `.../player-v1/weights/best.pt` vira `player-v1`."""
    caminho = Path(pesos)
    if caminho.parent.name == "weights" and caminho.stem in ("best", "last"):
        treino = caminho.parent.parent.name
        return treino if caminho.stem == "best" else f"{treino}/last"
    return caminho.stem


def rotulo_detector(cfg: Config) -> str:
    """Nome curto do detector da config: `rfdetr-base@896` ou o rótulo dos pesos YOLO."""
    if cfg.detector_tipo == "rfdetr":
        return f"rfdetr-{cfg.detector_modelo_rfdetr}@{cfg.detector_resolucao}"
    return rotulo_pesos(cfg.detector_pesos)


@lru_cache(maxsize=2)
def _modelo(pesos: str):
    from ultralytics import YOLO

    return YOLO(pesos)


@lru_cache(maxsize=2)
def _modelo_rfdetr(variante: str, resolucao: int, device: str):
    import rfdetr

    classe = getattr(rfdetr, f"RFDETR{variante.capitalize()}", None)
    if classe is None:
        raise ValueError(f"variante do RF-DETR desconhecida: '{variante}'")
    return classe(resolution=resolucao, device=device)


def resolver_device(device: str) -> str:
    if device.startswith("cuda"):
        import torch

        return device if torch.cuda.is_available() else "cpu"
    return device


def _modelo_rfdetr_da_config(cfg: Config):
    return _modelo_rfdetr(cfg.detector_modelo_rfdetr, cfg.detector_resolucao,
                          resolver_device(cfg.device))


def _classes_coco() -> dict[int, str]:
    try:
        from rfdetr.assets.coco_classes import COCO_CLASSES
    except ImportError:  # rfdetr < 1.9
        from rfdetr.util.coco_classes import COCO_CLASSES
    return COCO_CLASSES


def _eh_pessoa(dets) -> list[bool]:
    nomes = (dets.data or {}).get("class_name")
    if nomes is not None:
        return [n == "person" for n in nomes]
    ids_pessoa = {i for i, nome in _classes_coco().items() if nome == "person"}
    return [int(k) in ids_pessoa for k in dets.class_id]


def _detectar_rfdetr(img: np.ndarray, cfg: Config) -> list[Deteccao]:
    # o pipeline trabalha em BGR (OpenCV); o RF-DETR espera RGB
    rgb = Image.fromarray(np.ascontiguousarray(img[:, :, ::-1]))
    dets = _modelo_rfdetr_da_config(cfg).predict(rgb, threshold=cfg.detector_conf)
    pessoas = sorted(
        ((float(conf), tuple(float(v) for v in caixa))
         for caixa, conf, pessoa in zip(dets.xyxy, dets.confidence, _eh_pessoa(dets)) if pessoa),
        key=lambda p: -p[0],  # como o YOLO: det_id 0 é a mais confiante
    )
    return [Deteccao(det_id=i, bbox=caixa, confianca=conf)
            for i, (conf, caixa) in enumerate(pessoas)]


def _detectar_yolo(img: np.ndarray, cfg: Config) -> list[Deteccao]:
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


def detectar_pessoas(img: np.ndarray, cfg: Config) -> list[Deteccao]:
    """Pessoas detectadas, sem filtros; o detector vem de `cfg.detector_tipo`."""
    if cfg.detector_tipo == "rfdetr":
        return _detectar_rfdetr(img, cfg)
    return _detectar_yolo(img, cfg)


def caminho_pesos(cfg: Config) -> str | None:
    """Arquivo de pesos que o detector realmente carregou (pode ter sido baixado para outro lugar).

    YOLO: `ckpt_path` do ultralytics. RF-DETR: `model_config.pretrain_weights`
    (ex.: `~/.roboflow/models/rf-detr-base.pth`); None se o pacote não expuser o caminho.
    """
    if cfg.detector_tipo == "rfdetr":
        config_modelo = getattr(_modelo_rfdetr_da_config(cfg), "model_config", None)
        caminho = getattr(config_modelo, "pretrain_weights", None)
        return str(caminho) if caminho else None
    return getattr(_modelo(cfg.detector_pesos), "ckpt_path", None) or cfg.detector_pesos


def sha256_pesos(pesos: str | None) -> str | None:
    if not pesos:
        return None
    caminho = Path(pesos)
    return hashlib.sha256(caminho.read_bytes()).hexdigest() if caminho.exists() else None


def executar(estado) -> DetectOut:
    img = estado.imagem()
    cfg = estado.config
    return DetectOut(
        deteccoes=aplicar_filtros(detectar_pessoas(img, cfg), img, cfg),
        pesos_sha256=sha256_pesos(caminho_pesos(cfg)),
    )
```

(`regiao_do_campo`, `toca_borda_inferior`, `_fora_de_campo` e `aplicar_filtros` não mudam.)

Verificação empírica do caminho dos pesos (já feita ao escrever o plano, repetir se o rfdetr mudar de versão):

Run: `uv run python -c "from rfdetr import RFDETRBase; print(RFDETRBase(resolution=896, device='cpu').model_config.pretrain_weights)"`
Expected: última linha `C:\Users\User\.roboflow\models\rf-detr-base.pth`. Se o atributo não existir numa versão futura, `caminho_pesos` devolve `None` e o manifest grava `detector_pesos_sha256: null` (o teste `test_hash_dos_pesos_do_rfdetr_desconhecido_vira_none` cobre esse caso).

- [ ] **Step 5: `eval/preditores.py` passa a importar os itens movidos**

Para o pacote continuar importável nesta task (os preditores são reescritos na Task 3), em `pipeline/nfl_vision/eval/preditores.py`:

- trocar `from nfl_vision.stages.detect import aplicar_filtros, detectar_pessoas` por
  `from nfl_vision.stages.detect import _classes_coco, aplicar_filtros, detectar_pessoas, rotulo_pesos`;
- apagar as definições locais de `rotulo_pesos` e `_classes_coco` (corpos idênticos aos movidos).

- [ ] **Step 6: `coletar_versoes` registra o rfdetr**

Em `pipeline/nfl_vision/pipeline.py`:

```python
def coletar_versoes() -> dict[str, str]:
    return {p: _versao(p) for p in
            ("nfl-vision", "rfdetr", "ultralytics", "torch", "paddleocr", "nflreadpy")}
```

- [ ] **Step 7: Fixture `modelos_falsos` cobre o carregador do RF-DETR**

O despacho fica dentro de `detectar_pessoas`, então o patch de `detect.detectar_pessoas` da fixture continua valendo para os dois detectores. Falta só o hash: com o padrão RF-DETR, `executar` chama `caminho_pesos(cfg)`, que carregaria o modelo real. Em `pipeline/tests/conftest.py`, na fixture `modelos_falsos`, trocar a docstring e a linha

```python
    monkeypatch.setattr(detect, "_modelo", lambda p: SimpleNamespace(ckpt_path=str(pesos)))
```

por

```python
    modelo = SimpleNamespace(ckpt_path=str(pesos),
                             model_config=SimpleNamespace(pretrain_weights=str(pesos)))
    monkeypatch.setattr(detect, "_modelo", lambda p: modelo)
    monkeypatch.setattr(detect, "_modelo_rfdetr", lambda variante, resolucao, device: modelo)
```

com a docstring `"""Substitui os detectores (RF-DETR e YOLO) e o PaddleOCR. Conta as chamadas ao detector."""`.

Em `pipeline/tests/test_pipeline.py`, no teste `test_manifest_registra_hash_dos_pesos`, logo depois de `assert "nfl-vision" in manifest["versoes"]`, acrescentar:

```python
    assert "rfdetr" in manifest["versoes"]
```

- [ ] **Step 8: Rodar e ver passar**

Run: `uv run pytest tests/test_detect.py tests/test_pipeline.py -q`
Expected: PASS.

Run: `uv run pytest -q`
Expected: todos passam, sem carregar modelo real (a suíte rápida continua em ~30 s). `test_avaliacao.py::test_classes_coco_fallback_para_rfdetr_antigo` ainda passa porque `preditores._classes_coco` agora é o nome importado de `detect`; ele é removido na Task 3.

Run: `uv run pytest -m model tests/test_detect.py -q`
Expected: PASS nos dois testes `@model` de detecção (RF-DETR acha ≥ 3 pessoas em `bus.jpg`, caminho dos pesos termina em `.pth`).

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock nfl_vision/stages/detect.py nfl_vision/eval/preditores.py nfl_vision/pipeline.py tests/conftest.py tests/test_detect.py tests/test_pipeline.py
git commit -m "feat: RF-DETR na etapa detect, com despacho por tipo e hash dos pesos carregados"
```

---

### Task 3: Preditores de avaliação e `analise.json` sobre o detector compartilhado

**Files:**
- Modify: `pipeline/nfl_vision/eval/preditores.py`
- Modify: `pipeline/nfl_vision/montagem.py`
- Modify: `pipeline/tests/test_avaliacao.py`
- Modify: `pipeline/tests/test_pipeline.py`

- [ ] **Step 1: Escrever os testes que falham**

Em `pipeline/tests/test_avaliacao.py`:

1. Apagar os testes `test_rfdetr_filtra_pessoa`, `test_rfdetr_usa_imagem_com_orientacao_exif` e `test_classes_coco_fallback_para_rfdetr_antigo` (o filtro de pessoa e o fallback agora são testados em `test_detect.py`).

2. Trocar `test_nomes_dos_preditores_indicam_os_pesos` por:

```python
def test_nomes_dos_preditores_indicam_o_detector():
    padrao = Config()
    res = padrao.detector_resolucao
    assert preditores.PreditorNosso(padrao).nome == f"nosso (rfdetr-base@{res} + filtros + árbitro)"
    assert preditores.PreditorYoloBruto(padrao).nome == "yolo11m bruto (COCO, pessoa)"
    assert preditores.PreditorRFDETR(padrao).nome == f"rfdetr bruto (rfdetr-base@{res}, COCO, pessoa)"
    assert preditores.PreditorRFDETR(Config(detector_resolucao=560)).nome == (
        "rfdetr bruto (rfdetr-base@560, COCO, pessoa)")
    yolo = Config(detector_tipo="yolo")
    assert preditores.PreditorNosso(yolo).nome == "nosso (yolo11m + filtros + árbitro)"
    ajustado = Config(detector_tipo="yolo", detector_pesos="data/treinos/player-v1/weights/best.pt")
    assert preditores.PreditorNosso(ajustado).nome == "nosso (player-v1 + filtros + árbitro)"
    assert preditores.PreditorYoloBruto(ajustado).nome == "yolo bruto (player-v1)"
```

3. Acrescentar ao fim do arquivo:

```python
def _espiar_detector(monkeypatch, caixas):
    """Substitui detectar_pessoas e guarda a config com que cada preditor o chamou."""
    configs = []

    def detectar(img, cfg):
        configs.append(cfg)
        return [Deteccao(det_id=i, bbox=b, confianca=0.9) for i, b in enumerate(caixas)]

    monkeypatch.setattr(preditores, "detectar_pessoas", detectar)
    return configs


def test_preditor_nosso_segue_o_tipo_da_config(foto_sintetica, monkeypatch):
    caminho, caixas = foto_sintetica
    configs = _espiar_detector(monkeypatch, caixas)

    preditores.PreditorNosso(Config()).prever(caminho)
    preditores.PreditorNosso(Config(detector_tipo="yolo")).prever(caminho)

    assert [c.detector_tipo for c in configs] == ["rfdetr", "yolo"]


def test_preditor_rfdetr_usa_o_detector_compartilhado_sem_filtros(foto_sintetica, monkeypatch):
    caminho, caixas = foto_sintetica
    configs = _espiar_detector(monkeypatch, caixas)
    cfg = Config(detector_tipo="yolo", detector_resolucao=560, detector_conf=0.01)

    p = preditores.PreditorRFDETR(cfg)

    assert [caixa for _, caixa in p.prever(caminho)] == caixas  # árbitro e arquibancada incluídos
    assert p.pos_processamento == "nenhum"
    assert (configs[0].detector_tipo, configs[0].detector_resolucao, configs[0].detector_conf) == (
        "rfdetr", 560, 0.01)
    assert cfg.detector_tipo == "yolo"  # a config recebida não é alterada


def test_preditor_yolo_bruto_forca_yolo(foto_sintetica, monkeypatch):
    caminho, caixas = foto_sintetica
    configs = _espiar_detector(monkeypatch, caixas)

    preditores.PreditorYoloBruto(Config()).prever(caminho)

    assert configs[0].detector_tipo == "yolo"


def test_preditor_rfdetr_usa_imagem_com_orientacao_exif(tmp_path, monkeypatch):
    imagem = tmp_path / "x.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # girada 90°
    Image.new("RGB", (100, 50)).save(imagem, exif=exif)
    formas = []

    def detectar(img, cfg):
        formas.append(img.shape)
        return []

    monkeypatch.setattr(preditores, "detectar_pessoas", detectar)

    assert preditores.PreditorRFDETR(Config()).prever(imagem) == []
    assert formas == [(100, 50, 3)]  # carregar_imagem aplica o EXIF, como no pipeline
```

Em `pipeline/tests/test_pipeline.py`:

1. Acrescentar `from nfl_vision.config import Config` aos imports.
2. Em `test_analise_completa`, trocar
   `assert analise.modelos == {"detector": "yolo11m", "ocr": "paddleocr"}` por
   ```python
   assert analise.modelos == {"detector": f"rfdetr-base@{Config().detector_resolucao}",
                              "ocr": "paddleocr"}
   ```
3. Acrescentar ao fim:

```python
def test_analise_com_yolo_registra_os_pesos_como_detector(dados, foto_sintetica, modelos_falsos):
    run_dir, analise = pipeline.analisar(foto_sintetica[0], CTX, Config(detector_tipo="yolo"))

    assert analise.modelos["detector"] == "yolo11m"
    assert ler_manifest(run_dir)["config"]["detector_tipo"] == "yolo"
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_avaliacao.py tests/test_pipeline.py -q`
Expected: FAIL — nomes antigos (`nosso (yolo11m + filtros + árbitro)`), `PreditorRFDETR(cfg)` tentando abrir o RF-DETR real/`TypeError`, `modelos["detector"] == "yolo11m"` no padrão, `configs[0].detector_tipo == "rfdetr"` no yolo-bruto.

- [ ] **Step 3: Implementar os preditores**

Em `pipeline/nfl_vision/eval/preditores.py`:

1. Trocar os imports do topo por (sai `PIL`, entra `rotulo_detector`; `_classes_coco` não é mais usado aqui):

```python
"""Preditores comparados na avaliação: o nosso, YOLO bruto, RF-DETR bruto e o modelo NFL do Roboflow."""

from pathlib import Path
from typing import Protocol

from nfl_vision.config import Config
from nfl_vision.eval.datasets import chave_roboflow
from nfl_vision.schemas import BBox
from nfl_vision.stages.detect import aplicar_filtros, detectar_pessoas, rotulo_detector, rotulo_pesos
from nfl_vision.stages.ingest import carregar_imagem
from nfl_vision.stages.team import eh_arbitro_deteccao
```

(`rotulo_pesos` continua importado: `preditores.rotulo_pesos` segue disponível para `test_rotulo_dos_pesos`.)

2. Em `PreditorNosso.__init__`, trocar a linha do nome por:

```python
        self.nome = f"nosso ({rotulo_detector(cfg)} + filtros + árbitro)"
```

3. Substituir a classe `PreditorYoloBruto` inteira por:

```python
class PreditorYoloBruto:
    """O YOLO do pipeline, sem filtros de campo nem remoção de árbitro."""

    pos_processamento = "nenhum"

    def __init__(self, cfg: Config):
        self.cfg = cfg.model_copy(update={"detector_tipo": "yolo"})
        padrao = cfg.detector_pesos == Config().detector_pesos
        self.nome = ("yolo11m bruto (COCO, pessoa)" if padrao
                     else f"yolo bruto ({rotulo_pesos(cfg.detector_pesos)})")

    def prever(self, imagem: Path) -> list[Predicao]:
        return [(d.confianca, d.bbox) for d in detectar_pessoas(carregar_imagem(imagem), self.cfg)]
```

4. Substituir a classe `PreditorRFDETR` inteira (incluindo o método `_eh_pessoa`) por:

```python
class PreditorRFDETR:
    """O RF-DETR do pipeline (mesma função de detecção), sem filtros nem remoção de árbitro."""

    pos_processamento = "nenhum"

    def __init__(self, cfg: Config):
        self.cfg = cfg.model_copy(update={"detector_tipo": "rfdetr"})
        self.nome = f"rfdetr bruto ({rotulo_detector(self.cfg)}, COCO, pessoa)"

    def prever(self, imagem: Path) -> list[Predicao]:
        return [(d.confianca, d.bbox) for d in detectar_pessoas(carregar_imagem(imagem), self.cfg)]
```

O modelo é carregado na primeira predição (cache em `detect._modelo_rfdetr`); um erro de carga vira erro daquele preditor em `eval detect`, sem derrubar os outros.

5. Em `pipeline/nfl_vision/cli.py`, três ajustes mínimos para a suíte continuar verde (a Task 4 reescreve os dois últimos com `--detector-tipo`):

   - em `_criar_preditores`, trocar `lista.append(preditores.PreditorRFDETR(conf))` por `lista.append(preditores.PreditorRFDETR(cfg))` (`cfg.detector_conf` já é o `--conf`);
   - em `eval_detect_cmd`, `--pesos` passa a implicar YOLO (senão o `nosso` seria RF-DETR e `test_eval_detect_com_pesos_ajustados`, que espera `nosso (player-v1 + filtros + árbitro)`, falharia). Trocar

     ```python
         cfg = (Config(detector_conf=conf) if pesos is None
                else Config(detector_conf=conf, detector_pesos=str(pesos.resolve())))
     ```

     por

     ```python
         cfg = (Config(detector_conf=conf) if pesos is None
                else Config(detector_conf=conf, detector_tipo="yolo", detector_pesos=str(pesos.resolve())))
     ```

   - em `_analisar_ou_reprocessar`, `--detector` passa a implicar YOLO. Trocar

     ```python
         config = Config(detector_pesos=str(detector.resolve())) if detector is not None else None
     ```

     por

     ```python
         config = (Config(detector_tipo="yolo", detector_pesos=str(detector.resolve()))
                   if detector is not None else None)
     ```

- [ ] **Step 4: `montagem` usa o rótulo do detector**

Em `pipeline/nfl_vision/montagem.py`:

- trocar `from nfl_vision.eval.preditores import rotulo_pesos` por `from nfl_vision.stages.detect import rotulo_detector`;
- trocar `modelos={"detector": rotulo_pesos(estado.config.detector_pesos), "ocr": "paddleocr"},` por
  `modelos={"detector": rotulo_detector(estado.config), "ocr": "paddleocr"},`.

- [ ] **Step 5: Rodar e ver passar**

Run: `uv run pytest tests/test_avaliacao.py tests/test_pipeline.py -q`
Expected: PASS.

Run: `uv run pytest -q`
Expected: todos passam (inclusive `test_eval_detect_com_pesos_ajustados` e `test_analyze_com_detector_grava_os_pesos_na_config`, graças aos ajustes do item 5 do Step 3).

- [ ] **Step 6: Commit**

```bash
git add nfl_vision/eval/preditores.py nfl_vision/montagem.py nfl_vision/cli.py tests/test_avaliacao.py tests/test_pipeline.py
git commit -m "feat: preditores e analise.json seguem o tipo do detector; RF-DETR bruto usa a etapa detect"
```

---

### Task 4: CLI — `--detector-tipo` e `--resolucao`

**Files:**
- Modify: `pipeline/nfl_vision/cli.py`
- Modify: `pipeline/tests/test_cli.py`

- [ ] **Step 1: Escrever os testes que falham**

Em `pipeline/tests/test_cli.py`:

1. No teste `test_analyze_com_detector_grava_os_pesos_na_config`, acrescentar ao fim:

```python
    assert manifest["config"]["detector_tipo"] == "yolo"
```

2. Trocar `test_analyze_detector_erros` por:

```python
def test_analyze_detector_erros(dados, foto_sintetica, tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    pesos = tmp_path / "best.pt"
    pesos.write_bytes(b"ajustado")
    casos = [
        (["analyze", str(foto_sintetica[0]), *BASE, "--detector", str(tmp_path / "nao.pt")],
         "pesos não encontrados"),
        (["analyze", "--run", "x", "--from", "jersey", "--detector", str(foto_sintetica[0])],
         "--detector só vale para análise nova"),
        (["analyze", "--run", "x", "--from", "jersey", "--detector-tipo", "yolo"],
         "--detector-tipo só vale para análise nova"),
        (["analyze", str(foto_sintetica[0]), *BASE, "--detector-tipo", "detr"],
         "use rfdetr ou yolo"),
        (["analyze", str(foto_sintetica[0]), *BASE, "--detector-tipo", "rfdetr",
          "--detector", str(pesos)],
         "não combine"),
    ]
    for args, trecho in casos:
        r = runner.invoke(app, args)
        assert r.exit_code == 2, (args, r.output)
        assert trecho in r.output, (args, r.output)
```

3. Acrescentar ao fim do arquivo:

```python
def _manifest_da_unica_analise():
    import json

    run_dir = next(paths.runs_dir().iterdir())
    return run_dir, json.loads((run_dir / "manifest.json").read_text("utf-8"))


def test_analyze_usa_rfdetr_por_padrao(dados, foto_sintetica, modelos_falsos):
    import hashlib
    import json

    from nfl_vision.config import Config

    r = runner.invoke(app, ["analyze", str(foto_sintetica[0]), *BASE])

    assert r.exit_code == 0, r.output
    run_dir, manifest = _manifest_da_unica_analise()
    assert manifest["config"]["detector_tipo"] == "rfdetr"
    assert manifest["config"]["detector_resolucao"] == Config().detector_resolucao
    assert manifest["versoes"]["detector_pesos_sha256"] == hashlib.sha256(b"pesos falsos").hexdigest()
    analise = json.loads((run_dir / "analise.json").read_text("utf-8"))
    assert analise["modelos"]["detector"] == f"rfdetr-base@{Config().detector_resolucao}"


def test_analyze_detector_tipo_yolo(dados, foto_sintetica, modelos_falsos):
    r = runner.invoke(app, ["analyze", str(foto_sintetica[0]), *BASE, "--detector-tipo", "yolo"])

    assert r.exit_code == 0, r.output
    _, manifest = _manifest_da_unica_analise()
    assert manifest["config"]["detector_tipo"] == "yolo"
    assert manifest["config"]["detector_pesos"] == "yolo11m.pt"


def test_analyze_detector_tipo_yolo_com_pesos(dados, foto_sintetica, modelos_falsos, tmp_path):
    pesos = tmp_path / "best.pt"
    pesos.write_bytes(b"ajustado")

    r = runner.invoke(app, ["analyze", str(foto_sintetica[0]), *BASE, "--detector-tipo", "yolo",
                            "--detector", str(pesos)])

    assert r.exit_code == 0, r.output
    _, manifest = _manifest_da_unica_analise()
    assert manifest["config"]["detector_tipo"] == "yolo"
    assert manifest["config"]["detector_pesos"] == str(pesos.resolve())


def _espiar_detector_cli(monkeypatch):
    from nfl_vision.eval import preditores
    from nfl_vision.schemas import Deteccao

    configs = []

    def detectar(img, cfg):
        configs.append(cfg)
        return [Deteccao(det_id=0, bbox=(40.0, 30.0, 60.0, 70.0), confianca=0.9)]

    monkeypatch.setattr(preditores, "detectar_pessoas", detectar)
    return configs


def test_eval_detect_nosso_segue_rfdetr_e_resolucao(dados, tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    configs = _espiar_detector_cli(monkeypatch)
    ds = _dataset_deteccao(tmp_path / "ds")

    r = runner.invoke(app, ["eval", "detect", "--dataset", str(ds), "--resolucao", "560",
                            "--benchmark", "rfdetr", "--benchmark", "yolo-bruto"])

    assert r.exit_code == 0, r.output
    salvo = _ler_avaliacao("detect")
    assert salvo["config"]["detector_tipo"] == "rfdetr"
    assert salvo["config"]["detector_resolucao"] == 560
    assert [x["preditor"] for x in salvo["resultados"]] == [
        "nosso (rfdetr-base@560 + filtros + árbitro)",
        "rfdetr bruto (rfdetr-base@560, COCO, pessoa)",
        "yolo11m bruto (COCO, pessoa)",
    ]
    assert [(c.detector_tipo, c.detector_resolucao) for c in configs] == [
        ("rfdetr", 560), ("rfdetr", 560), ("yolo", 560)]


def test_eval_detect_detector_tipo_yolo(dados, tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    _espiar_detector_cli(monkeypatch)
    ds = _dataset_deteccao(tmp_path / "ds")

    r = runner.invoke(app, ["eval", "detect", "--dataset", str(ds), "--detector-tipo", "yolo"])

    assert r.exit_code == 0, r.output
    salvo = _ler_avaliacao("detect")
    assert salvo["config"]["detector_tipo"] == "yolo"
    assert salvo["resultados"][0]["preditor"] == "nosso (yolo11m + filtros + árbitro)"


def test_eval_detect_erros_de_detector(dados, tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    ds = _dataset_deteccao(tmp_path / "ds")
    pesos = tmp_path / "best.pt"
    pesos.write_bytes(b"ajustado")
    casos = [
        (["--resolucao", "900"], "múltiplo de 56"),
        (["--detector-tipo", "detr"], "use rfdetr ou yolo"),
        (["--detector-tipo", "rfdetr", "--pesos", str(pesos)], "não combine"),
    ]
    for args, trecho in casos:
        r = runner.invoke(app, ["eval", "detect", "--dataset", str(ds), *args])
        assert r.exit_code == 2, (args, r.output)
        assert trecho in r.output, (args, r.output)


def test_help_menciona_detector_tipo():
    for comando in (["analyze", "--help"], ["eval", "detect", "--help"]):
        r = runner.invoke(app, comando, env={"COLUMNS": "300"})
        assert r.exit_code == 0, r.output
        assert "--detector-tipo" in r.output
    r = runner.invoke(app, ["eval", "detect", "--help"], env={"COLUMNS": "300"})
    assert "--resolucao" in r.output
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_cli.py -q`
Expected: FAIL — `No such option: --detector-tipo` / `--resolucao` (saída 2 sem as mensagens esperadas).

- [ ] **Step 3: Implementar**

Em `pipeline/nfl_vision/cli.py`:

1. Logo depois de `console = Console()`, acrescentar:

```python
DETECTORES = ("rfdetr", "yolo")
AJUDA_DETECTOR_TIPO = "Detector: rfdetr (padrão) ou yolo"


def _validar_detector(detector_tipo: Optional[str], pesos: Optional[Path], opcao_pesos: str) -> None:
    """Tipo conhecido; pesos (.pt) são do YOLO e não combinam com rfdetr."""
    if detector_tipo is not None and detector_tipo not in DETECTORES:
        raise typer.BadParameter("use rfdetr ou yolo", param_hint="--detector-tipo")
    if pesos is not None and detector_tipo == "rfdetr":
        raise typer.BadParameter(
            f"{opcao_pesos} recebe pesos YOLO (.pt); não combine com --detector-tipo rfdetr",
            param_hint=opcao_pesos)


def _ajustes_do_detector(detector_tipo: Optional[str], pesos: Optional[Path]) -> dict:
    """Campos da Config para o detector escolhido; pesos .pt implicam yolo."""
    ajustes = {}
    if detector_tipo is not None:
        ajustes["detector_tipo"] = detector_tipo
    if pesos is not None:
        # caminho absoluto: o reprocessamento (--run) pode rodar de outro diretório
        ajustes.update(detector_tipo="yolo", detector_pesos=str(pesos.resolve()))
    return ajustes
```

(`cli.DETECTORES` repete `stages.detect.DETECTORES` para a CLI não importar a etapa no carregamento; os dois ficam coerentes com o `Literal` da `Config`.)

2. Substituir o comando `analyze` e `_analisar_ou_reprocessar` por:

```python
@app.command()
def analyze(
    foto: Optional[Path] = typer.Argument(None, help="Foto JPG ou PNG"),
    times: Tuple[str, str] = typer.Option((None, None), "--times", help="Siglas dos dois times"),
    temporada: Optional[int] = typer.Option(None, "--temporada"),
    semana: Optional[int] = typer.Option(None, "--semana"),
    run: Optional[str] = typer.Option(None, "--run", help="Reprocessar uma análise existente"),
    a_partir_de: Optional[str] = typer.Option(None, "--from", help="Etapa inicial do reprocessamento"),
    detector_tipo: Optional[str] = typer.Option(None, "--detector-tipo", help=AJUDA_DETECTOR_TIPO),
    detector: Optional[Path] = typer.Option(
        None, "--detector",
        help="Pesos YOLO (.pt) para esta análise; implica --detector-tipo yolo"),
) -> None:
    """Analisa uma foto ou reprocessa uma análise a partir de uma etapa."""
    with _tratando_falha_de_etapa(run):
        run_dir, analise = _analisar_ou_reprocessar(
            foto, times, temporada, semana, run, a_partir_de, detector, detector_tipo)
    _imprimir(analise, run_dir)


def _analisar_ou_reprocessar(foto, times, temporada, semana, run, a_partir_de, detector=None,
                             detector_tipo=None):
    if a_partir_de is not None and not run:
        raise typer.BadParameter("--from exige --run <id>", param_hint="--from")
    if run and foto is not None:
        raise typer.BadParameter("não informe a foto junto com --run", param_hint="FOTO")
    if run and a_partir_de is None:
        raise typer.BadParameter("--run exige --from <etapa>", param_hint="--run")
    if run and detector is not None:
        raise typer.BadParameter(
            "--detector só vale para análise nova; o reprocessamento usa a config gravada",
            param_hint="--detector")
    if run and detector_tipo is not None:
        raise typer.BadParameter(
            "--detector-tipo só vale para análise nova; o reprocessamento usa a config gravada",
            param_hint="--detector-tipo")
    _validar_detector(detector_tipo, detector, "--detector")
    if run:
        if a_partir_de not in pipeline.NOMES_ETAPAS:
            raise typer.BadParameter(
                f"use uma etapa: {', '.join(pipeline.NOMES_ETAPAS)}", param_hint="--from")
        try:
            return pipeline.reprocessar(run, a_partir_de, ao_arquivar=_avisar_arquivamento)
        except FileNotFoundError as exc:
            raise typer.BadParameter(str(exc), param_hint="--run") from exc
    if foto is None or None in times or temporada is None or semana is None:
        raise typer.BadParameter("informe a foto, --times, --temporada e --semana")
    if not foto.exists():
        raise typer.BadParameter(f"arquivo não encontrado: {foto}", param_hint="FOTO")
    try:
        ingest.validar_formato(foto)
    except ingest.FormatoNaoSuportado as exc:
        raise typer.BadParameter(str(exc), param_hint="FOTO") from exc
    _validar_imagem(foto)
    if detector is not None and not detector.is_file():
        raise typer.BadParameter(f"pesos não encontrados: {detector}", param_hint="--detector")
    contexto = _validar_contexto(times, temporada, semana)
    ajustes = _ajustes_do_detector(detector_tipo, detector)
    config = Config(**ajustes) if ajustes else None
    return pipeline.analisar(foto, contexto, config)
```

3. Em `eval_detect_cmd`, trocar a opção `pesos` e acrescentar as duas novas (depois de `conf`):

```python
    pesos: Optional[Path] = typer.Option(
        None, "--pesos",
        help="Pesos YOLO (.pt) para os preditores nosso e yolo-bruto; implica --detector-tipo yolo"),
    detector_tipo: Optional[str] = typer.Option(
        None, "--detector-tipo", help=f"{AJUDA_DETECTOR_TIPO} para o preditor nosso"),
    resolucao: Optional[int] = typer.Option(
        None, "--resolucao",
        help="Lado de entrada do RF-DETR (múltiplo de 56) para nosso e rfdetr; padrão: config"),
```

e, no corpo, trocar o bloco

```python
    if pesos is not None and not pesos.is_file():
        raise typer.BadParameter(f"pesos não encontrados: {pesos}", param_hint="--pesos")
    cfg = (Config(detector_conf=conf) if pesos is None
           else Config(detector_conf=conf, detector_tipo="yolo", detector_pesos=str(pesos.resolve())))
```

por

```python
    _validar_detector(detector_tipo, pesos, "--pesos")
    if resolucao is not None and (resolucao <= 0 or resolucao % 56):
        raise typer.BadParameter("--resolucao deve ser um múltiplo positivo de 56",
                                 param_hint="--resolucao")
    if pesos is not None and not pesos.is_file():
        raise typer.BadParameter(f"pesos não encontrados: {pesos}", param_hint="--pesos")
    ajustes = _ajustes_do_detector(detector_tipo, pesos)
    if resolucao is not None:
        ajustes["detector_resolucao"] = resolucao
    cfg = Config(detector_conf=conf, **ajustes)
```

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_cli.py -q`
Expected: PASS.

Run: `uv run pytest -q`
Expected: todos passam.

- [ ] **Step 5: Commit**

```bash
git add nfl_vision/cli.py tests/test_cli.py
git commit -m "feat: --detector-tipo no analyze e no eval detect, e --resolucao para o RF-DETR"
```

---

### Task 5: fixar padrões medidos

> **Controlador:** preencha `<RESOLUCAO_MEDIDA>` e `<CONF_MEDIDA>` com o resultado da medição (§4 do spec: maior mAP@0.5 sem filtros entre 560/896/1120, empate < 0,01 fica com a menor; limiar de maior F1 entre 0,2 e 0,6 na resolução escolhida) antes de despachar esta task. Este é o **único** passo com valores em aberto.

**Files:**
- Modify: `pipeline/nfl_vision/config.py`

- [ ] **Step 1: Gravar os padrões**

Em `pipeline/nfl_vision/config.py`, trocar

```python
    detector_resolucao: int = 896         # lado de entrada do RF-DETR; múltiplo de 56
    detector_conf: float = 0.3
```

por

```python
    detector_resolucao: int = <RESOLUCAO_MEDIDA>  # lado de entrada do RF-DETR; múltiplo de 56 (medido)
    detector_conf: float = <CONF_MEDIDA>          # limiar de maior F1 no split test (medido)
```

- [ ] **Step 2: Rodar a suíte**

Run: `uv run pytest -q`
Expected: todos passam (nenhum teste fixa os números; `test_config.py` confere que a resolução é múltiplo de 56).

Run: `uv run pytest -m model -q`
Expected: todos os testes `@model` passam (inclui `test_rfdetr_detecta_pessoas_em_imagem_real` com os padrões novos).

- [ ] **Step 3: Commit**

```bash
git add nfl_vision/config.py
git commit -m "feat: padrões medidos do RF-DETR (resolução <RESOLUCAO_MEDIDA>, limiar <CONF_MEDIDA>)"
```

---

## Execução (controlador)

Depois da Task 5, a partir de `pipeline/`:

1. Avaliação com os padrões finais (limiar 0,01, convenção do mAP): `nosso` = RF-DETR + filtros + árbitro, `rfdetr` bruto e `yolo-bruto`:

   ```bash
   uv run nfl-vision eval detect --dataset ../data/datasets/treino-player-v1 --split test --conf 0.01 --benchmark rfdetr --benchmark yolo-bruto
   ```

   Expected: tabela com três linhas; `rfdetr bruto` na resolução padrão com mAP@0.5 coerente com a medição; o JSON em `data/avaliacoes/detect-*.json`.

2. Análises reais com o padrão:

   ```bash
   uv run nfl-vision analyze ../fotos/cin_cle_2025_s1_broadcast_3.jpg --times CIN CLE --temporada 2025 --semana 1
   uv run nfl-vision analyze ../fotos/cin_cle_2025_s1_broadcast_4.jpg --times CIN CLE --temporada 2025 --semana 1
   ```

   Conferir em cada `data/runs/<id>/manifest.json`: `config.detector_tipo == "rfdetr"`, `config.detector_resolucao` = padrão, `versoes.detector_pesos_sha256` não nulo, `versoes.rfdetr == "1.11.1"`; em `analise.json`, `modelos.detector == "rfdetr-base@<resolução>"`; olhar `anotada.png` (comparar com as análises YOLO anteriores das mesmas fotos).

3. Escrever `docs/avaliacao/2026-10-detector-rfdetr.md`: tabela da medição (resolução × mAP sem/com filtros × tempo por imagem na RTX 5070 Ti; YOLO11m sem/com filtros), tabela de precisão/revocação/F1 por limiar (0,2–0,6), a escolha e o motivo, o resultado do passo 1 e observações das duas fotos.

4. Atualizar `docs/superpowers/specs/2026-10-03-pipeline-fotos-design.md` §6 `detect`: primeiro item passa a ser "RF-DETR base COCO (padrão; `detector_tipo`), classe pessoa, resolução `<R>`, `conf ≥ <C>`; YOLO11m (`imgsz = 1280`) como opção"; último item: o SHA-256 gravado é o do arquivo realmente carregado (`ckpt_path` no YOLO, `model_config.pretrain_weights` no RF-DETR, ex. `~/.roboflow/models/rf-detr-base.pth`).

5. Atualizar `README.md`: o detector padrão é o RF-DETR base (pesos baixados pelo pacote `rfdetr` em `~/.roboflow/models/`); `--detector-tipo yolo` volta ao YOLO11m; `--detector`/`--pesos` com `.pt` implicam YOLO; exemplo de `eval detect --resolucao`; a seção "Detector ajustado" deixa de dizer que o padrão é o YOLO11m; `pytest -m model` menciona RF-DETR. Link para `docs/avaliacao/2026-10-detector-rfdetr.md`.

6. Commit: `docs: avaliação do RF-DETR como detector padrão` e atualizar a página de resultados (critério de pronto 4 do spec).
