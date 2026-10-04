# Ajuste fino do detector de jogadores — Plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preparar um dataset de `player` sem vazamento entre treino e teste, ajustar o YOLO11m nele na GPU local e comparar o modelo ajustado com os atuais num jogo que nunca entra no treino, com a opção de usar os pesos ajustados no `analyze`.

**Architecture:** Novo subpacote `nfl_vision/treino/` com três módulos: `fontes.py` (declaração das fontes do Roboflow, mapeamento de classes para `player`, regra de split por clipe da base), `preparar.py` (conversão para uma classe, splits, painéis de triagem, manifest) e `rodar.py` (treino com Ultralytics e manifest do treino). A leitura de rótulos YOLO (caixa ou polígono) sai de `carregar_yolo` para uma função compartilhada `ler_rotulos` em `eval/datasets.py`. A CLI ganha o grupo `treino` (`preparar`, `rodar`), `eval detect --pesos` e `analyze --detector`. Ultralytics fica atrás de `rodar._yolo` e `rodar._gpu`, que os testes rápidos substituem por falsos.

**Tech Stack:** Python 3.12, uv, Ultralytics 8.4 (YOLO11), PyTorch cu128, OpenCV, Pillow, PyYAML, Typer + Rich, pytest. Download: `roboflow` (extra `eval`).

**Spec:** `docs/superpowers/specs/2026-10-04-detector-ajustado-design.md`

## Convenções

- Todos os comandos rodam a partir de `pipeline/`, num Git Bash com `export PATH="/c/Users/User/.local/bin:$PATH"` (onde está o `uv`).
- Testes rápidos: `uv run pytest`. Testes com modelos reais: `uv run pytest -m model`.
- Commits em português, **sem** linhas `Co-Authored-By` e sem menção a Claude/IA.
- Textos de interface, mensagens de erro e nomes de domínio em português.
- Erros de entrada do usuário viram `typer.BadParameter` (saída 2, sem traceback), como em `eval baixar`.

## Mapa de arquivos

```
pipeline/
  nfl_vision/
    paths.py                 # + treinos_dir()
    cli.py                   # + grupo `treino` (preparar, rodar); eval detect --pesos; analyze --detector
    eval/
      datasets.py            # + ler_rotulos() compartilhada; baixar() reusa pasta sem exigir chave
      preditores.py          # + rotulo_pesos(); nome dos preditores indica os pesos
    treino/
      __init__.py
      fontes.py              # Fonte, BASE, EXTERNAS, POR_NOME, clipe/split da base, validar_mapeamento, obter
      preparar.py            # Decisao, imagens_da_fonte, converter, split_externo, construir, painel, triagem
      rodar.py               # PARAMETROS, EntradaInvalida, validar_dataset, treinar, retomar
  tests/
    treino_sintetico.py      # datasets sintéticos (base, externos, preparado) e YOLO falso
    test_datasets.py         # + ler_rotulos, baixar reusando pasta
    test_treino_fontes.py
    test_treino_preparar.py
    test_treino_rodar.py     # + @model treino-relâmpago
    test_cli_treino.py
    test_cli.py              # + --pesos, --detector
    test_avaliacao.py        # + nomes dos preditores
README.md                    # seção "Detector ajustado"
docs/superpowers/specs/2026-10-04-detector-ajustado-design.md   # detalhes que a implementação fixou
```

---

### Task 1: Leitura compartilhada de rótulos YOLO e declaração das fontes

**Files:**
- Modify: `pipeline/nfl_vision/eval/datasets.py` (funções `baixar`, `_caixa_da_linha`, `carregar_yolo`; nova `ler_rotulos`)
- Create: `pipeline/nfl_vision/treino/__init__.py`
- Create: `pipeline/nfl_vision/treino/fontes.py`
- Test: `pipeline/tests/test_datasets.py`, `pipeline/tests/test_treino_fontes.py`

- [ ] **Step 1: Escrever os testes que falham de `ler_rotulos` e `baixar`**

Em `pipeline/tests/test_datasets.py`, troque a linha de import por:

```python
from nfl_vision.eval.datasets import baixar, carregar_pastas, carregar_yolo, ler_rotulos
```

e acrescente ao fim do arquivo:

```python
def test_ler_rotulos_normaliza_caixa_e_poligono(tmp_path):
    arquivo = tmp_path / "a.txt"
    arquivo.write_text("1 0.5 0.5 0.2 0.4\n\n0 0.1 0.2 0.3 0.2 0.3 0.6 0.1 0.6\n")

    rotulos = ler_rotulos(arquivo, ["ball", "player"])

    assert [classe for classe, _ in rotulos] == ["player", "ball"]
    assert rotulos[0][1] == pytest.approx((0.4, 0.3, 0.6, 0.7))
    assert rotulos[1][1] == pytest.approx((0.1, 0.2, 0.3, 0.6))  # polígono vira a caixa que o envolve


def test_ler_rotulos_linha_invalida(tmp_path):
    arquivo = tmp_path / "a.txt"
    arquivo.write_text("1 0.5 0.5\n")
    with pytest.raises(ValueError, match=r"a\.txt, linha 1"):
        ler_rotulos(arquivo, ["ball", "player"])


def test_baixar_reusa_pasta_existente_sem_chave(tmp_path, monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    alvo = tmp_path / "proj-v1-yolov11"
    alvo.mkdir()
    assert baixar("ws", "proj", 1, "yolov11", tmp_path) == alvo
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_datasets.py -v`
Expected: erro de coleta `ImportError: cannot import name 'ler_rotulos'`.

- [ ] **Step 3: Implementar `ler_rotulos` e ajustar `baixar` e `carregar_yolo`**

Em `pipeline/nfl_vision/eval/datasets.py`:

Substitua a função `baixar` inteira por (a pasta já baixada é reusada antes de exigir a chave):

```python
def baixar(workspace: str, projeto: str, versao: int, formato: str, destino: Path) -> Path:
    alvo = destino / f"{projeto}-v{versao}-{formato}"
    if alvo.exists():
        return alvo
    chave = chave_roboflow()
    from roboflow import Roboflow

    Roboflow(api_key=chave).workspace(workspace).project(projeto).version(versao).download(
        formato, location=str(alvo)
    )
    return alvo
```

Substitua `_caixa_da_linha` e todo o `carregar_yolo` por:

```python
CaixaNormalizada = tuple[float, float, float, float]  # x1, y1, x2, y2 em fração da imagem


def _caixa_da_linha(partes: list[str]) -> CaixaNormalizada | None:
    """Caixa normalizada de uma linha YOLO: `cx cy w h` ou polígono `x1 y1 x2 y2 ...`."""
    valores = list(map(float, partes[1:]))
    if len(valores) == 4:
        cx, cy, bw, bh = valores
        return (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
    if len(valores) >= 6 and len(valores) % 2 == 0:  # segmentação: caixa que envolve o polígono
        xs, ys = valores[0::2], valores[1::2]
        return (min(xs), min(ys), max(xs), max(ys))
    return None


def ler_rotulos(arquivo: Path, nomes: list) -> list[tuple[str, CaixaNormalizada]]:
    """(classe, caixa normalizada) de cada linha de um .txt YOLO; linhas vazias são ignoradas."""
    saida = []
    for n, linha in enumerate(arquivo.read_text().splitlines(), start=1):
        partes = linha.split()
        if not partes:
            continue
        try:
            caixa = _caixa_da_linha(partes)
            indice = int(partes[0])
        except ValueError:
            caixa = None
        if caixa is None:
            raise ValueError(f"{arquivo}, linha {n}: linha YOLO inválida: {linha!r}")
        if not 0 <= indice < len(nomes):
            raise ValueError(
                f"{arquivo}, linha {n}: classe {indice} fora de names ({len(nomes)} classes)")
        saida.append((nomes[indice], caixa))
    return saida


def carregar_yolo(raiz: Path, split: str = "test") -> list[AmostraDeteccao]:
    nomes = nomes_das_classes(raiz)
    pasta_imagens = raiz / split / "images"
    if not pasta_imagens.is_dir():
        existentes = ", ".join(splits_existentes(raiz)) or "nenhuma"
        raise FileNotFoundError(
            f"pasta {pasta_imagens} não encontrada; pastas em {raiz}: {existentes} "
            "(exports do Roboflow usam 'valid', não 'val')")
    amostras = []
    for caminho in sorted(pasta_imagens.iterdir()):
        if caminho.suffix.lower() not in IMAGENS:
            continue
        with Image.open(caminho) as im:
            w, h = ImageOps.exif_transpose(im).size
        caixas: dict[str, list[BBox]] = {}
        rotulos = raiz / split / "labels" / f"{caminho.stem}.txt"
        if rotulos.exists():
            for classe, (x1, y1, x2, y2) in ler_rotulos(rotulos, nomes):
                caixas.setdefault(classe, []).append((x1 * w, y1 * h, x2 * w, y2 * h))
        amostras.append(AmostraDeteccao(caminho, caixas))
    return amostras
```

- [ ] **Step 4: Rodar os testes de datasets**

Run: `uv run pytest tests/test_datasets.py -v`
Expected: todos PASS (os antigos de `carregar_yolo` continuam passando: mesma conta, só que normalizada antes de escalar).

- [ ] **Step 5: Escrever os testes que falham de `fontes`**

Crie `pipeline/tests/test_treino_fontes.py`:

```python
import pytest

from nfl_vision.treino import fontes
from nfl_vision.treino.fontes import (
    BASE, EXTERNAS, POR_NOME, FonteIndisponivel, MapeamentoInvalido, clipe_base, split_base,
    validar_mapeamento,
)


def test_clipe_base():
    assert clipe_base("cin_cle_wk1_burrow_td_broadcast_00015_jpg.rf.abc123.jpg") == \
        "cin_cle_wk1_burrow_td_broadcast"
    assert clipe_base("tb_atl_wk1_penix_pass_all22_00000_jpg.rf.f0.jpg") == "tb_atl_wk1_penix_pass_all22"


def test_clipe_base_fora_do_padrao():
    with pytest.raises(ValueError, match="padrão"):
        clipe_base("foto_qualquer.jpg")


@pytest.mark.parametrize("clipe, split", [
    ("cin_cle_wk1_burrow_td_broadcast", "test"),
    ("cin_cle_wk1_burrow_td_all22", "test"),
    ("tb_atl_wk1_penix_pass_all22", "valid"),
    ("tb_atl_wk1_penix_pass", "train"),
    ("phi_dal_wk1_williams_run", "train"),
    ("phi_dal_wk1_williams_run_endzone", "train"),
])
def test_split_base(clipe, split):
    assert split_base(f"{clipe}_00030_jpg.rf.deadbeef.jpg") == split


def test_fontes_declaradas():
    assert [f.nome for f in EXTERNAS] == ["pitchcamera", "fhtw", "evzn"]
    assert POR_NOME["base"] is BASE and not BASE.externa
    assert all(f.externa for f in EXTERNAS)
    assert {f.nome: f.classes_player for f in EXTERNAS} == {
        "pitchcamera": ("players",),
        "fhtw": ("american-football-players",),
        "evzn": ("football-players",),
    }


def test_pasta_usa_o_nome_do_download(tmp_path):
    assert BASE.pasta(tmp_path) == tmp_path / "nfl-player-model-ymsui-v1-yolov11"


def test_validar_mapeamento():
    validar_mapeamento(POR_NOME["evzn"], ["ball", "referee", "football-players"])
    with pytest.raises(MapeamentoInvalido) as exc:
        validar_mapeamento(POR_NOME["evzn"], ["ball", "referee", "players"])
    msg = str(exc.value)
    assert "evzn" in msg and "football-players" in msg and "ball, referee, players" in msg


def test_obter_erro_de_download_cita_a_fonte(tmp_path, monkeypatch):
    def quebrar(*a, **k):
        raise ConnectionError("timeout")

    monkeypatch.setattr(fontes, "baixar", quebrar)
    with pytest.raises(FonteIndisponivel) as exc:
        fontes.obter(POR_NOME["pitchcamera"], tmp_path)
    msg = str(exc.value)
    assert "pitchcamera" in msg and "ConnectionError" in msg and "timeout" in msg


def test_obter_sem_chave(tmp_path, monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    with pytest.raises(FonteIndisponivel, match="ROBOFLOW_API_KEY"):
        fontes.obter(POR_NOME["fhtw"], tmp_path)


def test_obter_reusa_download(tmp_path, monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    BASE.pasta(tmp_path).mkdir()
    assert fontes.obter(BASE, tmp_path) == BASE.pasta(tmp_path)
```

- [ ] **Step 6: Rodar e ver falhar**

Run: `uv run pytest tests/test_treino_fontes.py -v`
Expected: erro de coleta `ModuleNotFoundError: No module named 'nfl_vision.treino'`.

- [ ] **Step 7: Implementar o subpacote e `fontes.py`**

Crie `pipeline/nfl_vision/treino/__init__.py`:

```python
"""Ajuste fino do detector de jogadores: fontes, preparação do dataset e treino."""
```

Crie `pipeline/nfl_vision/treino/fontes.py`:

```python
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
```

- [ ] **Step 8: Rodar os testes da task e a suíte rápida**

Run: `uv run pytest tests/test_treino_fontes.py tests/test_datasets.py -v`
Expected: todos PASS.

Run: `uv run pytest`
Expected: todos PASS.

- [ ] **Step 9: Commit**

```bash
git add nfl_vision/eval/datasets.py nfl_vision/treino/__init__.py nfl_vision/treino/fontes.py tests/test_datasets.py tests/test_treino_fontes.py
git commit -m "feat: fontes do dataset de treino e leitura compartilhada de rótulos YOLO"
```

---

### Task 2: Montagem do dataset de `player` (conversão, splits, manifest)

**Files:**
- Create: `pipeline/nfl_vision/treino/preparar.py`
- Create: `pipeline/tests/treino_sintetico.py`
- Test: `pipeline/tests/test_treino_preparar.py`

- [ ] **Step 1: Criar os datasets sintéticos de teste**

Crie `pipeline/tests/treino_sintetico.py` (o diretório `tests/` já está no `sys.path`, como `sintetico.py`):

```python
"""Datasets sintéticos para os testes de treino e um YOLO falso."""

from pathlib import Path
from types import SimpleNamespace

import yaml
from PIL import Image, ImageDraw

from nfl_vision.treino.fontes import BASE, Fonte

CLIPES_BASE = (
    "cin_cle_wk1_burrow_td_broadcast", "cin_cle_wk1_burrow_td_all22",
    "tb_atl_wk1_penix_pass", "tb_atl_wk1_penix_pass_all22",
    "phi_dal_wk1_williams_run", "phi_dal_wk1_williams_run_endzone",
)
NOMES_EXTERNOS = {
    "pitchcamera": ["football", "referee", "players"],
    "fhtw": ["ball", "referee", "1", "2", "3", "4", "5", "american-football-players", "whitehat"],
    "evzn": ["ball", "referee", "football-players"],
}


def imagem(pasta: Path, nome: str, linhas: list[str], tamanho=(64, 48)) -> None:
    (pasta / "images").mkdir(parents=True, exist_ok=True)
    (pasta / "labels").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", tamanho, (40, 140, 40)).save(pasta / "images" / nome)
    (pasta / "labels" / f"{Path(nome).stem}.txt").write_text("\n".join(linhas) + "\n")


def dataset_base(datasets_dir: Path, quadros_por_clipe: int = 2) -> Path:
    """Export do fork: 6 clipes espalhados pelos splits train/valid/test, como no Roboflow.

    Cada imagem tem 1 player, 1 referee e 1 ball.
    """
    raiz = BASE.pasta(datasets_dir)
    raiz.mkdir(parents=True)
    (raiz / "data.yaml").write_text("names: ['ball', 'player', 'referee']\n", encoding="utf-8")
    splits = ("train", "valid", "test")
    k = 0
    for clipe in CLIPES_BASE:
        for q in range(quadros_por_clipe):
            nome = f"{clipe}_{q * 15:05d}_jpg.rf.{k:032x}.jpg"
            imagem(raiz / splits[k % 3], nome,
                   ["1 0.5 0.5 0.2 0.4", "2 0.2 0.2 0.1 0.2", "0 0.7 0.7 0.02 0.02"])
            k += 1
    return raiz


def dataset_externo(datasets_dir: Path, fonte: Fonte, originais: int = 20,
                    nomes: list[str] | None = None) -> Path:
    """`originais` imagens com 2 cópias aumentadas cada (player + referee), mais uma só com
    árbitro (fica fora) e uma com polígono de player: 2 * originais + 2 imagens."""
    raiz = fonte.pasta(datasets_dir)
    raiz.mkdir(parents=True)
    nomes = nomes or NOMES_EXTERNOS[fonte.nome]
    (raiz / "data.yaml").write_text(f"names: {nomes!r}\n", encoding="utf-8")
    classe = fonte.classes_player[0]
    jogador = nomes.index(classe) if classe in nomes else len(nomes) - 1
    for i in range(originais):
        for copia in range(2):
            imagem(raiz / "train", f"img{i:03d}_png.rf.{i:04d}{copia:028x}.jpg",
                   [f"{jogador} 0.5 0.5 0.2 0.4", "1 0.3 0.3 0.1 0.2"])
    imagem(raiz / "valid", "so_arbitro_jpg.rf.aa.jpg", ["1 0.5 0.5 0.2 0.4"])
    imagem(raiz / "valid", "poligono_jpg.rf.bb.jpg", [f"{jogador} 0.1 0.2 0.3 0.2 0.3 0.6 0.1 0.6"])
    return raiz


def dataset_preparado(raiz: Path, quantidades=(2, 1, 1), tamanho=(64, 64)) -> Path:
    """Dataset de uma classe como o de `treino preparar`: um retângulo claro em fundo verde."""
    for split, n in zip(("train", "valid", "test"), quantidades):
        pasta = raiz / split
        (pasta / "images").mkdir(parents=True, exist_ok=True)
        (pasta / "labels").mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = Image.new("RGB", tamanho, (40, 140, 40))
            ImageDraw.Draw(img).rectangle((24, 16, 40, 48), fill=(230, 230, 230))
            img.save(pasta / "images" / f"{split}{i}.jpg")
            (pasta / "labels" / f"{split}{i}.txt").write_text("0 0.5 0.5 0.25 0.5\n")
    (raiz / "data.yaml").write_text(yaml.safe_dump({
        "path": str(raiz.resolve()), "train": "train/images", "val": "valid/images",
        "test": "test/images", "nc": 1, "names": ["player"],
    }), encoding="utf-8")
    (raiz / "manifest.json").write_text('{"semente": 0}', encoding="utf-8")
    return raiz


class _YoloFalso:
    """Substitui `ultralytics.YOLO`: grava weights/best.pt e last.pt onde o Ultralytics gravaria."""

    def __init__(self, pesos: str, chamadas: list, falhar: BaseException | None):
        self.pesos, self.chamadas, self.falhar = pesos, chamadas, falhar

    def train(self, **kw):
        self.chamadas.append((self.pesos, kw))
        pasta = Path(self.pesos).parents[1] if kw.get("resume") else Path(kw["project"]) / kw["name"]
        (pasta / "weights").mkdir(parents=True, exist_ok=True)
        (pasta / "weights" / "best.pt").write_bytes(b"best")
        (pasta / "weights" / "last.pt").write_bytes(b"last")
        if self.falhar is not None:
            raise self.falhar
        self.trainer = SimpleNamespace(save_dir=pasta,
                                       metrics={"metrics/mAP50(B)": 0.91, "fitness": 0.8})


def instalar_yolo_falso(monkeypatch, falhar: BaseException | None = None) -> list:
    """Troca `rodar._yolo` e `rodar._gpu`; devolve a lista de chamadas (pesos, kwargs de train)."""
    from nfl_vision.treino import rodar

    chamadas: list = []
    monkeypatch.setattr(rodar, "_yolo", lambda pesos: _YoloFalso(str(pesos), chamadas, falhar))
    monkeypatch.setattr(rodar, "_gpu", lambda: "GPU falsa")
    return chamadas
```

(`dataset_preparado` e o YOLO falso são usados na Task 4; o import de `nfl_vision.treino.rodar` é preguiçoso, então o módulo já pode ser importado nesta task.)

- [ ] **Step 2: Escrever os testes que falham**

Crie `pipeline/tests/test_treino_preparar.py`:

```python
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from nfl_vision.treino import preparar
from nfl_vision.treino.fontes import BASE, EXTERNAS, POR_NOME, MapeamentoInvalido, clipe_base
from nfl_vision.treino.preparar import SPLITS, Decisao, construir, converter, split_externo
from treino_sintetico import NOMES_EXTERNOS, dataset_base, dataset_externo

PITCH = POR_NOME["pitchcamera"]


def test_converter_mapeia_player_e_descarta_o_resto(tmp_path):
    rotulos = tmp_path / "a.txt"
    rotulos.write_text("2 0.5 0.5 0.2 0.4\n1 0.3 0.3 0.1 0.1\n0 0.7 0.7 0.02 0.02\n"
                       "2 0.1 0.2 0.3 0.2 0.3 0.6 0.1 0.6\n")

    linhas, descartadas = converter(rotulos, NOMES_EXTERNOS["pitchcamera"], PITCH)

    assert linhas == ["0 0.500000 0.500000 0.200000 0.400000",
                      "0 0.200000 0.400000 0.200000 0.400000"]
    assert descartadas == Counter({"referee": 1, "football": 1})


def test_converter_corta_caixa_que_sai_da_imagem(tmp_path):
    rotulos = tmp_path / "a.txt"
    rotulos.write_text("2 0.95 0.5 0.2 0.2\n")
    linhas, _ = converter(rotulos, NOMES_EXTERNOS["pitchcamera"], PITCH)
    assert linhas == ["0 0.925000 0.500000 0.150000 0.200000"]


def test_converter_sem_arquivo_de_rotulos(tmp_path):
    assert converter(tmp_path / "nao.txt", ["players"], PITCH) == ([], Counter())


def test_converter_classe_numerica_no_yaml(tmp_path):
    # names: [..., 1, 2, ...] sem aspas vira int no YAML
    rotulos = tmp_path / "a.txt"
    rotulos.write_text("2 0.5 0.5 0.1 0.1\n7 0.5 0.5 0.2 0.4\n")
    nomes = ["ball", "referee", 1, 2, 3, 4, 5, "american-football-players", "whitehat"]
    linhas, descartadas = converter(rotulos, nomes, POR_NOME["fhtw"])
    assert len(linhas) == 1 and descartadas == Counter({"1": 1})


def test_split_externo_reprodutivel_e_agrupa_copias():
    nomes = [f"img{i:04d}_png.rf.{c}.jpg" for i in range(400) for c in ("aa", "bb")]
    a = [split_externo("evzn", n, 0) for n in nomes]
    assert a == [split_externo("evzn", n, 0) for n in nomes]
    assert all(a[k] == a[k + 1] for k in range(0, len(a), 2))  # cópias aumentadas juntas
    assert 0.10 < a.count("valid") / len(a) < 0.20
    assert a != [split_externo("evzn", n, 1) for n in nomes]


def _decisoes(datasets: Path, aprovadas=("pitchcamera", "fhtw", "evzn")) -> list[Decisao]:
    lista = [Decisao(BASE, True, "base", dataset_base(datasets))]
    for f in EXTERNAS:
        raiz = dataset_externo(datasets, f)
        aprovada = f.nome in aprovadas
        lista.append(Decisao(f, aprovada, "ok" if aprovada else "soccer", raiz if aprovada else None))
    return lista


def _arquivos(saida: Path, split: str, sub: str = "images") -> list[str]:
    return sorted(p.name for p in (saida / split / sub).iterdir())


def test_construir_separa_a_base_por_clipe(tmp_path):
    saida = tmp_path / "saida"
    construir(saida, _decisoes(tmp_path / "datasets"))

    teste, valid, treino = (_arquivos(saida, s) for s in ("test", "valid", "train"))
    assert len(teste) == 4 and all(n.startswith("base__cin_cle_") for n in teste)
    assert not any("cin_cle_" in n for n in valid + treino)
    base_valid = [n for n in valid if n.startswith("base__")]
    assert len(base_valid) == 2
    assert all(clipe_base(n.removeprefix("base__")) == "tb_atl_wk1_penix_pass_all22" for n in base_valid)
    assert {clipe_base(n.removeprefix("base__")) for n in treino if n.startswith("base__")} == {
        "tb_atl_wk1_penix_pass", "phi_dal_wk1_williams_run", "phi_dal_wk1_williams_run_endzone"}
    assert not any(n.split("__")[0] in {"pitchcamera", "fhtw", "evzn"} for n in teste)


def test_construir_converte_prefixa_e_conta(tmp_path):
    saida = tmp_path / "saida"
    manifest = construir(saida, _decisoes(tmp_path / "datasets"))

    todas = [n for s in SPLITS for n in _arquivos(saida, s)]
    assert all(n.split("__")[0] in {"base", "pitchcamera", "fhtw", "evzn"} for n in todas)
    assert not any("so_arbitro" in n for n in todas)  # sem player fica fora
    for s in SPLITS:
        assert [Path(n).stem for n in _arquivos(saida, s)] == \
            [Path(n).stem for n in _arquivos(saida, s, "labels")]
        for rotulo in (saida / s / "labels").iterdir():
            assert all(linha.split()[0] == "0" for linha in rotulo.read_text().splitlines())
    assert yaml.safe_load((saida / "data.yaml").read_text("utf-8"))["names"] == ["player"]

    assert manifest["descartadas"]["base"] == {"ball": 12, "referee": 12}
    # 40 cópias com 1 árbitro cada + a imagem só com árbitro
    assert manifest["descartadas"]["pitchcamera"] == {"referee": 41}
    assert manifest["sem_player"] == {"base": 0, "pitchcamera": 1, "fhtw": 1, "evzn": 1}
    assert manifest["contagens"]["test"] == {"base": {"imagens": 4, "caixas": 4}}
    for s in SPLITS:
        assert sum(c["imagens"] for c in manifest["contagens"][s].values()) == len(_arquivos(saida, s))
    for f in EXTERNAS:
        total = sum(manifest["contagens"][s].get(f.nome, {"imagens": 0})["imagens"]
                    for s in ("train", "valid"))
        assert total == 41


def test_manifest_registra_decisoes(tmp_path):
    saida = tmp_path / "saida"
    manifest = construir(saida, _decisoes(tmp_path / "datasets", aprovadas=("evzn",)))

    por_nome = {f["nome"]: f for f in manifest["fontes"]}
    assert list(por_nome) == ["base", "pitchcamera", "fhtw", "evzn"]
    assert por_nome["fhtw"]["aprovada"] is False and por_nome["fhtw"]["motivo"] == "soccer"
    assert por_nome["fhtw"]["pasta"] is None
    assert por_nome["evzn"]["aprovada"] is True
    assert (por_nome["evzn"]["workspace"], por_nome["evzn"]["projeto"], por_nome["evzn"]["versao"]) == \
        ("evzn", "american-football-analyst", 4)
    assert por_nome["evzn"]["mapeamento"] == {"football-players": "player"}
    assert all("fhtw" not in c and "pitchcamera" not in c for c in manifest["contagens"].values())
    assert manifest["semente"] == 0 and manifest["versao"]
    assert json.loads((saida / "manifest.json").read_text("utf-8")) == manifest


def test_construir_reprodutivel(tmp_path):
    decisoes = _decisoes(tmp_path / "datasets")
    construir(tmp_path / "a", decisoes)
    construir(tmp_path / "b", decisoes)
    for s in SPLITS:
        assert _arquivos(tmp_path / "a", s) == _arquivos(tmp_path / "b", s)


def test_construir_recusa_dataset_existente(tmp_path):
    decisoes = _decisoes(tmp_path / "datasets")
    construir(tmp_path / "saida", decisoes)
    with pytest.raises(FileExistsError, match="já tem um dataset"):
        construir(tmp_path / "saida", decisoes)


def test_construir_aceita_pasta_so_com_triagem(tmp_path):
    (tmp_path / "saida" / "triagem").mkdir(parents=True)
    construir(tmp_path / "saida", _decisoes(tmp_path / "datasets"))
    assert (tmp_path / "saida" / "data.yaml").exists()


def test_construir_exige_base(tmp_path):
    with pytest.raises(ValueError, match="base"):
        construir(tmp_path / "saida", _decisoes(tmp_path / "datasets")[1:])


def test_construir_mapeamento_invalido_nao_grava_nada(tmp_path):
    datasets = tmp_path / "datasets"
    evzn = POR_NOME["evzn"]
    lista = [Decisao(BASE, True, "base", dataset_base(datasets)),
             Decisao(evzn, True, "ok", dataset_externo(datasets, evzn, nomes=["ball", "referee", "players"]))]
    with pytest.raises(MapeamentoInvalido, match="football-players"):
        construir(tmp_path / "saida", lista)
    assert not (tmp_path / "saida" / "train").exists()
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `uv run pytest tests/test_treino_preparar.py -v`
Expected: erro de coleta `ModuleNotFoundError: No module named 'nfl_vision.treino.preparar'`.

- [ ] **Step 4: Implementar `preparar.py`**

Crie `pipeline/nfl_vision/treino/preparar.py`:

```python
"""Montagem do dataset de treino do detector: conversão para `player`, splits sem vazamento e manifest."""

import hashlib
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from nfl_vision.eval.datasets import IMAGENS, ler_rotulos, nomes_das_classes
from nfl_vision.pipeline import _versao
from nfl_vision.runner import gravar_json, gravar_texto
from nfl_vision.treino.fontes import (
    CLIPE_VALID, PREFIXO_TESTE, Fonte, split_base, validar_mapeamento,
)

SPLITS = ("train", "valid", "test")
FRACAO_VALID_EXTERNOS = 0.15
PASTA_PADRAO = "treino-player-v1"
MOTIVO_BASE = "base: jogo de teste (cin_cle) e treino"


@dataclass(frozen=True)
class Decisao:
    fonte: Fonte
    aprovada: bool
    motivo: str
    raiz: Path | None = None  # pasta baixada; None para fonte rejeitada


def imagens_da_fonte(raiz: Path) -> list[tuple[Path, Path]]:
    """(imagem, rótulo) de todos os splits do export (train, valid, test...), em ordem estável."""
    pares = []
    for pasta in sorted(p for p in raiz.glob("*/images") if p.is_dir()):
        for imagem in sorted(pasta.iterdir()):
            if imagem.suffix.lower() in IMAGENS:
                pares.append((imagem, pasta.parent / "labels" / f"{imagem.stem}.txt"))
    return pares


def converter(rotulos: Path, nomes: list, fonte: Fonte) -> tuple[list[str], Counter]:
    """Linhas YOLO `0 cx cy w h` das caixas de player e a contagem das descartadas por classe.

    Polígonos viram a caixa que os envolve; caixas são cortadas aos limites da imagem.
    """
    linhas: list[str] = []
    descartadas: Counter = Counter()
    if not rotulos.exists():
        return linhas, descartadas
    for classe, caixa in ler_rotulos(rotulos, nomes):
        if str(classe) not in fonte.classes_player:
            descartadas[str(classe)] += 1
            continue
        x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in caixa)
        if x2 <= x1 or y2 <= y1:
            descartadas["(caixa degenerada)"] += 1
            continue
        linhas.append(f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
    return linhas, descartadas


def grupo_externo(nome_arquivo: str) -> str:
    """Imagem original: o Roboflow exporta cópias aumentadas como `<original>.rf.<hash>.<ext>`."""
    return nome_arquivo.split(".rf.")[0]


def split_externo(fonte: str, nome_arquivo: str, semente: int,
                  fracao: float = FRACAO_VALID_EXTERNOS) -> str:
    """Sorteio reprodutível por imagem original (independe da ordem dos arquivos)."""
    chave = f"{semente}:{fonte}:{grupo_externo(nome_arquivo)}"
    sorteio = int(hashlib.sha256(chave.encode()).hexdigest()[:12], 16) / 16**12
    return "valid" if sorteio < fracao else "train"


def _descrever(d: Decisao) -> dict:
    f = d.fonte
    return {
        "nome": f.nome, "workspace": f.workspace, "projeto": f.projeto, "versao": f.versao,
        "aprovada": d.aprovada, "motivo": d.motivo,
        "mapeamento": {c: "player" for c in f.classes_player},
        "pasta": str(d.raiz) if d.raiz is not None else None,
    }


def construir(saida: Path, decisoes: list[Decisao], semente: int = 0) -> dict:
    """Monta o dataset YOLO de uma classe em `saida`, grava data.yaml e manifest.json e devolve o manifest."""
    if not any(not d.fonte.externa and d.aprovada for d in decisoes):
        raise ValueError("a fonte base é obrigatória: é dela que vem o split de teste")
    existentes = [n for n in (*SPLITS, "data.yaml") if (saida / n).exists()]
    if existentes:
        raise FileExistsError(
            f"{saida} já tem um dataset ({', '.join(existentes)}); "
            "apague essas pastas ou use outra --saida")
    aprovadas = [d for d in decisoes if d.aprovada]
    nomes_por_fonte = {}
    for d in aprovadas:  # valida tudo antes de gravar qualquer arquivo
        if d.raiz is None:
            raise ValueError(f"fonte aprovada '{d.fonte.nome}' sem pasta baixada")
        nomes = nomes_das_classes(d.raiz)
        validar_mapeamento(d.fonte, nomes)
        nomes_por_fonte[d.fonte.nome] = nomes

    for split in SPLITS:
        for sub in ("images", "labels"):
            (saida / split / sub).mkdir(parents=True, exist_ok=True)
    contagens: dict[str, dict] = {split: {} for split in SPLITS}
    descartadas: dict[str, dict] = {}
    sem_player: dict[str, int] = {}
    for d in aprovadas:
        fonte, nomes = d.fonte, nomes_por_fonte[d.fonte.nome]
        descartes: Counter = Counter()
        sem = 0
        for imagem, rotulos in imagens_da_fonte(d.raiz):
            linhas, desc = converter(rotulos, nomes, fonte)
            descartes.update(desc)
            if not linhas:
                sem += 1
                continue
            split = (split_externo(fonte.nome, imagem.name, semente) if fonte.externa
                     else split_base(imagem.name))
            nome = f"{fonte.nome}__{imagem.name}"
            if any((saida / s / "images" / nome).exists() for s in SPLITS):
                raise ValueError(f"fonte '{fonte.nome}': imagem repetida no export: {imagem.name}")
            alvo = saida / split / "images" / nome
            shutil.copy2(imagem, alvo)
            gravar_texto(saida / split / "labels" / f"{alvo.stem}.txt", "\n".join(linhas) + "\n")
            c = contagens[split].setdefault(fonte.nome, {"imagens": 0, "caixas": 0})
            c["imagens"] += 1
            c["caixas"] += len(linhas)
        descartadas[fonte.nome] = dict(sorted(descartes.items()))
        sem_player[fonte.nome] = sem

    gravar_texto(saida / "data.yaml", yaml.safe_dump({
        # path absoluto: o Ultralytics não depende do diretório atual nem do seu datasets_dir
        "path": str(saida.resolve()), "train": "train/images", "val": "valid/images",
        "test": "test/images", "nc": 1, "names": ["player"],
    }, sort_keys=False, allow_unicode=True))
    manifest = {
        "versao": _versao("nfl-vision"),
        "criado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "semente": semente,
        "fracao_valid_externos": FRACAO_VALID_EXTERNOS,
        "regras_de_split": {
            "test": f"clipes {PREFIXO_TESTE}* da base",
            "valid": (f"clipe {CLIPE_VALID} da base + {FRACAO_VALID_EXTERNOS:.0%} de cada externa "
                      "aprovada (sorteio por imagem original)"),
            "train": "o restante",
        },
        "fontes": [_descrever(d) for d in decisoes],
        "contagens": contagens,
        "descartadas": descartadas,
        "sem_player": sem_player,
    }
    gravar_json(saida / "manifest.json", manifest)
    return manifest
```

- [ ] **Step 5: Rodar os testes da task e a suíte rápida**

Run: `uv run pytest tests/test_treino_preparar.py -v`
Expected: todos PASS.

Run: `uv run pytest`
Expected: todos PASS.

- [ ] **Step 6: Commit**

```bash
git add nfl_vision/treino/preparar.py tests/treino_sintetico.py tests/test_treino_preparar.py
git commit -m "feat: monta o dataset de player com split por clipe e manifest"
```

---

### Task 3: Painéis de triagem e comando `treino preparar`

**Files:**
- Modify: `pipeline/nfl_vision/treino/preparar.py` (imports; novas `painel` e `triagem`)
- Modify: `pipeline/nfl_vision/cli.py` (grupo `treino`, comando `preparar`)
- Test: `pipeline/tests/test_treino_preparar.py`, `pipeline/tests/test_cli_treino.py`

- [ ] **Step 1: Escrever os testes que falham dos painéis**

Acrescente ao fim de `pipeline/tests/test_treino_preparar.py`:

```python
def test_painel_de_triagem(tmp_path):
    import cv2
    import numpy as np

    raiz = dataset_externo(tmp_path / "datasets", PITCH)
    destino = tmp_path / "triagem" / "pitchcamera.jpg"

    n = preparar.painel(PITCH, raiz, destino)

    assert n == 42
    img = cv2.imdecode(np.fromfile(str(destino), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (3 * 320, 4 * 480, 3)
    assert (img[..., 1] > 180).any()  # caixas de player em verde


def test_triagem_gera_um_painel_por_fonte(tmp_path):
    datasets = tmp_path / "datasets"
    pares = [(f, dataset_externo(datasets, f)) for f in EXTERNAS]

    resumo = preparar.triagem(pares, tmp_path / "saida")

    assert [r["fonte"] for r in resumo] == ["pitchcamera", "fhtw", "evzn"]
    for r in resumo:
        assert r["imagens"] == 42
        assert r["painel"] == tmp_path / "saida" / "triagem" / f"{r['fonte']}.jpg"
        assert r["painel"].exists()
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_treino_preparar.py -k "painel or triagem" -v`
Expected: FAIL com `AttributeError: module 'nfl_vision.treino.preparar' has no attribute 'painel'`.

- [ ] **Step 3: Implementar `painel` e `triagem`**

Em `pipeline/nfl_vision/treino/preparar.py`, troque o bloco de imports do topo por:

```python
import hashlib
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

from nfl_vision.eval.datasets import IMAGENS, ler_rotulos, nomes_das_classes
from nfl_vision.pipeline import _versao
from nfl_vision.runner import gravar_json, gravar_texto
from nfl_vision.stages.ingest import carregar_imagem
from nfl_vision.treino.fontes import (
    CLIPE_VALID, PREFIXO_TESTE, Fonte, split_base, validar_mapeamento,
)
```

e acrescente ao fim do arquivo:

```python
MINIATURA = (480, 320)  # largura, altura de cada imagem do painel
COLUNAS = 4
VERDE, CINZA = (0, 200, 0), (170, 170, 170)


def painel(fonte: Fonte, raiz: Path, destino: Path, n: int = 12, semente: int = 0) -> int:
    """Grade com `n` imagens sorteadas da fonte e suas caixas: player em verde, as outras
    classes em cinza com o nome. Devolve o total de imagens da fonte."""
    nomes = nomes_das_classes(raiz)
    validar_mapeamento(fonte, nomes)
    pares = imagens_da_fonte(raiz)
    if not pares:
        raise ValueError(f"fonte '{fonte.nome}': nenhuma imagem em {raiz}/*/images")
    escolhidas = random.Random(semente).sample(pares, min(n, len(pares)))
    largura, altura = MINIATURA
    linhas_grade = -(-len(escolhidas) // COLUNAS)
    tela = np.zeros((altura * linhas_grade, largura * COLUNAS, 3), np.uint8)
    for k, (imagem, rotulos) in enumerate(escolhidas):
        img = carregar_imagem(imagem)
        h, w = img.shape[:2]
        espessura = max(2, round(max(h, w) / 400))
        for classe, (x1, y1, x2, y2) in (ler_rotulos(rotulos, nomes) if rotulos.exists() else []):
            jogador = str(classe) in fonte.classes_player
            cor = VERDE if jogador else CINZA
            p1, p2 = (round(x1 * w), round(y1 * h)), (round(x2 * w), round(y2 * h))
            cv2.rectangle(img, p1, p2, cor, espessura)
            if not jogador:
                cv2.putText(img, str(classe), (p1[0], max(p1[1] - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                            max(0.5, max(h, w) / 1600), cor, espessura)
        escala = min(largura / w, altura / h)
        mini = cv2.resize(img, (max(1, round(w * escala)), max(1, round(h * escala))),
                          interpolation=cv2.INTER_AREA)
        lin, col = divmod(k, COLUNAS)
        tela[lin * altura:lin * altura + mini.shape[0], col * largura:col * largura + mini.shape[1]] = mini
    destino.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", tela)
    if not ok:
        raise RuntimeError(f"falha ao gerar {destino}")
    buf.tofile(str(destino))
    return len(pares)


def triagem(pares: list[tuple[Fonte, Path]], saida: Path, semente: int = 0) -> list[dict]:
    """Um painel por fonte em `<saida>/triagem/<fonte>.jpg`."""
    resumo = []
    for fonte, raiz in pares:
        destino = saida / "triagem" / f"{fonte.nome}.jpg"
        resumo.append({"fonte": fonte.nome, "imagens": painel(fonte, raiz, destino, semente=semente),
                       "painel": destino})
    return resumo
```

- [ ] **Step 4: Rodar os testes dos painéis**

Run: `uv run pytest tests/test_treino_preparar.py -v`
Expected: todos PASS.

- [ ] **Step 5: Escrever os testes que falham da CLI**

Crie `pipeline/tests/test_cli_treino.py`:

```python
import json

from typer.testing import CliRunner

from nfl_vision import cli, paths
from nfl_vision.cli import app
from nfl_vision.treino.fontes import EXTERNAS, POR_NOME
from nfl_vision.treino.preparar import MOTIVO_BASE
from treino_sintetico import dataset_base, dataset_externo

runner = CliRunner()
DECISOES = ["--aprovar", "evzn:futebol americano, caixas boas",
            "--rejeitar", "pitchcamera:soccer", "--rejeitar", "fhtw:caixas ruins"]


def _fontes_locais():
    datasets = paths.datasets_dir()
    dataset_base(datasets)
    for f in EXTERNAS:
        dataset_externo(datasets, f)


def _pasta():
    return paths.datasets_dir() / "treino-player-v1"


def test_treino_help_lista_comandos():
    r = runner.invoke(app, ["treino", "--help"])
    assert r.exit_code == 0, r.output
    assert "preparar" in r.output


def test_preparar_so_triagem(dados, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    _fontes_locais()

    r = runner.invoke(app, ["treino", "preparar", "--so-triagem"])

    assert r.exit_code == 0, r.output
    assert sorted(p.name for p in (_pasta() / "triagem").iterdir()) == \
        ["evzn.jpg", "fhtw.jpg", "pitchcamera.jpg"]
    assert not (_pasta() / "data.yaml").exists()
    assert "--aprovar" in r.output


def test_preparar_com_decisoes(dados, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    _fontes_locais()

    r = runner.invoke(app, ["treino", "preparar", *DECISOES])

    assert r.exit_code == 0, r.output
    assert "Dataset em:" in r.output
    manifest = json.loads((_pasta() / "manifest.json").read_text("utf-8"))
    assert {f["nome"]: (f["aprovada"], f["motivo"]) for f in manifest["fontes"]} == {
        "base": (True, MOTIVO_BASE),
        "pitchcamera": (False, "soccer"),
        "fhtw": (False, "caixas ruins"),
        "evzn": (True, "futebol americano, caixas boas"),
    }


def test_preparar_erros_de_entrada(dados, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    _fontes_locais()
    casos = [
        (["--aprovar", "evzn:ok"], "faltam: pitchcamera, fhtw"),
        (["--aprovar", "xyz:ok", *DECISOES], "fonte desconhecida 'xyz'"),
        (["--aprovar", "evzn", "--rejeitar", "pitchcamera:x", "--rejeitar", "fhtw:y"], "motivo"),
        (["--rejeitar", "evzn:x", *DECISOES], "mais de uma vez"),
        (["--so-triagem", "--aprovar", "evzn:ok"], "--so-triagem"),
    ]
    for args, trecho in casos:
        r = runner.invoke(app, ["treino", "preparar", *args])
        assert r.exit_code == 2, (args, r.output)
        assert trecho in r.output, (args, r.output)

    assert runner.invoke(app, ["treino", "preparar", *DECISOES]).exit_code == 0
    r = runner.invoke(app, ["treino", "preparar", *DECISOES])
    assert r.exit_code == 2, r.output
    assert "já tem um dataset" in r.output


def test_preparar_fonte_sem_acesso(dados, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)  # um .env local não pode trazer a chave
    dataset_base(paths.datasets_dir())  # externas ausentes: precisariam de download

    r = runner.invoke(app, ["treino", "preparar", "--so-triagem"])

    assert r.exit_code == 2, r.output
    assert "pitchcamera" in r.output and "ROBOFLOW_API_KEY" in r.output


def test_preparar_mapeamento_invalido_lista_classes(dados, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    datasets = paths.datasets_dir()
    dataset_base(datasets)
    dataset_externo(datasets, POR_NOME["pitchcamera"])
    dataset_externo(datasets, POR_NOME["fhtw"])
    dataset_externo(datasets, POR_NOME["evzn"], nomes=["ball", "referee", "players"])

    r = runner.invoke(app, ["treino", "preparar", "--so-triagem"])

    assert r.exit_code == 2, r.output
    assert "football-players" in r.output and "ball, referee, players" in r.output
```

- [ ] **Step 6: Rodar e ver falhar**

Run: `uv run pytest tests/test_cli_treino.py -v`
Expected: FAIL — `No such command 'treino'` (saída 2) em todos.

- [ ] **Step 7: Implementar o comando `treino preparar`**

Em `pipeline/nfl_vision/cli.py`, logo depois de `app.add_typer(eval_app, name="eval")`, acrescente:

```python
treino_app = typer.Typer(help="Preparação do dataset e ajuste fino do detector de jogadores.",
                         no_args_is_help=True)
app.add_typer(treino_app, name="treino")
```

E ao fim do arquivo acrescente:

```python
def _decisoes_externas(aprovar: List[str], rejeitar: List[str]) -> dict[str, tuple[bool, str]]:
    """{fonte: (aprovada, motivo)} a partir de valores 'nome:motivo'."""
    from nfl_vision.treino.fontes import EXTERNAS

    validas = [f.nome for f in EXTERNAS]
    decisoes: dict[str, tuple[bool, str]] = {}
    for opcao, aprovada, valores in (("--aprovar", True, aprovar), ("--rejeitar", False, rejeitar)):
        for valor in valores:
            nome, _, motivo = valor.partition(":")
            nome, motivo = nome.strip(), motivo.strip()
            if nome not in validas:
                raise typer.BadParameter(
                    f"fonte desconhecida '{nome}'; use: {', '.join(validas)}", param_hint=opcao)
            if not motivo:
                raise typer.BadParameter(f"informe o motivo: '{nome}:<motivo>'", param_hint=opcao)
            if nome in decisoes:
                raise typer.BadParameter(f"fonte '{nome}' decidida mais de uma vez", param_hint=opcao)
            decisoes[nome] = (aprovada, motivo)
    return decisoes


@treino_app.command("preparar")
def treino_preparar(
    saida: Optional[Path] = typer.Option(
        None, "--saida", help="Pasta do dataset; padrão: data/datasets/treino-player-v1"),
    so_triagem: bool = typer.Option(
        False, "--so-triagem", help="Só baixa as fontes externas e gera os painéis de triagem"),
    aprovar: List[str] = typer.Option(
        [], "--aprovar", help="Fonte externa aprovada, 'nome:motivo' (repetível)"),
    rejeitar: List[str] = typer.Option(
        [], "--rejeitar", help="Fonte externa rejeitada, 'nome:motivo' (repetível)"),
    semente: int = typer.Option(0, "--semente", help="Semente do sorteio de validação dos externos"),
) -> None:
    """Prepara o dataset de player: triagem das fontes externas (--so-triagem) e depois a montagem."""
    from nfl_vision.treino import fontes, preparar

    datasets = paths.datasets_dir()
    saida = saida or datasets / preparar.PASTA_PADRAO
    erros = (fontes.FonteIndisponivel, FileNotFoundError, FileExistsError, ValueError)

    if so_triagem:
        if aprovar or rejeitar:
            raise typer.BadParameter("--so-triagem não aceita --aprovar nem --rejeitar",
                                     param_hint="--so-triagem")
        try:
            pares = [(f, fontes.obter(f, datasets)) for f in fontes.EXTERNAS]
            resumo = preparar.triagem(pares, saida, semente)
        except erros as exc:
            raise typer.BadParameter(str(exc)) from exc
        tabela = Table(title="Triagem das fontes externas")
        for coluna in ("fonte", "imagens", "painel"):
            tabela.add_column(coluna)
        for r in resumo:
            tabela.add_row(r["fonte"], str(r["imagens"]), escape(str(r["painel"])))
        console.print(tabela)
        console.print("Veja os painéis e decida cada fonte: nfl-vision treino preparar "
                      "--aprovar <fonte>:<motivo> --rejeitar <fonte>:<motivo>")
        return

    decididas = _decisoes_externas(aprovar, rejeitar)
    faltam = [f.nome for f in fontes.EXTERNAS if f.nome not in decididas]
    if faltam:
        raise typer.BadParameter(
            f"decida todas as fontes externas (faltam: {', '.join(faltam)}); "
            "gere os painéis antes com --so-triagem", param_hint="--aprovar")
    try:
        decisoes = [preparar.Decisao(fontes.BASE, True, preparar.MOTIVO_BASE,
                                     fontes.obter(fontes.BASE, datasets))]
        for f in fontes.EXTERNAS:
            aprovada, motivo = decididas[f.nome]
            decisoes.append(preparar.Decisao(
                f, aprovada, motivo, fontes.obter(f, datasets) if aprovada else None))
        manifest = preparar.construir(saida, decisoes, semente)
    except erros as exc:
        raise typer.BadParameter(str(exc)) from exc

    tabela = Table(title=f"Dataset {saida.name}")
    for coluna in ("split", "fonte", "imagens", "caixas"):
        tabela.add_column(coluna)
    for split, por_fonte in manifest["contagens"].items():
        for nome, c in por_fonte.items():
            tabela.add_row(split, nome, str(c["imagens"]), str(c["caixas"]))
    console.print(tabela)
    console.print(f"Dataset em: {escape(str(saida))}")
```

- [ ] **Step 8: Rodar os testes da task e a suíte rápida**

Run: `uv run pytest tests/test_cli_treino.py tests/test_treino_preparar.py -v`
Expected: todos PASS.

Run: `uv run pytest`
Expected: todos PASS.

- [ ] **Step 9: Commit**

```bash
git add nfl_vision/treino/preparar.py nfl_vision/cli.py tests/test_treino_preparar.py tests/test_cli_treino.py
git commit -m "feat: painéis de triagem e comando treino preparar"
```

---

### Task 4: Treino com Ultralytics e comando `treino rodar`

**Files:**
- Modify: `pipeline/nfl_vision/paths.py` (nova `treinos_dir`)
- Create: `pipeline/nfl_vision/treino/rodar.py`
- Modify: `pipeline/nfl_vision/cli.py` (comando `treino rodar`)
- Test: `pipeline/tests/test_treino_rodar.py`, `pipeline/tests/test_cli_treino.py`

Notas para quem implementa:
- `project` precisa ser absoluto; com caminho relativo o Ultralytics grava em `runs/detect/...`.
- Com `exist_ok=False` o Ultralytics **não** dá erro com pasta existente: cria `player-v12` em silêncio. Por isso o código recusa antes um `--nome` repetido, cria a pasta (para gravar o manifest desde o início) e passa `exist_ok=True`.
- Windows: os workers do dataloader usam `spawn` e reimportam o módulo principal. O executável `nfl-vision` gerado pelo uv chama `app()` sob `if __name__ == "__main__"`, então `workers=2` é seguro pela CLI. No teste `@model` (dentro do pytest) use `workers=0`.
- `YOLO(last.pt).train(resume=True)` lê dataset, parâmetros e pasta do checkpoint.

- [ ] **Step 1: Escrever os testes que falham**

Crie `pipeline/tests/test_treino_rodar.py`:

```python
import hashlib
import json

import pytest

from nfl_vision.treino import rodar
from nfl_vision.treino.rodar import EntradaInvalida
from treino_sintetico import dataset_preparado, instalar_yolo_falso


def _manifest(pasta):
    return json.loads((pasta / "manifest.json").read_text("utf-8"))


def test_validar_dataset_ok(tmp_path):
    ds = dataset_preparado(tmp_path / "ds")
    assert rodar.validar_dataset(ds) == ds / "data.yaml"


def test_validar_dataset_erros(tmp_path):
    with pytest.raises(EntradaInvalida, match="data.yaml"):
        rodar.validar_dataset(tmp_path / "nada")
    ds = dataset_preparado(tmp_path / "ds")
    for p in (ds / "valid" / "images").iterdir():
        p.unlink()
    with pytest.raises(EntradaInvalida, match="split 'valid' sem imagens"):
        rodar.validar_dataset(ds)


def test_validar_dataset_exige_so_player(tmp_path):
    ds = dataset_preparado(tmp_path / "ds")
    (ds / "data.yaml").write_text("names: ['ball', 'player']\n", encoding="utf-8")
    with pytest.raises(EntradaInvalida, match="só a classe player"):
        rodar.validar_dataset(ds)


def test_treinar_passa_parametros_e_grava_manifest(tmp_path, monkeypatch):
    chamadas = instalar_yolo_falso(monkeypatch)
    ds = dataset_preparado(tmp_path / "ds")

    pasta = rodar.treinar(ds, "player-v1", tmp_path / "treinos", parametros={"epochs": 3})

    assert pasta == (tmp_path / "treinos" / "player-v1").resolve()
    pesos, kw = chamadas[0]
    assert pesos == "yolo11m.pt"
    assert kw == {"data": str((ds / "data.yaml").resolve()), "project": str(pasta.parent),
                  "name": "player-v1", "exist_ok": True, **rodar.PARAMETROS, "epochs": 3}
    m = _manifest(pasta)
    assert m["status"] == "concluido" and m["nome"] == "player-v1" and m["mensagem"] is None
    assert m["parametros"]["epochs"] == 3 and m["parametros"]["imgsz"] == 1280
    assert m["parametros"]["single_cls"] is True and m["parametros"]["workers"] == 2
    assert m["data_yaml_sha256"] == hashlib.sha256((ds / "data.yaml").read_bytes()).hexdigest()
    assert m["dataset_manifest_sha256"] == hashlib.sha256((ds / "manifest.json").read_bytes()).hexdigest()
    assert set(m["versoes"]) == {"nfl-vision", "ultralytics", "torch"}
    assert m["gpu"] == "GPU falsa" and m["duracao_s"] >= 0 and m["retomadas"] == []
    assert m["best"] == {"caminho": str(pasta / "weights" / "best.pt"),
                         "sha256": hashlib.sha256(b"best").hexdigest()}
    assert m["metricas"]["metrics/mAP50(B)"] == 0.91


def test_treinar_recusa_nome_existente_e_invalido(tmp_path, monkeypatch):
    instalar_yolo_falso(monkeypatch)
    ds = dataset_preparado(tmp_path / "ds")
    rodar.treinar(ds, "player-v1", tmp_path / "treinos")
    with pytest.raises(EntradaInvalida, match="já existe"):
        rodar.treinar(ds, "player-v1", tmp_path / "treinos")
    with pytest.raises(EntradaInvalida, match="nome de treino inválido"):
        rodar.treinar(ds, "../fora", tmp_path / "treinos")


def test_treinar_com_erro_registra_no_manifest(tmp_path, monkeypatch):
    instalar_yolo_falso(monkeypatch, falhar=RuntimeError("CUDA out of memory"))
    ds = dataset_preparado(tmp_path / "ds")
    with pytest.raises(RuntimeError, match="out of memory"):
        rodar.treinar(ds, "x", tmp_path / "treinos")
    m = _manifest(tmp_path / "treinos" / "x")
    assert m["status"] == "erro" and "CUDA out of memory" in m["mensagem"]


def test_retomar_sem_last(tmp_path):
    with pytest.raises(EntradaInvalida, match="last.pt"):
        rodar.retomar("player-v1", tmp_path / "treinos")


def test_retomar_continua_do_last(tmp_path, monkeypatch):
    instalar_yolo_falso(monkeypatch, falhar=KeyboardInterrupt())
    ds = dataset_preparado(tmp_path / "ds")
    with pytest.raises(KeyboardInterrupt):
        rodar.treinar(ds, "player-v1", tmp_path / "treinos")
    pasta = (tmp_path / "treinos" / "player-v1").resolve()
    assert _manifest(pasta)["status"] == "interrompido"

    chamadas = instalar_yolo_falso(monkeypatch)
    assert rodar.retomar("player-v1", tmp_path / "treinos") == pasta

    assert chamadas == [(str(pasta / "weights" / "last.pt"), {"resume": True})]
    m = _manifest(pasta)
    assert m["status"] == "concluido" and len(m["retomadas"]) == 1
    assert m["parametros"]["imgsz"] == 1280  # preservado do treino original
    with pytest.raises(EntradaInvalida, match="já terminou"):
        rodar.retomar("player-v1", tmp_path / "treinos")


@pytest.mark.model
def test_treino_relampago(tmp_path):
    ds = dataset_preparado(tmp_path / "ds", quantidades=(2, 1, 1))

    pasta = rodar.treinar(ds, "relampago", tmp_path / "treinos", parametros={
        "epochs": 1, "imgsz": 64, "batch": 2, "workers": 0, "amp": False, "plots": False})

    best = pasta / "weights" / "best.pt"
    assert best.exists()
    m = _manifest(pasta)
    assert m["status"] == "concluido"
    assert m["best"]["sha256"] == hashlib.sha256(best.read_bytes()).hexdigest()
    assert m["versoes"]["ultralytics"] != "não instalado"
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_treino_rodar.py -v`
Expected: erro de coleta `ImportError: cannot import name 'rodar'`.

- [ ] **Step 3: Implementar `treinos_dir` e `rodar.py`**

Em `pipeline/nfl_vision/paths.py`, acrescente ao fim:

```python
def treinos_dir() -> Path:
    return dados_dir() / "treinos"
```

Crie `pipeline/nfl_vision/treino/rodar.py`:

```python
"""Ajuste fino do YOLO11m para `player` com o Ultralytics e o manifest do treino."""

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from nfl_vision.eval.datasets import IMAGENS, nomes_das_classes
from nfl_vision.pipeline import _versao
from nfl_vision.runner import gravar_json

MODELO_INICIAL = "yolo11m.pt"
PARAMETROS = {
    "imgsz": 1280,
    "epochs": 100,
    "patience": 20,
    "batch": -1,           # automático (~60% da VRAM)
    "amp": True,
    "single_cls": True,
    "close_mosaic": 10,
    "workers": 2,          # estabilidade dos dataloaders no Windows
    "seed": 0,
    "deterministic": True,
}
SPLITS = ("train", "valid", "test")
_NOME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class EntradaInvalida(Exception):
    """Erro de entrada (dataset, nome, retomada): vira mensagem sem traceback na CLI."""


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _yolo(pesos: str):
    from ultralytics import YOLO

    return YOLO(pesos)


def _gpu() -> str | None:
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def _pasta(nome: str, destino: Path) -> Path:
    if not _NOME.fullmatch(nome):
        raise EntradaInvalida(
            f"nome de treino inválido: {nome!r} (use letras, números, '.', '_' ou '-')")
    return (destino / nome).resolve()


def validar_dataset(dataset: Path) -> Path:
    """data.yaml de um dataset de uma classe com imagens em train, valid e test."""
    try:
        nomes = nomes_das_classes(dataset)
    except (FileNotFoundError, ValueError) as exc:
        raise EntradaInvalida(str(exc)) from exc
    if [str(n) for n in nomes] != ["player"]:
        raise EntradaInvalida(
            f"o dataset deve ter só a classe player (names: {nomes}); "
            "gere-o com nfl-vision treino preparar")
    for split in SPLITS:
        pasta = dataset / split / "images"
        if not pasta.is_dir() or not any(p.suffix.lower() in IMAGENS for p in pasta.iterdir()):
            raise EntradaInvalida(f"split '{split}' sem imagens em {pasta}")
    return dataset / "data.yaml"


def _metricas(modelo) -> dict:
    brutas = getattr(getattr(modelo, "trainer", None), "metrics", None) or {}
    saida = {}
    for chave, valor in brutas.items():
        try:
            saida[chave] = round(float(valor), 5)
        except (TypeError, ValueError):
            continue
    return saida


def _executar(pasta: Path, manifest: dict, treino: Callable[[], object]) -> None:
    """Roda o treino registrando status, duração acumulada, best.pt e métricas no manifest."""
    arquivo = pasta / "manifest.json"
    manifest.update(status="em_andamento", mensagem=None)
    gravar_json(arquivo, manifest)
    t0 = time.perf_counter()

    def somar_duracao() -> None:
        manifest["duracao_s"] = round(manifest.get("duracao_s", 0.0) + time.perf_counter() - t0, 1)

    try:
        modelo = treino()
        best = pasta / "weights" / "best.pt"
        if not best.exists():
            raise RuntimeError(f"o Ultralytics terminou sem gerar {best}")
    except BaseException as exc:
        somar_duracao()
        manifest["status"] = "interrompido" if isinstance(exc, KeyboardInterrupt) else "erro"
        manifest["mensagem"] = f"{type(exc).__name__}: {exc}"
        gravar_json(arquivo, manifest)
        raise
    somar_duracao()
    manifest.update(status="concluido", best={"caminho": str(best), "sha256": _sha256(best)},
                    metricas=_metricas(modelo))
    gravar_json(arquivo, manifest)


def treinar(dataset: Path, nome: str, destino: Path, parametros: dict | None = None,
            modelo: str = MODELO_INICIAL) -> Path:
    """Treino novo em `<destino>/<nome>`; devolve a pasta do treino."""
    pasta = _pasta(nome, destino)
    if pasta.exists():
        raise EntradaInvalida(
            f"o treino '{nome}' já existe em {pasta}; use --retomar (se houver weights/last.pt), "
            "apague a pasta ou escolha outro --nome")
    data_yaml = validar_dataset(dataset).resolve()
    params = {**PARAMETROS, **(parametros or {})}
    manifest_dataset = dataset / "manifest.json"
    pasta.mkdir(parents=True)
    manifest = {
        "nome": nome,
        "dataset": str(dataset.resolve()),
        "data_yaml_sha256": _sha256(data_yaml),
        "dataset_manifest_sha256": _sha256(manifest_dataset) if manifest_dataset.exists() else None,
        "modelo_inicial": modelo,
        "parametros": params,
        "versoes": {p: _versao(p) for p in ("nfl-vision", "ultralytics", "torch")},
        "gpu": _gpu(),
        "inicio": _agora(),
        "duracao_s": 0.0,
        "retomadas": [],
    }

    def treino():
        m = _yolo(modelo)
        # project absoluto (senão vai para runs/detect/); exist_ok=True porque a pasta já foi
        # criada acima, depois de recusar nome repetido (exist_ok=False criaria "<nome>2").
        m.train(data=str(data_yaml), project=str(pasta.parent), name=pasta.name, exist_ok=True,
                **params)
        return m

    _executar(pasta, manifest, treino)
    return pasta


def retomar(nome: str, destino: Path) -> Path:
    """Continua o treino `<destino>/<nome>` a partir de weights/last.pt."""
    pasta = _pasta(nome, destino)
    last = pasta / "weights" / "last.pt"
    if not last.exists():
        raise EntradaInvalida(f"não há {last} para retomar; confira o --nome ou comece um treino novo")
    arquivo = pasta / "manifest.json"
    manifest = json.loads(arquivo.read_text("utf-8")) if arquivo.exists() else {"nome": nome}
    if manifest.get("status") == "concluido":
        raise EntradaInvalida(f"o treino '{nome}' já terminou; não há o que retomar")
    manifest.setdefault("retomadas", []).append(_agora())

    def treino():
        m = _yolo(str(last))
        m.train(resume=True)  # dataset, parâmetros e pasta vêm do checkpoint
        return m

    _executar(pasta, manifest, treino)
    return pasta
```

- [ ] **Step 4: Rodar os testes do treino**

Run: `uv run pytest tests/test_treino_rodar.py -v`
Expected: todos PASS (o `test_treino_relampago` aparece como deselected).

- [ ] **Step 5: Escrever os testes que falham da CLI**

Em `pipeline/tests/test_cli_treino.py`, troque o import de `treino_sintetico` por:

```python
from treino_sintetico import dataset_base, dataset_externo, dataset_preparado, instalar_yolo_falso
```

troque `test_treino_help_lista_comandos` por:

```python
def test_treino_help_lista_comandos():
    r = runner.invoke(app, ["treino", "--help"])
    assert r.exit_code == 0, r.output
    assert "preparar" in r.output and "rodar" in r.output
```

e acrescente ao fim:

```python
def test_rodar_treina_e_mostra_pesos(dados, tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    chamadas = instalar_yolo_falso(monkeypatch)
    ds = dataset_preparado(tmp_path / "ds")

    r = runner.invoke(app, ["treino", "rodar", "--dataset", str(ds), "--nome", "player-v1",
                            "--epocas", "5", "--imgsz", "640"])

    assert r.exit_code == 0, r.output
    assert chamadas[0][1]["epochs"] == 5 and chamadas[0][1]["imgsz"] == 640
    assert (paths.treinos_dir() / "player-v1" / "weights" / "best.pt").exists()
    assert "best.pt" in r.output


def test_rodar_interrompido_e_retomado(dados, tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    ds = dataset_preparado(tmp_path / "ds")
    instalar_yolo_falso(monkeypatch, falhar=KeyboardInterrupt())
    r = runner.invoke(app, ["treino", "rodar", "--dataset", str(ds), "--nome", "player-v1"])
    assert r.exit_code == 130, r.output
    assert "--retomar" in r.output

    chamadas = instalar_yolo_falso(monkeypatch)
    r = runner.invoke(app, ["treino", "rodar", "--nome", "player-v1", "--retomar"])
    assert r.exit_code == 0, r.output
    assert chamadas[-1][1] == {"resume": True}


def test_rodar_falha_no_treino_sai_com_1(dados, tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    instalar_yolo_falso(monkeypatch, falhar=RuntimeError("CUDA out of memory"))
    ds = dataset_preparado(tmp_path / "ds")

    r = runner.invoke(app, ["treino", "rodar", "--dataset", str(ds), "--nome", "player-v1"])

    assert r.exit_code == 1, r.output
    assert "CUDA out of memory" in r.output and "--retomar" in r.output


def test_rodar_erros_de_entrada(dados, tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    instalar_yolo_falso(monkeypatch)
    ds = dataset_preparado(tmp_path / "ds")
    sem_teste = dataset_preparado(tmp_path / "sem_teste")
    for p in (sem_teste / "test" / "images").iterdir():
        p.unlink()
    casos = [
        (["--nome", "x"], "--dataset"),
        (["--dataset", str(tmp_path / "nada"), "--nome", "x"], "data.yaml"),
        (["--dataset", str(sem_teste), "--nome", "x"], "split 'test' sem imagens"),
        (["--nome", "x", "--retomar"], "last.pt"),
        (["--nome", "x", "--retomar", "--dataset", str(ds)], "--retomar usa os parâmetros"),
    ]
    for args, trecho in casos:
        r = runner.invoke(app, ["treino", "rodar", *args])
        assert r.exit_code == 2, (args, r.output)
        assert trecho in r.output, (args, r.output)
```

- [ ] **Step 6: Rodar e ver falhar**

Run: `uv run pytest tests/test_cli_treino.py -v`
Expected: os testes novos FAIL com `No such command 'rodar'`.

- [ ] **Step 7: Implementar o comando `treino rodar`**

Ao fim de `pipeline/nfl_vision/cli.py`, acrescente:

```python
@treino_app.command("rodar")
def treino_rodar(
    nome: str = typer.Option(..., "--nome", help="Nome do treino (pasta em data/treinos/)"),
    dataset: Optional[Path] = typer.Option(
        None, "--dataset", help="Pasta gerada por `treino preparar`"),
    epocas: Optional[int] = typer.Option(
        None, "--epocas", min=1, help="Padrão: 100 (com parada antecipada, patience 20)"),
    imgsz: Optional[int] = typer.Option(None, "--imgsz", min=32, help="Padrão: 1280"),
    retomar: bool = typer.Option(
        False, "--retomar", help="Continua do weights/last.pt do treino --nome"),
) -> None:
    """Ajuste fino do YOLO11m para player (Ultralytics, GPU local)."""
    from nfl_vision.treino import rodar

    if retomar and (dataset is not None or epocas is not None or imgsz is not None):
        raise typer.BadParameter(
            "--retomar usa os parâmetros do treino original; não informe --dataset, --epocas nem --imgsz",
            param_hint="--retomar")
    if not retomar and dataset is None:
        raise typer.BadParameter("informe --dataset (ou --retomar para continuar um treino)",
                                 param_hint="--dataset")
    destino = paths.treinos_dir()
    try:
        if retomar:
            pasta = rodar.retomar(nome, destino)
        else:
            ajustes = {k: v for k, v in (("epochs", epocas), ("imgsz", imgsz)) if v is not None}
            pasta = rodar.treinar(dataset, nome, destino, parametros=ajustes)
    except rodar.EntradaInvalida as exc:
        raise typer.BadParameter(str(exc)) from exc
    except KeyboardInterrupt:
        console.print("[yellow]interrompido; para continuar: "
                      f"nfl-vision treino rodar --nome {escape(nome)} --retomar[/yellow]")
        raise typer.Exit(130)
    except Exception as exc:  # erro do Ultralytics/CUDA: mensagem curta; o manifest guarda o status
        console.print(f"[red]treino falhou: {escape(type(exc).__name__)}: {escape(str(exc))}[/red]")
        console.print("Se houver weights/last.pt, continue com: "
                      f"nfl-vision treino rodar --nome {escape(nome)} --retomar")
        raise typer.Exit(1)
    console.print(f"Pesos: {escape(str(pasta / 'weights' / 'best.pt'))}")
    console.print(f"Manifest: {escape(str(pasta / 'manifest.json'))}")
```

- [ ] **Step 8: Rodar os testes da task, a suíte rápida e o treino-relâmpago**

Run: `uv run pytest tests/test_cli_treino.py tests/test_treino_rodar.py -v`
Expected: todos PASS.

Run: `uv run pytest`
Expected: todos PASS.

Run: `uv run pytest -m model tests/test_treino_rodar.py -v`
Expected: `test_treino_relampago` PASS (1 época em imgsz 64, alguns segundos na GPU; usa `pipeline/yolo11m.pt`).

- [ ] **Step 9: Commit**

```bash
git add nfl_vision/paths.py nfl_vision/treino/rodar.py nfl_vision/cli.py tests/test_treino_rodar.py tests/test_cli_treino.py
git commit -m "feat: comando treino rodar com manifest e retomada"
```

---

### Task 5: `eval detect --pesos` e `analyze --detector`

**Files:**
- Modify: `pipeline/nfl_vision/eval/preditores.py` (nova `rotulo_pesos`; `nome` de `PreditorNosso` e `PreditorYoloBruto`)
- Modify: `pipeline/nfl_vision/cli.py` (`eval_detect_cmd`, `analyze`, `_analisar_ou_reprocessar`)
- Test: `pipeline/tests/test_avaliacao.py`, `pipeline/tests/test_cli.py`

- [ ] **Step 1: Escrever os testes que falham dos nomes dos preditores**

Acrescente ao fim de `pipeline/tests/test_avaliacao.py`:

```python
def test_rotulo_dos_pesos():
    assert preditores.rotulo_pesos("yolo11m.pt") == "yolo11m"
    assert preditores.rotulo_pesos("../data/treinos/player-v1/weights/best.pt") == "player-v1"
    assert preditores.rotulo_pesos("/x/treinos/player-v1/weights/last.pt") == "player-v1/last"
    assert preditores.rotulo_pesos("pesos/meu-detector.pt") == "meu-detector"


def test_nomes_dos_preditores_indicam_os_pesos():
    padrao = Config()
    assert preditores.PreditorNosso(padrao).nome == "nosso (yolo11m + filtros + árbitro)"
    assert preditores.PreditorYoloBruto(padrao).nome == "yolo11m bruto (COCO, pessoa)"
    ajustado = Config(detector_pesos="data/treinos/player-v1/weights/best.pt")
    assert preditores.PreditorNosso(ajustado).nome == "nosso (player-v1 + filtros + árbitro)"
    assert preditores.PreditorYoloBruto(ajustado).nome == "yolo bruto (player-v1)"
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_avaliacao.py -k "rotulo or nomes_dos" -v`
Expected: FAIL com `AttributeError: module 'nfl_vision.eval.preditores' has no attribute 'rotulo_pesos'`.

- [ ] **Step 3: Implementar**

Em `pipeline/nfl_vision/eval/preditores.py`, logo depois de `class Preditor(Protocol): ...` (antes de `class PreditorNosso`), acrescente:

```python
def rotulo_pesos(pesos: str) -> str:
    """Nome curto dos pesos para o nome do preditor: `.../player-v1/weights/best.pt` vira `player-v1`."""
    caminho = Path(pesos)
    if caminho.parent.name == "weights" and caminho.stem in ("best", "last"):
        treino = caminho.parent.parent.name
        return treino if caminho.stem == "best" else f"{treino}/last"
    return caminho.stem
```

Substitua o cabeçalho de `PreditorNosso` (da linha `class PreditorNosso:` até o fim do `__init__`) por:

```python
class PreditorNosso:
    pos_processamento = "filtros de campo + remoção de árbitro"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.nome = f"nosso ({rotulo_pesos(cfg.detector_pesos)} + filtros + árbitro)"
```

E o cabeçalho de `PreditorYoloBruto` (da linha `class PreditorYoloBruto:` até o fim do `__init__`) por:

```python
class PreditorYoloBruto:
    """O mesmo detector do pipeline, sem filtros de campo nem remoção de árbitro."""

    pos_processamento = "nenhum"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        padrao = cfg.detector_pesos == Config().detector_pesos
        self.nome = ("yolo11m bruto (COCO, pessoa)" if padrao
                     else f"yolo bruto ({rotulo_pesos(cfg.detector_pesos)})")
```

(os métodos `prever` ficam como estão).

- [ ] **Step 4: Rodar os testes de avaliação**

Run: `uv run pytest tests/test_avaliacao.py -v`
Expected: todos PASS.

- [ ] **Step 5: Escrever os testes que falham da CLI**

Acrescente ao fim de `pipeline/tests/test_cli.py`:

```python
def test_eval_detect_com_pesos_ajustados(dados, tmp_path, monkeypatch):
    import hashlib

    monkeypatch.setenv("COLUMNS", "300")
    _detector_falso(monkeypatch)
    pesos = tmp_path / "treinos" / "player-v1" / "weights" / "best.pt"
    pesos.parent.mkdir(parents=True)
    pesos.write_bytes(b"ajustado")
    ds = _dataset_deteccao(tmp_path / "ds")

    r = runner.invoke(app, ["eval", "detect", "--dataset", str(ds), "--pesos", str(pesos),
                            "--benchmark", "yolo-bruto"])

    assert r.exit_code == 0, r.output
    salvo = _ler_avaliacao("detect")
    assert salvo["config"]["detector_pesos"] == str(pesos.resolve())
    assert salvo["pesos_sha256"] == hashlib.sha256(b"ajustado").hexdigest()
    assert [x["preditor"] for x in salvo["resultados"]] == [
        "nosso (player-v1 + filtros + árbitro)", "yolo bruto (player-v1)"]


def test_eval_detect_pesos_inexistentes(dados, tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    ds = _dataset_deteccao(tmp_path / "ds")
    r = runner.invoke(app, ["eval", "detect", "--dataset", str(ds), "--pesos", str(tmp_path / "nao.pt")])
    assert r.exit_code == 2, r.output
    assert "pesos não encontrados" in r.output


def test_analyze_com_detector_grava_os_pesos_na_config(dados, foto_sintetica, modelos_falsos, tmp_path):
    import json

    pesos = tmp_path / "best.pt"
    pesos.write_bytes(b"ajustado")

    r = runner.invoke(app, ["analyze", str(foto_sintetica[0]), *BASE, "--detector", str(pesos)])

    assert r.exit_code == 0, r.output
    run_dir = next(paths.runs_dir().iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    assert manifest["config"]["detector_pesos"] == str(pesos.resolve())


def test_analyze_detector_erros(dados, foto_sintetica, tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    casos = [
        (["analyze", str(foto_sintetica[0]), *BASE, "--detector", str(tmp_path / "nao.pt")],
         "pesos não encontrados"),
        (["analyze", "--run", "x", "--from", "jersey", "--detector", str(foto_sintetica[0])],
         "--detector só vale para análise nova"),
    ]
    for args, trecho in casos:
        r = runner.invoke(app, args)
        assert r.exit_code == 2, (args, r.output)
        assert trecho in r.output, (args, r.output)
```

- [ ] **Step 6: Rodar e ver falhar**

Run: `uv run pytest tests/test_cli.py -k "pesos or detector" -v`
Expected: FAIL com `No such option: --pesos` / `No such option: --detector` (saída 2 sem o trecho esperado, ou `KeyError` no JSON).

- [ ] **Step 7: Implementar `--pesos` em `eval detect`**

Em `pipeline/nfl_vision/cli.py`, na assinatura de `eval_detect_cmd`, acrescente o parâmetro depois de `conf`:

```python
    pesos: Optional[Path] = typer.Option(
        None, "--pesos",
        help="Pesos do detector (.pt) para os preditores nosso e yolo-bruto; padrão: yolo11m.pt COCO"),
```

e substitua a linha

```python
    cfg = Config(detector_conf=conf)
```

por:

```python
    if pesos is not None and not pesos.is_file():
        raise typer.BadParameter(f"pesos não encontrados: {pesos}", param_hint="--pesos")
    cfg = (Config(detector_conf=conf) if pesos is None
           else Config(detector_conf=conf, detector_pesos=str(pesos.resolve())))
```

e, logo depois do bloco `if "roboflow-nfl" in benchmark: cabecalho["modelo_roboflow"] = modelo_roboflow`, acrescente:

```python
    if pesos is not None:
        from nfl_vision.stages.detect import sha256_pesos

        cabecalho["pesos_sha256"] = sha256_pesos(cfg.detector_pesos)
```

- [ ] **Step 8: Implementar `--detector` em `analyze`**

Em `pipeline/nfl_vision/cli.py`, substitua a função `analyze` inteira por:

```python
@app.command()
def analyze(
    foto: Optional[Path] = typer.Argument(None, help="Foto JPG ou PNG"),
    times: Tuple[str, str] = typer.Option((None, None), "--times", help="Siglas dos dois times"),
    temporada: Optional[int] = typer.Option(None, "--temporada"),
    semana: Optional[int] = typer.Option(None, "--semana"),
    run: Optional[str] = typer.Option(None, "--run", help="Reprocessar uma análise existente"),
    a_partir_de: Optional[str] = typer.Option(None, "--from", help="Etapa inicial do reprocessamento"),
    detector: Optional[Path] = typer.Option(
        None, "--detector", help="Pesos do detector (.pt) para esta análise; padrão: yolo11m.pt COCO"),
) -> None:
    """Analisa uma foto ou reprocessa uma análise a partir de uma etapa."""
    with _tratando_falha_de_etapa(run):
        run_dir, analise = _analisar_ou_reprocessar(
            foto, times, temporada, semana, run, a_partir_de, detector)
    _imprimir(analise, run_dir)
```

Em `_analisar_ou_reprocessar`, troque a assinatura por:

```python
def _analisar_ou_reprocessar(foto, times, temporada, semana, run, a_partir_de, detector=None):
```

acrescente, logo depois do teste `if run and a_partir_de is None: ...`:

```python
    if run and detector is not None:
        raise typer.BadParameter(
            "--detector só vale para análise nova; o reprocessamento usa a config gravada",
            param_hint="--detector")
```

e substitua as duas últimas linhas da função

```python
    contexto = _validar_contexto(times, temporada, semana)
    return pipeline.analisar(foto, contexto)
```

por:

```python
    if detector is not None and not detector.is_file():
        raise typer.BadParameter(f"pesos não encontrados: {detector}", param_hint="--detector")
    contexto = _validar_contexto(times, temporada, semana)
    # caminho absoluto: o reprocessamento (--run) pode rodar de outro diretório
    config = Config(detector_pesos=str(detector.resolve())) if detector is not None else None
    return pipeline.analisar(foto, contexto, config)
```

- [ ] **Step 9: Rodar os testes da task e a suíte rápida**

Run: `uv run pytest tests/test_cli.py tests/test_avaliacao.py -v`
Expected: todos PASS.

Run: `uv run pytest`
Expected: todos PASS.

- [ ] **Step 10: Commit**

```bash
git add nfl_vision/eval/preditores.py nfl_vision/cli.py tests/test_avaliacao.py tests/test_cli.py
git commit -m "feat: eval detect --pesos e analyze --detector"
```

---

### Task 6: README e ajustes na spec

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-10-04-detector-ajustado-design.md`

- [ ] **Step 1: README**

Em `README.md`, insira esta seção entre a seção `## Avaliação` (depois do parágrafo que termina em "O Roboflow exporta o split de validação como `valid`.") e `## Testes`:

```markdown
## Detector ajustado (opcional)

O padrão é o YOLO11m do COCO (`yolo11m.pt`). Para ajustá-lo à classe `player` na GPU local:

    uv run nfl-vision treino preparar --so-triagem
    # veja ../data/datasets/treino-player-v1/triagem/*.jpg e decida cada fonte externa
    uv run nfl-vision treino preparar --aprovar "<fonte>:<motivo>" --rejeitar "<fonte>:<motivo>"
    uv run nfl-vision treino rodar --dataset ../data/datasets/treino-player-v1 --nome player-v1
    uv run nfl-vision treino rodar --nome player-v1 --retomar     # se o treino parar no meio

O dataset junta o fork `elyas-carvalho/nfl-player-model-ymsui` v1 e as fontes externas aprovadas (`pitchcamera`, `fhtw`, `evzn`; ver `nfl_vision/treino/fontes.py`). O split `test` é só o jogo CIN × CLE, que nunca entra no treino. Os pesos ficam em `../data/treinos/<nome>/weights/best.pt`, fora do git, com um `manifest.json` (dataset, parâmetros, versões, GPU, sha256).

    uv run nfl-vision eval detect --dataset ../data/datasets/treino-player-v1 --split test --pesos ../data/treinos/player-v1/weights/best.pt --benchmark yolo-bruto
    uv run nfl-vision analyze foto.jpg --times CIN CLE --temporada 2025 --semana 1 --detector ../data/treinos/player-v1/weights/best.pt

`--pesos` troca os pesos dos preditores `nosso` e `yolo-bruto`; `--detector` vale para uma análise nova e fica gravado no manifest dela.
```

E troque a última linha do README

```markdown
Datasets de avaliação: `nflplayerdetection-mjrl1/nfl-player-model` e `taiseis-workspace/jersey-number-ijbaq` (Roboflow Universe, CC BY 4.0).
```

por:

```markdown
Datasets de avaliação: `nflplayerdetection-mjrl1/nfl-player-model` e `taiseis-workspace/jersey-number-ijbaq` (Roboflow Universe, CC BY 4.0).
Datasets de treino do detector: os acima e as fontes externas aprovadas na triagem (Roboflow Universe; licença de cada uma no `README.roboflow.txt` do download). Pesos treinados não são publicados.
```

- [ ] **Step 2: Spec**

Em `docs/superpowers/specs/2026-10-04-detector-ajustado-design.md`:

Troque

```markdown
A lista de fontes aprovadas é um parâmetro do comando, para a decisão ser explícita e reprodutível.
```

por

```markdown
A decisão é parâmetro do comando, para ser explícita e reprodutível: `treino preparar --so-triagem` baixa as fontes e gera os painéis; depois `treino preparar --aprovar <fonte>:<motivo> --rejeitar <fonte>:<motivo>` monta o dataset, e toda fonte externa precisa ser decidida. No painel, caixas da classe mapeada para `player` aparecem em verde e as demais em cinza com o nome da classe.
```

Troque

```markdown
- Linhas de polígono viram caixa (mín./máx. das coordenadas), como em `eval/datasets.py`.
```

por

```markdown
- Linhas de polígono viram caixa (mín./máx. das coordenadas), com a mesma função de `eval/datasets.py` (`ler_rotulos`). Caixas são cortadas aos limites da imagem; as degeneradas são descartadas e contadas.
```

Troque

```markdown
- `valid`: o clipe `tb_atl_wk1_penix_pass_all22` inteiro mais 15% de cada externo aprovado (sorteio com semente fixa por imagem; esses datasets não têm identificador de clipe).
```

por

```markdown
- `valid`: o clipe `tb_atl_wk1_penix_pass_all22` inteiro mais 15% de cada externo aprovado (sorteio com semente fixa por imagem original: as cópias aumentadas que o Roboflow exporta como `<original>.rf.<hash>` ficam no mesmo split; esses datasets não têm identificador de clipe).
```

Troque

```markdown
`data/datasets/treino-player-v1/` no formato YOLO (`train|valid|test/images|labels`, `data.yaml` com `names: ['player']`)
```

por

```markdown
`data/datasets/treino-player-v1/` no formato YOLO (`train|valid|test/images|labels`, `data.yaml` com `path` absoluto e `names: ['player']`)
```

Depois da frase `Tempo estimado: 30–60 min.` (fim do parágrafo do comando, §3), acrescente:

```markdown
O comando recusa um `--nome` que já existe e passa `project`/`name` absolutos com `exist_ok=True` ao Ultralytics (com `exist_ok=False` ele criaria `player-v12` em silêncio). O manifest do treino tem `status` (`em_andamento`, `concluido`, `erro`, `interrompido`), duração acumulada entre retomadas e a lista de retomadas; `--retomar` num treino concluído é recusado.
```

Depois do parágrafo `Ressalva registrada no resultado: ...` (§4), acrescente:

```markdown
O split `test` preparado só tem a classe `player`, então a coluna de árbitros da avaliação fica vazia; árbitro marcado como jogador aparece como falso positivo no mAP. A comparação sai em duas execuções do `eval detect` no mesmo split: uma com `--benchmark yolo-bruto --benchmark rfdetr` (pesos COCO) e outra com `--pesos <best.pt> --benchmark yolo-bruto`.
```

- [ ] **Step 3: Conferir e commitar**

Run: `uv run pytest`
Expected: todos PASS (nada de código mudou).

```bash
git add ../README.md ../docs/superpowers/specs/2026-10-04-detector-ajustado-design.md
git commit -m "docs: README e spec do detector ajustado"
```

---

## Execução (controlador)

Passos com dados e GPU reais, feitos pelo controlador (não por subagentes) depois das Tasks 1–6. Rodar de `pipeline/` com `export PATH="/c/Users/User/.local/bin:$PATH"`; `.env` na raiz com `ROBOFLOW_API_KEY`.

- [ ] **E1: Pré-voo**

Run: `uv run pytest` e `uv run pytest -m model`
Expected: tudo PASS (inclui `test_treino_relampago` e `test_gpu_blackwell_disponivel`).

- [ ] **E2: Triagem (download + painéis)**

Run: `uv run nfl-vision treino preparar --so-triagem`
Expected: tabela com `pitchcamera`, `fhtw` e `evzn`, o número de imagens de cada e o caminho do painel em `../data/datasets/treino-player-v1/triagem/`.

Se der `MapeamentoInvalido` (classe ausente) ou `FonteIndisponivel` (versão que não existe): corrija a versão ou o nome da classe em `nfl_vision/treino/fontes.py` conforme as classes listadas no erro (ou o `data.yaml` baixado), atualize `test_fontes_declaradas` em `tests/test_treino_fontes.py`, rode `uv run pytest`, commite (`fix: ajusta a fonte <nome> ao data.yaml publicado`) e repita o E2. Se uma fonte não puder ser baixada de jeito nenhum, remova-a de `EXTERNAS` (mesmo commit) e registre o motivo no documento de avaliação.

- [ ] **E3: Decidir cada fonte olhando os painéis**

Abra com o Read tool `../data/datasets/treino-player-v1/triagem/pitchcamera.jpg`, `fhtw.jpg` e `evzn.jpg`. Aprovar só se:

1. é futebol americano (capacetes, ombreiras, linhas de jarda), não soccer;
2. as caixas verdes cobrem os jogadores de corpo inteiro, uma por jogador, sem muitos jogadores sem caixa;
3. árbitros (listras) não estão em verde.

Escreva um motivo curto por fonte (ex.: `futebol americano, broadcast, caixas coerentes` ou `soccer`).

- [ ] **E4: Montar o dataset**

Run (com as decisões do E3):

```bash
uv run nfl-vision treino preparar \
  --aprovar "<fonte>:<motivo>" \
  --rejeitar "<fonte>:<motivo>" ...
```

Expected: tabela por split e fonte. Para a base: `test` = 102 imagens (só `cin_cle_*`), `valid` = 56 (`tb_atl_wk1_penix_pass_all22`), `train` = 180 (menos as imagens sem `player`, ver `sem_player` no manifest). Externas aprovadas só em `train`/`valid`. Confira `../data/datasets/treino-player-v1/manifest.json` (fontes com `aprovada`/`motivo`, `descartadas`).

- [ ] **E5: Treinar (longo, em background)**

Run em background (Bash `run_in_background`):

```bash
uv run nfl-vision treino rodar --dataset ../data/datasets/treino-player-v1 --nome player-v1
```

Acompanhe com `tail -n 3 ../data/treinos/player-v1/results.csv` (uma linha por época). Estimativa: 30–60 min, mais se as externas forem grandes. Se parar no meio: `uv run nfl-vision treino rodar --nome player-v1 --retomar`. Se faltar memória logo no início, apague `../data/treinos/player-v1` e rode de novo com `--imgsz 960` (e registre isso no documento).

Expected no fim: `Pesos: ...\data\treinos\player-v1\weights\best.pt`; `manifest.json` com `status: concluido`, `best.sha256`, `gpu`, `duracao_s`, `metricas`.

- [ ] **E6: Avaliar no jogo separado (5 preditores, limiar 0,25)**

```bash
uv run nfl-vision eval detect --dataset ../data/datasets/treino-player-v1 --split test --benchmark yolo-bruto --benchmark rfdetr
uv run nfl-vision eval detect --dataset ../data/datasets/treino-player-v1 --split test --pesos ../data/treinos/player-v1/weights/best.pt --benchmark yolo-bruto
```

Expected: a primeira traz `nosso (yolo11m + filtros + árbitro)`, `yolo11m bruto (COCO, pessoa)` e `rf-detr base (COCO, pessoa)`; a segunda, `nosso (player-v1 + filtros + árbitro)` e `yolo bruto (player-v1)`. Anote os dois JSONs `../data/avaliacoes/detect-*.json`.

- [ ] **E7: `analyze --detector` numa foto real**

As fotos em `../fotos/` são do jogo CIN × CLE (semana 1 de 2025), que não entra no treino:

```bash
uv run nfl-vision analyze ../fotos/cin_cle_2025_s1_broadcast_2.jpg --times CIN CLE --temporada 2025 --semana 1
uv run nfl-vision analyze ../fotos/cin_cle_2025_s1_broadcast_2.jpg --times CIN CLE --temporada 2025 --semana 1 --detector ../data/treinos/player-v1/weights/best.pt
```

Confira no manifest da segunda análise que `config.detector_pesos` aponta para o `best.pt` e que `versoes.detector_pesos_sha256` é igual a `best.sha256` do manifest do treino. Compare as duas `anotada.png` (jogadores achados/perdidos, árbitro, arquibancada).

- [ ] **E8: Documento de avaliação**

Crie `docs/avaliacao/2026-10-detector-ajustado.md` no estilo de `docs/avaliacao/2026-10-linha-de-base.md`:

1. Cabeçalho: data, commit, hardware, caminhos dos JSONs do E6 e do manifest do treino.
2. **Dados:** fontes com a decisão e o motivo da triagem; contagens por split e fonte; descartes; ressalva de que o teste é uma jogada de um jogo (102 imagens, 2 câmeras).
3. **Treino:** parâmetros efetivos, épocas até a parada, duração, GPU, sha256 do `best.pt`.
4. **Resultado:** tabela com os 5 preditores (preditor, pós-processamento, mAP@0.5) e a pergunta que cada um responde (spec §4). Nota: coluna de árbitros vazia porque o teste preparado só tem `player`.
5. **Foto real:** o que mudou entre as duas `anotada.png` do E7.
6. **Decisão:** usar ou não o ajustado como recomendado (meta mAP@0.5 ≥ 0,85); se não atingir, o próximo passo (RF-DETR ajustado ou mais dados).

Run: `uv run pytest`
Expected: tudo PASS.

```bash
git add ../docs/avaliacao/2026-10-detector-ajustado.md
git commit -m "docs: avaliação do detector ajustado"
```
