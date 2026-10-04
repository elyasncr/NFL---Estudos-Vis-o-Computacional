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
