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
