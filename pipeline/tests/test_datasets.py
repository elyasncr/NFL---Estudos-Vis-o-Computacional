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
