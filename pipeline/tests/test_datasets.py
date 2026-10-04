import pytest
from PIL import Image

from nfl_vision.eval.datasets import baixar, carregar_pastas, carregar_yolo, ler_rotulos


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
    # polígono (segmentação) vira a caixa que o envolve
    assert amostras[0].caixas["ball"] == [pytest.approx((10.0, 5.0, 50.0, 25.0))]
    assert amostras[1].caixas == {}


def _dataset_yolo(raiz, nomes="names: ['ball', 'player']\n", split="test"):
    if nomes is not None:
        (raiz / "data.yaml").write_text(nomes, encoding="utf-8")
    (raiz / split / "images").mkdir(parents=True)
    (raiz / split / "labels").mkdir(parents=True)
    return raiz / split


def test_carregar_yolo_sem_data_yaml(tmp_path):
    _dataset_yolo(tmp_path, nomes=None)
    with pytest.raises(FileNotFoundError, match="data.yaml"):
        carregar_yolo(tmp_path, "test")


def test_carregar_yolo_sem_names(tmp_path):
    _dataset_yolo(tmp_path, nomes="nc: 2\n")
    with pytest.raises(ValueError, match="names"):
        carregar_yolo(tmp_path, "test")


def test_carregar_yolo_split_ausente_lista_existentes(tmp_path):
    _dataset_yolo(tmp_path, split="valid")
    (tmp_path / "train" / "images").mkdir(parents=True)
    with pytest.raises(FileNotFoundError) as exc:
        carregar_yolo(tmp_path, "val")
    msg = str(exc.value)
    assert "valid" in msg and "train" in msg and "Roboflow" in msg


def test_carregar_yolo_classe_fora_de_names(tmp_path):
    pasta = _dataset_yolo(tmp_path)
    Image.new("RGB", (10, 10)).save(pasta / "images" / "a.jpg")
    (pasta / "labels" / "a.txt").write_text("1 0.5 0.5 0.2 0.2\n7 0.5 0.5 0.2 0.2\n")
    with pytest.raises(ValueError, match=r"a\.txt.*linha 2"):
        carregar_yolo(tmp_path, "test")


def test_carregar_yolo_respeita_orientacao_exif(tmp_path):
    pasta = _dataset_yolo(tmp_path)
    exif = Image.Exif()
    exif[0x0112] = 6  # girar 90°: 100x50 armazenada vira 50x100 exibida
    Image.new("RGB", (100, 50)).save(pasta / "images" / "a.jpg", exif=exif)
    (pasta / "labels" / "a.txt").write_text("1 0.5 0.5 1.0 1.0\n")

    caixa = carregar_yolo(tmp_path, "test")[0].caixas["player"][0]

    assert caixa == pytest.approx((0.0, 0.0, 50.0, 100.0))


def test_carregar_pastas(tmp_path):
    for rotulo in ("87", "9"):
        (tmp_path / "test" / rotulo).mkdir(parents=True)
        Image.new("RGB", (10, 10)).save(tmp_path / "test" / rotulo / "x.jpg")
    assert [r for _, r in carregar_pastas(tmp_path, "test")] == ["87", "9"]


def test_baixar_exige_chave(tmp_path, monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ROBOFLOW_API_KEY"):
        baixar("ws", "proj", 1, "yolov11", tmp_path)


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
