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
