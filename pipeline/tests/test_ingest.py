import hashlib

import pytest
from PIL import Image

from nfl_vision.config import Config
from nfl_vision.runner import Runner
from nfl_vision.schemas import Contexto
from nfl_vision.stages import ingest
from nfl_vision.stages.ingest import FormatoNaoSuportado, carregar_imagem, validar_formato


def test_aplica_orientacao_exif(tmp_path):
    caminho = tmp_path / "celular.jpg"
    # metade esquerda vermelha, metade direita azul: a cor no canto
    # superior esquerdo após a correção de orientação denuncia se o
    # sentido do giro foi aplicado corretamente.
    img = Image.new("RGB", (40, 20))
    for x in range(40):
        for y in range(20):
            img.putpixel((x, y), (255, 0, 0) if x < 20 else (0, 0, 255))
    exif = Image.Exif()
    exif[0x0112] = 6  # girar 90° no sentido horário ao exibir
    img.save(caminho, exif=exif)

    saida = carregar_imagem(caminho)

    assert saida.shape == (40, 20, 3)  # altura 40, largura 20 após girar
    # orientação 6 -> ROTATE_270 (90° horário); a coluna esquerda original
    # (vermelha) vira a linha do topo da imagem corrigida.
    assert tuple(saida[0, 0]) == pytest.approx((0, 0, 254), abs=3)  # BGR


def test_png_aceito(tmp_path):
    caminho = tmp_path / "a.PNG"
    Image.new("RGB", (10, 8)).save(caminho)
    assert carregar_imagem(caminho).shape == (8, 10, 3)


@pytest.mark.parametrize("nome", ["video.mp4", "foto.gif", "doc.pdf"])
def test_formatos_recusados(tmp_path, nome):
    with pytest.raises(FormatoNaoSuportado, match="use JPG ou PNG"):
        validar_formato(tmp_path / nome)


def test_executar_grava_caminho_relativo_e_metadados(tmp_path):
    foto = tmp_path / "foto.png"
    img = Image.new("RGB", (10, 8))
    for x in range(10):
        for y in range(8):
            img.putpixel((x, y), (255, 0, 0) if x < 5 else (0, 0, 255))
    img.save(foto)

    runner = Runner([], tmp_path / "runs")
    ctx = Contexto(temporada=2025, semana=11, times=("KC", "BUF"))
    run_dir = runner.nova_analise(foto, ctx, Config(), {})
    estado = runner.carregar_estado(run_dir)

    saida = ingest.executar(estado)

    assert saida.caminho == "input.png"
    assert saida.largura == 10
    assert saida.altura == 8
    assert saida.sha256 == hashlib.sha256((run_dir / "input.png").read_bytes()).hexdigest()
