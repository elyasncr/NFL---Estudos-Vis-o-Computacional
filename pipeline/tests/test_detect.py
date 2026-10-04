import cv2
import numpy as np
import pytest

from nfl_vision.config import Config
from nfl_vision.schemas import Deteccao
from nfl_vision.stages.detect import aplicar_filtros, regiao_do_campo
from sintetico import AZUL_BUF, BRANCO, CINZA, VERDE, VERMELHO_KC, campo, jogador


def _det(i, bbox):
    return Deteccao(det_id=i, bbox=bbox, confianca=0.9)


def _motivos(saida):
    return [(d.descartado, d.motivo_descarte) for d in saida]


def test_filtros_de_fora_de_campo_e_tamanho():
    img = campo()
    em_campo = jogador(img, 100, 300, VERMELHO_KC)
    na_arquibancada = jogador(img, 300, 0, VERMELHO_KC, h=100)  # pés em y=100, campo começa em 150
    pequeno = (500.0, 400.0, 510.0, 420.0)
    # jogador cortado pela borda inferior: a parte de baixo da caixa é camisa/perna
    na_borda = jogador(img, 600, 480, VERMELHO_KC, h=120)

    saida = aplicar_filtros(
        [_det(0, em_campo), _det(1, na_arquibancada), _det(2, pequeno), _det(3, na_borda)],
        img, Config(),
    )

    assert _motivos(saida) == [
        (False, None), (True, "fora_de_campo"), (True, "pequeno"), (False, None),
    ]


def test_jogadores_sobre_end_zone_pintada_sao_mantidos():
    img = campo()
    img[330:560, 350:760] = VERMELHO_KC          # end zone pintada dentro do campo
    for x in range(380, 740, 70):                 # letras brancas
        img[360:530, x:x + 40] = BRANCO
    sobre_letra = jogador(img, 385, 300, AZUL_BUF)      # pés em y=420, sobre o branco
    sobre_pintura = jogador(img, 600, 380, AZUL_BUF)    # pés em y=500

    saida = aplicar_filtros([_det(0, sobre_letra), _det(1, sobre_pintura)], img, Config())

    assert _motivos(saida) == [(False, None), (False, None)]


def test_jogador_sobre_linha_branca_e_sombra_e_mantido():
    img = campo()
    for y in range(200, 600, 60):                 # linhas de jarda
        img[y:y + 6] = BRANCO
    sobre_linha = jogador(img, 100, 140, VERMELHO_KC)   # pés em y=260, na linha
    img[440:470, 280:380] = (20, 40, 20)                # sombra escura
    sobre_sombra = jogador(img, 300, 330, VERMELHO_KC)  # pés em y=450

    saida = aplicar_filtros([_det(0, sobre_linha), _det(1, sobre_sombra)], img, Config())

    assert _motivos(saida) == [(False, None), (False, None)]


def test_close_sem_campo_visivel_nao_descarta_ninguem():
    img = np.full((600, 800, 3), CINZA, np.uint8)
    img[500:560, 0:150] = VERDE                    # retalho de gramado < 5% da imagem
    a = jogador(img, 100, 50, VERMELHO_KC, w=200, h=400)
    b = jogador(img, 450, 20, AZUL_BUF, w=200, h=400)

    saida = aplicar_filtros([_det(0, a), _det(1, b)], img, Config())

    assert _motivos(saida) == [(False, None), (False, None)]


def test_close_com_faixa_de_gramado_nao_descarta_ninguem():
    # pilha em close: só uma faixa de gramado entre as pernas forma componente grande
    img = np.full((600, 800, 3), CINZA, np.uint8)
    img[:, 700:] = VERDE
    a = jogador(img, 50, 20, VERMELHO_KC, w=250, h=500)
    b = jogador(img, 350, 40, AZUL_BUF, w=250, h=480)

    saida = aplicar_filtros([_det(0, a), _det(1, b)], img, Config())

    assert _motivos(saida) == [(False, None), (False, None)]


def test_caixa_na_borda_inferior_e_mantida_mesmo_fora_do_campo():
    img = campo()
    img[450:] = CINZA                               # primeiro plano: sideline/arquibancada
    na_borda = jogador(img, 300, 400, VERMELHO_KC, h=200)   # cortado pela borda inferior
    fora = jogador(img, 500, 430, VERMELHO_KC, h=130)       # pés em y=560, fora do campo

    saida = aplicar_filtros([_det(0, na_borda), _det(1, fora)], img, Config())

    assert _motivos(saida) == [(False, None), (True, "fora_de_campo")]


def test_regiao_do_campo():
    img = campo()
    poligono = regiao_do_campo(img, Config())
    assert poligono is not None
    assert cv2.pointPolygonTest(poligono, (400.0, 400.0), False) > 0
    assert cv2.pointPolygonTest(poligono, (400.0, 50.0), False) < 0
    assert regiao_do_campo(np.full((600, 800, 3), CINZA, np.uint8), Config()) is None


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


def test_hash_dos_pesos_usa_o_caminho_carregado(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace

    from nfl_vision.stages import detect

    arquivo = tmp_path / "baixado" / "yolo11m.pt"
    arquivo.parent.mkdir()
    arquivo.write_bytes(b"pesos")
    monkeypatch.chdir(tmp_path)  # cwd não tem yolo11m.pt
    monkeypatch.setattr(detect, "_modelo", lambda pesos: SimpleNamespace(ckpt_path=str(arquivo)))
    monkeypatch.setattr(detect, "detectar_pessoas", lambda img, cfg: [])
    estado = SimpleNamespace(imagem=lambda: campo(), config=Config())

    saida = detect.executar(estado)

    assert saida.pesos_sha256 == hashlib.sha256(b"pesos").hexdigest()
