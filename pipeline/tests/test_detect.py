import pytest

from nfl_vision.config import Config
from nfl_vision.schemas import Deteccao
from nfl_vision.stages.detect import aplicar_filtros
from sintetico import CINZA, VERMELHO_KC, campo, jogador


def _det(i, bbox):
    return Deteccao(det_id=i, bbox=bbox, confianca=0.9)


def test_filtros_de_fora_de_campo_e_tamanho():
    img = campo()
    em_campo = jogador(img, 100, 300, VERMELHO_KC)
    na_arquibancada = jogador(img, 300, 10, VERMELHO_KC)
    pequeno = (500.0, 400.0, 510.0, 420.0)
    # jogador cortado pela borda inferior: a parte de baixo da caixa é camisa/perna
    na_borda = jogador(img, 600, 480, VERMELHO_KC, h=120)

    saida = aplicar_filtros(
        [_det(0, em_campo), _det(1, na_arquibancada), _det(2, pequeno), _det(3, na_borda)],
        img, Config(),
    )

    assert [(d.descartado, d.motivo_descarte) for d in saida] == [
        (False, None), (True, "fora_de_campo"), (True, "pequeno"), (False, None),
    ]


def test_perto_da_borda_usa_faixa_externa_recortada():
    img = campo()
    sobre_gramado = jogador(img, 100, 475, VERMELHO_KC)   # termina em y=595
    sobre_cinza = jogador(img, 300, 475, VERMELHO_KC)
    img[595:, 300:360] = CINZA

    saida = aplicar_filtros([_det(0, sobre_gramado), _det(1, sobre_cinza)], img, Config())

    assert [(d.descartado, d.motivo_descarte) for d in saida] == [
        (False, None), (True, "fora_de_campo"),
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
