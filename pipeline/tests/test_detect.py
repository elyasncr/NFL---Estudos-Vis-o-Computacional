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
    estado = SimpleNamespace(imagem=lambda: campo(), config=Config(detector_tipo="yolo"))

    saida = detect.executar(estado)

    assert saida.pesos_sha256 == hashlib.sha256(b"pesos").hexdigest()


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
