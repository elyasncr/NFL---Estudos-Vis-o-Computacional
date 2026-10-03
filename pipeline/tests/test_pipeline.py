import hashlib
import json

import pytest

from nfl_vision import pipeline
from nfl_vision.runner import ler_manifest
from nfl_vision.schemas import Contexto

CTX = Contexto(temporada=2025, semana=11, times=("KC", "BUF"))


def _por_id(analise):
    return {j.track_id: j for j in analise.jogadores}


def test_analise_completa(dados, foto_sintetica, modelos_falsos):
    run_dir, analise = pipeline.analisar(foto_sintetica[0], CTX)

    jogadores = _por_id(analise)
    assert sorted(jogadores) == [0, 1, 2, 3]  # sem árbitro (4) e arquibancada (5)
    assert (jogadores[0].time, jogadores[0].numero, jogadores[0].posicao, jogadores[0].nome) == (
        "KC", 87, "TE", "Jogador KC 87")
    assert jogadores[2].nome == "Jogador BUF 17"
    assert jogadores[3].time == "BUF" and jogadores[3].numero is None
    assert analise.midia == {"tipo": "foto", "largura": 800, "altura": 600}
    assert analise.modelos == {"detector": "yolo11m", "ocr": "paddleocr"}

    assert (run_dir / "anotada.png").exists()
    salvo = json.loads((run_dir / "analise.json").read_text("utf-8"))
    assert salvo["analise_id"] == run_dir.name
    assert all(e["status"] == "ok" for e in ler_manifest(run_dir)["etapas"].values())


def test_manifest_registra_hash_dos_pesos(dados, foto_sintetica, modelos_falsos):
    run_dir, _ = pipeline.analisar(foto_sintetica[0], CTX)

    esperado = hashlib.sha256(b"pesos falsos").hexdigest()
    manifest = ler_manifest(run_dir)
    assert manifest["versoes"]["detector_pesos_sha256"] == esperado
    assert "nfl-vision" in manifest["versoes"]

    pipeline.reprocessar(run_dir.name, "roster")
    assert ler_manifest(run_dir)["versoes"]["detector_pesos_sha256"] == esperado


def test_reprocessar_nao_roda_detector_de_novo(dados, foto_sintetica, modelos_falsos):
    run_dir, _ = pipeline.analisar(foto_sintetica[0], CTX)
    pipeline.reprocessar(run_dir.name, "roster")
    assert modelos_falsos["detect"] == 1


def test_corrigir_numero(dados, foto_sintetica, modelos_falsos):
    run_dir, _ = pipeline.analisar(foto_sintetica[0], CTX)

    analise = pipeline.corrigir(run_dir.name, det_id=3, numero=14)

    j = _por_id(analise)[3]
    assert (j.numero, j.nome, j.confianca_numero, j.corrigido_pelo_usuario) == (
        14, "Jogador BUF 14 Ativo", 1.0, True)
    assert json.loads((run_dir / "corrections.json").read_text("utf-8"))[0]["numero"] == 14


def test_corrigir_valida_entrada(dados, foto_sintetica, modelos_falsos):
    run_dir, _ = pipeline.analisar(foto_sintetica[0], CTX)
    with pytest.raises(ValueError, match="não é um jogador"):
        pipeline.corrigir(run_dir.name, det_id=4, numero=10)
    with pytest.raises(ValueError, match="time deve ser"):
        pipeline.corrigir(run_dir.name, det_id=0, time="LA")
    with pytest.raises(ValueError, match="informe"):
        pipeline.corrigir(run_dir.name, det_id=0)
    with pytest.raises(FileNotFoundError):
        pipeline.corrigir("2000-01-01-001", det_id=0, numero=1)


def _quebrar_roster(monkeypatch):
    from nfl_vision.stages import roster

    def quebrar(*a, **k):
        raise roster.RosterIndisponivel("sem rede")

    monkeypatch.setattr(roster, "carregar_roster", quebrar)


def test_reprocessamento_falho_remove_saidas_finais(dados, foto_sintetica, modelos_falsos, monkeypatch):
    from nfl_vision.runner import EtapaFalhou

    run_dir, _ = pipeline.analisar(foto_sintetica[0], CTX)
    _quebrar_roster(monkeypatch)
    with pytest.raises(EtapaFalhou):
        pipeline.reprocessar(run_dir.name, "jersey")
    assert not (run_dir / "analise.json").exists()
    assert not (run_dir / "anotada.png").exists()


def test_corrigir_exige_analise_completa(dados, foto_sintetica, modelos_falsos, monkeypatch):
    from nfl_vision.runner import EtapaFalhou

    run_dir, _ = pipeline.analisar(foto_sintetica[0], CTX)
    _quebrar_roster(monkeypatch)
    with pytest.raises(EtapaFalhou):
        pipeline.reprocessar(run_dir.name, "jersey")

    esperado = f"análise incompleta; rode nfl-vision analyze --run {run_dir.name} --from roster antes"
    with pytest.raises(ValueError, match=esperado):
        pipeline.corrigir(run_dir.name, det_id=3, numero=14)
    assert not (run_dir / "corrections.json").exists()


def test_finalizar_falha_ao_gerar_png(dados, foto_sintetica, modelos_falsos, monkeypatch):
    monkeypatch.setattr(pipeline.cv2, "imencode", lambda ext, img: (False, None))
    with pytest.raises(RuntimeError, match="falha ao gerar anotada.png"):
        pipeline.analisar(foto_sintetica[0], CTX)
