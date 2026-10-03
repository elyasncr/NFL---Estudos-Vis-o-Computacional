import json
from datetime import date

import pytest
from pydantic import BaseModel

from nfl_vision.config import Config
from nfl_vision.runner import Etapa, EtapaFalhou, Runner, ler_manifest, proximo_id
from nfl_vision.schemas import Contexto

CTX = Contexto(temporada=2025, semana=11, times=("KC", "BUF"))


class Numero(BaseModel):
    valor: int


def _etapas(chamadas, falhar_b=False):
    def a(estado):
        chamadas.append("a")
        return Numero(valor=1)

    def b(estado):
        chamadas.append("b")
        if falhar_b:
            raise RuntimeError("quebrou")
        return Numero(valor=estado.saidas["a"].valor + 1)

    return [Etapa("a", Numero, a), Etapa("b", Numero, b)]


def _etapas_controlaveis(chamadas, flags):
    """Como `_etapas`, mas a e b podem falhar a qualquer momento via `flags`."""

    def a(estado):
        chamadas.append("a")
        if flags.get("falhar_a"):
            raise RuntimeError("quebrou a")
        return Numero(valor=1)

    def b(estado):
        chamadas.append("b")
        if flags.get("falhar_b"):
            raise RuntimeError("quebrou b")
        return Numero(valor=estado.saidas["a"].valor + 1)

    return [Etapa("a", Numero, a), Etapa("b", Numero, b)]


def _etapas_abc(chamadas):
    def a(estado):
        chamadas.append("a")
        return Numero(valor=1)

    def b(estado):
        chamadas.append("b")
        return Numero(valor=2)

    def c(estado):
        chamadas.append("c")
        return Numero(valor=3)

    return [Etapa("a", Numero, a), Etapa("b", Numero, b), Etapa("c", Numero, c)]


@pytest.fixture
def foto(tmp_path):
    p = tmp_path / "foto.JPG"
    p.write_bytes(b"conteudo")
    return p


def test_executa_todas_e_grava_artefatos(tmp_path, foto):
    chamadas = []
    runner = Runner(_etapas(chamadas), tmp_path / "runs")
    run_dir = runner.nova_analise(foto, CTX, Config(), {"pacote": "0.1"})
    estado = runner.executar(run_dir)

    assert chamadas == ["a", "b"]
    assert estado.saidas["b"].valor == 2
    assert (run_dir / "input.jpg").read_bytes() == b"conteudo"
    assert json.loads((run_dir / "b.json").read_text())["valor"] == 2
    manifest = ler_manifest(run_dir)
    assert manifest["etapas"]["a"]["status"] == "ok"
    assert manifest["config"]["detector_pesos"] == "yolo11m.pt"
    assert manifest["versoes"] == {"pacote": "0.1"}


def test_retoma_a_partir_de_uma_etapa(tmp_path, foto):
    chamadas = []
    runner = Runner(_etapas(chamadas), tmp_path / "runs")
    run_dir = runner.nova_analise(foto, CTX, Config(), {})
    runner.executar(run_dir)
    chamadas.clear()

    estado = runner.executar(run_dir, a_partir_de="b")

    assert chamadas == ["b"]
    assert estado.saidas["a"].valor == 1


def test_falha_registra_erro_e_preserva_anteriores(tmp_path, foto):
    runner = Runner(_etapas([], falhar_b=True), tmp_path / "runs")
    run_dir = runner.nova_analise(foto, CTX, Config(), {})

    with pytest.raises(EtapaFalhou) as erro:
        runner.executar(run_dir)

    assert erro.value.etapa == "b"
    assert erro.value.analise_id == run_dir.name
    assert (run_dir / "a.json").exists()
    manifest = ler_manifest(run_dir)
    assert manifest["etapas"]["b"]["status"] == "erro"
    assert "quebrou" in manifest["etapas"]["b"]["mensagem"]


def test_resume_apos_falha_recria_manifest_consistente(tmp_path, foto):
    chamadas = []
    flags = {"falhar_b": True}
    runner = Runner(_etapas_controlaveis(chamadas, flags), tmp_path / "runs")
    run_dir = runner.nova_analise(foto, CTX, Config(), {})

    with pytest.raises(EtapaFalhou):
        runner.executar(run_dir)

    flags["falhar_b"] = False
    chamadas.clear()
    estado = runner.executar(run_dir, a_partir_de="b")

    assert chamadas == ["b"]
    assert estado.saidas["b"].valor == 2
    manifest = ler_manifest(run_dir)
    assert manifest["etapas"]["a"]["status"] == "ok"
    assert manifest["etapas"]["b"]["status"] == "ok"
    assert (run_dir / "a.json").exists()
    assert (run_dir / "b.json").exists()


def test_etapa_falhada_nao_pode_ser_pulada(tmp_path, foto):
    chamadas = []
    runner = Runner(_etapas_abc(chamadas), tmp_path / "runs")
    run_dir = runner.nova_analise(foto, CTX, Config(), {})
    runner.executar(run_dir)

    manifest = ler_manifest(run_dir)
    manifest["etapas"]["b"] = {"status": "erro", "mensagem": "falhou antes", "duracao_s": 0}
    (run_dir / "b.json").write_text('{"valor": 2}', encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(EtapaFalhou) as erro:
        runner.executar(run_dir, a_partir_de="c")

    assert erro.value.etapa == "b"


def test_resume_remove_artefatos_posteriores_antes_de_rodar(tmp_path, foto):
    chamadas = []
    flags = {}
    runner = Runner(_etapas_controlaveis(chamadas, flags), tmp_path / "runs")
    run_dir = runner.nova_analise(foto, CTX, Config(), {})
    runner.executar(run_dir)

    flags["falhar_a"] = True
    with pytest.raises(EtapaFalhou):
        runner.executar(run_dir, a_partir_de="a")

    manifest = ler_manifest(run_dir)
    assert "b" not in manifest["etapas"]
    assert not (run_dir / "b.json").exists()


def test_etapa_desconhecida(tmp_path, foto):
    runner = Runner(_etapas([]), tmp_path / "runs")
    run_dir = runner.nova_analise(foto, CTX, Config(), {})
    with pytest.raises(ValueError, match="etapa desconhecida"):
        runner.executar(run_dir, a_partir_de="zzz")


def test_proximo_id_sequencial(tmp_path):
    runs = tmp_path / "runs"
    hoje = date(2026, 10, 3)
    assert proximo_id(runs, hoje) == "2026-10-03-001"
    (runs / "2026-10-03-001").mkdir(parents=True)
    (runs / "2026-10-03-007").mkdir()
    assert proximo_id(runs, hoje) == "2026-10-03-008"


def test_le_correcoes(tmp_path, foto):
    runner = Runner(_etapas([]), tmp_path / "runs")
    run_dir = runner.nova_analise(foto, CTX, Config(), {})
    (run_dir / "corrections.json").write_text(
        json.dumps([{"det_id": 3, "numero": 87, "timestamp": "t"}]), encoding="utf-8"
    )
    estado = runner.carregar_estado(run_dir)
    assert estado.correcoes()[0].numero == 87
