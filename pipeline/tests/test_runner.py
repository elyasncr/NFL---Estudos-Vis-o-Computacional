import hashlib
import json
from datetime import date
from pathlib import Path

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


def test_proximo_id_ignora_nomes_fora_do_padrao(tmp_path):
    runs = tmp_path / "runs"
    hoje = date(2026, 10, 3)
    (runs / "2026-10-03-001").mkdir(parents=True)
    (runs / "2026-10-03-007").mkdir()
    (runs / "2026-10-03-001-old").mkdir()
    (runs / "2026-10-03-099.txt").write_text("nao e diretorio", encoding="utf-8")
    assert proximo_id(runs, hoje) == "2026-10-03-008"


def test_manifest_contem_hash_da_entrada(tmp_path, foto):
    runner = Runner(_etapas([]), tmp_path / "runs")
    run_dir = runner.nova_analise(foto, CTX, Config(), {})

    manifest = ler_manifest(run_dir)

    assert manifest["input_sha256"] == hashlib.sha256(foto.read_bytes()).hexdigest()


def test_escrita_de_artefatos_e_atomica(tmp_path, foto, monkeypatch):
    import nfl_vision.runner as runner_mod

    chamadas = []
    runner = Runner(_etapas(chamadas), tmp_path / "runs")
    run_dir = runner.nova_analise(foto, CTX, Config(), {})

    substituicoes = []
    original_replace = runner_mod.os.replace

    def replace_espiao(origem, destino):
        substituicoes.append(Path(origem).name)
        return original_replace(origem, destino)

    monkeypatch.setattr(runner_mod.os, "replace", replace_espiao)

    runner.executar(run_dir)

    assert substituicoes
    assert all(nome.endswith(".tmp") for nome in substituicoes)
    assert not list(run_dir.glob("*.tmp"))


def test_artefato_corrompido_ao_retomar(tmp_path, foto):
    chamadas = []
    runner = Runner(_etapas_abc(chamadas), tmp_path / "runs")
    run_dir = runner.nova_analise(foto, CTX, Config(), {})
    runner.executar(run_dir)

    (run_dir / "a.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(EtapaFalhou) as erro:
        runner.executar(run_dir, a_partir_de="c")

    assert erro.value.etapa == "a"
    assert "artefato inválido" in erro.value.mensagem


def test_caminho_imagem_ausente_da_erro_claro(tmp_path, foto):
    runner = Runner(_etapas([]), tmp_path / "runs")
    run_dir = runner.nova_analise(foto, CTX, Config(), {})
    (run_dir / f"input{foto.suffix.lower()}").unlink()

    estado = runner.carregar_estado(run_dir)
    with pytest.raises(FileNotFoundError, match="input"):
        estado.caminho_imagem


def test_saida_com_tipo_errado_falha_como_etapa(tmp_path, foto):
    class Outro(BaseModel):
        x: int = 1

    def errada(estado):
        return Outro()

    etapas = [Etapa("a", Numero, errada)]
    runner = Runner(etapas, tmp_path / "runs")
    run_dir = runner.nova_analise(foto, CTX, Config(), {})

    with pytest.raises(EtapaFalhou) as erro:
        runner.executar(run_dir)

    assert erro.value.etapa == "a"
    assert "Outro" in erro.value.mensagem
    assert "Numero" in erro.value.mensagem
    manifest = ler_manifest(run_dir)
    assert manifest["etapas"]["a"]["status"] == "erro"


def test_config_efetiva_grava_campo_ausente_e_guarda_original(tmp_path, foto):
    chamadas = []
    runner = Runner(_etapas(chamadas), tmp_path / "runs")
    run_dir = runner.nova_analise(foto, CTX, Config(), {})
    runner.executar(run_dir)

    manifest = ler_manifest(run_dir)
    antigo = dict(manifest["config"])
    del antigo["detector_conf"]
    manifest["config"] = antigo
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    estado = runner.executar(run_dir, a_partir_de="b")

    assert estado.config.detector_conf == Config().detector_conf
    novo = ler_manifest(run_dir)
    assert novo["config"]["detector_conf"] == Config().detector_conf
    assert novo["config_original"] == antigo


def test_config_original_nao_e_sobrescrita_em_execucoes_seguintes(tmp_path, foto):
    chamadas = []
    runner = Runner(_etapas(chamadas), tmp_path / "runs")
    run_dir = runner.nova_analise(foto, CTX, Config(), {})
    runner.executar(run_dir)

    manifest = ler_manifest(run_dir)
    antigo = dict(manifest["config"])
    del antigo["detector_conf"]
    manifest["config"] = antigo
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    runner.executar(run_dir, a_partir_de="b")

    runner.executar(run_dir, a_partir_de="b")  # segunda retomada: config já é a efetiva

    novo = ler_manifest(run_dir)
    assert novo["config_original"] == antigo


def test_config_com_campo_extra_desconhecido_carrega_sem_erro(tmp_path, foto):
    chamadas = []
    runner = Runner(_etapas(chamadas), tmp_path / "runs")
    run_dir = runner.nova_analise(foto, CTX, Config(), {})
    runner.executar(run_dir)

    manifest = ler_manifest(run_dir)
    manifest["config"]["campo_removido_no_futuro"] = "valor antigo"
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    estado = runner.executar(run_dir, a_partir_de="b")

    assert estado.saidas["b"].valor == 2


def test_le_correcoes(tmp_path, foto):
    runner = Runner(_etapas([]), tmp_path / "runs")
    run_dir = runner.nova_analise(foto, CTX, Config(), {})
    (run_dir / "corrections.json").write_text(
        json.dumps([{"det_id": 3, "numero": 87, "timestamp": "t"}]), encoding="utf-8"
    )
    estado = runner.carregar_estado(run_dir)
    assert estado.correcoes()[0].numero == 87


def test_manifest_antigo_sem_campos_do_rfdetr_ganha_padroes(tmp_path, foto):
    chamadas = []
    runner = Runner(_etapas(chamadas), tmp_path / "runs")
    run_dir = runner.nova_analise(foto, CTX, Config(), {})
    runner.executar(run_dir)

    manifest = ler_manifest(run_dir)
    antigo = {k: v for k, v in manifest["config"].items()
              if k not in ("detector_tipo", "detector_modelo_rfdetr", "detector_resolucao")}
    manifest["config"] = antigo
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    estado = runner.executar(run_dir, a_partir_de="b")

    assert estado.config.detector_tipo == "rfdetr"
    novo = ler_manifest(run_dir)
    assert novo["config"]["detector_tipo"] == "rfdetr"
    assert novo["config"]["detector_resolucao"] == Config().detector_resolucao
    assert novo["config_original"] == antigo
