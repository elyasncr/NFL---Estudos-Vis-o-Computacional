import json

from typer.testing import CliRunner

from nfl_vision import cli, paths
from nfl_vision.cli import app
from nfl_vision.treino.fontes import EXTERNAS, POR_NOME
from nfl_vision.treino.preparar import MOTIVO_BASE
from treino_sintetico import dataset_base, dataset_externo

runner = CliRunner()
DECISOES = ["--aprovar", "evzn:futebol americano, caixas boas",
            "--rejeitar", "pitchcamera:soccer", "--rejeitar", "fhtw:caixas ruins"]


def _fontes_locais():
    datasets = paths.datasets_dir()
    dataset_base(datasets)
    for f in EXTERNAS:
        dataset_externo(datasets, f)


def _pasta():
    return paths.datasets_dir() / "treino-player-v1"


def test_treino_help_lista_comandos():
    r = runner.invoke(app, ["treino", "--help"])
    assert r.exit_code == 0, r.output
    assert "preparar" in r.output


def test_preparar_so_triagem(dados, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    _fontes_locais()

    r = runner.invoke(app, ["treino", "preparar", "--so-triagem"])

    assert r.exit_code == 0, r.output
    assert sorted(p.name for p in (_pasta() / "triagem").iterdir()) == \
        ["evzn.jpg", "fhtw.jpg", "pitchcamera.jpg"]
    assert not (_pasta() / "data.yaml").exists()
    assert "--aprovar" in r.output


def test_preparar_com_decisoes(dados, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    _fontes_locais()

    r = runner.invoke(app, ["treino", "preparar", *DECISOES])

    assert r.exit_code == 0, r.output
    assert "Dataset em:" in r.output
    manifest = json.loads((_pasta() / "manifest.json").read_text("utf-8"))
    assert {f["nome"]: (f["aprovada"], f["motivo"]) for f in manifest["fontes"]} == {
        "base": (True, MOTIVO_BASE),
        "pitchcamera": (False, "soccer"),
        "fhtw": (False, "caixas ruins"),
        "evzn": (True, "futebol americano, caixas boas"),
    }


def test_preparar_erros_de_entrada(dados, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    _fontes_locais()
    casos = [
        (["--aprovar", "evzn:ok"], "faltam: pitchcamera, fhtw"),
        (["--aprovar", "xyz:ok", *DECISOES], "fonte desconhecida 'xyz'"),
        (["--aprovar", "evzn", "--rejeitar", "pitchcamera:x", "--rejeitar", "fhtw:y"], "motivo"),
        (["--rejeitar", "evzn:x", *DECISOES], "mais de uma vez"),
        (["--so-triagem", "--aprovar", "evzn:ok"], "--so-triagem"),
    ]
    for args, trecho in casos:
        r = runner.invoke(app, ["treino", "preparar", *args])
        assert r.exit_code == 2, (args, r.output)
        assert trecho in r.output, (args, r.output)

    assert runner.invoke(app, ["treino", "preparar", *DECISOES]).exit_code == 0
    r = runner.invoke(app, ["treino", "preparar", *DECISOES])
    assert r.exit_code == 2, r.output
    assert "já tem um dataset" in r.output


def test_preparar_fonte_sem_acesso(dados, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)  # um .env local não pode trazer a chave
    dataset_base(paths.datasets_dir())  # externas ausentes: precisariam de download

    r = runner.invoke(app, ["treino", "preparar", "--so-triagem"])

    assert r.exit_code == 2, r.output
    assert "pitchcamera" in r.output and "ROBOFLOW_API_KEY" in r.output


def test_preparar_mapeamento_invalido_lista_classes(dados, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    datasets = paths.datasets_dir()
    dataset_base(datasets)
    dataset_externo(datasets, POR_NOME["pitchcamera"])
    dataset_externo(datasets, POR_NOME["fhtw"])
    dataset_externo(datasets, POR_NOME["evzn"], nomes=["ball", "referee", "players"])

    r = runner.invoke(app, ["treino", "preparar", "--so-triagem"])

    assert r.exit_code == 2, r.output
    assert "football-players" in r.output and "ball, referee, players" in r.output
