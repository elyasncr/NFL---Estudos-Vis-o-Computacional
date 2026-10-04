import json

from typer.testing import CliRunner

from nfl_vision import cli, paths
from nfl_vision.cli import app
from nfl_vision.treino.fontes import EXTERNAS, POR_NOME
from nfl_vision.treino.preparar import MOTIVO_BASE
from treino_sintetico import dataset_base, dataset_externo, dataset_preparado, instalar_yolo_falso

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
    assert "preparar" in r.output and "rodar" in r.output


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


def test_rodar_treina_e_mostra_pesos(dados, tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    chamadas = instalar_yolo_falso(monkeypatch)
    ds = dataset_preparado(tmp_path / "ds")

    r = runner.invoke(app, ["treino", "rodar", "--dataset", str(ds), "--nome", "player-v1",
                            "--epocas", "5", "--imgsz", "640"])

    assert r.exit_code == 0, r.output
    assert chamadas[0][1]["epochs"] == 5 and chamadas[0][1]["imgsz"] == 640
    assert (paths.treinos_dir() / "player-v1" / "weights" / "best.pt").exists()
    assert "best.pt" in r.output


def test_rodar_interrompido_e_retomado(dados, tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    ds = dataset_preparado(tmp_path / "ds")
    instalar_yolo_falso(monkeypatch, falhar=KeyboardInterrupt())
    r = runner.invoke(app, ["treino", "rodar", "--dataset", str(ds), "--nome", "player-v1"])
    assert r.exit_code == 130, r.output
    assert "--retomar" in r.output

    chamadas = instalar_yolo_falso(monkeypatch)
    r = runner.invoke(app, ["treino", "rodar", "--nome", "player-v1", "--retomar"])
    assert r.exit_code == 0, r.output
    assert chamadas[-1][1] == {"resume": True}


def test_rodar_falha_no_treino_sai_com_1(dados, tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    instalar_yolo_falso(monkeypatch, falhar=RuntimeError("CUDA out of memory"))
    ds = dataset_preparado(tmp_path / "ds")

    r = runner.invoke(app, ["treino", "rodar", "--dataset", str(ds), "--nome", "player-v1"])

    assert r.exit_code == 1, r.output
    assert "CUDA out of memory" in r.output and "--retomar" in r.output


def test_rodar_erros_de_entrada(dados, tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    instalar_yolo_falso(monkeypatch)
    ds = dataset_preparado(tmp_path / "ds")
    sem_teste = dataset_preparado(tmp_path / "sem_teste")
    for p in (sem_teste / "test" / "images").iterdir():
        p.unlink()
    casos = [
        (["--nome", "x"], "--dataset"),
        (["--dataset", str(tmp_path / "nada"), "--nome", "x"], "data.yaml"),
        (["--dataset", str(sem_teste), "--nome", "x"], "split 'test' sem imagens"),
        (["--nome", "x", "--retomar"], "last.pt"),
        (["--nome", "x", "--retomar", "--dataset", str(ds)], "--retomar usa os parâmetros"),
    ]
    for args, trecho in casos:
        r = runner.invoke(app, ["treino", "rodar", *args])
        assert r.exit_code == 2, (args, r.output)
        assert trecho in r.output, (args, r.output)
