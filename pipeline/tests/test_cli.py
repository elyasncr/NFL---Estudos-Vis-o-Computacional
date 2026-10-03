from PIL import Image
from typer.testing import CliRunner

from nfl_vision import cli, paths, teams
from nfl_vision.cli import app

runner = CliRunner()
BASE = ["--times", "KC", "BUF", "--temporada", "2025", "--semana", "11"]


def test_analyze_ok_e_correct(dados, foto_sintetica, modelos_falsos):
    r = runner.invoke(app, ["analyze", str(foto_sintetica[0]), *BASE])
    assert r.exit_code == 0, r.output
    run_dir = next(paths.runs_dir().iterdir())
    assert (run_dir / "analise.json").exists()

    r = runner.invoke(app, ["correct", run_dir.name, "--det", "3", "--numero", "14"])
    assert r.exit_code == 0, r.output
    assert (run_dir / "corrections.json").exists()


def test_analyze_reprocessa_com_from(dados, foto_sintetica, modelos_falsos):
    runner.invoke(app, ["analyze", str(foto_sintetica[0]), *BASE])
    run_id = next(paths.runs_dir().iterdir()).name
    r = runner.invoke(app, ["analyze", "--run", run_id, "--from", "jersey"])
    assert r.exit_code == 0, r.output


def test_validacoes(dados, tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    video = tmp_path / "jogo.mp4"
    video.write_bytes(b"x")
    corrompida = tmp_path / "corrompida.jpg"
    corrompida.write_bytes(b"x")
    foto = tmp_path / "f.jpg"
    Image.new("RGB", (32, 32), (40, 140, 40)).save(foto, "JPEG")

    casos = [
        (["analyze", str(video), *BASE], "use JPG ou PNG"),
        (["analyze", str(tmp_path / "nao.jpg"), *BASE], "não encontrado"),
        (["analyze", str(corrompida), *BASE], "imagem inválida ou corrompida"),
        (["analyze", str(foto), "--times", "KC", "XYZ", "--temporada", "2025", "--semana", "11"], "XYZ"),
        (["analyze", str(foto), "--times", "KC", "kc", "--temporada", "2025", "--semana", "11"], "diferentes"),
        (["analyze", str(foto), "--times", "KC", "BUF", "--temporada", "2025", "--semana", "30"], "semana"),
        (["analyze", str(foto), "--times", "KC", "BUF", "--temporada", "1990", "--semana", "1"], "temporada"),
        (["analyze", "--run", "x", "--from", "zzz"], "--from"),
    ]
    for args, trecho in casos:
        r = runner.invoke(app, args)
        assert r.exit_code == 2, (args, r.output)
        assert trecho in r.output, (args, r.output)


def test_times_indisponiveis(dados, tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    foto = tmp_path / "f.jpg"
    Image.new("RGB", (32, 32), (40, 140, 40)).save(foto, "JPEG")

    def sem_times(cache_dir):
        raise teams.TimesIndisponiveis("times indisponíveis (sem rede)")

    monkeypatch.setattr(cli.teams, "carregar_times", sem_times)
    r = runner.invoke(app, ["analyze", str(foto), *BASE])
    assert r.exit_code == 2, r.output
    assert "times indisponíveis (sem rede)" in r.output


def test_etapa_com_erro_sugere_from(dados, foto_sintetica, modelos_falsos, monkeypatch):
    from nfl_vision.stages import roster

    def quebrar(*a, **k):
        raise roster.RosterIndisponivel("sem rede")

    monkeypatch.setattr(roster, "carregar_roster", quebrar)
    r = runner.invoke(app, ["analyze", str(foto_sintetica[0]), *BASE])
    assert r.exit_code == 1
    assert "--from roster" in r.output


def _quebrar_roster(monkeypatch):
    from nfl_vision.stages import roster

    def quebrar(*a, **k):
        raise roster.RosterIndisponivel("sem rede")

    monkeypatch.setattr(roster, "carregar_roster", quebrar)


def test_correct_com_erro_de_etapa_sugere_from(dados, foto_sintetica, modelos_falsos, monkeypatch):
    runner.invoke(app, ["analyze", str(foto_sintetica[0]), *BASE])
    run_id = next(paths.runs_dir().iterdir()).name
    _quebrar_roster(monkeypatch)

    r = runner.invoke(app, ["correct", run_id, "--det", "3", "--numero", "14"])

    assert r.exit_code == 1, r.output
    assert f"--run {run_id} --from roster" in r.output


def test_correct_apos_reprocessamento_falho(dados, foto_sintetica, modelos_falsos, monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    runner.invoke(app, ["analyze", str(foto_sintetica[0]), *BASE])
    run_id = next(paths.runs_dir().iterdir()).name
    _quebrar_roster(monkeypatch)
    assert runner.invoke(app, ["analyze", "--run", run_id, "--from", "jersey"]).exit_code == 1

    r = runner.invoke(app, ["correct", run_id, "--det", "3", "--numero", "14"])

    assert r.exit_code == 2, r.output
    assert "análise incompleta" in r.output and "--from roster" in r.output


def test_interrompido(dados, monkeypatch):
    def interromper(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.pipeline, "corrigir", interromper)
    r = runner.invoke(app, ["correct", "2025-01-01-001", "--det", "1", "--numero", "2"])
    assert r.exit_code == 130
    assert "interrompido" in r.output


def test_from_detect_avisa_correcoes_arquivadas(dados, foto_sintetica, modelos_falsos, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    runner.invoke(app, ["analyze", str(foto_sintetica[0]), *BASE])
    run_id = next(paths.runs_dir().iterdir()).name
    runner.invoke(app, ["correct", run_id, "--det", "3", "--numero", "14"])

    r = runner.invoke(app, ["analyze", "--run", run_id, "--from", "detect"])

    assert r.exit_code == 0, r.output
    assert "correções anteriores arquivadas em" in r.output


def test_combinacoes_invalidas_de_opcoes(dados, tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    foto = tmp_path / "f.jpg"
    Image.new("RGB", (32, 32), (40, 140, 40)).save(foto, "JPEG")

    casos = [
        (["analyze", str(foto), *BASE, "--from", "jersey"], "--from exige --run"),
        (["analyze", str(foto), "--run", "x", "--from", "jersey"], "FOTO"),
        (["analyze", "--run", "x"], "--run exige --from <etapa>"),
    ]
    for args, trecho in casos:
        r = runner.invoke(app, args)
        assert r.exit_code == 2, (args, r.output)
        assert trecho in r.output, (args, r.output)


def test_eval_help_lista_comandos():
    r = runner.invoke(app, ["eval", "--help"])
    assert r.exit_code == 0, r.output
    assert all(c in r.output for c in ("baixar", "detect", "jersey"))


def test_eval_detect_benchmark_invalido(dados, tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    r = runner.invoke(app, ["eval", "detect", "--dataset", str(tmp_path), "--benchmark", "xyz"])
    assert r.exit_code == 2, r.output
    assert "rfdetr ou roboflow-nfl" in r.output


def test_eval_jersey_grava_resultado(dados, tmp_path, monkeypatch):
    import json

    from nfl_vision.stages import jersey
    from nfl_vision.stages.jersey import Leitura

    (tmp_path / "ds" / "test" / "87").mkdir(parents=True)
    Image.new("RGB", (20, 20)).save(tmp_path / "ds" / "test" / "87" / "a.jpg")

    class Leitor:
        def ler(self, img):
            return [Leitura("87", 0.9)]

    monkeypatch.setattr(jersey, "leitor_padrao", lambda device: Leitor())
    r = runner.invoke(app, ["eval", "jersey", "--dataset", str(tmp_path / "ds")])

    assert r.exit_code == 0, r.output
    arquivo = next(paths.avaliacoes_dir().glob("jersey-*.json"))
    assert json.loads(arquivo.read_text("utf-8"))["acuracia_geral"] == 1.0
