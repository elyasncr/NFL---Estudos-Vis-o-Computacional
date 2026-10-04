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


def test_finalizar_falha_sem_id_conhecido_nao_sugere_from(
    dados, foto_sintetica, modelos_falsos, monkeypatch
):
    def quebrar(estado):
        raise RuntimeError("falha ao gerar anotada.png")

    monkeypatch.setattr(cli.pipeline, "finalizar", quebrar)
    r = runner.invoke(app, ["analyze", str(foto_sintetica[0]), *BASE])

    assert r.exit_code == 1, r.output
    assert "falha ao gerar anotada.png" in r.output
    assert "--from" not in r.output


def test_analyze_from_com_finalizar_falho_sugere_from_roster(
    dados, foto_sintetica, modelos_falsos, monkeypatch
):
    runner.invoke(app, ["analyze", str(foto_sintetica[0]), *BASE])
    run_id = next(paths.runs_dir().iterdir()).name

    def quebrar(estado):
        raise RuntimeError("falha ao gerar anotada.png")

    monkeypatch.setattr(cli.pipeline, "finalizar", quebrar)
    r = runner.invoke(app, ["analyze", "--run", run_id, "--from", "jersey"])

    assert r.exit_code == 1, r.output
    assert "falha ao gerar anotada.png" in r.output
    assert f"--run {run_id} --from roster" in r.output


def test_correct_com_finalizar_falho_sugere_from_roster(
    dados, foto_sintetica, modelos_falsos, monkeypatch
):
    runner.invoke(app, ["analyze", str(foto_sintetica[0]), *BASE])
    run_id = next(paths.runs_dir().iterdir()).name

    def quebrar(estado):
        raise teams.TimesIndisponiveis("times indisponíveis (sem rede)")

    monkeypatch.setattr(cli.pipeline, "finalizar", quebrar)
    r = runner.invoke(app, ["correct", run_id, "--det", "3", "--numero", "14"])

    assert r.exit_code == 1, r.output
    assert "times indisponíveis" in r.output
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


def test_eval_baixar_sem_chave_da_erro_claro(dados, tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)  # um .env local não pode trazer a chave

    r = runner.invoke(
        app, ["eval", "baixar", "--workspace", "ws", "--projeto", "proj", "--versao", "1"])

    assert r.exit_code == 2, r.output
    assert "ROBOFLOW_API_KEY" in r.output


def test_eval_baixar_erro_de_rede_reporta_tipo_e_mensagem(dados, tmp_path, monkeypatch):
    from nfl_vision.eval import datasets as eval_datasets

    monkeypatch.setenv("COLUMNS", "300")

    def quebrar(*a, **k):
        raise ConnectionError("timeout")

    monkeypatch.setattr(eval_datasets, "baixar", quebrar)
    r = runner.invoke(
        app, ["eval", "baixar", "--workspace", "ws", "--projeto", "proj", "--versao", "1"])

    assert r.exit_code == 2, r.output
    assert "ConnectionError" in r.output and "timeout" in r.output


def test_eval_help_lista_comandos():
    r = runner.invoke(app, ["eval", "--help"])
    assert r.exit_code == 0, r.output
    assert all(c in r.output for c in ("baixar", "detect", "jersey"))


def test_eval_detect_benchmark_invalido(dados, tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    r = runner.invoke(app, ["eval", "detect", "--dataset", str(tmp_path), "--benchmark", "xyz"])
    assert r.exit_code == 2, r.output
    assert "yolo-bruto, rfdetr ou roboflow-nfl" in r.output


def _dataset_deteccao(raiz, nomes="['ball', 'player', 'referee']", split="test"):
    (raiz / split / "images").mkdir(parents=True)
    (raiz / "data.yaml").write_text(f"names: {nomes}\n", encoding="utf-8")
    (raiz / split / "labels").mkdir(parents=True)
    Image.new("RGB", (100, 100), (40, 140, 40)).save(raiz / split / "images" / "a.jpg")
    (raiz / split / "labels" / "a.txt").write_text("1 0.5 0.5 0.2 0.4\n")
    return raiz


def _detector_falso(monkeypatch):
    from nfl_vision.eval import preditores
    from nfl_vision.schemas import Deteccao

    def detectar(img, cfg):
        return [Deteccao(det_id=0, bbox=(40.0, 30.0, 60.0, 70.0), confianca=0.9)]

    monkeypatch.setattr(preditores, "detectar_pessoas", detectar)


def _ler_avaliacao(prefixo):
    import json

    return json.loads(next(paths.avaliacoes_dir().glob(f"{prefixo}-*.json")).read_text("utf-8"))


def test_eval_detect_grava_contexto_e_pos_processamento(dados, tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    _detector_falso(monkeypatch)
    ds = _dataset_deteccao(tmp_path / "ds")

    r = runner.invoke(app, ["eval", "detect", "--dataset", str(ds), "--benchmark", "yolo-bruto",
                            "--conf", "0.3"])

    assert r.exit_code == 0, r.output
    assert "pós-processamento" in r.output
    assert "árbitros cobertos por caixa de jogador (IoU ≥ 0,5)" in r.output
    salvo = _ler_avaliacao("detect")
    assert salvo["dataset"] == str(ds) and salvo["split"] == "test"
    assert salvo["conf"] == 0.3 and salvo["config"]["detector_conf"] == 0.3
    assert salvo["versao"]
    assert "modelo_roboflow" not in salvo
    assert [x["pos_processamento"] for x in salvo["resultados"]] == [
        "filtros de campo + remoção de árbitro", "nenhum"]
    assert salvo["resultados"][1]["map50"] == 1.0


def test_eval_detect_conf_padrao_vem_da_config(dados, tmp_path, monkeypatch):
    from nfl_vision.config import Config

    _detector_falso(monkeypatch)
    r = runner.invoke(app, ["eval", "detect", "--dataset", str(_dataset_deteccao(tmp_path / "ds"))])
    assert r.exit_code == 0, r.output
    assert _ler_avaliacao("detect")["conf"] == Config().detector_conf


def test_eval_detect_preditor_com_erro_nao_para_os_outros(dados, tmp_path, monkeypatch):
    from nfl_vision.eval import preditores

    _detector_falso(monkeypatch)
    ds = _dataset_deteccao(tmp_path / "ds")
    vistos = []

    def quebrar(self, imagem):
        raise RuntimeError("GPU sem memória")

    def bruto_espiando(self, imagem):
        vistos.append(_ler_avaliacao("detect")["resultados"])  # já gravou o preditor anterior
        return [(0.9, (40.0, 30.0, 60.0, 70.0))]

    monkeypatch.setattr(preditores.PreditorNosso, "prever", quebrar)
    monkeypatch.setattr(preditores.PreditorYoloBruto, "prever", bruto_espiando)
    r = runner.invoke(app, ["eval", "detect", "--dataset", str(ds), "--benchmark", "yolo-bruto"])

    assert r.exit_code == 1, r.output
    assert vistos and vistos[0][0]["erro"] == "RuntimeError: GPU sem memória"
    resultados = _ler_avaliacao("detect")["resultados"]
    assert resultados[0]["erro"] == "RuntimeError: GPU sem memória"
    assert resultados[1]["map50"] == 1.0


def test_eval_detect_erros_de_entrada(dados, tmp_path, monkeypatch):
    from nfl_vision.eval import preditores

    monkeypatch.setenv("COLUMNS", "300")
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)  # um .env local não pode trazer a chave
    ds =_dataset_deteccao(tmp_path / "ds")
    sem_player = _dataset_deteccao(tmp_path / "sem_player", nomes="['ball', 'goleiro']")
    so_valid = _dataset_deteccao(tmp_path / "so_valid", split="valid")

    def sem_extra(*a, **k):
        raise ImportError("No module named 'rfdetr'")

    monkeypatch.setattr(preditores, "PreditorRFDETR", sem_extra)
    casos = [
        (["--dataset", str(tmp_path / "nada")], "data.yaml"),
        (["--dataset", str(so_valid), "--split", "val"], "valid"),
        (["--dataset", str(sem_player)], "goleiro"),
        (["--dataset", str(ds), "--benchmark", "rfdetr"], "uv sync --extra ocr --extra eval"),
        (["--dataset", str(ds), "--benchmark", "roboflow-nfl"], "ROBOFLOW_API_KEY"),
    ]
    for args, trecho in casos:
        r = runner.invoke(app, ["eval", "detect", *args])
        assert r.exit_code == 2, (args, r.output)
        assert trecho in r.output, (args, r.output)


def test_eval_detect_help_menciona_valid():
    r = runner.invoke(app, ["eval", "detect", "--help"], env={"COLUMNS": "300"})
    assert r.exit_code == 0, r.output
    assert "valid" in r.output and "--conf" in r.output and "yolo-bruto" in r.output


def _leitor_falso(monkeypatch):
    from nfl_vision.stages import jersey
    from nfl_vision.stages.jersey import Leitura

    class Leitor:
        def ler(self, img):
            return [Leitura("87", 0.9)]

    monkeypatch.setattr(jersey, "leitor_padrao", lambda device: Leitor())


def test_eval_jersey_grava_resultado(dados, tmp_path, monkeypatch):
    from nfl_vision.config import Config

    for rotulo in ("87", "-1"):
        (tmp_path / "ds" / "test" / rotulo).mkdir(parents=True)
        Image.new("RGB", (20, 20)).save(tmp_path / "ds" / "test" / rotulo / "a.jpg")
    _leitor_falso(monkeypatch)

    r = runner.invoke(app, ["eval", "jersey", "--dataset", str(tmp_path / "ds")])

    assert r.exit_code == 0, r.output
    salvo = _ler_avaliacao("jersey")
    cfg = Config()
    assert salvo["dataset"] == str(tmp_path / "ds") and salvo["split"] == "test"
    assert salvo["limiar"] == cfg.limiar_numero and salvo["altura_min"] == cfg.numero_altura_min
    assert salvo["versao"]
    assert salvo["resultado"]["acuracia_geral"] == 1.0
    assert salvo["resultado"]["excluidas"] == 1


def test_eval_jersey_erros_de_entrada(dados, tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    _leitor_falso(monkeypatch)
    (tmp_path / "ds" / "valid" / "87").mkdir(parents=True)
    Image.new("RGB", (20, 20)).save(tmp_path / "ds" / "valid" / "87" / "a.jpg")
    (tmp_path / "vazio" / "test").mkdir(parents=True)

    casos = [
        (["--dataset", str(tmp_path / "ds"), "--split", "val"], "valid"),
        (["--dataset", str(tmp_path / "vazio")], "nenhuma imagem"),
    ]
    for args, trecho in casos:
        r = runner.invoke(app, ["eval", "jersey", *args])
        assert r.exit_code == 2, (args, r.output)
        assert trecho in r.output, (args, r.output)


def test_file_not_found_na_analise_nova_nao_culpa_run(dados, foto_sintetica, monkeypatch):
    def sem_pesos(*a, **k):
        raise FileNotFoundError("pesos ausentes")

    monkeypatch.setattr(cli.pipeline, "analisar", sem_pesos)
    r = runner.invoke(app, ["analyze", str(foto_sintetica[0]), *BASE])

    assert r.exit_code != 2, r.output
    assert isinstance(r.exception, FileNotFoundError)


def test_eval_detect_com_pesos_ajustados(dados, tmp_path, monkeypatch):
    import hashlib

    monkeypatch.setenv("COLUMNS", "300")
    _detector_falso(monkeypatch)
    pesos = tmp_path / "treinos" / "player-v1" / "weights" / "best.pt"
    pesos.parent.mkdir(parents=True)
    pesos.write_bytes(b"ajustado")
    ds = _dataset_deteccao(tmp_path / "ds")

    r = runner.invoke(app, ["eval", "detect", "--dataset", str(ds), "--pesos", str(pesos),
                            "--benchmark", "yolo-bruto"])

    assert r.exit_code == 0, r.output
    salvo = _ler_avaliacao("detect")
    assert salvo["config"]["detector_pesos"] == str(pesos.resolve())
    assert salvo["pesos_sha256"] == hashlib.sha256(b"ajustado").hexdigest()
    assert [x["preditor"] for x in salvo["resultados"]] == [
        "nosso (player-v1 + filtros + árbitro)", "yolo bruto (player-v1)"]


def test_eval_detect_pesos_inexistentes(dados, tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    ds = _dataset_deteccao(tmp_path / "ds")
    r = runner.invoke(app, ["eval", "detect", "--dataset", str(ds), "--pesos", str(tmp_path / "nao.pt")])
    assert r.exit_code == 2, r.output
    assert "pesos não encontrados" in r.output


def test_analyze_com_detector_grava_os_pesos_na_config(dados, foto_sintetica, modelos_falsos, tmp_path):
    import json

    pesos = tmp_path / "best.pt"
    pesos.write_bytes(b"ajustado")

    r = runner.invoke(app, ["analyze", str(foto_sintetica[0]), *BASE, "--detector", str(pesos)])

    assert r.exit_code == 0, r.output
    run_dir = next(paths.runs_dir().iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    assert manifest["config"]["detector_pesos"] == str(pesos.resolve())


def test_analyze_detector_erros(dados, foto_sintetica, tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    casos = [
        (["analyze", str(foto_sintetica[0]), *BASE, "--detector", str(tmp_path / "nao.pt")],
         "pesos não encontrados"),
        (["analyze", "--run", "x", "--from", "jersey", "--detector", str(foto_sintetica[0])],
         "--detector só vale para análise nova"),
    ]
    for args, trecho in casos:
        r = runner.invoke(app, args)
        assert r.exit_code == 2, (args, r.output)
        assert trecho in r.output, (args, r.output)
