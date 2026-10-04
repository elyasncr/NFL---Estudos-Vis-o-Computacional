import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from nfl_vision.treino import preparar
from nfl_vision.treino.fontes import BASE, EXTERNAS, POR_NOME, MapeamentoInvalido, clipe_base
from nfl_vision.treino.preparar import SPLITS, Decisao, construir, converter, split_externo
from treino_sintetico import NOMES_EXTERNOS, dataset_base, dataset_externo

PITCH = POR_NOME["pitchcamera"]


def test_converter_mapeia_player_e_descarta_o_resto(tmp_path):
    rotulos = tmp_path / "a.txt"
    rotulos.write_text("2 0.5 0.5 0.2 0.4\n1 0.3 0.3 0.1 0.1\n0 0.7 0.7 0.02 0.02\n"
                       "2 0.1 0.2 0.3 0.2 0.3 0.6 0.1 0.6\n")

    linhas, descartadas = converter(rotulos, NOMES_EXTERNOS["pitchcamera"], PITCH)

    assert linhas == ["0 0.500000 0.500000 0.200000 0.400000",
                      "0 0.200000 0.400000 0.200000 0.400000"]
    assert descartadas == Counter({"referee": 1, "football": 1})


def test_converter_corta_caixa_que_sai_da_imagem(tmp_path):
    rotulos = tmp_path / "a.txt"
    rotulos.write_text("2 0.95 0.5 0.2 0.2\n")
    linhas, _ = converter(rotulos, NOMES_EXTERNOS["pitchcamera"], PITCH)
    assert linhas == ["0 0.925000 0.500000 0.150000 0.200000"]


def test_converter_sem_arquivo_de_rotulos(tmp_path):
    assert converter(tmp_path / "nao.txt", ["players"], PITCH) == ([], Counter())


def test_converter_classe_numerica_no_yaml(tmp_path):
    # names: [..., 1, 2, ...] sem aspas vira int no YAML
    rotulos = tmp_path / "a.txt"
    rotulos.write_text("2 0.5 0.5 0.1 0.1\n7 0.5 0.5 0.2 0.4\n")
    nomes = ["ball", "referee", 1, 2, 3, 4, 5, "american-football-players", "whitehat"]
    linhas, descartadas = converter(rotulos, nomes, POR_NOME["fhtw"])
    assert len(linhas) == 1 and descartadas == Counter({"1": 1})


def test_split_externo_reprodutivel_e_agrupa_copias():
    nomes = [f"img{i:04d}_png.rf.{c}.jpg" for i in range(400) for c in ("aa", "bb")]
    a = [split_externo("evzn", n, 0, fracao=0.15) for n in nomes]
    assert a == [split_externo("evzn", n, 0, fracao=0.15) for n in nomes]
    assert all(a[k] == a[k + 1] for k in range(0, len(a), 2))  # cópias aumentadas juntas
    assert 0.10 < a.count("valid") / len(a) < 0.20
    assert a != [split_externo("evzn", n, 1, fracao=0.15) for n in nomes]


def test_externas_vao_inteiras_para_o_treino_por_padrao():
    # quadros vizinhos do mesmo vídeo vazariam entre train e valid; a validação fica só com a base
    nomes = [f"{i}_jpg.rf.x.jpg" for i in range(200)]
    assert {split_externo("fhtw", n, 0) for n in nomes} == {"train"}


def _decisoes(datasets: Path, aprovadas=("pitchcamera", "fhtw", "evzn")) -> list[Decisao]:
    lista = [Decisao(BASE, True, "base", dataset_base(datasets))]
    for f in EXTERNAS:
        raiz = dataset_externo(datasets, f)
        aprovada = f.nome in aprovadas
        lista.append(Decisao(f, aprovada, "ok" if aprovada else "soccer", raiz if aprovada else None))
    return lista


def _arquivos(saida: Path, split: str, sub: str = "images") -> list[str]:
    return sorted(p.name for p in (saida / split / sub).iterdir())


def test_construir_separa_a_base_por_clipe(tmp_path):
    saida = tmp_path / "saida"
    construir(saida, _decisoes(tmp_path / "datasets"))

    teste, valid, treino = (_arquivos(saida, s) for s in ("test", "valid", "train"))
    assert len(teste) == 4 and all(n.startswith("base__cin_cle_") for n in teste)
    assert not any("cin_cle_" in n for n in valid + treino)
    base_valid = [n for n in valid if n.startswith("base__")]
    assert len(base_valid) == 2
    assert all(clipe_base(n.removeprefix("base__")) == "tb_atl_wk1_penix_pass_all22" for n in base_valid)
    assert {clipe_base(n.removeprefix("base__")) for n in treino if n.startswith("base__")} == {
        "tb_atl_wk1_penix_pass", "phi_dal_wk1_williams_run", "phi_dal_wk1_williams_run_endzone"}
    assert not any(n.split("__")[0] in {"pitchcamera", "fhtw", "evzn"} for n in teste)


def test_construir_converte_prefixa_e_conta(tmp_path):
    saida = tmp_path / "saida"
    manifest = construir(saida, _decisoes(tmp_path / "datasets"))

    todas = [n for s in SPLITS for n in _arquivos(saida, s)]
    assert all(n.split("__")[0] in {"base", "pitchcamera", "fhtw", "evzn"} for n in todas)
    assert not any("so_arbitro" in n for n in todas)  # sem player fica fora
    for s in SPLITS:
        assert [Path(n).stem for n in _arquivos(saida, s)] == \
            [Path(n).stem for n in _arquivos(saida, s, "labels")]
        for rotulo in (saida / s / "labels").iterdir():
            assert all(linha.split()[0] == "0" for linha in rotulo.read_text().splitlines())
    assert yaml.safe_load((saida / "data.yaml").read_text("utf-8"))["names"] == ["player"]

    assert manifest["descartadas"]["base"] == {"ball": 12, "referee": 12}
    # 40 cópias com 1 árbitro cada + a imagem só com árbitro
    assert manifest["descartadas"]["pitchcamera"] == {"referee": 41}
    assert manifest["sem_player"] == {"base": 0, "pitchcamera": 1, "fhtw": 1, "evzn": 1}
    assert manifest["contagens"]["test"] == {"base": {"imagens": 4, "caixas": 4}}
    for s in SPLITS:
        assert sum(c["imagens"] for c in manifest["contagens"][s].values()) == len(_arquivos(saida, s))
    for f in EXTERNAS:
        total = sum(manifest["contagens"][s].get(f.nome, {"imagens": 0})["imagens"]
                    for s in ("train", "valid"))
        assert total == 41


def test_manifest_registra_decisoes(tmp_path):
    saida = tmp_path / "saida"
    manifest = construir(saida, _decisoes(tmp_path / "datasets", aprovadas=("evzn",)))

    por_nome = {f["nome"]: f for f in manifest["fontes"]}
    assert list(por_nome) == ["base", "pitchcamera", "fhtw", "evzn"]
    assert por_nome["fhtw"]["aprovada"] is False and por_nome["fhtw"]["motivo"] == "soccer"
    assert por_nome["fhtw"]["pasta"] is None
    assert por_nome["evzn"]["aprovada"] is True
    assert (por_nome["evzn"]["workspace"], por_nome["evzn"]["projeto"], por_nome["evzn"]["versao"]) == \
        ("evzn", "american-football-analyst", 4)
    assert por_nome["evzn"]["mapeamento"] == {"football-players": "player"}
    assert all("fhtw" not in c and "pitchcamera" not in c for c in manifest["contagens"].values())
    assert manifest["semente"] == 0 and manifest["versao"]
    assert json.loads((saida / "manifest.json").read_text("utf-8")) == manifest


def test_construir_reprodutivel(tmp_path):
    decisoes = _decisoes(tmp_path / "datasets")
    construir(tmp_path / "a", decisoes)
    construir(tmp_path / "b", decisoes)
    for s in SPLITS:
        assert _arquivos(tmp_path / "a", s) == _arquivos(tmp_path / "b", s)


def test_construir_recusa_dataset_existente(tmp_path):
    decisoes = _decisoes(tmp_path / "datasets")
    construir(tmp_path / "saida", decisoes)
    with pytest.raises(FileExistsError, match="já tem um dataset"):
        construir(tmp_path / "saida", decisoes)


def test_construir_aceita_pasta_so_com_triagem(tmp_path):
    (tmp_path / "saida" / "triagem").mkdir(parents=True)
    construir(tmp_path / "saida", _decisoes(tmp_path / "datasets"))
    assert (tmp_path / "saida" / "data.yaml").exists()


def test_construir_exige_base(tmp_path):
    with pytest.raises(ValueError, match="base"):
        construir(tmp_path / "saida", _decisoes(tmp_path / "datasets")[1:])


def test_construir_mapeamento_invalido_nao_grava_nada(tmp_path):
    datasets = tmp_path / "datasets"
    evzn = POR_NOME["evzn"]
    lista = [Decisao(BASE, True, "base", dataset_base(datasets)),
             Decisao(evzn, True, "ok", dataset_externo(datasets, evzn, nomes=["ball", "referee", "players"]))]
    with pytest.raises(MapeamentoInvalido, match="football-players"):
        construir(tmp_path / "saida", lista)
    assert not (tmp_path / "saida" / "train").exists()


def test_painel_de_triagem(tmp_path):
    import cv2
    import numpy as np

    raiz = dataset_externo(tmp_path / "datasets", PITCH)
    destino = tmp_path / "triagem" / "pitchcamera.jpg"

    n = preparar.painel(PITCH, raiz, destino)

    assert n == 42
    img = cv2.imdecode(np.fromfile(str(destino), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (3 * 320, 4 * 480, 3)
    assert (img[..., 1] > 180).any()  # caixas de player em verde


def test_triagem_gera_um_painel_por_fonte(tmp_path):
    datasets = tmp_path / "datasets"
    pares = [(f, dataset_externo(datasets, f)) for f in EXTERNAS]

    resumo = preparar.triagem(pares, tmp_path / "saida")

    assert [r["fonte"] for r in resumo] == ["pitchcamera", "fhtw", "evzn"]
    for r in resumo:
        assert r["imagens"] == 42
        assert r["painel"] == tmp_path / "saida" / "triagem" / f"{r['fonte']}.jpg"
        assert r["painel"].exists()
