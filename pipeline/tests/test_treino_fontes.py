import pytest

from nfl_vision.treino import fontes
from nfl_vision.treino.fontes import (
    BASE, EXTERNAS, POR_NOME, FonteIndisponivel, MapeamentoInvalido, clipe_base, split_base,
    validar_mapeamento,
)


def test_clipe_base():
    assert clipe_base("cin_cle_wk1_burrow_td_broadcast_00015_jpg.rf.abc123.jpg") == \
        "cin_cle_wk1_burrow_td_broadcast"
    assert clipe_base("tb_atl_wk1_penix_pass_all22_00000_jpg.rf.f0.jpg") == "tb_atl_wk1_penix_pass_all22"


def test_clipe_base_fora_do_padrao():
    with pytest.raises(ValueError, match="padrão"):
        clipe_base("foto_qualquer.jpg")


@pytest.mark.parametrize("clipe, split", [
    ("cin_cle_wk1_burrow_td_broadcast", "test"),
    ("cin_cle_wk1_burrow_td_all22", "test"),
    ("tb_atl_wk1_penix_pass_all22", "valid"),
    ("tb_atl_wk1_penix_pass", "train"),
    ("phi_dal_wk1_williams_run", "train"),
    ("phi_dal_wk1_williams_run_endzone", "train"),
])
def test_split_base(clipe, split):
    assert split_base(f"{clipe}_00030_jpg.rf.deadbeef.jpg") == split


def test_fontes_declaradas():
    assert [f.nome for f in EXTERNAS] == ["pitchcamera", "fhtw", "evzn"]
    assert POR_NOME["base"] is BASE and not BASE.externa
    assert all(f.externa for f in EXTERNAS)
    assert {f.nome: f.classes_player for f in EXTERNAS} == {
        "pitchcamera": ("players",),
        "fhtw": ("american-football-players",),
        "evzn": ("football-players",),
    }


def test_pasta_usa_o_nome_do_download(tmp_path):
    assert BASE.pasta(tmp_path) == tmp_path / "nfl-player-model-ymsui-v1-yolov11"


def test_validar_mapeamento():
    validar_mapeamento(POR_NOME["evzn"], ["ball", "referee", "football-players"])
    with pytest.raises(MapeamentoInvalido) as exc:
        validar_mapeamento(POR_NOME["evzn"], ["ball", "referee", "players"])
    msg = str(exc.value)
    assert "evzn" in msg and "football-players" in msg and "ball, referee, players" in msg


def test_obter_erro_de_download_cita_a_fonte(tmp_path, monkeypatch):
    def quebrar(*a, **k):
        raise ConnectionError("timeout")

    monkeypatch.setattr(fontes, "baixar", quebrar)
    with pytest.raises(FonteIndisponivel) as exc:
        fontes.obter(POR_NOME["pitchcamera"], tmp_path)
    msg = str(exc.value)
    assert "pitchcamera" in msg and "ConnectionError" in msg and "timeout" in msg


def test_obter_sem_chave(tmp_path, monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    with pytest.raises(FonteIndisponivel, match="ROBOFLOW_API_KEY"):
        fontes.obter(POR_NOME["fhtw"], tmp_path)


def test_obter_reusa_download(tmp_path, monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    BASE.pasta(tmp_path).mkdir()
    assert fontes.obter(BASE, tmp_path) == BASE.pasta(tmp_path)
