import json

import pytest
from pydantic import ValidationError

from nfl_vision.schemas import Analise, Contexto, Jogador, NumeroDet


def _jogador(**kwargs):
    base = dict(
        track_id=1, time="KC", numero=87, confianca_numero=0.9,
        posicao="TE", nome="Fulano", frames_visiveis=[0],
        corrigido_pelo_usuario=False,
    )
    base.update(kwargs)
    return Jogador(**base)


def test_analise_serializa_no_formato_do_sdd():
    analise = Analise(
        analise_id="2026-10-03-001",
        midia={"tipo": "foto", "largura": 1280, "altura": 720},
        contexto=Contexto(temporada=2025, semana=11, times=("KC", "BUF")),
        modelos={"detector": "yolo11m", "ocr": "paddleocr"},
        jogadores=[
            Jogador(
                track_id=7, time="KC", numero=87, confianca_numero=0.91,
                posicao="TE", nome="Fulano", frames_visiveis=[0],
                corrigido_pelo_usuario=False,
            )
        ],
    )
    dados = json.loads(analise.model_dump_json())
    assert dados["contexto"] == {"temporada": 2025, "semana": 11, "times": ["KC", "BUF"]}
    assert set(dados["jogadores"][0]) == {
        "track_id", "time", "numero", "confianca_numero", "posicao", "nome",
        "frames_visiveis", "corrigido_pelo_usuario",
    }
    assert Analise.model_validate_json(analise.model_dump_json()) == analise


def test_numero_fora_de_0_a_99_e_rejeitado():
    with pytest.raises(ValidationError):
        NumeroDet(det_id=1, numero=100, confianca=0.9)


def test_contexto_temporada_minima():
    with pytest.raises(ValidationError):
        Contexto(temporada=2001, semana=1, times=("KC", "BUF"))


def test_contexto_semana_fora_do_intervalo():
    with pytest.raises(ValidationError):
        Contexto(temporada=2025, semana=23, times=("KC", "BUF"))
    with pytest.raises(ValidationError):
        Contexto(temporada=2025, semana=0, times=("KC", "BUF"))


def test_contexto_times_iguais_e_invalido():
    with pytest.raises(ValidationError):
        Contexto(temporada=2025, semana=11, times=("KC", "KC"))


def test_jogador_numero_e_obrigatorio_mas_aceita_null():
    with pytest.raises(ValidationError):
        Jogador(
            track_id=1, time="KC", confianca_numero=0.5,
            posicao=None, nome=None, frames_visiveis=[0],
            corrigido_pelo_usuario=False,
        )
    jogador = _jogador(numero=None)
    assert jogador.numero is None


def test_jogador_numero_fora_de_0_a_99_e_rejeitado():
    with pytest.raises(ValidationError):
        _jogador(numero=150)
