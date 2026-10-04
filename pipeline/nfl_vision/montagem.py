"""Monta o analise.json (formato do SDD) a partir das saídas das etapas."""

from pathlib import Path

from nfl_vision.eval.preditores import rotulo_pesos
from nfl_vision.schemas import Analise, Jogador
from nfl_vision.stages.roster import aplicar_correcoes


def montar(estado) -> Analise:
    ingest = estado.saidas["ingest"]
    numeros = {n.det_id: n for n in estado.saidas["jersey"].itens}
    roster = {r.det_id: r for r in estado.saidas["roster"].itens}
    correcoes = estado.correcoes()
    time_final, numero_final = aplicar_correcoes(
        estado.saidas["team"].itens, numeros.values(), correcoes
    )
    corrigidos = {c.det_id for c in correcoes}
    numero_corrigido = {c.det_id for c in correcoes if c.numero is not None}

    jogadores = []
    for det_id in sorted(time_final):
        leitura, r = numeros.get(det_id), roster.get(det_id)
        if det_id in numero_corrigido:
            conf = 1.0
        else:
            conf = leitura.confianca if leitura else 0.0
        jogadores.append(Jogador(
            track_id=det_id,
            time=time_final[det_id],
            numero=numero_final.get(det_id),
            confianca_numero=conf,
            posicao=r.posicao if r else None,
            nome=r.nome if r else None,
            frames_visiveis=[0],
            corrigido_pelo_usuario=det_id in corrigidos,
        ))

    return Analise(
        analise_id=estado.run_dir.name,
        midia={"tipo": "foto", "largura": ingest.largura, "altura": ingest.altura},
        contexto=estado.contexto,
        modelos={"detector": rotulo_pesos(estado.config.detector_pesos), "ocr": "paddleocr"},
        jogadores=jogadores,
    )
