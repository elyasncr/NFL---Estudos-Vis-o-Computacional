# Avaliação — detector ajustado (player-v1)

Data: 2026-10-04 · Código: branch `feat/detector-ajustado` (commit `127b84b` no treino) · Hardware: RTX 5070 Ti 16 GB

Resultados brutos (fora do git): `data/avaliacoes/detect-20261004-133707.json` (COCO e RF-DETR), `detect-20261004-133747.json` (player-v1), `detect-20261004-135206.json` (player-v2-freeze10), todos com limiar 0,01; os mesmos preditores com limiar 0,25 em `detect-20261004-133504.json` e `detect-20261004-133540.json`. Treinos em `data/treinos/player-v1/` e `data/treinos/player-v2-freeze10/` (com `manifest.json`).

## Resumo

**O ajuste fino não superou o YOLO11m COCO no jogo separado.** O melhor detector continua sendo o **RF-DETR base COCO, sem ajuste**. A decisão é manter o `yolo11m.pt` COCO como padrão do pipeline e não recomendar o `player-v1`.

## Dados

Dataset `data/datasets/treino-player-v1/` (`nfl-vision treino preparar`):

| Fonte | Decisão da triagem | train | valid | test |
| --- | --- | --- | --- | --- |
| base (fork `nfl-player-model-ymsui` v1) | clipes PHI×DAL e TB×ATL no treino; TB×ATL All-22 na validação; CIN×CLE no teste | 180 imagens / 2.974 caixas | 56 / 812 | 101 / 1.377 |
| `fhtw` (american-football-player-detection v3) | aprovada: futebol americano europeu, câmera alta de estádio, caixas coerentes | 171 / 3.369 | — | — |
| `evzn` (american-football-analyst v4) | aprovada: transmissões da NFL, caixas coerentes; inclui jogos do CIN que não são o do teste | 207 / 2.086 | — | — |
| `pitchcamera` (football-player-referee v7) | rejeitada: é soccer | — | — | — |

- Descartados na conversão: 668 árbitros, 561 caixas de números e "whitehat" e 124 bolas.
- Os externos vão inteiros para o treino. Na primeira montagem, 15% deles iam para a validação, mas a revisão mostrou que quadros vizinhos do mesmo vídeo caíam em train e valid.
- **Ressalva:** o teste é **uma jogada de um jogo** (CIN×CLE, semana 1 de 2025, 101 imagens em 2 câmeras). Serve para comparar modelos, não para fixar a qualidade absoluta.
- **Limitação descoberta depois do treino:** a validação (TB×ATL All-22) é a mesma jogada de um clipe de treino (TB×ATL na transmissão), em outra câmera. A validação ficou inflada (≈0,87) e a parada antecipada escolheu o modelo por um sinal enganoso.

## Treino

| | player-v1 | player-v2-freeze10 |
| --- | --- | --- |
| Diferença | — | backbone congelado (`freeze=10`) |
| Épocas até a parada | 56 (melhor checkpoint: época 36) | 50 |
| Duração | 21 min | 13 min |
| mAP@0.5 na validação | 0,867 | 0,871 |
| sha256 do `best.pt` | `88b9c6370afa…` | `9d57045c7ff8…` |

Parâmetros: YOLO11m COCO, `imgsz` 1280, `batch` 4, AMP, `single_cls`, `patience` 20, `seed` 0.

O primeiro disparo usou `batch=-1` (AutoBatch). No Windows, o AutoBatch mediu errado por causa da memória compartilhada da GPU e escolheu batch 1. Com batch 8 a VRAM transbordou para a RAM e cada época levou ~130 s; com batch 4 caiu para ~22 s. O padrão do comando passou a ser 4.

## Resultado no jogo separado (mAP@0.5, limiar 0,01)

| Preditor | Pós-processamento | mAP@0.5 | Pergunta que responde |
| --- | --- | --- | --- |
| RF-DETR base (COCO, pessoa) | nenhum | **0,663** | melhor base genérica |
| YOLO11m bruto (COCO, pessoa) | nenhum | 0,611 | ponto de partida |
| YOLO11m player-v1 | nenhum | 0,606 | ganho do treino |
| YOLO11m player-v2-freeze10 | nenhum | 0,602 | treino sem alterar o backbone |
| YOLO11m player-v1 + filtros | filtros de campo + árbitro | 0,598 | filtros depois do treino |
| YOLO11m COCO + filtros (pipeline atual) | filtros de campo + árbitro | 0,596 | pipeline atual |

Com limiar 0,25 (o valor usado pelo pipeline), o ajustado cai mais: 0,515 contra 0,585 do COCO. O modelo ajustado dá notas de confiança mais baixas e perde detecções boas nesse corte. Por isso a comparação principal usa 0,01, que é a convenção do mAP.

Por tipo de câmera (sem filtros):

| | All-22 (aérea) | Transmissão |
| --- | --- | --- |
| COCO, limiar 0,25 | 0,457 | 0,740 |
| player-v1, limiar 0,25 | 0,484 | 0,574 |
| COCO, limiar 0,01 | 0,481 | 0,768 |
| player-v1, limiar 0,01 | **0,536** | 0,727 |

O ajuste melhora a câmera aérea e piora a transmissão. As duas mudanças se anulam no total.

A meta do SDD (≥ 0,85) não foi atingida por nenhum preditor neste teste. O teste com CIN×CLE é mais difícil que o da linha de base: a metade All-22 tem jogadores muito pequenos.

## Foto real (CIN×CLE, quadros 3 e 4 da transmissão)

- **Plano fechado:** o ajustado deixou de marcar o segurança da lateral como jogador; o COCO marcava. O #43 (Mohamoud Diabate, LB) continua identificado. Apareceu uma caixa duplicada sobre um jogador do CLE.
- **Plano aberto:** o ajustado listou 15 pessoas contra 20 do COCO. Ele perde jogadores amontoados na linha de scrimmage. O árbitro continua aparecendo como jogador nos dois.

## Decisão e próximos passos

- **Padrão do pipeline:** continua `yolo11m.pt` COCO. O `player-v1` fica disponível com `--detector`, sem recomendação.
- **Ganho mais barato e imediato:** trocar o detector do pipeline pelo **RF-DETR base COCO** (+0,05 de mAP no jogo separado, sem treino).
- **Para o ajuste fino valer a pena, falta variedade de dados**, não épocas nem parâmetros. Hoje o treino tem 3 jogadas da NFL e 2 fontes externas. Seria preciso:
  - quadros de dezenas de jogos diferentes, anotados (por exemplo, pelo auto-label do Roboflow com revisão);
  - validação feita com jogos que não estão no treino, nem em outra câmera.
- Com esse conjunto, repetir a comparação ajustando também o RF-DETR.
