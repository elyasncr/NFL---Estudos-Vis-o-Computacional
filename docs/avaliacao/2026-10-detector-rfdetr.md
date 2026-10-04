# Avaliação — RF-DETR como detector padrão

Data: 2026-10-04 · Código: branch `feat/detector-rfdetr` (commit `9f7e3da`) · Hardware: RTX 5070 Ti 16 GB

Resultados brutos (fora do git): `data/avaliacoes/medicao-rfdetr-resolucao.json` (escolha de resolução e limiar), `data/avaliacoes/detect-20261004-151412.json` (RF-DETR com e sem filtros, YOLO bruto) e `detect-20261004-151508.json` (YOLO com filtros), todos no split `test` de `data/datasets/treino-player-v1` (CIN×CLE, 101 quadros, jogo fora de qualquer treino), limiar 0,01.

## Resumo

O pipeline passa a usar o **RF-DETR base COCO em 1120 px com limiar 0,4**. No jogo separado, o mAP@0.5 do pipeline completo (detector + filtros) **sobe de 0,596 para 0,663** (+0,067), sem nenhum treino.

## Escolha da resolução

| Resolução | mAP@0.5 sem filtros | mAP@0.5 com filtros | Tempo por imagem |
| --- | --- | --- | --- |
| 560 | 0,663 | 0,648 | 37 ms |
| 896 | 0,667 | 0,652 | 38 ms |
| **1120** | **0,679** | **0,663** | 60 ms |

Regra da spec: maior mAP sem filtros; diferenças abaixo de 0,01 contam como empate e ficam com a menor resolução. O 1120 supera o 896 por 0,0115, então fica o 1120. Os 60 ms por imagem não pesam num pipeline offline.

## Escolha do limiar (1120 px, com filtros, IoU ≥ 0,5)

| Limiar | Precisão | Revocação | F1 |
| --- | --- | --- | --- |
| 0,2 | 0,426 | 0,757 | 0,545 |
| 0,3 | 0,556 | 0,728 | 0,630 |
| **0,4** | **0,667** | **0,656** | **0,662** |
| 0,5 | 0,789 | 0,552 | 0,650 |
| 0,6 | 0,907 | 0,395 | 0,550 |

O limiar 0,4 maximiza o F1 e passa a ser o padrão do RF-DETR. O `detector_conf` pode ser definido para qualquer um dos dois detectores, mas cada um mantém seu próprio padrão quando ele não é definido (método `limiar()`): o YOLO (`--detector-tipo yolo`) continua com 0,25, o padrão de antes do RF-DETR.

## Comparação no jogo separado (limiar 0,01)

| Preditor | Pós-processamento | mAP@0.5 |
| --- | --- | --- |
| RF-DETR 1120 | nenhum | 0,678 |
| **RF-DETR 1120 + filtros (novo padrão)** | filtros de campo + árbitro | **0,663** |
| YOLO11m COCO | nenhum | 0,611 |
| YOLO11m COCO + filtros (padrão anterior) | filtros de campo + árbitro | 0,596 |

Os filtros custam cerca de 0,015 de mAP com os dois detectores. O custo vem de pessoas na lateral que o dataset rotula como jogador e que o filtro descarta de propósito. Por isso eles continuam ligados.

A meta do SDD (≥ 0,85) segue fora de alcance neste teste. A metade All-22, com jogadores muito pequenos, é a parte difícil.

## Foto real (CIN×CLE, quadros 3 e 4)

- **Plano fechado:** o #43 continua identificado (Mohamoud Diabate, LB, confiança 0,89) e as caixas ficaram mais justas. O recebedor do CIN ficou sem time: a caixa mais justa mudou o recorte do tronco e a confiança da cor caiu abaixo do limiar. Duas pessoas da equipe na lateral foram detectadas.
- **Plano aberto:** 15 pessoas listadas contra 20 do YOLO. Com limiar 0,4, o detector perde jogadores amontoados na linha. É a troca de revocação por precisão escolhida pelo F1.

## Próximos passos

- Ajustar o recorte do tronco à geometria das caixas do RF-DETR, que são mais justas que as do YOLO, e medir a acurácia de time.
- O `RFDETRBase` está marcado como obsoleto no rfdetr 1.11 e sai na 2.0. Migrar para a classe nova antes de atualizar o pacote.
- Para chegar perto de 0,85: ajuste fino do RF-DETR com dados de muitos jogos diferentes (ver `2026-10-detector-ajustado.md`).
