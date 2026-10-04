# Linha de base — pipeline em fotos

Data: 2026-10-04 · Código: commit `6ed67ff` (branch `feat/pipeline-fotos`) · Hardware: RTX 5070 Ti (detecção), CPU (OCR)

Resultados brutos em `data/avaliacoes/detect-20261004-002931.json` e `data/avaliacoes/jersey-20261004-002217.json` (fora do git).

## Detecção

Dataset: fork `elyas-carvalho/nfl-player-model-ymsui` v1 do Universe `nflplayerdetection-mjrl1/nfl-player-model` (CC BY 4.0), só auto-orientação, split de **teste com 34 imagens** (quadros de transmissão, All-22 e end zone de três jogos da semana 1 de 2025). Métrica: mAP@0.5 da classe `player`, limiar de confiança 0,25 para todos.

| Preditor | Pós-processamento | mAP@0.5 |
| --- | --- | --- |
| nosso (yolo11m + filtros + árbitro) | filtros de campo + remoção de árbitro | 0,748 |
| yolo11m bruto (COCO, pessoa) | nenhum | 0,757 |
| rf-detr base (COCO, pessoa) | nenhum | 0,787 |
| roboflow `nfl-player-model/4` | modelo treinado em NFL (classe player) | 0,844 |

No split de validação (67 imagens): nosso 0,777 · yolo11m bruto 0,786.

Meta do SDD: ≥ 0,85. **Ainda não atingida.**

Como ler a tabela:

- **Arquitetura** (yolo-bruto × rf-detr): o RF-DETR base ganha 0,03 sem nenhum ajuste.
- **Efeito dos filtros** (nosso × yolo-bruto): −0,009. Os descartes restantes que "custam" mAP foram inspecionados um a um: são cinegrafista, pessoas sentadas na lateral e um árbitro na lateral que o dataset rotula como `player`. A primeira versão do filtro (faixa de gramado abaixo dos pés) custava −0,142 e foi trocada pela regra da região do campo.
- **Modelo treinado em NFL**: 0,844 está **inflado**. Os 6 clipes do split de teste também aparecem no de treino (quadros amostrados a cada 15 frames das mesmas jogadas), então o modelo do Roboflow, treinado no split de treino, viu cenas praticamente iguais às de teste. O nosso pipeline não foi treinado nesse dataset, então não tem esse vazamento.
- **Árbitros**: a coluna "árbitros cobertos por caixa de jogador" deu 100% para todos e não é informativa. O teste tem 1 caixa de árbitro (de 513), e o dataset rotula vários árbitros como `player`. O teste de listras não remove árbitros em imagens reais (pose, compressão, branco que vira cinza). Ver "Limitações conhecidas" na spec.

## Número da camisa (OCR)

Dataset: `taiseis-workspace/jersey-number-ijbaq` v1 (CC BY 4.0), recortes de camisa de **outros esportes**, split de teste. É uma aproximação até existirem recortes próprios de NFL.

| Métrica | Valor |
| --- | --- |
| Recortes avaliados (números legíveis em NFL) | 396 (39 excluídos: "00", "07" etc.) |
| Acurácia entre os lidos | **0,902** |
| Taxa de desconhecido | 0,303 |
| Acurácia geral | 0,629 |

Meta do SDD: ≥ 0,80 entre os legíveis. **Atingida** na leitura, com 30% marcados como desconhecido — o comportamento de "precisão antes de cobertura" pedido no SDD.

## Ponta a ponta (foto real)

Quadros da transmissão de CIN × CLE, temporada 2025, semana 1 (do próprio dataset), com `--times CIN CLE --temporada 2025 --semana 1`:

- **Plano fechado** (2 quadros): o #43 do CLE foi identificado como **Mohamoud Diabate (LB)**, com confiança 0,99–1,00. Os jogadores de costas ou com o número escondido ficaram como desconhecidos. Os times estão corretos. Um segurança em pé na lateral foi marcado como "CLE ?".
- **Plano aberto** (1 quadro, 20 pessoas): os times estão corretos para os jogadores, inclusive sobre a pintura da end zone. Nenhum número foi lido, porque os jogadores têm cerca de 40 px de altura. O árbitro listrado foi marcado como "CLE ?".

## Próximos passos sugeridos

1. Ajuste fino do detector com as classes `player` e `referee` (resolve árbitro e lateral), avaliado num split sem quadros vizinhos das mesmas jogadas.
2. Conjunto próprio de avaliação de ponta a ponta (10–20 capturas de jogos conhecidos rotuladas à mão) e o teste golden do `analise.json`.
3. Vídeo (sub-projeto 2): o voto entre frames deve aumentar bastante a cobertura de números.
