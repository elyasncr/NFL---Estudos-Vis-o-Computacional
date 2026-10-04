# Design — RF-DETR como detector padrão

2026-10-04 · @elyas · Referências: `docs/superpowers/specs/2026-10-03-pipeline-fotos-design.md` (§6 detect), `docs/avaliacao/2026-10-detector-ajustado.md`

## 1. Contexto e objetivo

No jogo separado (CIN×CLE, 101 quadros, split `test` de `data/datasets/treino-player-v1`), o RF-DETR base COCO sem ajuste teve mAP@0.5 0,663, contra 0,611 do YOLO11m COCO e 0,606 do YOLO11m ajustado (limiar 0,01). Este sub-projeto torna o RF-DETR o detector padrão da etapa `detect`, mantendo o YOLO como opção, e escolhe por medição a resolução e o limiar de confiança.

**Fora do escopo:** ajuste fino do RF-DETR, classe árbitro, vídeo.

## 2. Configuração

`Config` ganha:

| Campo | Padrão | Significado |
| --- | --- | --- |
| `detector_tipo` | `"rfdetr"` | `"rfdetr"` ou `"yolo"` |
| `detector_modelo_rfdetr` | `"base"` | variante do RF-DETR (classe do pacote `rfdetr`) |
| `detector_resolucao` | definido pela medição (§4) | lado de entrada do RF-DETR; múltiplo de 56 |

`detector_pesos`, `detector_imgsz` e `detector_conf` continuam valendo para o YOLO; `detector_conf` passa a valer para os dois (limiar de confiança do detector), com o padrão também definido pela medição. Manifests antigos (sem os campos novos) carregam com os padrões e o manifest registra a config efetiva (mecanismo atual de `config_efetiva`). Análises antigas reprocessadas com `--from detect` passam a usar o detector da config gravada nelas; como os campos novos não existiam, elas recebem o padrão novo. Isso fica registrado em `config_original`.

## 3. Etapa `detect`

- `detectar_pessoas(img, cfg)` despacha por `cfg.detector_tipo`. O caminho YOLO não muda.
- Caminho RF-DETR: carrega o modelo uma vez (cache por variante, resolução e dispositivo), converte BGR→RGB, roda `predict(..., threshold=cfg.detector_conf)` e mantém só a classe pessoa do COCO, devolvendo `Deteccao` como hoje.
- Os filtros (`pequeno`, região do campo, close) são aplicados depois, iguais para os dois detectores.
- O hash dos pesos (`pesos_sha256`) é calculado a partir do arquivo de pesos realmente carregado pelo RF-DETR.
- A lógica do RF-DETR sai de `eval/preditores.py` (`PreditorRFDETR`) e passa a ficar na etapa `detect`; o preditor de avaliação passa a chamar a função compartilhada, para avaliação e pipeline usarem o mesmo código. O mesmo vale para o preditor `nosso`, que passa a seguir `detector_tipo`.

## 4. Medição (decide os padrões)

No split `test` de `treino-player-v1`, mAP@0.5 da classe jogador com limiar 0,01 (convenção do mAP):

- RF-DETR base nas resoluções 560, 896 e 1120, sem filtros e com filtros;
- YOLO11m COCO sem filtros e com filtros como referência.

Escolhe-se a resolução com maior mAP sem filtros; empate técnico (diferença < 0,01) fica com a menor resolução, que é mais rápida. Mede-se também o tempo médio por imagem na RTX 5070 Ti.

Para o limiar do pipeline (`detector_conf`), na resolução escolhida, mede-se precisão e revocação da classe jogador (IoU ≥ 0,5) nos limiares 0,2 a 0,6. O padrão é o limiar com maior F1; se os filtros forem usados, a medição é feita com eles.

Resultado em `docs/avaliacao/2026-10-detector-rfdetr.md` e na página de resultados.

## 5. CLI

- `analyze` ganha `--detector-tipo [rfdetr|yolo]`. `--detector <arquivo.pt>` continua significando pesos YOLO e implica `--detector-tipo yolo`.
- `eval detect`: o preditor `nosso` segue a config (RF-DETR por padrão). `--benchmark yolo-bruto` e `--benchmark rfdetr` continuam. Ganha `--resolucao` para o RF-DETR (usado na medição).

## 6. Testes

- Rápidos: despacho por tipo com detectores falsos; filtro de classe pessoa e conversão de caixas do RF-DETR com um modelo falso; carga de manifest antigo sem os campos novos; CLI `--detector-tipo` e combinação com `--detector`.
- `@model`: RF-DETR detecta pessoas em `ultralytics/assets/bus.jpg` na GPU.

## 7. Critério de pronto

1. `analyze` usa o RF-DETR por padrão numa foto real, com o manifest registrando tipo, resolução e hash.
2. Medição registrada com a escolha de resolução e limiar.
3. Testes rápidos e `@model` passando.
4. Página de resultados atualizada com o RF-DETR.
