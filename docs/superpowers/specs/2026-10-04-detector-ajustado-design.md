# Design — Ajuste fino do detector de jogadores

2026-10-04 · @elyas · Referências: `SDD — Visão computacional NFL.md`, `docs/superpowers/specs/2026-10-03-pipeline-fotos-design.md`, `docs/avaliacao/2026-10-linha-de-base.md`

## 1. Contexto e objetivo

O pipeline em fotos usa o YOLO11m pré-treinado no COCO (classe pessoa). Na linha de base, o detector ficou em mAP@0.5 0,748 contra a meta de 0,85 do SDD. Este sub-projeto ajusta (fine-tuning) o YOLO11m para a classe `player` com dados de futebol americano, treinando na RTX 5070 Ti local, e mede o ganho num jogo que nunca entra no treino.

**Dentro do escopo:** preparação do dataset de treino (fontes, triagem, conversão para uma classe, split sem vazamento), comando de treino, avaliação comparativa no jogo separado, opção de usar os pesos ajustados no `analyze`.

**Fora do escopo:** classe `referee` (fica para uma rodada com anotação própria), RF-DETR ajustado, vídeo, publicação dos pesos.

## 2. Dados

### Fontes

| Fonte | Uso |
| --- | --- |
| Fork `elyas-carvalho/nfl-player-model-ymsui` v1 (YOLO) | Treino: clipes `phi_dal_*` e `tb_atl_*` (4 câmeras, 236 imagens). Teste: clipes `cin_cle_*` (2 câmeras, 102 imagens) |
| Universe `pitchcamera/football-player-referee` | Treino, se aprovado na triagem |
| Universe `fh-technikum-wien-m15r2/american-football-player-detection` | Treino, se aprovado na triagem |
| Universe `evzn/american-football-analyst` | Treino, se aprovado na triagem |

O dataset base tem só 6 clipes (3 jogadas, 2 câmeras cada). Separar por quadro vazaria quadros vizinhos da mesma jogada entre treino e teste, por isso a separação é sempre por clipe ou fonte.

### Triagem dos externos

Para cada dataset externo, o comando de preparação gera um painel com ~12 imagens e as caixas desenhadas, salvo em `data/datasets/treino-player-v1/triagem/`. O dataset só entra se for futebol americano (não soccer) e se as caixas de jogador forem coerentes. A decisão e o motivo ficam no `manifest.json` do dataset (`fontes[].aprovada`, `fontes[].motivo`). A decisão é parâmetro do comando, para ser explícita e reprodutível: `treino preparar --so-triagem` baixa as fontes e gera os painéis; depois `treino preparar --aprovar <fonte>:<motivo> --rejeitar <fonte>:<motivo>` monta o dataset, e toda fonte externa precisa ser decidida. No painel, caixas da classe mapeada para `player` aparecem em verde e as demais em cinza com o nome da classe.

### Conversão

- Cada fonte declara o mapeamento das suas classes para `player`. Exemplos: `player`, `players`, `american-football-players`, `football-players` → `player`. Todas as outras classes (bola, árbitro, números, capacetes, `whitehat` etc.) são descartadas. Descartar o árbitro é intencional: o modelo aprende a não marcá-lo como jogador.
- Linhas de polígono viram caixa (mín./máx. das coordenadas), com a mesma função de `eval/datasets.py` (`ler_rotulos`). Caixas são cortadas aos limites da imagem; as degeneradas são descartadas e contadas.
- Imagens sem nenhuma caixa de `player` após a conversão ficam fora.
- Nomes de arquivo recebem o prefixo da fonte para não colidir.

### Splits

- `test`: só os clipes `cin_cle_*` do fork. Nunca usados no treino nem na validação.
- `valid`: o clipe `tb_atl_wk1_penix_pass_all22` inteiro mais 15% de cada externo aprovado (sorteio com semente fixa por imagem original: as cópias aumentadas que o Roboflow exporta como `<original>.rf.<hash>` ficam no mesmo split; esses datasets não têm identificador de clipe).
- `train`: o restante.

### Saída

`data/datasets/treino-player-v1/` no formato YOLO (`train|valid|test/images|labels`, `data.yaml` com `path` absoluto e `names: ['player']`) e `manifest.json` com: fontes (workspace, projeto, versão, aprovada, motivo, mapeamento), contagens de imagens e caixas por split e fonte, caixas descartadas por classe, semente e versão do `nfl-vision`. Tudo fica fora do git (`data/`).

## 3. Treino

| Parâmetro | Valor |
| --- | --- |
| Modelo inicial | `yolo11m.pt` (COCO) |
| `imgsz` | 1280 |
| `single_cls` | true |
| `epochs` / `patience` | 100 / 20 |
| `batch` | -1 (automático, ~60% da VRAM) |
| `amp` | true |
| Augmentation | padrão do Ultralytics; `close_mosaic=10` |
| `workers` | 2 (estabilidade no Windows) |
| `seed` / `deterministic` | 0 / true |

Comando: `nfl-vision treino rodar --dataset <pasta> --nome player-v1 [--epocas N] [--imgsz N] [--retomar]`. Saída em `data/treinos/<nome>/`: `weights/best.pt`, `weights/last.pt`, curvas e `args.yaml` do Ultralytics, e `manifest.json` com o sha256 do `data.yaml` e do manifest do dataset, os parâmetros, as versões (`ultralytics`, `torch`), a GPU, a duração e o caminho e o sha256 do `best.pt`. `--retomar` continua a partir do `last.pt` com o `resume` do Ultralytics. Tempo estimado: 30–60 min.

O comando recusa um `--nome` que já existe e passa `project`/`name` absolutos com `exist_ok=True` ao Ultralytics (com `exist_ok=False` ele criaria `player-v12` em silêncio). O manifest do treino tem `status` (`em_andamento`, `concluido`, `erro`, `interrompido`), duração acumulada entre retomadas e a lista de retomadas; `--retomar` num treino concluído é recusado.

## 4. Avaliação

`eval detect` ganha `--pesos <arquivo.pt>`, que troca os pesos usados pelos preditores `nosso` e `yolo-bruto` (o nome do preditor passa a indicar os pesos, ex.: `yolo bruto (player-v1)`). No split `test` do dataset preparado, todos com limiar 0,25:

| Preditor | Pergunta que responde |
| --- | --- |
| YOLO11m COCO sem filtros | ponto de partida |
| RF-DETR COCO sem filtros | melhor base genérica |
| YOLO11m COCO + filtros | pipeline atual |
| YOLO11m ajustado sem filtros | ganho do treino |
| YOLO11m ajustado + filtros | se os filtros ainda ajudam depois do treino |

Ressalva registrada no resultado: o teste é uma jogada de um jogo; compara modelos, não fixa a qualidade absoluta.

O split `test` preparado só tem a classe `player`, então a coluna de árbitros da avaliação fica vazia; árbitro marcado como jogador aparece como falso positivo no mAP. A comparação sai em duas execuções do `eval detect` no mesmo split: uma com `--benchmark yolo-bruto --benchmark rfdetr` (pesos COCO) e outra com `--pesos <best.pt> --benchmark yolo-bruto`.

Resultado e decisão (usar ou não o ajustado como recomendado) em `docs/avaliacao/2026-10-detector-ajustado.md`.

## 5. Integração

- `analyze` ganha `--detector <caminho.pt>`, que sobrescreve `Config.detector_pesos` na análise nova; caminho e sha256 ficam no manifest (mecanismo atual).
- O padrão continua `yolo11m.pt` COCO: os pesos ajustados não vão para o git, e um clone novo precisa funcionar sem eles. O README explica como preparar, treinar e usar.
- A etapa `detect` não muda: o modelo ajustado tem a classe 0 = `player` e o código já filtra `classes=[0]`.

## 6. Estrutura

```
pipeline/nfl_vision/
  treino/
    __init__.py
    fontes.py      # declaração das fontes e mapeamentos de classe
    preparar.py    # conversão, splits, triagem, manifest
    rodar.py       # treino com Ultralytics e manifest do treino
  cli.py           # subcomandos `treino preparar`, `treino rodar`; --pesos; --detector
```

## 7. Erros

- Fonte sem acesso, chave ausente ou download falhando: erro claro com o nome da fonte, sem traceback (mesmo padrão de `eval baixar`).
- Classe declarada no mapeamento que não existe no `data.yaml` da fonte: erro listando as classes disponíveis.
- Dataset preparado sem imagens em algum split: erro antes de treinar.
- `--retomar` sem `last.pt`: erro claro.

## 8. Testes

- Rápidos, com mini-datasets sintéticos: mapeamento de classes e descarte; polígono → caixa; imagens sem `player` excluídas; split por clipe (nenhum clipe de teste em train/valid) e sorteio reprodutível dos externos; prefixo de nomes; manifest com contagens e descartes; CLI (`treino preparar` com fontes locais, erros de entrada, `--pesos`, `--detector`).
- `@model`: treino-relâmpago de 1 época em `imgsz=64` com 4 imagens sintéticas, verificando `best.pt` e o manifest do treino.

## 9. Critério de pronto

1. `treino preparar` gera o dataset e o manifest; a triagem dos externos fica registrada.
2. O treino completa e salva `best.pt` com o manifest.
3. A avaliação no jogo separado fica registrada em `docs/avaliacao/2026-10-detector-ajustado.md`, com a decisão.
4. `analyze --detector` funciona numa foto real.
5. Testes rápidos e `@model` passando.

Meta: mAP@0.5 ≥ 0,85 no jogo separado. Não atingir não bloqueia: os números ficam registrados e o próximo passo indicado (RF-DETR ajustado ou mais dados).
