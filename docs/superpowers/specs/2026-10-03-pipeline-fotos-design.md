# Design — Sub-projeto 1: pipeline de identificação em fotos

2026-10-03 · @elyas · Referência: `SDD — Visão computacional NFL.md`

## 1. Contexto e escopo

A fase 1 do SDD (identificação) foi decomposta em quatro sub-projetos, cada um com seu próprio ciclo de spec → plano → implementação:

1. **Pipeline em fotos (CLI)** — este documento.
2. Vídeo: amostragem de frames, detecção de cortes, ByteTrack, voto entre frames.
3. API + armazenamento: FastAPI, jobs em segundo plano, SQLite.
4. Web: as quatro telas e o design system.

**Dentro do escopo:** ingestão de foto (JPG, PNG), detecção, filtro de fora de campo, identificação de time e de árbitro, leitura de número, consulta ao roster, imagem anotada, `analise.json`, correção manual via CLI, scripts de avaliação de detecção e de número.

**Fora do escopo:** vídeo e tracking, API, front-end, ajuste fino de modelos, modelo dedicado de números.

Requisitos do SDD cobertos: RF01 (só foto), RF02, RF03, RF04, RF05, RF06, RF07, RF08 (imagem + lista), RF09 (via CLI).

## 2. Ambiente

- Windows 11, GPU NVIDIA RTX 5070 Ti (Blackwell, sm_120).
- Python 3.12, gerenciado com **uv**.
- **PyTorch com CUDA ≥ 12.8** (obrigatório para sm_120).
- PaddleOCR em GPU se a instalação do PaddlePaddle-GPU funcionar no Windows com a placa; caso contrário, em CPU (recortes pequenos, custo baixo). O dispositivo do OCR é um parâmetro de config.
- Colab continua disponível para treino e avaliação pesada; o pacote é instalável lá via `pip install`.

## 3. Estrutura

```
nfl-vision/                 (raiz do repositório)
  pipeline/
    pyproject.toml
    nfl_vision/
      schemas.py            # contratos Pydantic entre etapas
      runner.py             # executa etapas, grava artefatos, retoma com --from
      config.py             # parâmetros; gravados em cada análise
      cli.py                # comando `nfl-vision`
      render.py             # imagem anotada
      teams.py              # siglas e cores oficiais (nflverse)
      stages/
        ingest.py  detect.py  team.py  jersey.py  roster.py
      eval/
        detect.py  jersey.py
  notebooks/                # exploração e avaliação; importam o pacote
  tests/
    fixtures/               # imagens sintéticas, parquet de roster
  data/                     # ignorado pelo git
    runs/<analise_id>/
    cache/
    datasets/
  docs/
```

Notebooks nunca duplicam lógica do pacote.

## 4. Contratos de dados

Cada etapa é uma função `run(ctx, inputs, config) -> Output`, sem efeitos colaterais além de ler a imagem e o cache. Toda detecção tem um `det_id` inteiro estável, atribuído em `detect`.

| Etapa | Saída por detecção |
| --- | --- |
| `ingest` | (por imagem) caminho, largura, altura, sha256 |
| `detect` | `det_id`, `bbox` xyxy em pixels, `confianca`, `classe` (`pessoa`), `descartado` (bool), `motivo_descarte` (`fora_de_campo` \| `pequeno` \| null) |
| `team` | `det_id`, `time` (sigla \| null), `confianca`, `cor_lab` [L, a, b], `arbitro` (bool) |
| `jersey` | `det_id`, `numero` (0–99 \| null), `confianca`, `texto_bruto` |
| `roster` | `det_id`, `posicao`, `nome`, `gsis_id` (todos \| null), `motivo` (`ok` \| `sem_numero` \| `sem_time` \| `numero_fora_do_roster` \| `ambiguo`) |

Etapas seguintes ignoram detecções com `descartado = true` ou `arbitro = true`.

### Artefatos de uma análise

```
data/runs/<analise_id>/
  input.<ext>          # cópia da imagem
  manifest.json        # contexto, config completa, versões, hash, status por etapa
  ingest.json  detect.json  team.json  jersey.json  roster.json
  corrections.json     # opcional; correções do usuário
  analise.json         # saída final, formato do SDD §6
  anotada.png
```

- `analise_id` = `AAAA-MM-DD-NNN`, sequencial por dia.
- `manifest.json` registra: versão do pacote, nome e hash dos pesos do detector, versão do PaddleOCR, todos os parâmetros de `config`, e por etapa `status` (`ok` \| `erro`), mensagem de erro e duração.
- `analise.json` segue exatamente o exemplo do SDD. Em fotos, `track_id = det_id` e `frames_visiveis = [0]`; `midia = {"tipo": "foto"}`. Jogadores com time ou número desconhecido aparecem com esses campos `null`. Árbitros e descartados não entram na lista.

### Correções

`corrections.json` é uma lista de `{det_id, time?, numero?, timestamp}`. Na montagem do resultado, correções sobrescrevem as saídas de `team`/`jersey`, a etapa `roster` é refeita para os jogadores afetados e `corrigido_pelo_usuario = true`. As saídas originais dos modelos não são alteradas, para servir de dataset depois.

## 5. CLI

```
nfl-vision analyze <foto> --times KC BUF --temporada 2025 --semana 11
nfl-vision analyze --run <analise_id> --from <etapa>
nfl-vision correct <analise_id> --det <id> [--time KC] [--numero 87]
nfl-vision eval detect --dataset <pasta>
nfl-vision eval jersey --dataset <pasta>
```

`analyze` imprime a tabela de jogadores e o caminho dos artefatos. `--from` reaproveita os artefatos anteriores à etapa indicada e refaz dela em diante.

## 6. Módulos

### `ingest`
Lê com Pillow, aplica `ImageOps.exif_transpose`, converte para array BGR (OpenCV). Aceita JPG e PNG; outros formatos (incluindo vídeo) são recusados com mensagem clara.

### `detect`
- YOLO11m pré-treinado COCO, classe pessoa, `imgsz = 1280`, `conf ≥ 0,25`.
- Filtro `pequeno`: altura da caixa < 0,4 × mediana das alturas das detecções.
- Filtro `fora_de_campo`: na faixa logo abaixo da caixa (altura = 10% da caixa, mesma largura), fração de pixels verdes de gramado (máscara HSV) < 0,3. Caixas que tocam a borda inferior da imagem usam a faixa interna inferior da própria caixa.
- Pesos do detector definidos em config, para trocar pelo modelo ajustado no futuro.

### `team`
1. Recorte do tronco: 10–50% da altura e 20–80% da largura da caixa.
2. Máscara HSV remove os pixels de gramado. Recortes com menos de 50 pixels restantes → `time = null`.
3. **Árbitro:** fração de pixels muito escuros (L < 30) e fração de pixels muito claros (L > 80), ambas ≥ 0,25 → `arbitro = true`.
4. Cor dominante do recorte: K-means k=3 nos pixels restantes, centro do maior grupo, em LAB.
5. Agrupamento dos jogadores: K-means k=2 nas cores dominantes. Se ΔE (CIEDE2000) entre os centros < 15, todos ficam num único grupo.
6. **Grupo → time:** cada time tem a paleta {`team_color`, `team_color2`} do nflverse. Custo de grupo↔time = menor ΔE entre o centro e a paleta. Escolhe a atribuição dos 2 grupos aos 2 times com menor custo total. Um grupo "branco" (L > 85, croma < 10) não entra no custo: o outro grupo decide, e o branco fica com o time restante. Com um único grupo, ele vai para o time de menor custo (se for branco, `time = null`).
7. **Confiança por jogador:** `d_outro / (d_proprio + d_outro)`, sendo `d` o ΔE até cada centro; com grupo único, 1 − ΔE até o centro / 50, limitado a [0, 1]. Abaixo de 0,60 → `time = null`.

### `jersey`
- Recorte da região do número: 15–60% da altura e 10–90% da largura da caixa, ampliado para pelo menos 128 px de altura.
- PaddleOCR (detecção + reconhecimento) no recorte. Entre os textos que batem com `^\d{1,2}$`, fica o de maior confiança.
- Confiança < 0,60 → `numero = null`.

### `roster`
- `nflreadpy.load_rosters_weekly([temporada])`, cache em `data/cache/rosters/<temporada>.parquet`.
- Siglas normalizadas pelo módulo `teams.py` (ex.: `LAR` → `LA`); siglas inválidas falham na validação da CLI.
- Busca por temporada + semana + time + número. Mais de um resultado → prefere `status = ACT`; continuando ambíguo → `motivo = ambiguo`, campos `null`.
- Número sem correspondência → `motivo = numero_fora_do_roster`, campos `null`.

### `render`
Caixa de 2 px na cor `team_color` do time, rótulo `KC 87 TE`. Jogador com time ou número desconhecido em `#6B7480` com os campos conhecidos (ex.: `KC ?`). Árbitros e descartados não são desenhados.

## 7. Tratamento de erros

- A CLI valida antes de executar: arquivo existe e tem formato aceito, temporada ≥ 2002, semana 1–22, siglas existem.
- Falha numa etapa: as etapas anteriores ficam salvas, o manifest registra o erro, a CLI sai com código ≠ 0 e sugere `--from <etapa>`.
- Sem rede: `roster` usa o cache; sem cache para a temporada, só `roster` falha.
- Nenhuma detecção: análise válida com lista vazia.

## 8. Testes

- **pytest**, ciclo rápido sem modelos: `pytest`.
- Testes com modelos reais marcados `@pytest.mark.model`, executados com `pytest -m model`.

| Alvo | Como |
| --- | --- |
| `runner` | etapas falsas: ordem, artefatos, `--from`, manifest, falha com retomada, correções |
| `ingest` | imagem com EXIF rotacionado; formato recusado |
| `detect` (filtros) | caixas sintéticas sobre fundo verde/não verde; filtro de altura |
| `team` | retângulos sintéticos de cores conhecidas sobre verde; listrado → árbitro; grupo branco; grupo único; mapeamento por paleta |
| `jersey` | parsing e filtro de texto com saída de OCR simulada; com `model`: dígitos renderizados |
| `roster` | parquet de fixture: busca, duplicado com `ACT`, ambíguo, fora do roster, sigla `LA` |
| `render` | gera imagem com dimensões corretas sem erro |
| ponta a ponta | teste golden de `analise.json`, adicionado quando houver capturas de jogos conhecidos |

## 9. Avaliação

- `eval detect`: mAP@0.5 da classe jogador num dataset do Roboflow Universe, via `ultralytics val`.
- `eval jersey`: acurácia de número num dataset de números de camisa do Roboflow, entre os recortes legíveis; reporta também a taxa de `null`.
- Datasets baixados por script com a API key do Roboflow (variável `ROBOFLOW_API_KEY`), em `data/datasets/`.
- Acurácia de time e ponta a ponta dependem de 10–20 capturas de jogos conhecidos rotuladas pelo usuário; ficam para quando existirem.

## 10. Critério de pronto

1. `nfl-vision analyze` gera `analise.json` no formato do SDD e `anotada.png` para uma foto real.
2. `--from` e `correct` funcionam.
3. Testes rápidos passando.
4. Linha de base de detecção e de número medida e registrada em `docs/`. As metas do SDD (0,85 e 0,80) não são requisito desta etapa: a primeira medição calibra as metas.
