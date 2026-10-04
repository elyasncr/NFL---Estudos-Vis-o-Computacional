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
      schemas.py  config.py  paths.py  cores.py  geometria.py  teams.py
      runner.py  render.py  montagem.py  pipeline.py  cli.py
      stages/  ingest.py  detect.py  team.py  jersey.py  roster.py
      eval/    metricas.py  datasets.py  preditores.py  detect.py  jersey.py
    tests/                  # testes do pipeline (rápidos e @model)
  notebooks/                # exploração e avaliação; importam o pacote (futuro)
  data/                     # ignorado pelo git
    runs/<analise_id>/
    cache/
    datasets/
    avaliacoes/
  docs/
```

Notebooks nunca duplicam lógica do pacote.

## 4. Contratos de dados

Cada etapa é uma função `executar(estado) -> <Etapa>Out` (ex.: `DetectOut`, `TeamOut`), registrada como `runner.Etapa(nome, saida, executar)`. `estado` é um `runner.Estado`: contexto, config e as saídas das etapas anteriores em `estado.saidas`; a imagem só é decodificada na primeira chamada de `estado.imagem()` (lazy — etapas que não precisam dela, como `roster`, não pagam o custo). Os únicos efeitos colaterais são ler a imagem e ler/gravar os caches (`rosters`, `teams`). Toda detecção tem um `det_id` inteiro estável, atribuído em `detect`.

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
- `manifest.json` registra: versão do pacote, nome e hash dos pesos do detector, versão do PaddleOCR, todos os parâmetros de `config`, e por etapa `status` (`ok` \| `erro`), mensagem de erro e duração. Ao retomar com `--from`, se a config efetivamente usada (campo novo ausente ganha o padrão, campo removido é ignorado) difere da gravada, o campo `config` passa a registrar essa config efetiva e a versão antiga fica em `config_original` — só na primeira vez que isso acontece.
- `analise.json` segue exatamente o exemplo do SDD. Em fotos, `track_id = det_id` e `frames_visiveis = [0]`; `midia = {"tipo": "foto", "largura": ..., "altura": ...}`. Jogadores com time ou número desconhecido aparecem com esses campos `null`. Árbitros e descartados não entram na lista.

### Correções

`corrections.json` é uma lista de `{det_id, time?, numero?, timestamp}`. Na montagem do resultado, correções sobrescrevem as saídas de `team`/`jersey`, a etapa `roster` é refeita para os jogadores afetados e `corrigido_pelo_usuario = true`. As saídas originais dos modelos não são alteradas, para servir de dataset depois.

## 5. CLI

```
nfl-vision analyze <foto> --times KC BUF --temporada 2025 --semana 11
nfl-vision analyze --run <analise_id> --from <etapa>
nfl-vision correct <analise_id> --det <id> [--time KC] [--numero 87]
nfl-vision eval detect --dataset <pasta>
nfl-vision eval jersey --dataset <pasta>
nfl-vision eval baixar --workspace <ws> --projeto <slug> --versao <n> --formato <yolov11|folder>
```

`analyze` imprime a tabela de jogadores e o caminho dos artefatos. `--from` reaproveita os artefatos anteriores à etapa indicada e refaz dela em diante.

## 6. Módulos

### `ingest`
Lê com Pillow, aplica `ImageOps.exif_transpose`, converte para array BGR (OpenCV). Aceita JPG e PNG; outros formatos (incluindo vídeo) são recusados com mensagem clara.

### `detect`
- YOLO11m pré-treinado COCO, classe pessoa, `imgsz = 1280`, `conf ≥ 0,25`.
- Filtro `pequeno`: altura da caixa < 0,4 × mediana das alturas das detecções.
- Região do campo (`regiao_do_campo`), estimada uma vez por imagem: máscara HSV de gramado numa cópia reduzida (lado maior 640 px), fechamento morfológico (une as faixas de grama separadas por linhas de jarda) e abertura (remove ruído), núcleo elíptico de 2% do lado maior; ficam os componentes conexos com área ≥ `campo_area_min` (0,05) da imagem; o polígono é a envoltória convexa da união desses componentes, levada de volta à resolução original. Pintura de end zone, letras, logos, linhas brancas e sombras dentro do campo ficam dentro da envoltória. Sem componente grande o bastante → sem região (nenhum descarte por campo).
- Filtro `fora_de_campo`: os pés (centro da base da caixa) estão fora da envoltória por mais que `campo_margem_rel` (0,02) × diagonal da imagem (`cv2.pointPolygonTest` com distância).
- Close: se a mediana das alturas das caixas > `campo_close_altura_rel` (0,5) × altura da imagem, o filtro de campo é pulado (o gramado visível é só retalho entre pernas e não há arquibancada na mesma escala). No conjunto de avaliação os closes têm mediana ≥ 0,84 e os planos abertos/médios ≤ 0,37.
- Caixa que toca a borda inferior (`y2 ≥ altura da imagem − 2`): o jogador está cortado e os pés não aparecem; o teste de campo é pulado e a caixa é mantida (só o filtro `pequeno` vale).
- Histórico: a regra anterior (fração de gramado na faixa logo abaixo dos pés < 0,3) descartava 83 jogadores reais contra 44 não-jogadores no split `test` do dataset NFL do Roboflow (All-22: jogadores minúsculos sobre letras pintadas, linhas e a própria sombra) e derrubava o mAP@0,5 de 0,757 (YOLO bruto) para 0,615. Com a região do campo: 0,748 no `test` (2 jogadores perdidos por campo, ambos o mesmo árbitro rotulado como `player` em frames duplicados) e 0,777 contra 0,786 do bruto no `valid`. A diferença restante vem de pessoas da sideline (árbitro na linha lateral, cinegrafista) rotuladas como `player` no dataset: descartá-las é o comportamento desejado.
- Coordenadas float → índices inteiros por `geometria.caixa_inteira` (arredonda e limita à imagem), usada também no recorte do tronco.
- Pesos do detector definidos em config, para trocar pelo modelo ajustado no futuro. O SHA-256 gravado é o do arquivo que o ultralytics de fato carregou (`ckpt_path`), não o do nome relativo ao diretório atual.

### `team`
1. Recorte do tronco: 10–50% da altura e 20–80% da largura da caixa (frações calculadas em float, depois arredondadas e limitadas à imagem).
2. Máscara HSV remove os pixels de gramado. Se a máscara cobre mais de `mascara_gramado_max_tronco` (0,6) do recorte, a camisa é verde (NYJ, GB, SEA…) e nada é removido. Recortes com menos de 50 pixels restantes → `time = null`, confiança 0, não árbitro.
3. **Árbitro:** teste espacial de listras verticais no recorte 2D. Cada pixel é escuro (L < 30), claro (L > 80) ou outro. Por coluna: "escura" se ≥ 70% dos pixels são escuros, "clara" se ≥ 70% são claros, senão "mista". É árbitro se ≥ 70% das colunas são puras (escuras ou claras), as frações de colunas escuras e de colunas claras são ambas ≥ 0,25 e a sequência de colunas puras (ignorando as mistas) alterna escura/clara pelo menos 4 vezes. Camisa branca com número preto, ou preta com número branco, não forma listras e não é árbitro.
4. Cor dominante do recorte: K-means k=3 nos pixels restantes (amostra de até 3000 pixels, semente fixa), centro do maior grupo, em LAB.
5. Agrupamento dos jogadores: K-means k=2 nas cores dominantes. Se ΔE (CIEDE2000) entre os centros < 15, todos ficam num único grupo (centro = média). Com 5 ou mais jogadores, se o grupo menor tem menos de `max(2, ⌈0,15 × n⌉)` jogadores, é tratado como outlier: um único grupo, centrado no grupo maior; o outlier recebe confiança baixa pela regra de grupo único.
6. **Grupo → time:** cada time tem a paleta {`team_color`, `team_color2`} do nflverse. Custo de grupo↔time = menor ΔE entre o centro e a paleta. Escolhe a atribuição dos 2 grupos aos 2 times com menor custo total. Um grupo "branco" (L > 85, croma < 10) não entra no custo: o outro grupo decide, e o branco fica com o time restante. Com um único grupo, ele vai para o time de menor custo (se for branco, `time = null`).
7. **Confiança por jogador:** `d_outro / (d_proprio + d_outro)`, sendo `d` o ΔE até cada centro; com grupo único, 1 − ΔE até o centro / 50, limitado a [0, 1]. Abaixo de 0,60 → `time = null`.

### Limitações conhecidas (a medir em `eval`)
- A envoltória convexa inclui a grama da sideline e o que estiver entre componentes de gramado: staff, jogadores no banco e pessoas sobre grama da sideline passam no filtro de campo; já a faixa branca da linha lateral fica fora, e um árbitro em pé sobre ela pode ser descartado.
- Campo com pouca grama visível (tomada só da end zone pintada, neve, grama seca fora do matiz) pode não formar componente ≥ 5% da imagem: o filtro de campo é pulado.
- Arquibancada ou torcida com muito verde contíguo ao campo pode alargar a envoltória.
- O teste de listras do árbitro não funciona em transmissão real (medido no dataset NFL do Roboflow, 11 caixas `referee` detectadas pelo YOLO em train/test; o valid tem 2 caixas que o YOLO não detecta): nenhum árbitro é reconhecido. O recorte do tronco tem 21–78 px de largura (as listras seriam resolúveis), mas pose, braços e compressão quebram a pureza por coluna (0–52% de colunas puras, exige 70%) e o branco sai cinza (só 2–29% dos pixels com L > 80 e croma baixo); em árbitro agachado ou com braços erguidos o recorte pega calça e joelhos. Critérios relaxados (frações de escuro e claro, alternâncias por linha) não separam árbitros de jogadores: o melhor candidato pegou 3 de 11 árbitros e 890 de 4.472 jogadores (camisa branca do CIN com listras pretas, camisas escuras com número branco). A regra fica como está; árbitros são limitação conhecida, a resolver com detector ajustado com classe `referee`. O mesmo dataset rotula vários árbitros e pessoas da sideline como `player`, o que impede medir a remoção de árbitro por mAP.
- Grama natural amarelada/seca fora do matiz 35–85 não entra na máscara.
- O filtro `pequeno` usa a mediana global das alturas: em fotos de ângulo alto, jogadores distantes podem ser descartados.
- A confiança de time mede a separação entre grupos, não a qualidade do mapeamento grupo → time (uniformes alternativos e color rush podem ser mapeados ao time errado com confiança alta).

### `jersey`
- Recorte da região do número: 15–60% da altura e 10–90% da largura da caixa, ampliado para pelo menos 128 px de altura.
- PaddleOCR (detecção + reconhecimento) no recorte; cada leitura traz texto, confiança e caixa (`rec_boxes`, ou o retângulo de `rec_polys`).
- Normalização do texto: remove espaços e tira `.`, `#` e `-` das pontas (`#87` e `8 7` viram `87`).
- Dígitos soltos: duas leituras de um dígito na mesma linha (centros verticais a menos de 0,5 × a maior altura, alturas a até ±40%) e vizinhas (espaço horizontal menor que a maior altura) geram também a leitura juntada, na ordem do x, com a menor das duas confianças. As leituras originais continuam valendo.
- Válidos: 0–99 sem zero à esquerda (`0` vale; `07` e `00` não).
- Escolha: o válido de maior confiança; se for de 1 dígito e houver um de 2 dígitos que o contém com confiança no máximo 0,15 abaixo, fica o de 2 dígitos (o OCR costuma ler só metade do número).
- Confiança < 0,60 → `numero = null`.

### `roster`
- `nflreadpy.load_rosters_weekly([temporada])`, cache em `data/cache/rosters/<temporada>.parquet` (gravado de forma atômica). Se a semana pedida é maior que a última semana do cache (temporada em andamento), baixa de novo e regrava o cache; se a semana continua ausente após baixar, ou se não há rede e o cache está desatualizado, a etapa falha com `RosterIndisponivel`.
- Siglas normalizadas pelo módulo `teams.py` (ex.: `LAR` → `LA`); siglas inválidas falham na validação da CLI.
- Busca por temporada + semana + time + número, ignorando quem está fora do elenco na semana (`status` em `CUT`, `RET`, `TRD`, `EXE`, `TRC`). Mais de um resultado → prefere `status = ACT`; continuando ambíguo → `motivo = ambiguo`, campos `null`.
- Número sem correspondência → `motivo = numero_fora_do_roster`, campos `null`.

### `render`
Caixa de 2 px na cor `team_color` do time, rótulo `KC 87 TE` (deslocado para a esquerda quando não cabe na borda direita; texto preto se a luminância da cor for > 150, senão branco). Jogador com time ou número desconhecido em `#6B7480` com os campos conhecidos (ex.: `KC ?`). Árbitros e descartados não são desenhados.

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

- **Dataset de detecção:** fork no workspace `elyas-carvalho` do Universe `nflplayerdetection-mjrl1/nfl-player-model` (338 imagens, classes `player`, `referee`, `ball`; CC BY 4.0, citar a fonte). `ball` é ignorada.
- `eval detect` e `eval jersey` recebem `--split` (padrão `test`; exports do Roboflow usam `valid`, não `val`). A linha de base (`docs/avaliacao/`) mede os dois splits de detecção, `test` e `valid`.
- `eval detect`: mAP@0.5 da classe jogador (as detecções de `pessoa` não descartadas pelo nosso pipeline contra `player`) e a fração de árbitros cobertos por caixa de jogador (IoU ≥ 0,5). Todos os preditores usam a mesma confiança mínima (`--conf`, padrão `detector_conf` da configuração); o JSON salvo registra dataset, split, `conf`, configuração, versão e, por preditor, o pós-processamento aplicado. É regravado após cada preditor, e a falha de um preditor fica registrada sem interromper os outros.
- `eval jersey`: acurácia de número num dataset de recortes de números do Roboflow Universe (outro esporte, como linha de base aproximada) até existirem recortes próprios de NFL; reporta também a taxa de `null` e quantos rótulos ilegíveis foram excluídos.
- Datasets baixados por script com o pacote `roboflow` e a API key em `ROBOFLOW_API_KEY` (`.env`), em `data/datasets/`. O MCP do Roboflow é usado só durante o desenvolvimento, para inspecionar e exportar.
- **Benchmarks** (opcionais, fora de `analyze`), todos no mesmo split (`--split`) e com a mesma métrica:
  - `--benchmark yolo-bruto`: o mesmo YOLO do pipeline (pessoa, COCO), sem filtros de campo nem remoção de árbitro. "nosso" contra `yolo-bruto` mede o efeito dos filtros.
  - `--benchmark rfdetr`: RF-DETR pré-treinado COCO rodando localmente (pacote `rfdetr`), sem pós-processamento. A comparação de arquitetura é `yolo-bruto` contra `rfdetr` (os dois sem filtros).
  - `--benchmark roboflow-nfl`: um modelo treinado do próprio projeto `nfl-player-model`, via inferência hospedada do Roboflow (`inference-sdk`, API key no header). Só envia imagens do split de teste do dataset público, nunca mídias do usuário. Compara "detector genérico + filtros" com "modelo treinado em NFL".
- Acurácia de time e ponta a ponta dependem de 10–20 capturas de jogos conhecidos rotuladas pelo usuário; ficam para quando existirem.

## 10. Critério de pronto

1. `nfl-vision analyze` gera `analise.json` no formato do SDD e `anotada.png` para uma foto real.
2. `--from` e `correct` funcionam.
3. Testes rápidos passando.
4. Linha de base de detecção e de número medida e registrada em `docs/`, junto com o benchmark do Roboflow na detecção. As metas do SDD (0,85 e 0,80) não são requisito desta etapa: a primeira medição calibra as metas.
