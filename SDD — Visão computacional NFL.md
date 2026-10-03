# SDD — Visão computacional NFL

Oct 2, 2026 · @maury

## 1. Visão geral

O sistema recebe uma foto ou um vídeo de até 2 minutos de uma partida da NFL e identifica cada jogador visível: time, número, posição e nome. Em fases futuras, analisa partidas inteiras e sugere jogadas alternativas às que foram executadas.

É um projeto pessoal com dois objetivos:

- **Aprendizado:** praticar, de ponta a ponta, os conceitos centrais de visão computacional (detecção, rastreamento, classificação, OCR, homografia).
- **Acompanhamento da NFL:** revisar partidas depois que acontecem e entender que outras opções de jogada existiam em cada momento.

O processamento é offline (não em tempo real). Isso permite usar modelos mais pesados e priorizar precisão sobre velocidade.

## 2. Escopo e fases

O projeto tem três fases. A fase 1 é o MVP e o foco deste documento; as fases 2 e 3 estão descritas para orientar decisões de arquitetura.

| Fase | Objetivo | Entrada | Saída |
| --- | --- | --- | --- |
| 1. Identificação (MVP) | Saber quem é cada jogador | Foto ou vídeo ≤ 2 min | Jogadores com time, número, posição e nome |
| 2. Contexto do campo | Saber onde cada jogador está no campo e em que jogada | Vídeo de partida | Posições em jardas, jogadas segmentadas, formação, down e distância |
| 3. Análise de jogadas | Avaliar a jogada feita e as alternativas | Trajetórias da fase 2 | EPA da jogada executada e de cada opção alternativa |

**Fora de escopo:** processamento em tempo real, apostas, distribuição pública de vídeos da NFL e uso comercial.

## 3. Requisitos

### Funcionais (fase 1)

| ID | Requisito |
| --- | --- |
| RF01 | Aceitar upload de foto (JPG, PNG) ou vídeo (MP4, MOV) de até 2 minutos |
| RF02 | Receber do usuário os dois times da partida e a temporada/semana |
| RF03 | Detectar os jogadores e descartar árbitros e pessoas fora de campo |
| RF04 | Atribuir cada jogador a um dos dois times |
| RF05 | Ler o número da camisa de cada jogador, com nível de confiança |
| RF06 | Consultar o roster e retornar posição e nome |
| RF07 | Marcar como "desconhecido" qualquer jogador abaixo do limiar de confiança |
| RF08 | Gerar a mídia anotada (caixas + rótulos) e uma lista de jogadores identificados |
| RF09 | Permitir que o usuário corrija uma identificação errada |

### Não funcionais

- **Processamento offline:** um vídeo de 2 minutos deve terminar em até 10 minutos numa GPU do Google Colab gratuito.
- **Precisão antes de cobertura:** é preferível marcar "desconhecido" a errar com confiança alta.
- **Reprodutibilidade:** versões de modelos e parâmetros ficam registradas em cada análise.
- **Modularidade:** cada etapa do pipeline é um módulo testável e substituível isoladamente.
- **Privacidade:** vídeos ficam armazenados localmente e não são publicados.

## 4. Arquitetura do sistema

A aplicação separa a interface do processamento pesado: a API recebe a mídia, dispara o pipeline em segundo plano e serve os resultados quando ficam prontos.

&#91;embedded content: arquitetura · 3 serviços, pipeline de 5 módulos\]

Cada módulo do pipeline lê a saída do anterior e grava a sua, então qualquer etapa pode ser trocada ou reprocessada sem refazer as outras. As caixas tracejadas são dependências externas.

## 5. Pipeline da fase 1

O pipeline transforma mídia bruta em jogadores identificados em cinco etapas sequenciais. Cada etapa grava sua saída, para que possa ser inspecionada e reprocessada isoladamente.

1. **Entrada e pré-processamento**
   - Foto: usada diretamente.
   - Vídeo: extração de frames com OpenCV, amostrando 5 a 10 fps.
   - Detecção de cortes de câmera (mudança brusca entre frames) para reiniciar o tracking.
2. **Detecção e tracking**
   - YOLO (Ultralytics) pré-treinado na classe "pessoa"; depois, ajuste fino com dataset do Roboflow que separa jogador e árbitro.
   - Filtro de pessoas fora de campo por tamanho e posição da caixa.
   - ByteTrack atribui um ID fixo por jogador ao longo dos frames.
3. **Identificação do time**
   - Recorte do tronco (metade superior da caixa).
   - Máscara HSV remove os pixels verdes do gramado.
   - Cor dominante de cada recorte agrupada com K-means (k = 2, ou 3 incluindo árbitros).
   - Grupos mapeados para os times informados pelo usuário; no vídeo, voto majoritário por ID.
4. **Leitura do número**
   - Recorte do tronco ampliado e passado ao PaddleOCR, restrito a números de 0 a 99.
   - No vídeo, leitura em vários frames do mesmo ID e voto ponderado pela confiança.
   - Abaixo do limiar, resultado "desconhecido".
   - Evolução prevista: modelo dedicado a números de camisa, caso o OCR genérico fique abaixo da meta.
5. **Consulta ao roster**
   - Chave: temporada + semana + time + número.
   - Fonte: roster semanal do nflverse, porque números mudam com trocas durante a temporada.
   - Saída: posição e nome do jogador.

## 6. Modelo de dados e fontes externas

Cada análise gera um registro em JSON com a configuração usada e os jogadores identificados. Exemplo da saída da fase 1:

```json
{
  "analise_id": "2026-10-02-001",
  "midia": {"tipo": "video", "duracao_s": 95, "fps_amostrado": 8},
  "contexto": {"temporada": 2026, "semana": 4, "times": ["KC", "BUF"]},
  "modelos": {"detector": "yolo11m", "tracker": "bytetrack", "ocr": "paddleocr"},
  "jogadores": [
    {
      "track_id": 7,
      "time": "KC",
      "numero": 87,
      "confianca_numero": 0.91,
      "posicao": "TE",
      "nome": "(do roster)",
      "frames_visiveis": [12, 13, 14, 20],
      "corrigido_pelo_usuario": false
    }
  ]
}
```

### Fontes externas

| Fonte | Uso | Fase |
| --- | --- | --- |
| nflverse (biblioteca `nflreadpy`) | Rosters semanais: time, número, posição, nome | 1 |
| Roboflow Universe | Datasets anotados para ajuste fino do detector e leitura de números | 1 |
| nflverse play-by-play | Down, distância e EPA de cada jogada real | 2 e 3 |
| NFL Big Data Bowl (Kaggle) | Dados de rastreamento x/y dos 22 jogadores a 10 Hz para treinar o modelo de jogadas | 3 |
| NFL+ Premium (All-22) | Vídeo aéreo com todos os jogadores em campo | 2 e 3 |

## 7. Interface da aplicação

A interface é uma aplicação web local com quatro telas, percorridas nesta ordem:

1. **Nova análise:** upload da mídia, seleção dos dois times e da temporada/semana.
2. **Processamento:** progresso por etapa do pipeline (detecção, time, número, roster), com tempo estimado.
3. **Resultado:** player de vídeo (ou imagem) com caixas e rótulos sobrepostos, ao lado da lista de jogadores. Tocar num jogador destaca sua caixa e mostra os frames em que aparece.
4. **Histórico:** análises anteriores, com miniatura, times, data e taxa de identificação.

Na tela de resultado, cada jogador pode ser corrigido manualmente (time ou número). As correções são salvas e alimentam um conjunto de dados para melhorar os modelos depois.

Nas fases 2 e 3, a tela de resultado ganha uma visão de cima do campo com as posições em jardas e um painel de jogadas alternativas com o EPA de cada uma.

## 8. Design system

O conceito visual é a **sala de análise de vídeo** (film room): interface escura e discreta, onde o vídeo é o protagonista e a cor serve para informar, não para decorar.

### Princípios

- **O vídeo em primeiro lugar:** fundo escuro, sem elementos que disputem atenção com a mídia.
- **Cor dos times vem dos dados:** caixas e rótulos usam a cor oficial de cada time; a interface fica neutra para não conflitar com nenhum deles.
- **Confiança sempre visível:** todo dado inferido mostra seu nível de confiança.
- **Números são dados:** números de camisa, jardas e EPA usam fonte tabular para alinhar e comparar.

### Cores

A cor de destaque é o amarelo da linha de first down da transmissão, usada só para seleção e foco.

| Token | Escuro | Claro | Uso |
| --- | --- | --- | --- |
| `bg/base` | #0E1116 | #F7F8FA | Fundo da aplicação |
| `bg/surface` | #161B22 | #FFFFFF | Painéis e cartões |
| `bg/raised` | #1F2630 | #EEF1F5 | Hover, menus |
| `border/default` | #2A323D | #D8DDE4 | Divisórias e contornos |
| `text/primary` | #E8EBEF | #12161C | Texto principal |
| `text/secondary` | #9AA4B2 | #5A6472 | Rótulos e legendas |
| `accent/first-down` | #F5D90A | #B89C00 | Seleção, foco, jogador ativo |
| `field/turf` | #2E6B3F | #2E6B3F | Visão de cima do campo (fases 2 e 3) |
| `confidence/high` | #2FBF8F | #12875F | Confiança ≥ 0,85 |
| `confidence/medium` | #F08A3C | #B85A12 | Confiança entre 0,60 e 0,85 |
| `confidence/unknown` | #6B7480 | #8A93A0 | Abaixo de 0,60 ("desconhecido") |
| `team/home`, `team/away` | dinâmica | dinâmica | Cor oficial de cada time, vinda do nflverse |

### Tipografia

| Papel | Fonte | Tamanho / peso |
| --- | --- | --- |
| Títulos de tela | Barlow Condensed | 28 px / 600 |
| Número de camisa em destaque | Barlow Condensed | 22 px / 700 |
| Texto de interface | Inter | 14 px / 400 e 500 |
| Legendas | Inter | 12 px / 400 |
| Dados (confiança, jardas, EPA, timecode) | JetBrains Mono | 13 px / 400, tabular |

### Espaçamento e forma

- Escala de espaçamento com base 4 px: 4, 8, 12, 16, 24, 32, 48.
- Raio de borda: 4 px em controles, 8 px em cartões.
- Sem sombras; a hierarquia vem das três camadas de fundo.
- Caixas sobre o vídeo: contorno de 2 px na cor do time; 3 px em `accent/first-down` quando selecionadas.

### Componentes

| Componente | Descrição |
| --- | --- |
| `MediaPlayer` | Player com camada de anotações e linha do tempo marcando onde o jogador selecionado aparece |
| `BoundingBox` | Caixa sobre o jogador com rótulo compacto: time, número, posição |
| `PlayerRow` | Linha da lista: número grande, nome, posição, time e `ConfidenceBadge` |
| `ConfidenceBadge` | Selo com o valor (ex.: 0,91) na cor da faixa de confiança |
| `TeamSelector` | Seletor de time com logo, sigla e cor |
| `StageProgress` | Progresso das etapas do pipeline, com status e tempo de cada uma |
| `CorrectionPopover` | Painel para corrigir time ou número de um jogador |
| `FieldView` | Campo visto de cima com jogadores em jardas (fases 2 e 3) |

## 9. Stack tecnológica e ambiente

O projeto é todo em Python no back-end, com front-end web leve e execução dos modelos no Google Colab durante o desenvolvimento.

| Camada | Tecnologia | Motivo |
| --- | --- | --- |
| Detecção e tracking | Ultralytics (YOLO11) + ByteTrack | Modelos prontos, tracking embutido, fácil ajuste fino |
| Processamento de imagem | OpenCV | Leitura de vídeo, recortes, máscaras HSV |
| Agrupamento por cor | scikit-learn (K-means) | Simples e suficiente para 2–3 grupos |
| OCR | PaddleOCR | Bom desempenho em texto curto e em ângulo |
| Dados da NFL | nflreadpy (Polars) | Rosters semanais, play-by-play e cores dos times |
| API | FastAPI | Expõe o pipeline e processa análises em segundo plano |
| Armazenamento | Sistema de arquivos + SQLite | Mídias e JSONs em disco, índice de análises no SQLite |
| Front-end | React + Vite | Player com camada de anotações e lista de jogadores |
| Experimentos | Google Colab (GPU gratuita) + Jupyter | Treino e avaliação sem hardware próprio |
| Anotação de dados | Roboflow | Rotular frames próprios para ajuste fino |

Estrutura de pastas sugerida:

```
nfl-vision/
  pipeline/      # ingest, detect, team, jersey, roster
  api/           # FastAPI
  web/           # React
  notebooks/     # experimentos e avaliação
  data/          # mídias, saídas JSON, cache de rosters
  tests/         # testes por etapa com imagens de referência
```

## 10. Avaliação, riscos e limitações

### Métricas da fase 1

A avaliação usa um conjunto próprio de 200 frames rotulados à mão, de partidas e ângulos variados. As metas abaixo são pontos de partida, a revisar após a primeira medição.

| Etapa | Métrica | Meta inicial |
| --- | --- | --- |
| Detecção | mAP@0.5 da classe jogador | ≥ 0,85 |
| Tracking | IDF1 (consistência de ID) | ≥ 0,70 |
| Time | Acurácia de atribuição | ≥ 0,95 |
| Número | Acurácia entre os jogadores com número legível | ≥ 0,80 |
| Ponta a ponta | % de jogadores identificados corretamente / % marcados como desconhecido | ≥ 70% / ≤ 25% |

### Riscos

| Risco | Impacto | Mitigação |
| --- | --- | --- |
| Número ilegível (oclusão, ângulo, borrão) | Alto | Voto entre frames; resposta "desconhecido"; modelo dedicado depois |
| Uniformes de cores parecidas | Médio | Usar espaço de cor LAB; considerar capacete e calça no agrupamento |
| Troca de IDs no tracking após contato entre jogadores | Médio | Reconciliação de IDs pelo número já lido |
| Cortes de câmera e replays | Médio | Detectar corte de cena e reiniciar o tracking |
| Roster desatualizado | Baixo | Usar roster semanal e permitir correção manual |

### Limitações conhecidas

- A câmera de transmissão raramente mostra os 22 jogadores; as fases 2 e 3 dependem do vídeo All-22.
- Desde 2021 a numeração não indica posição com segurança, então a posição depende sempre do roster.
- Vídeos da NFL são protegidos por direitos autorais; o uso fica restrito ao estudo pessoal.

## 11. Roadmap

O projeto avança por etapas, e cada portão exige um critério cumprido antes da etapa seguinte. Sem prazos fixos, por ser um projeto pessoal.

&#91;embedded content: roadmap · 4 etapas, 3 portões\]

A etapa destacada é o ponto de partida: tudo em fotos, sem tracking, para validar cada módulo antes de lidar com vídeo.
