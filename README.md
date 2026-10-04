# NFL — Visão computacional

Identifica jogadores da NFL em fotos de partidas: time, número, posição e nome.
É um projeto de estudo; os detalhes estão no [SDD](SDD%20—%20Visão%20computacional%20NFL.md), na
[spec do pipeline em fotos](docs/superpowers/specs/2026-10-03-pipeline-fotos-design.md) e na
[linha de base](docs/avaliacao/2026-10-linha-de-base.md).

## Setup (Windows + GPU NVIDIA)

1. Instale o [uv](https://docs.astral.sh/uv/).
2. Na pasta `pipeline/`:

       uv sync --extra ocr --extra eval

   Se o `import cv2` quebrar depois de um `uv sync` (o projeto mantém só o `opencv-contrib-python`), rode uma vez:

       uv sync --extra ocr --extra eval --reinstall-package opencv-contrib-python

3. Para baixar datasets e usar o benchmark do Roboflow, crie `.env` na raiz com `ROBOFLOW_API_KEY=...`.

O PyTorch vem do índice CUDA 12.8 (necessário para placas Blackwell, como a RTX 50xx). O OCR (PaddleOCR) roda em CPU.

## Uso

    uv run nfl-vision analyze foto.jpg --times KC BUF --temporada 2025 --semana 11
    uv run nfl-vision analyze --run 2026-10-03-001 --from jersey
    uv run nfl-vision correct 2026-10-03-001 --det 4 --numero 87

Os resultados ficam em `data/runs/<id>/`: `analise.json`, `anotada.png`, `manifest.json` e a saída de cada etapa.

O detector padrão é o RF-DETR base (COCO) em 1120 px, com limiar 0,4; os pesos (~355 MB) são baixados na primeira execução. Para usar o YOLO11m: `--detector-tipo yolo`. Comparação e escolha da resolução em `docs/avaliacao/2026-10-detector-rfdetr.md`.

## Avaliação

    uv run nfl-vision eval baixar --workspace <ws> --projeto <slug> --versao <n> --formato yolov11
    uv run nfl-vision eval baixar --workspace taiseis-workspace --projeto jersey-number-ijbaq --versao 1 --formato folder
    uv run nfl-vision eval detect --dataset ../data/datasets/<pasta> --split test --benchmark yolo-bruto --benchmark rfdetr
    uv run nfl-vision eval jersey --dataset ../data/datasets/<pasta> --split test
    uv run nfl-vision eval time --gabarito ../data/avaliacoes/time-gabarito-cin-cle.json --dataset ../data/datasets/treino-player-v1 --split test

Os downloads ficam em `../data/datasets/<projeto>-v<n>-<formato>`, o caminho que vai em `--dataset`. O Roboflow exporta o split de validação como `valid`. `eval time` mede acurácia e cobertura do time (cor do tronco) contra um gabarito rotulado à mão.

## Detector ajustado (opcional)

O ajuste fino é feito sobre o YOLO11m do COCO (`yolo11m.pt`); no jogo separado ele não superou o modelo original (`docs/avaliacao/2026-10-detector-ajustado.md`). Para ajustá-lo à classe `player` na GPU local:

    uv run nfl-vision treino preparar --so-triagem
    # veja ../data/datasets/treino-player-v1/triagem/*.jpg e decida cada fonte externa
    uv run nfl-vision treino preparar --aprovar "<fonte>:<motivo>" --rejeitar "<fonte>:<motivo>"
    uv run nfl-vision treino rodar --dataset ../data/datasets/treino-player-v1 --nome player-v1
    uv run nfl-vision treino rodar --nome player-v1 --retomar     # se o treino parar no meio

O dataset junta o fork `elyas-carvalho/nfl-player-model-ymsui` v1 e as fontes externas aprovadas (`pitchcamera`, `fhtw`, `evzn`; ver `nfl_vision/treino/fontes.py`). O split `test` é só o jogo CIN × CLE, que nunca entra no treino. Os pesos ficam em `../data/treinos/<nome>/weights/best.pt`, fora do git, com um `manifest.json` (dataset, parâmetros, versões, GPU, sha256).

    uv run nfl-vision eval detect --dataset ../data/datasets/treino-player-v1 --split test --pesos ../data/treinos/player-v1/weights/best.pt --benchmark yolo-bruto
    uv run nfl-vision analyze foto.jpg --times CIN CLE --temporada 2025 --semana 1 --detector ../data/treinos/player-v1/weights/best.pt

`--pesos` troca os pesos dos preditores `nosso` e `yolo-bruto`; `--detector` vale para uma análise nova e fica gravado no manifest dela.

## Testes

    uv run pytest            # rápidos, sem modelos
    uv run pytest -m model   # com GPU, YOLO e PaddleOCR

## Dados e direitos

Fotos e vídeos da NFL são usados só para estudo pessoal e não são publicados (`data/` e `fotos/` ficam fora do git).
Datasets de avaliação: `nflplayerdetection-mjrl1/nfl-player-model` e `taiseis-workspace/jersey-number-ijbaq` (Roboflow Universe, CC BY 4.0).
Datasets de treino do detector: os acima e as fontes externas aprovadas na triagem (Roboflow Universe; licença de cada uma no `README.roboflow.txt` do download). Pesos treinados não são publicados.
