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

## Avaliação

    uv run nfl-vision eval baixar --workspace <ws> --projeto <slug> --versao <n> --formato yolov11
    uv run nfl-vision eval detect --dataset ../data/datasets/<pasta> --split test --benchmark yolo-bruto --benchmark rfdetr
    uv run nfl-vision eval jersey --dataset ../data/datasets/<pasta> --split test

O Roboflow exporta o split de validação como `valid`.

## Testes

    uv run pytest            # rápidos, sem modelos
    uv run pytest -m model   # com GPU, YOLO e PaddleOCR

## Dados e direitos

Fotos e vídeos da NFL são usados só para estudo pessoal e não são publicados (`data/` e `fotos/` ficam fora do git).
Datasets de avaliação: `nflplayerdetection-mjrl1/nfl-player-model` e `taiseis-workspace/jersey-number-ijbaq` (Roboflow Universe, CC BY 4.0).
