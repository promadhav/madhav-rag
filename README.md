## AMD AI Academy — Challenge Repositories

This submission links to two separate challenge projects:

- **Challenge 2 — OCR:** https://github.com/promadhav/madhav-ocr
- **Challenge 3 — RAG:** https://github.com/promadhav/madhav-rag

The OCR implementation remains in its original repository.
This repository contains the RAG implementation.



# MADHAV RAG

Offline document question answering for AMD × lablab AI Academy Mini Challenge 3.

## Features

- Supports PDF, DOCX, XLSX, CSV, TXT/LOG, Python source files, and images.
- Uses Qwen3-VL-4B-Instruct for visual document understanding and answering.
- Returns short answers with corpus-relative source filenames.
- Skips unsupported, encrypted, and unreadable files.
- Returns {"answer": "", "citations": []} when sufficient evidence is unavailable.

## Model and runtime

- Model: Qwen/Qwen3-VL-4B-Instruct
- Revision: ebb281ec70b05090aa6165b016eac8ec08e71b17
- AMD ROCm PyTorch runtime.
- Model files are packaged at /models/qwen in the container.
- Runtime is configured for offline use.
- Model weights are intentionally excluded from GitHub.

## Container interface

The default container command starts the persistent worker.

Index a corpus inside the running container:

```bash
python3 /app/app.py --index /app/corpus
```

Ask a question:

```bash
python3 /app/app.py --corpus /app/corpus \
  --query-id query_01 \
  --query "What is the requested value?"
```

Output is written to:

```text
/app/output/query_01_output.json
```

## Validation status

- Local Docker build: passed.
- Offline model-file, processor, and application-import checks: passed.
- Uncompressed image layers: 37.746 GiB.
- Docker GPU inference: not yet tested.
- These checks do not establish a hidden-evaluation score.

See BUILD_REPORT.json for the build-time report.
