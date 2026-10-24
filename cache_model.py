"""Inspect local weights only. This module never contacts Hugging Face."""
import json
import os
from pathlib import Path

MODEL_ID = 'Qwen/Qwen3-VL-4B-Instruct'
CACHE = Path('/workspace/madhav-rag/models/qwen3vl4b')


def model_files(path):
    path = Path(path)
    config = json.loads((path / 'config.json').read_text())
    if config.get('model_type') != 'qwen3_vl':
        raise ValueError(f'Wrong model type: {path}')
    required = ['config.json', 'preprocessor_config.json', 'tokenizer_config.json', 'tokenizer.json', 'chat_template.json']
    index = path / 'model.safetensors.index.json'
    if index.is_file():
        required += ['model.safetensors.index.json']
        required += sorted(set(json.loads(index.read_text())['weight_map'].values()))
    else:
        required += ['model.safetensors']
    for name in required:
        p = path / name
        if not p.is_file() or p.stat().st_size == 0:
            raise FileNotFoundError(f'Missing/incomplete model file: {p}')
        if name.endswith('.safetensors'):
            with p.open('rb') as f:
                header_bytes = int.from_bytes(f.read(8), 'little')
                if not 0 < header_bytes < 100_000_000:
                    raise ValueError(f'Invalid safetensors header: {p}')
                header = json.loads(f.read(header_bytes))
            data_end = max(v['data_offsets'][1] for k,v in header.items() if k != '__metadata__')
            if p.stat().st_size != 8 + header_bytes + data_end:
                raise ValueError(f'Truncated weights: {p}')
    return required


def locate_model():
    override = os.environ.get('RAG_MODEL_DIR')
    if override:
        model_files(override)
        return Path(override).resolve()
    candidates = [Path('/models/qwen'), CACHE]
    for root in (CACHE, CACHE / 'hub'):
        repo = root / 'models--Qwen--Qwen3-VL-4B-Instruct'
        ref = repo / 'refs/main'
        if ref.is_file():
            candidates.append(repo / 'snapshots' / ref.read_text().strip())
        candidates += sorted((repo / 'snapshots').glob('*'), reverse=True)
    failures = []
    for p in candidates:
        if not (p / 'config.json').is_file():
            continue
        try:
            model_files(p)
            return p.resolve()
        except Exception as e:
            failures.append(str(e))
    raise FileNotFoundError('Complete cached Qwen model not found. NO download attempted. '
                            + '; '.join(failures))
