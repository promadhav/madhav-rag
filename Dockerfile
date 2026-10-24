FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0@sha256:3174cb7061d94c427da96c0edef4adea28046fa3f3b2ff3948dc4e995665ff8c
WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY requirements-lock.txt /app/requirements-lock.txt
# Every dependency is explicit; pip must never resolve/replace the ROCm torch.
RUN python3 -m pip install --no-cache-dir --only-binary=:all: --no-deps -r /app/requirements-lock.txt \
 && python3 -c "import torch; assert torch.version.hip and '+rocm' in torch.__version__; print(torch.__version__)" \
 && python3 -c "from transformers import Qwen3VLForConditionalGeneration, AutoProcessor; import pymupdf"
COPY models/qwen/ /models/qwen/
COPY app.py worker.py cache_model.py parsers.py retrieval.py answer_v2.py vision_v3.py /app/
RUN mkdir -p /app/corpus /app/output /app/state
ENV RAG_MODEL_DIR=/models/qwen HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
RUN python3 -c "from cache_model import locate_model; print(locate_model())"
# No exposed TCP port. The harness execs app.py; its resident model uses a Unix socket.
CMD ["python3", "-u", "/app/worker.py"]
