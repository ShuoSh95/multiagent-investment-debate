# HuggingFace Spaces (Docker SDK) image for the public demo.
# Spaces runs the container as UID 1000, so caches/data must be
# writable by that user.

FROM python:3.11-slim

RUN useradd -m -u 1000 user

WORKDIR /app

# CPU-only torch first — avoids pulling multi-GB CUDA wheels
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .
RUN mkdir -p /app/data /app/.cache && chown -R user:user /app

USER user
ENV HOME=/home/user \
    HF_HOME=/app/.cache/huggingface \
    DEMO_MODE=1

# Pre-download the embedding model at build time (public, ~2.3GB) so
# the Space is ready to answer immediately after boot.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# Bake the knowledge base using the HF_TOKEN build secret. If the secret
# is unavailable at build, rag/fetch_kb.py retries at runtime instead.
RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=false \
    sh -c 'export HF_TOKEN=$(cat /run/secrets/HF_TOKEN 2>/dev/null || true); \
           python -m rag.fetch_kb || echo "[build] KB fetch deferred to runtime"'

EXPOSE 8501
CMD ["streamlit", "run", "web/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
