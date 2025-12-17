# =========================
# Stage 1: builder
# =========================
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Dependências de build (não vão para o runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    curl \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências Python DIRETAMENTE (SEM pip wheel)
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copia código da aplicação
COPY app ./app
COPY scripts ./scripts
COPY llama.cpp ./llama.cpp

# Caso llama.cpp não venha no contexto
RUN if [ ! -f /build/llama.cpp/CMakeLists.txt ]; then \
      echo "[llama.cpp] não encontrado no contexto, clonando..."; \
      git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /build/llama.cpp; \
    fi

# Compila llama.cpp
RUN set -eux; \
    rm -rf /build/llama.cpp/build; \
    cmake -S /build/llama.cpp -B /build/llama.cpp/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_BLAS=ON \
      -DGGML_OPENMP=ON \
      -DBLAS=OpenBLAS \
      -DLLAMA_CURL=OFF; \
    cmake --build /build/llama.cpp/build -j "$(nproc)"; \
    BIN="/build/llama.cpp/build/bin"; \
    if    [ -x "$BIN/llama-cli"    ]; then TGT="$BIN/llama-cli"; \
    elif [ -x "$BIN/llama-bin"    ]; then TGT="$BIN/llama-bin"; \
    elif [ -x "$BIN/llama-simple" ]; then TGT="$BIN/llama-simple"; \
    elif [ -x "$BIN/main"         ]; then TGT="$BIN/main"; \
    elif [ -x "$BIN/llama"        ]; then TGT="$BIN/llama"; \
    else \
        echo "Nenhum binário do llama.cpp encontrado"; \
        ls -la "$BIN"; \
        exit 1; \
    fi; \
    ln -sf "$TGT" "$BIN/llama-bin"; \
    "$BIN/llama-bin" -h || true


# =========================
# Stage 2: runtime
# =========================
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Dependências mínimas de runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copia Python + site-packages do builder
COPY --from=builder /usr/local /usr/local

# Copia aplicação e build do llama.cpp
COPY --from=builder /build/app /app/app
COPY --from=builder /build/scripts /app/scripts
COPY --from=builder /build/llama.cpp/build /app/llama.cpp/build

# Usuário não-root
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Variáveis de ambiente
ENV LD_LIBRARY_PATH=/app/llama.cpp/build:/app/llama.cpp/build/lib:/app/llama.cpp/build/bin:$LD_LIBRARY_PATH \
    LLAMA_CPP_PATH=/app/llama.cpp/build/bin/llama-bin \
    APP_MODULE=app.main:app \
    HOST=0.0.0.0 \
    PORT=8000 \
    RELOAD=false

CMD ["sh", "-c", "uvicorn ${APP_MODULE} --host ${HOST} --port ${PORT} $( [ \"$RELOAD\" = \"true\" ] && echo --reload )"]
