import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

# CORREÇÃO CRÍTICA: A importação do router de análise deve ser do ficheiro 'app_analysis'.
from app.routes import auth, training as train, ask, app_analysis


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from app.services.opensearch_service import OpenSearchService
        osvc = OpenSearchService()
        try:
            osvc._ensure_index()
        except Exception:
            # Em ambiente de teste ou desenvolvimento sem OpenSearch, isso pode falhar.
            print("AVISO: Falha ao garantir o índice do OpenSearch no lifespan.")
            pass
    except Exception:
        # Se OpenSearchService não puder ser importado (ex: dependências ausentes), ignora.
        print("AVISO: OpenSearchService não disponível ou inicialização falhou.")
        pass
    yield


app = FastAPI(
    title="ThemisAI Security API",
    version="0.1.0",
    description="API para RAG com OpenSearch + LLaMA",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    inicio = time.time()
    resposta = await call_next(request)
    fim = time.time()
    duracao = fim - inicio
    resposta.headers["X-Process-Time"] = str(duracao)
    return resposta


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


# Rotas
app.include_router(auth.router)
app.include_router(train.router)
app.include_router(ask.router)
# CORREÇÃO: Inclusão do router com o prefixo /analysis.
app.include_router(app_analysis.router, prefix="/analysis")


# Expondo a aplicação com o nome esperado pelo módulo de teste (app_main)
app_main = app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )