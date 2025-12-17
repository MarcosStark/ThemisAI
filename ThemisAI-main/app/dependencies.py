from __future__ import annotations

from app.services.opensearch_service import OpenSearchService
from app.services.llama_service import LlamaService


# -------------------------
# OpenSearch Dependency
# -------------------------
def get_opensearch_service() -> OpenSearchService:
    """
    Provider oficial para FastAPI.
    Chamado automaticamente via Depends().
    """
    return OpenSearchService()


# -------------------------
# LLaMA Dependency
# -------------------------
def get_llama_service() -> LlamaService:
    """
    Provider oficial para o serviço de LLM.
    """
    return LlamaService()
