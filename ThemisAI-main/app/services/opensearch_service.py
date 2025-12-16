from __future__ import annotations

from typing import Any, Dict, List, Optional

from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer

from app.config.settings import settings


# =============================================================================
# 🔒 MODELO DE EMBEDDINGS GLOBAL (carregado uma única vez)
# =============================================================================
EMBED_MODEL = SentenceTransformer(settings.EMBED_MODEL_NAME)
EMBED_DIM = EMBED_MODEL.get_sentence_embedding_dimension()


class OpenSearchService:
    """
    Serviço RAG com OpenSearch.
    - Criação idempotente de índice vetorial (HNSW/FAISS).
    - Indexação em lote.
    - Busca KNN (slim).
    - Busca híbrida BM25 + KNN via RRF (slim).
    """

    def __init__(
        self,
        host: Optional[str] = None,
        index_name: Optional[str] = None,
    ) -> None:

        self.host = host or settings.OPENSEARCH_HOST
        self.index = index_name or settings.OPENSEARCH_INDEX

        # Cliente OpenSearch
        self.client = OpenSearch(
            hosts=[self.host],
            http_compress=True,
            timeout=30,
            max_retries=3,
            retry_on_timeout=True,
        )

        # 🔥 Reutiliza modelo global
        self.model = EMBED_MODEL
        self.dim = EMBED_DIM

    # =========================================================================
    # INFRA
    # =========================================================================

    def _ensure_index(self) -> None:
        """Criação idempotente do índice."""
        try:
            if self.client.indices.exists(index=self.index):
                return

            body = {
                "settings": {
                    "index": {
                        "knn": True
                    }
                },
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "text": {"type": "text"},
                        "metadata": {"type": "object", "enabled": True},
                        "vector": {
                            "type": "knn_vector",
                            "dimension": self.dim,
                            "method": {
                                "name": "hnsw",
                                "space_type": "innerproduct",  # Alterado para "innerproduct"
                                "engine": "faiss",
                                "parameters": {
                                    "ef_construction": 128,
                                    "m": 16
                                },
                            },
                        },
                    }
                },
            }

            self.client.indices.create(index=self.index, body=body)

        except Exception as e:
            print(f"❌ Erro ao garantir índice OpenSearch: {e}")
            raise

    # =========================================================================
    # INDEXAÇÃO
    # =========================================================================

    def index_docs(self, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Indexação em lote: docs = [{id, text, metadata}]"""
        if not docs:
            return {"indexed": 0}

        self._ensure_index()

        texts = [d["text"] for d in docs]
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalização do vetor
        )

        actions = []
        for d, vec in zip(docs, vectors):
            _id = d.get("id")
            body = {
                "id": _id,
                "text": d.get("text"),
                "metadata": d.get("metadata") or {},
                "vector": vec.tolist(),
            }
            actions.append({
                "_op_type": "index",
                "_index": self.index,
                "_id": _id,
                "_source": body
            })

        success, _ = helpers.bulk(self.client, actions, stats_only=True)
        return {"indexed": success}

    # =========================================================================
    # BUSCA KNN
    # =========================================================================

    def search_knn_slim(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retorna hits slim: id, score, text, meta."""
        self._ensure_index()

        vec = self.model.encode(
            query,
            normalize_embeddings=True  # Normalização do vetor
        ).tolist()

        body = {
            "size": top_k,
            "query": {
                "knn": {
                    "vector": {
                        "vector": vec,
                        "k": top_k
                    }
                }
            }
        }

        resp = self.client.search(index=self.index, body=body)
        hits = resp.get("hits", {}).get("hits", [])

        out = []
        for h in hits:
            src = h.get("_source", {})
            out.append({
                "id": src.get("id") or h.get("_id"),
                "score": h.get("_score"),
                "text": src.get("text", ""),
                "meta": src.get("metadata") or {},
            })
        return out

    # =========================================================================
    # BUSCA HÍBRIDA (BM25 + KNN via RRF)
    # =========================================================================

    def search_hybrid_slim(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Combina BM25 + KNN usando Reciprocal Rank Fusion."""
        self._ensure_index()

        # BM25
        bm25_body = {
            "size": top_k,
            "query": {"match": {"text": query}}
        }
        r1 = self.client.search(index=self.index, body=bm25_body)
        bm25_hits = r1.get("hits", {}).get("hits", [])

        # KNN
        vec = self.model.encode(
            query,
            normalize_embeddings=True  # Normalização do vetor
        ).tolist()
        knn_body = {
            "size": top_k,
            "query": {
                "knn": {
                    "vector": {
                        "vector": vec,
                        "k": top_k
                    }
                }
            }
        }
        r2 = self.client.search(index=self.index, body=knn_body)
        knn_hits = r2.get("hits", {}).get("hits", [])

        # RRF
        scores: Dict[str, float] = {}

        def add_rrf(hlist):
            for rank, h in enumerate(hlist, start=1):
                _id = h.get("_id")
                if not _id:
                    continue
                scores.setdefault(_id, 0.0)
                scores[_id] += 1.0 / (60 + rank)

        add_rrf(bm25_hits)
        add_rrf(knn_hits)

        if not scores:
            return []

        merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_ids = [m[0] for m in merged[:top_k]]

        mresp = self.client.mget(index=self.index, body={"ids": top_ids})

        hits = []
        for doc in mresp.get("docs", []):
            if not doc.get("found"):
                continue
            src = doc.get("_source", {})
            hits.append({
                "id": src.get("id") or doc.get("_id"),
                "score": scores.get(doc.get("_id"), 0),
                "text": src.get("text", ""),
                "meta": src.get("metadata") or {},
            })

        return hits


# Dependency para FastAPI
def get_opensearch_service() -> OpenSearchService:
    return OpenSearchService()
