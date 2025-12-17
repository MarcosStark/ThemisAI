# app/services/androguard_rag_service.py

# IMPORTAÇÕES CORRIGIDAS: Agora importamos os serviços reais da sua pasta 'services'
from .llama_service import LlamaService         # Faz o trabalho do LLM e Prompt
from .opensearch_service import OpenSearchService # Faz o trabalho da Busca RAG
from ..models.app_analysis import ThemisAIReport, AndroguardAnalysis # Corrigido o nome do arquivo model

class AndroguardAnalysisService:

    # 1. Construtor com Tipos Reais (Correção de Nomes)
    def __init__(
            self,
            # Injetamos os serviços que o perform_analysis precisará
            # O LlamaService fará a chamada final ao LLM
            llama_service: LlamaService,
            # O OpenSearchService fará a busca no Banco de Dados Vetorial (MITRE)
            opensearch_service: OpenSearchService
    ):
        # 2. Atribuição das Dependências
        self.llama_service = llama_service
        self.opensearch_service = opensearch_service

    # O método perform_analysis usará estas variáveis injetadas.
    # Ex: self.opensearch_service.search_vector(...)
    # Ex: self.llama_service.call_model(...)