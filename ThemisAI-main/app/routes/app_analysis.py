# app/routes/app_analysis.py

# Garante compatibilidade de type hints em runtime (para tipos futuros) e em compilação.
from __future__ import annotations
# Módulos padrão para type hinting de coleções (Lista) e valores opcionais.
from typing import List, Optional, Annotated

# --- Importações Essenciais do FastAPI ---
from fastapi import APIRouter, Depends, HTTPException, status
# Importação da classe BaseModel (base para modelos de dados) e Field (para validação avançada).
from pydantic import BaseModel, Field

# --- Importações de Segurança e Serviços Existentes (Infraestrutura) ---
# Importa a função que valida o Bearer Token do usuário na requisição.
from app.config.security import get_current_user
# Serviço que lida com a busca vetorial (componente 'Retriever' do RAG).
from app.services.opensearch_service import OpenSearchService
# Serviço que lida com a API do LLM (Llama) e a geração de texto (componente 'Generator').
from app.services.llama_service import LlamaService
# Assumimos que as funções get_opensearch_service e get_llama_service existem em app.dependencies
from app.dependencies import get_opensearch_service, get_llama_service

# --- Importações do Domínio e Modelos (Lógica de Negócio) ---
# Importa a classe de coordenação que orquestra o fluxo de RAG Duplo (Lógica de Negócio).
from app.domain import AnalysisDomain
# Importa os modelos Pydantic de entrada (Payload) e saída (Relatório).
from ..models.app_analysis import AndroguardAnalysis, ThemisAIReport

# --------------------------------------------------------------------------------------
# 1. INICIALIZAÇÃO DO ROUTER (O SETOR DE ROTAS)
# --------------------------------------------------------------------------------------

# Cria o objeto roteador.
# prefix="/analysis": Todas as rotas neste arquivo começarão com /analysis (Ex: /analysis/app-analysis).
# tags=["security-analysis"]: Define a categoria na documentação do Swagger UI.
router = APIRouter(prefix="/analysis", tags=["security-analysis"])


# --------------------------------------------------------------------------------------
# 2. FUNÇÃO AUXILIAR DE DOMÍNIO (FÁBRICA DE INJEÇÃO DE DEPENDÊNCIA)
# --------------------------------------------------------------------------------------

def get_analysis_domain(
        # Injeção Aninhada (Dependency Injection):
        # O FastAPI injeta o serviço OpenSearch (retriever) e Llama (generator) usando as funções auxiliares.
        retriever: OpenSearchService = Depends(get_opensearch_service),
        generator: LlamaService = Depends(get_llama_service)
) -> AnalysisDomain:
    """
    Constrói o objeto AnalysisDomain, injetando as dependências de infraestrutura necessárias.
    Esta função é a 'fábrica' que o FastAPI chama via Depends() na rota principal.
    """
    # Retornamos uma nova instância do Domínio, passando os serviços injetados para o seu __init__.
    # O Domínio agora tem acesso total às ferramentas OpenSearch e Llama.
    return AnalysisDomain(retriever=retriever, generator=generator)


# --------------------------------------------------------------------------------------
# 3. ENDPOINT PRINCIPAL (A ROTA POST)
# --------------------------------------------------------------------------------------

@router.post(
    "/app-analysis",
    response_model=ThemisAIReport  # Garante que a saída (o relatório) adere estritamente ao modelo ThemisAIReport.
)
async def analyze_app(
        # PAYLOAD DE ENTRADA: FastAPI valida o JSON do corpo da requisição contra o modelo Pydantic.
        req: AndroguardAnalysis,

        # SEGURANÇA: Injeção da função de validação de token.
        # Se o token for inválido, o código da função para aqui e retorna 401 Unauthorized.
        _user: dict = Depends(get_current_user),

        # DOMÍNIO: Injeta a classe de coordenação. O objeto 'domain' está pronto para ser usado.
        domain: AnalysisDomain = Depends(get_analysis_domain),
):
    """
    Processa os dados de análise do Androguard e retorna o relatório de segurança.
    """
    try:
        # LÓGICA DE NEGÓCIO ÚNICA: Delega o trabalho de RAG Duplo ao objeto Domain.
        # A chamada é 'await' porque o Domínio realiza operações assíncronas (busca no DB, chamada ao LLM).
        report: ThemisAIReport = await domain.perform_analysis(req)

        # Retorna o relatório Pydantic validado.
        return report

    except Exception as e:
        # TRATAMENTO DE ERRO: Captura qualquer falha (LLM, DB, validação) e retorna um erro HTTP 500
        # de forma controlada, impedindo o servidor de travar.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Falha ao gerar análise: {e}",
        )