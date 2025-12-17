# tests/unit/test_analysis_domain.py

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, ANY
import json

# Supondo que você tenha uma maneira de importar o AnalysisDomain
from app.domain.analysis_domain import AnalysisDomain
from app.models.app_analysis import AndroguardAnalysis, ThemisAIReport


# --------------------
# FIXTURES (Dados Mock)
# --------------------

# Mock de entrada (payload que o cliente enviaria)

@pytest.fixture
def mock_androguard_input() -> AndroguardAnalysis:
    return AndroguardAnalysis(
        package_name="com.example.vulnerableapp",
        # CORREÇÃO AQUI: Mudando o nome do campo para 'permissions_list'
        permissions_list=["INTERNET", "READ_EXTERNAL_STORAGE"],
        dalvik_analysis_log="Vulnerabilidade detectada em Logcat..."
        # Adicione aqui todos os outros campos OBRIGATÓRIOS do seu modelo
    )


# Mock de saída (relatório perfeito que o LLM retornaria)
# NOVO: mock_themis_report_json (ASSUMINDO CAMPOS OBRIGATÓRIOS NO NÍVEL SUPERIOR)

# tests/unit/test_analysis_domain.py (MOCK CORRIGIDO FINALMENTE)

@pytest.fixture
def mock_themis_report_json() -> str:
    report_data = {
        "package_name": "com.example.vulnerableapp",
        "score": 7.5, # Agora o Pydantic vai aceitar isso
        "summary": "O app apresenta risco médio devido a permissões excessivas.",
        "vulnerability_name": "Insegurança na Comunicação",
        "severity_level": "High",
        "mitigation_actions": ["Implementar TLS/SSL em todas as comunicações."],
        "findings": [{"risk": "HIGH", "description": "Permissão INTERNET sem uso de TLS."}]
    }
    return json.dumps(report_data)

@pytest.fixture
def mock_themis_report_pydantic(mock_themis_report_json: str) -> ThemisAIReport:
    """Retorna a versão Pydantic do relatório mockado para simular o retorno do LLM."""
    # Deserializa o JSON mockado para o objeto Pydantic
    return ThemisAIReport.model_validate_json(mock_themis_report_json)

# --------------------
# TESTES DA LÓGICA DE CACHE
# --------------------

@pytest.mark.asyncio
async def test_perform_analysis_cache_hit(
        mock_androguard_input: AndroguardAnalysis,
        mock_themis_report_json: str
):
    """
    Testa o cenário onde o relatório é encontrado no cache (CACHE HIT).
    O LLM NUNCA deve ser chamado.
    """
    # 1. Configurar Mocks dos Serviços

    # Criamos Mocks para todos os serviços injetados
    mock_retriever = AsyncMock()
    mock_generator = AsyncMock()
    mock_cache = AsyncMock()

    # Configuramos o Mock do Cache para SIMULAR O ACERTO:
    mock_cache.get.return_value = mock_themis_report_json

    # 2. Instanciar a Unidade de Teste (AnalysisDomain)
    domain = AnalysisDomain(
        retriever=mock_retriever,
        generator=mock_generator,
        cache=mock_cache
    )

    # 3. Executar o Método
    report = await domain.perform_analysis(mock_androguard_input)

    # 4. Asserts (Verificações)

    # VERIFICAÇÃO 1: O LLM NÃO FOI CHAMADO! (Mais importante)
    mock_generator.generate_response_async.assert_not_called()

    # VERIFICAÇÃO 2: O método de busca RAG (OpenSearch) TAMBÉM NÃO FOI CHAMADO
    mock_retriever.search_vector.assert_not_called()

    # VERIFICAÇÃO 3: O resultado é do tipo e valor esperado
    assert isinstance(report, ThemisAIReport)
    assert report.score == 7.5
    assert report.summary == "O app apresenta risco médio devido a permissões excessivas."

    # VERIFICAÇÃO 4: O método 'get' do cache foi chamado corretamente
    mock_cache.get.assert_called_once()

    # VERIFICAÇÃO 5: O método 'set' do cache (escrita) não foi chamado, pois foi lido
    mock_cache.set.assert_not_called()

@pytest.mark.asyncio
async def test_perform_analysis_cache_miss(
        mock_androguard_input: AndroguardAnalysis,
        mock_themis_report_json: str,
        mock_themis_report_pydantic: ThemisAIReport
):
    """
    Testa o cenário onde o relatório NÃO é encontrado no cache (CACHE MISS).
    O fluxo RAG Duplo e a escrita no cache DEVEM ser chamados.
    """
    # 1. Configurar Mocks dos Serviços
    mock_retriever = AsyncMock()
    mock_generator = AsyncMock()
    mock_cache = AsyncMock()

    # Configuração de CACHE MISS:
    mock_cache.get.return_value = None  # <--- O cache retorna None

    # Configuração do LLM (o que ele deve retornar)
    mock_generator.generate_response_async.return_value = mock_themis_report_json

    # Configuração do RAG (OpenSearch) - O RAG deve retornar contexto (simulando a busca)
    mock_retriever.search_vector.return_value = ["Contexto de RAG 1", "Contexto de RAG 2"]

    # 2. Instanciar a Unidade de Teste
    domain = AnalysisDomain(
        retriever=mock_retriever,
        generator=mock_generator,
        cache=mock_cache
    )

    # 3. Executar o Método
    report = await domain.perform_analysis(mock_androguard_input)

    # 4. Asserts (Verificações)

    # VERIFICAÇÃO 1: RAG (Retriever) FOI CHAMADO (Para buscar o contexto)
    mock_retriever.search_vector.assert_called_once()

    # VERIFICAÇÃO 2: O LLM (Generator) FOI CHAMADO (Para gerar o relatório)
    mock_generator.generate_response_async.assert_called_once()

    # VERIFICAÇÃO 3: O resultado foi salvo no CACHE (Escrita)
    # Usamos o 'ANY' para ignorar o valor exato do hash de cache, que é dinâmico.
    mock_cache.set.assert_called_once_with(
        ANY,
        mock_themis_report_json,
        # Você pode adicionar o 'ex' (tempo de expiração) se o seu 'cache.set' for assim:
        ex=ANY
    )

    # VERIFICAÇÃO 4: O resultado é o esperado (Verificando a saída do domínio)
    assert isinstance(report, ThemisAIReport)
    assert report.score == 7.5