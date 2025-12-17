# app/domain/analysis_domain.py (VERSÃO CORRIGIDA E FINALIZADA)

from __future__ import annotations
from typing import TYPE_CHECKING, List
import hashlib
import json  # Usado para manipulação e validação de strings JSON.

# --- Importações de Modelos de I/O ---
from ..models.app_analysis import AndroguardAnalysis, ThemisAIReport

# (Importações de serviços omitidas por brevidade, mas devem estar no topo)
if TYPE_CHECKING:
    from app.services.opensearch_service import OpenSearchService
    from app.services.llama_service import LlamaService
    # Adicionando o serviço de cache que discutiremos a seguir (RedisService)
    from app.services.redis_service import RedisService


class AnalysisDomain:
    """
    Classe de Domínio responsável por orquestrar a Análise de Segurança RAG Duplo.
    Coordena a busca (OpenSearch) e a geração (LlamaService) para produzir o relatório.
    """

    # ----------------------------------------------------------------------
    # 1. CONSTRUTOR (__init__) - (Pronto para receber o Cache)
    # ----------------------------------------------------------------------
    def __init__(
            self,
            retriever: OpenSearchService,
            generator: LlamaService,
            # Placeholder para o RedisService que vamos injetar:
            cache: RedisService
    ):
        """Recebe os serviços de infraestrutura injetados."""
        self.retriever = retriever
        self.generator = generator
        self.cache = cache  # Serviço de cache injetado

    # ----------------------------------------------------------------------
    # 2. MÉTODO PRINCIPAL (perform_analysis) - Lógica de RAG Duplo
    # ----------------------------------------------------------------------
    async def perform_analysis(self, req: AndroguardAnalysis) -> ThemisAIReport:
        """
        Executa a análise de segurança de ponta a ponta, com verificação de Cache (Redis).
        """

        # ----------------------------------------------------------------------
        # ETAPA 0: VERIFICAÇÃO DE CACHE E HASHING
        # ----------------------------------------------------------------------

        # 1. Gera uma chave única (hash) a partir do JSON de entrada.
        # Isso garante que se o payload for idêntico, a chave será a mesma.
        input_hash = hashlib.sha256(req.model_dump_json().encode()).hexdigest()
        cache_key = f"themisai:analysis:{input_hash}"  # Exemplo: themisai:analysis:a3b4c5d6...

        # 2. Tenta obter o relatório do cache (Redis)
        cached_report = await self.cache.get(cache_key)

        if cached_report:
            # Se existir no cache, retorna o relatório instantaneamente (milissegundos)
            print(f"CACHE HIT: Retornando relatório para chave {cache_key}")
            # Desserializa a string JSON do Redis de volta para o modelo Pydantic.
            return ThemisAIReport.model_validate_json(cached_report)

            # ----------------------------------------------------------------------
        # ETAPA 1 a 3: PROCESSAMENTO (CACHE MISS)
        # ----------------------------------------------------------------------
        # Se chegarmos aqui, o cache falhou (CACHE MISS), e o processamento é necessário.

        # --- ETAPA 1: FORMATAR CONTEXTO ESPECÍFICO (INPUT) ---
        app_data_json = req.model_dump_json(indent=2)
        specific_context = f"""
        # CONTEXTO ESPECÍFICO DA APLICAÇÃO (ANDROGUARD)
        Pacote: {req.package_name}.
        Dados Analíticos Brutos (JSON):
        {app_data_json}
        """

        # --- ETAPA 2: BUSCA RAG (RECUPERAÇÃO DO MITRE) ---
        mitre_chunks: List[str] = await self.retriever.search_vector(specific_context)

        # --- ETAPA 3: GERAÇÃO DO PROMPT FINAL ---
        SYSTEM_PROMPT = self._get_system_prompt()
        final_prompt = self._format_final_prompt(
            system_prompt=SYSTEM_PROMPT,
            app_context=specific_context,
            mitre_context=mitre_chunks
        )

        # --- ETAPA 4: CHAMADA AO LLM (Corrigida) ---
        json_response_str: str = await self.generator.generate_response_async(
            prompt=final_prompt,
            max_tokens=512
        )

        # ----------------------------------------------------------------------
        # ETAPA 5: ARMAZENAMENTO NO CACHE E RETORNO
        # ----------------------------------------------------------------------

        # 3. Salvar o resultado no cache antes de retornar.
        # Tempo de Vida (TTL): 3600 segundos = 1 hora.
        TTL_SECONDS = 3600
        await self.cache.set(cache_key, json_response_str, ex=TTL_SECONDS)

        # 4. VALIDAÇÃO PYDANTIC E RETORNO
        try:
            # Finaliza a validação e retorna o relatório ao chamador.
            return ThemisAIReport.model_validate_json(json_response_str)
        except json.JSONDecodeError as e:
            # Lida com o erro de JSON malformado do LLM.
            raise ValueError(f"LLM retornou JSON inválido. Erro: {e}. Output: {json_response_str[:200]}...")

    # ----------------------------------------------------------------------
    # MÉTODOS DE ENGENHARIA DE PROMPT (Implementação)
    # ----------------------------------------------------------------------

    def _get_system_prompt(self) -> str:
        """Define o papel da IA e a restrição de formato de saída JSON."""
        # 1. Obtém o esquema JSON do modelo Pydantic
        schema = ThemisAIReport.model_json_schema()
        # 2. Converte o esquema para string JSON formatada
        schema_json_str = json.dumps(schema, indent=2)

        # 3. Monta o Prompt de Sistema com as instruções de Persona e o Schema
        return f"""
        Você é um Analista de Segurança Sênior, especializado em análise estática de aplicativos Android.

        Sua única e estrita tarefa é ANALISAR os dados brutos do aplicativo fornecidos
        e usar o CONTEXTO MITRE fornecido para identificar vulnerabilidades e mitigá-las.

        SUA RESPOSTA DEVE SER ESTREITAMENTE O OBJETO JSON COMPLETO, SEM QUALQUER TEXTO EXPLICATIVO, 
        CABEÇALHO, MARCAÇÃO DE CÓDIGO (```JSON) OU INTRODUÇÃO.

        FORMATO DE SAÍDA OBRIGATÓRIO (JSON Schema):
        ---
        {schema_json_str}
        ---
        """

    def _format_final_prompt(self, system_prompt: str, app_context: str, mitre_context: List[str]) -> str:
        """Cria o prompt final combinando todos os componentes do RAG Duplo."""
        mitre_context_str = "\n".join([f"  - {c}" for c in mitre_context])

        return f"""
        {system_prompt}

        ################################################################################
        # CONTEXTO GERAL DE SEGURANÇA (MITRE ATT&CK Mobile)
        ################################################################################
        Use este contexto para identificar e categorizar vulnerabilidades presentes nos 
        dados do aplicativo e sugerir mitigações.

        Chunk de Conhecimento:
        {mitre_context_str}


        ################################################################################
        # DADOS BRUTOS DA APLICAÇÃO (ANDROGUARD)
        ################################################################################
        Analise o conteúdo abaixo (logs, permissões, resultados de análise estática)
        para correlacionar com as técnicas MITRE.

        {app_context}


        ################################################################################
        # TAREFA FINAL E RESTRIÇÕES
        ################################################################################
        Sua análise deve ser completa, baseada estritamente nos dados fornecidos. 
        Gere o relatório de segurança completo no formato JSON ESPECIFICADO NO INÍCIO 
        (ThemisAIReport), sem adicionar nenhuma outra informação.
        """