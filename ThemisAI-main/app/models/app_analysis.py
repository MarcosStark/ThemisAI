from pydantic import BaseModel
from typing import List


# ----------------------------------------------------
# 1. Modelo de ENTRADA (Input)
# Recebe os dados brutos da análise do Androguard
# ----------------------------------------------------
class AndroguardAnalysis(BaseModel):
    # Identificação única do aplicativo
    package_name: str

    # Contexto de Segurança: Lista de permissões que o app solicita
    permissions_list: List[str]

    # Dados Brutos: Log ou resumo das chamadas suspeitas (o que o LLM vai analisar)
    dalvik_analysis_log: str


# ----------------------------------------------------
# 2. Modelo de SAÍDA (Output)
# O contrato JSON que o LLM deve gerar e a API vai retornar
# ----------------------------------------------------
# app/models/app_analysis.py

class ThemisAIReport(BaseModel):
    # Campos obrigatórios adicionais
    package_name: str # Já estava no seu mock, mas garanta que está no modelo
    score: float      # <-- CAMPO NOVO E OBRIGATÓRIO PARA O TESTE
    summary: str      # <-- CAMPO NOVO (Também estava no seu mock)

    # O que foi encontrado (ex: "Exposição de Token em Log")
    vulnerability_name: str

    # Nível de risco (ex: "Crítico", "Médio")
    severity_level: str

    # A solução: Lista de ações de mitigação para o desenvolvedor
    mitigation_actions: List[str]

    # O campo 'findings' também deve estar aqui, se for obrigatório
    # findings: List[FindingModel]