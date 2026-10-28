# assistente_ia.py

# Assistente de vendas (Gemini) para o painel_vendas.py, em DUAS requisições:

#

#   1) PLANEJAMENTO: a IA recebe a pergunta (anonimizada) e o esquema dos dados (colunas, períodos,

#      métricas) e devolve um plano em JSON dizendo QUAIS dados precisa consultar. Não recebe linhas.

#   2) EXECUÇÃO LOCAL: este código valida o plano (lista de dimensões/métricas permitidas), executa as

#      consultas na base completa e gera tabelas agregadas, sem herdar filtros visuais do painel.

#   3) RESPOSTA: a IA recebe a pergunta + as tabelas calculadas e escreve a resposta.

#

# Privacidade: nomes de clientes, vendedores, representantes e gerentes nunca são enviados; viram

# "Cliente 1", "Vendedor 2"... (consistentes durante a conversa) e são trocados de volta localmente.

#

# Chave da API (uma das opções):

#   - .streamlit/secrets.toml:  GEMINI_API_KEY = "sua-chave"

#   - variável de ambiente:     GEMINI_API_KEY

# Modelo (opcional): variável de ambiente GEMINI_MODEL.

import json
import difflib

import os

import re

import time

import unicodedata

from datetime import datetime

from functools import lru_cache

import pandas as pd

import streamlit as st

from google import genai

from google.genai import types

MODELO_GEMINI = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

MAX_CONSULTAS = 4            # consultas por pergunta

LIMITE_PADRAO = 20           # linhas por consulta quando o plano não diz

LIMITE_MAXIMO = 200          # máximo de linhas por consulta

MAX_LINHAS_TOTAIS = 300      # máximo de linhas somando todas as consultas

MAX_VALORES_LISTADOS = 40    # valores listados no esquema para dimensões pequenas
MIN_REGISTROS_GRUPO_PRIVACIDADE = 0  # bloqueia divulgação de grupos com menos de cinco registros

TURNOS_HISTORICO = 4         # turnos anteriores enviados ao planejador

# ============================================================

# INSTRUÇÕES DA IA

# ============================================================

INSTRUCOES_PLANEJADOR = """

Você é o PLANEJADOR de consultas de um assistente de análise de vendas.

Você NÃO responde à pergunta. Sua tarefa é decidir quais dados agregados precisam ser

consultados no sistema local para que outra etapa consiga responder.

Responda SOMENTE com um objeto JSON válido, sem texto fora dele.

FORMATO:

{

  "consultas": [

    {

      "descricao": "o que esta consulta responde",

      "agrupar_por": ["produto", "mes"],

      "metricas": ["faturamento", "quantidade", "preco_medio"],

      "filtros": [{"dimensao": "sku", "operador": "igual", "valores": ["GP2000"]}],

      "periodo": {"inicio": "2026-01-01", "fim": "2026-12-31"},

      "somente_ultima_data": false,

      "ordenar_por": "faturamento",

      "direcao": "desc",

      "limite": 20

    }

  ],

  "esclarecimento": null

}

REGRAS:

1. Use somente as dimensões, métricas e operadores listados no esquema recebido. Todas as colunas adicionais estão disponíveis com IDs como campo_5; a descrição ao lado informa o nome original da coluna. Use o ID, não o nome do cabeçalho, nos planos. As colunas adicionais aparecem como campo_1, campo_2 etc.; use esses IDs literalmente para agrupar e filtrar.

   Operadores de filtro: "igual", "contem", "diferente".

2. Clientes, vendedores, representantes e gerentes aparecem como identificadores

   genéricos ("Cliente 3", "Vendedor 1"). Use-os exatamente assim nos filtros, com "igual".

   Nunca tente descobrir nomes reais.

3. Quando a pergunta trouxer um código identificador exato, como BP500, SKU ou código de produto, use o operador "igual" e filtre pela dimensão mais específica disponível (priorize SKU/código; se não houver, use produto). Não use "contem" para códigos exatos. Use "contem" somente quando o usuário pedir uma busca parcial por descrição/nome. Se a pergunta trouxer um identificador pseudonimizado como "Dado 1", use esse identificador exatamente como valor do filtro. Não invente códigos nem substitua o código citado por outro.

4. Use "periodo" (datas ISO AAAA-MM-DD) para intervalos. Datas relativas ("mês passado",

   "este ano") devem ser calculadas a partir de "hoje" no esquema. Se a pergunta não cita

   período, não envie "periodo": considere toda a base recebida. Os filtros visuais
   do painel não se aplicam às consultas do assistente; filtre apenas conforme a pergunta.

5. Para "última compra" ou "última data", use "somente_ultima_data": true com a métrica

   "faturamento". Use "data_ultima" somente se essa métrica estiver disponível no esquema.
   Se o esquema informar que Data Pedido é sensível, nunca solicite dia exato nem data_ultima;
   responda em granularidade mensal, trimestral ou anual. Para a última compra de cada item,
   agrupe por esse item.

6. "preco_medio" é faturamento ÷ quantidade (média ponderada). Para preço médio por

   mês, agrupe por "mes".

7. Para listas ("quais produtos...", "quem comprou..."), agrupe pela dimensão pedida e use

   um "limite" suficiente (até 200). Não limite a 10 quando a pergunta pede a lista completa.

8. Use no máximo 4 consultas. Prefira poucas consultas bem escolhidas. Inclua uma consulta de

   total geral (sem agrupar_por) quando a pergunta envolver participação ou comparação.

9. Use o histórico recente para resolver referências como "essa compra", "ele", "e em 2025?",

   reaproveitando os filtros da consulta anterior.

10. Se a pergunta for ambígua ou não puder ser respondida com as dimensões disponíveis,

    devolva "consultas": [] e escreva em "esclarecimento" uma pergunta curta ao usuário,

    ou explique o que falta nos dados.

11. Alguns grupos com menos de 5 registros serão ocultados por privacidade. Não tente contornar

    essa regra com consultas adicionais ou combinações alternativas de dimensões.

"""

INSTRUCOES_IA = """

Você é um assistente especializado em análise comercial e vendas.

Responda sempre em português do Brasil.

Você recebe a pergunta do usuário e RESULTADOS de consultas já calculadas localmente

sobre toda a base de vendas recebida. Os filtros visuais do painel não se aplicam às
consultas do assistente; somente os filtros explicitamente definidos na pergunta são usados.

REGRAS OBRIGATÓRIAS:

1. Utilize exclusivamente os resultados fornecidos. Não invente valores, quantidades,

   datas, produtos ou resultados.

2. Se os resultados forem insuficientes ou uma consulta tiver "avisos", informe a limitação

   e diga qual dado faltou. Se uma consulta voltou com 0 linhas, informe que não há vendas

   para aqueles critérios na base de vendas consultada; não use outros números para preencher.

3. Valores monetários em reais (R$), com duas casas decimais, no padrão brasileiro.

4. Não tente identificar pessoas, empresas, produtos, pedidos ou documentos. Preserve os

   identificadores genéricos como "Cliente 1", "Vendedor 2", "Representante 1", "Gerente 1"

   e "Dado 1" exatamente como aparecem; os valores reais serão restaurados localmente.

5. Não afirme que uma linha representa um pedido único. Diferencie registros, produtos

   e pedidos quando isso for relevante.

6. Cada consulta traz "totais_do_filtro". Se "truncado" for verdadeiro, avise que só as

   primeiras linhas foram trazidas e que há mais ("linhas_totais").

7. Preço médio = faturamento ÷ quantidade (média ponderada). Não recalcule métricas

   que já vieram prontas; apenas cite-as e explique o cálculo quando ajudar.

8. Não invente motivos para aumentos ou quedas. Não confunda faturamento com lucro ou margem.

9. Para listas, apresente uma tabela em Markdown. Cite o período considerado quando houver.

10. Se faltar informação para responder, faça uma pergunta objetiva de esclarecimento.

"""

# ============================================================

# NORMALIZAÇÃO E CONVERSÃO

# ============================================================

def normalizar(texto):

    """Remove acentos, diferenças de caixa e espaços repetidos."""

    texto = unicodedata.normalize("NFKD", str(texto))

    texto = "".join(c for c in texto if not unicodedata.combining(c))

    return re.sub(r"\s+", " ", texto.casefold().strip())

@lru_cache(maxsize=100_000)

def _normalizar_posicional(texto):

    """Como normalizar(), mas com exatamente 1 caractere de saída por caractere de entrada.

    Permite achar nomes ignorando acentos/caixa e trocar o trecho no texto original.

    """

    saida = []

    for c in str(texto):

        base = "".join(x for x in unicodedata.normalize("NFKD", c) if not unicodedata.combining(x))

        base = base.casefold()

        saida.append(base if len(base) == 1 else c)

    return "".join(saida)

def localizar_coluna(df, colunas_possiveis):

    """Localiza uma coluna ignorando acentos e maiúsculas (aceita um nome ou uma lista)."""

    if isinstance(colunas_possiveis, str):

        colunas_possiveis = [colunas_possiveis]

    mapa = {normalizar(coluna): coluna for coluna in df.columns}

    for coluna in colunas_possiveis:

        chave = normalizar(coluna)

        if chave in mapa:

            return mapa[chave]

    raise ValueError("Nenhuma das colunas foi encontrada: " + ", ".join(colunas_possiveis))

def localizar_coluna_opcional(df, colunas_possiveis):

    try:

        return localizar_coluna(df, colunas_possiveis)

    except ValueError:

        return None

def converter_valor(valor):

    """Converte valores monetários (formato brasileiro ou internacional) para float."""

    if pd.isna(valor):

        return None

    if isinstance(valor, (int, float)):

        return float(valor)

    texto = str(valor).strip()

    if not texto:

        return None

    texto = texto.replace("R$", "").replace("\u00a0", "").replace(" ", "")

    if "," in texto and "." in texto:

        if texto.rfind(",") > texto.rfind("."):

            texto = texto.replace(".", "").replace(",", ".")

        else:

            texto = texto.replace(",", "")

    elif "," in texto:

        texto = texto.replace(",", ".")

    try:

        return float(texto)

    except (ValueError, TypeError):

        return None

def _valores(serie):

    """Série numérica (float; NaN quando inválido). Colunas já numéricas não passam por conversão linha a linha."""

    if pd.api.types.is_numeric_dtype(serie):

        return pd.to_numeric(serie, errors="coerce").astype(float)

    return serie.map(converter_valor).astype(float)

def sanitizar_valor_textual(valor):
    """Remove padrões comuns de dados pessoais antes de qualquer texto chegar ao Gemini."""
    texto = str(valor)
    texto = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[EMAIL_REMOVIDO]", texto)
    texto = re.sub(r"(?<!\d)\d{3}\.?\d{3}\.?\d{3}-?\d{2}(?!\d)", "[DOCUMENTO_REMOVIDO]", texto)
    texto = re.sub(r"(?<!\d)\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}(?!\d)", "[DOCUMENTO_REMOVIDO]", texto)
    texto = re.sub(r"(?<!\d)(?:\+?55\s*)?(?:\(\d{2}\)\s*|\d{2}\s+)9?\d{4}[-.\s]?\d{4}(?!\d)", "[TELEFONE_REMOVIDO]", texto)
    return texto

def formatar_moeda(valor):

    return f"R$ {float(valor):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# ============================================================

# DIMENSÕES E MÉTRICAS

# ============================================================

# LISTA PERMITIDA: cabeçalhos fora desta lista não são disponibilizados ao assistente.
COLUNAS_PERMITIDAS = {normalizar(nome) for nome in (
    "Classe Produto", "Cliente Gerencial", "Grupo Empresarial", "Empresa", "Ano-Mês", "Ano-Mes",
    "Ano Fat.", "Mês Fat.", "Mes Fat.", "Trimestre", "Nº Pedido", "No Pedido", "Data Pedido",
    "Produto", "Voltagem", "Qtde", "Quantidade", "Vlr unitário", "Vlr unitario",
    "Vlr total dos produtos", "Data Faturamento", "IPI", "Valor Bruto da NF", "ST", "Total Bruto",
    "Comissão Gerada", "Comissao Gerada", "Representante", "Tipo de Operação", "Tipo de Operacao",
    "Departamento", "Vendedor atual do cadastro", "Gerente região Venda", "Gerente regiao Venda",
    "Gerente região Cadastro", "Gerente regiao Cadastro", "Status Pedido", "Status Faturamento",
    "Segmento", "Porte", "SKU", "Cidade", "Estado", "Região", "Regiao",
)}
COLUNAS_TEMPO_PRESERVADAS = {normalizar(nome) for nome in (
    "Ano-Mês", "Ano-Mes", "Ano Fat.", "Mês Fat.", "Mes Fat.", "Trimestre", "Data Pedido", "Data Faturamento",
)}
COLUNAS_VALOR_PRESERVADAS = {normalizar(nome) for nome in (
    "Qtde", "Quantidade", "Vlr unitário", "Vlr unitario", "Vlr total dos produtos", "IPI",
    "Valor Bruto da NF", "ST", "Total Bruto", "Comissão Gerada", "Comissao Gerada",
)}

# Colunas auxiliares criadas localmente pelo painel_vendas.py.
# São aceitas apenas quando existem no DataFrame recebido; não liberam outras colunas da base.
COLUNAS_INTERNAS_PERMITIDAS = {
    normalizar(nome) for nome in (
        "_valor", "_quantidade", "_data", "_cliente", "_produto",
        "_vendedor", "_segmento",
    )
}

# prefixo != None: dado pessoal/comercial sensível, sempre enviado como "Prefixo N".

DIMENSOES_TEXTO = {

    "cliente": {"rotulo": "cliente gerencial", "aliases": ["Cliente Gerencial", "Cliente"], "prefixo": "Cliente"},

    "produto": {"rotulo": "produto/descrição comercial", "aliases": ["Produto", "Descrição Produto", "Descricao Produto", "Nome Produto", "Nome do Produto", "Descrição", "Descricao", "Produto/Descrição", "Produto/Descricao"], "prefixo": "Dado"},

    "sku": {"rotulo": "código SKU do produto", "aliases": ["SKU", "Código Produto", "Codigo Produto", "Código do Produto", "Codigo do Produto", "Cod Produto", "Cod. Produto", "Cod do Produto", "Cod. do Produto", "Código SKU", "Codigo SKU", "Cod SKU", "Cod. SKU", "Referência", "Referencia", "Código Item", "Codigo Item", "Cod Item", "Cod. Item"], "prefixo": "Dado"},

    "vendedor": {"rotulo": "vendedor", "aliases": ["Vendedor", "Representante", "Vendedor atual do cadastro"], "prefixo": "Vendedor"},

    "representante": {"rotulo": "representante", "aliases": ["Representante"], "prefixo": "Representante"},

    "vendedor_cadastro": {"rotulo": "vendedor atual do cadastro", "aliases": ["Vendedor atual do cadastro"], "prefixo": "Vendedor"},

    "gerente_regiao": {"rotulo": "gerente de região (cadastro)", "aliases": ["Gerente região Cadastro", "Gerente região"], "prefixo": "Gerente"},

    "segmento": {"rotulo": "segmento", "aliases": ["Segmento"], "prefixo": "Dado"},

    "empresa": {"rotulo": "empresa", "aliases": ["Empresa"], "prefixo": "Dado"},

    "status_faturamento": {"rotulo": "status de faturamento", "aliases": ["Status Faturamento"], "prefixo": "Dado"},

    "departamento": {"rotulo": "departamento", "aliases": ["Departamento"], "prefixo": "Dado"},

}

# Somente campos com dados pessoais, credenciais ou identificadores financeiros são bloqueados.
# Campos comerciais (cliente, produto, SKU, vendedor, empresa, pedido, departamento etc.)
# continuam disponíveis; dimensões que possuem pseudônimo continuam pseudonimizadas.
COLUNAS_SENSIVEIS = {
    normalizar(nome) for nome in (
        "CPF", "CNPJ", "CPF/CNPJ", "CNPJ Matriz", "CPF do Cliente",
        "Documento", "Documento Pessoal", "E-mail", "Email",
        "Telefone", "Celular", "Endereço", "Endereco",
        "Chave Pix", "Conta Bancária", "Agência Bancária",
        "Senha", "Login", "Data de Nascimento", "Nome Completo",
        "Nome da Pessoa", "Nome do Contato", "Contato Pessoal",
    )
}

# Termos específicos de dados sensíveis. Não incluir termos comerciais genéricos
# como cliente, produto, SKU, pedido, empresa, vendedor ou departamento.
TERMOS_BLOQUEADOS = (
    "cpf", "cnpj", "documento pessoal", "documento do titular",
    "email", "e-mail", "telefone", "celular", "endereco",
    "chave pix", "conta bancaria", "agencia bancaria", "senha",
    "login", "data de nascimento", "nome completo", "nome da pessoa",
    "nome do contato", "contato pessoal",
)

def coluna_bloqueada(nome):
    """Identifica apenas colunas que contêm dados pessoais/financeiros sensíveis."""
    cabecalho = normalizar(nome)
    if cabecalho in COLUNAS_SENSIVEIS:
        return True
    # Evita bloquear colunas comerciais só porque contêm palavras como cliente/produto.
    return any(termo in cabecalho for termo in TERMOS_BLOQUEADOS)

# A data do pedido não é, por si só, um dado pessoal sensível.
def _data_pedido_sensivel(cols):
    return False

TEMPO_AGRUPAR = ["ano", "trimestre", "mes", "dia"]

FILTROS_TEMPO = ["ano", "trimestre_do_ano", "mes_do_ano"]

OPERADORES = {"igual": "igual", "contem": "contem", "diferente": "diferente", "=": "igual", "!=": "diferente"}

METRICA_SINONIMOS = {

    "valor": "faturamento", "vendas": "faturamento", "valor_total": "faturamento",

    "qtde": "quantidade", "qtd": "quantidade", "preco": "preco_medio", "preco_medio_unitario": "preco_medio",

    "linhas": "registros",

}

PLACEHOLDERS = {"(nao informado)", "(sem informacao)"}

REGEX_PSEUDONIMO = re.compile(r"^(cliente|vendedor|representante|gerente|dado) \d+$")

REGEX_PSEUDONIMO_TEXTO = re.compile(r"(?<!\w)(Cliente|Vendedor|Representante|Gerente|Dado) (\d+)(?!\w)", re.IGNORECASE)

def resolver_colunas(df, explicitas):

    """Descobre as colunas de cada dimensão. Parâmetros explícitos têm prioridade; senão, nomes padrão."""

    cols = {}

    for dim, cfg in DIMENSOES_TEXTO.items():

        explicita = explicitas.get(dim)

        achada = localizar_coluna_opcional(df, [explicita]) if explicita else None

        cols[dim] = achada or localizar_coluna_opcional(df, cfg["aliases"])

    for dim in ("representante", "vendedor_cadastro"):  # não duplicar a coluna de vendedor

        if cols[dim] and cols[dim] == cols["vendedor"]:

            cols[dim] = None

    ev = explicitas.get("valor")

    cols["valor"] = (localizar_coluna_opcional(df, [ev]) if ev else None) or localizar_coluna_opcional(

        df, ["Vlr total dos produtos", "Valor Total (Sem IPI)", "Valor Bruto da NF"]

    )

    eq = explicitas.get("qtd")

    cols["qtd"] = (localizar_coluna_opcional(df, [eq]) if eq else None) or localizar_coluna_opcional(df, ["Qtde", "Quantidade"])

    ed = explicitas.get("data")

    cols["data"] = (localizar_coluna_opcional(df, [ed]) if ed else None) or localizar_coluna_opcional(

        df, ["Data Faturamento", "Data Pedido"]

    )

    # Métricas e datas também precisam pertencer à lista permitida.
    for chave in ("valor", "qtd", "data"):
        origem = cols.get(chave)
        if origem and (
            normalizar(origem) not in COLUNAS_PERMITIDAS
            and normalizar(origem) not in COLUNAS_INTERNAS_PERMITIDAS
        ):
            cols[chave] = None

    # Defesa adicional: uma coluna sensível nunca pode ser usada como valor,
    # quantidade ou data, mesmo quando informada explicitamente na configuração.
    for chave in ("valor", "qtd", "data"):
        origem = cols.get(chave)
        if origem and coluna_bloqueada(origem):
            cols[chave] = None

    # Guarda localmente as colunas sensíveis apenas para anonimizar a pergunta.
    # Elas não ficam disponíveis para agrupamento, filtro ou consulta pelo Gemini.
    colunas_anonimizacao = {}
    for dim, cfg in DIMENSOES_TEXTO.items():
        origem = cols.get(dim)
        prefixo = cfg.get("prefixo")
        if origem and prefixo and coluna_bloqueada(origem):
            colunas_anonimizacao.setdefault(prefixo, set()).add(origem)

    # Também considera colunas bloqueadas que não estejam mapeadas como dimensão padrão.
    for nome in df.columns:
        if coluna_bloqueada(nome):
            colunas_anonimizacao.setdefault("Dado", set()).add(nome)

    cols["_colunas_anonimizacao"] = {
        prefixo: sorted(nomes) for prefixo, nomes in colunas_anonimizacao.items()
    }

    # Bloqueia dimensões sensíveis antes de construir o esquema enviado ao Gemini.
    # O DataFrame original permanece local e não é alterado.
    for dim in DIMENSOES_TEXTO:
        origem = cols.get(dim)
        if origem and (
            coluna_bloqueada(origem)
            or (
                normalizar(origem) not in COLUNAS_PERMITIDAS
                and normalizar(origem) not in COLUNAS_INTERNAS_PERMITIDAS
            )
        ):
            cols[dim] = None
        elif origem and normalizar(origem) not in COLUNAS_TEMPO_PRESERVADAS and normalizar(origem) not in COLUNAS_VALOR_PRESERVADAS:
            # Todas as dimensões categóricas permitidas são pseudonimizadas.
            cfg = DIMENSOES_TEXTO[dim]
            if not cfg.get("prefixo"):
                cfg["prefixo"] = "Dado"

    reservadas = {v for k, v in cols.items() if isinstance(v, str) and not k.startswith("_")}
    dinamicas, prefixos_sensiveis, numericas = {}, {}, {}
    for indice, nome in enumerate(df.columns, start=1):
        cabecalho = normalizar(nome)
        # Lista permitida: colunas não listadas não são expostas ao assistente.
        if cabecalho not in COLUNAS_PERMITIDAS or nome in reservadas or coluna_bloqueada(nome):
            continue
        chave = f"campo_{indice}"
        dinamicas[chave] = nome
        serie = df[nome]
        if cabecalho in COLUNAS_TEMPO_PRESERVADAS:
            continue
        if cabecalho in COLUNAS_VALOR_PRESERVADAS:
            # Valores podem vir como texto formatado (ex.: "R$ 1.234,56"); a conversão numérica é local.
            numericas[chave] = nome
        else:
            # Identificadores e categorias (inclusive números de pedido) viram Dado N.
            prefixos_sensiveis[chave] = "Dado"
    cols["_dinamicas"] = dinamicas
    cols["_prefixos_sensiveis"] = prefixos_sensiveis
    cols["_numericas"] = numericas
    return cols

# ============================================================

# PSEUDÔNIMOS (consistentes durante a conversa)

# ============================================================

def _pseudonimo(mapa, prefixo, nome):

    """Devolve 'Prefixo N' para o nome, criando o identificador se ainda não existir."""

    if normalizar(nome) in PLACEHOLDERS:

        return nome

    grupo = mapa.setdefault(prefixo, {})

    if nome not in grupo:

        grupo[nome] = len(grupo) + 1

    return f"{prefixo} {grupo[nome]}"

def _mapa_reverso(mapa):

    """{'cliente 3': 'NOME REAL'}"""

    reverso = {}

    for prefixo, nomes in mapa.items():

        for nome, numero in nomes.items():

            reverso[normalizar(f"{prefixo} {numero}")] = nome

    return reverso

def desanonimizar(texto, mapa):

    """Troca 'Cliente 3', 'Vendedor 1'... pelos nomes reais, localmente."""

    reverso = _mapa_reverso(mapa)

    if not reverso:

        return str(texto)

    def trocar(m):

        return reverso.get(normalizar(f"{m.group(1)} {m.group(2)}"), m.group(0))

    return REGEX_PSEUDONIMO_TEXTO.sub(trocar, str(texto))

def anonimizar_pergunta(pergunta, nomes_por_prefixo, mapa):

    """Troca, na pergunta, nomes de clientes/vendedores/etc. por identificadores genéricos.

    A busca ignora acentos e maiúsculas. Retorna (pergunta_segura, [(nome, identificador)]).

    """

    pergunta_segura = str(pergunta)

    pn = _normalizar_posicional(pergunta_segura)

    candidatos, vistos = [], set()

    for prefixo in ("Cliente", "Vendedor", "Representante", "Gerente", "Dado"):

        for nome in nomes_por_prefixo.get(prefixo, ()):

            nome = str(nome).strip()

            if len(nome) >= 2 and nome not in vistos and normalizar(nome) not in PLACEHOLDERS:

                vistos.add(nome)

                candidatos.append((nome, prefixo))

    candidatos.sort(key=lambda item: len(item[0]), reverse=True)

    substituicoes = []

    for nome, prefixo in candidatos:

        nn = _normalizar_posicional(nome)

        if nn not in pn:

            continue

        padrao = re.compile(r"(?<!\w)" + re.escape(nn) + r"(?!\w)")

        while True:

            m = padrao.search(pn)

            if not m:

                break

            indice = next((i for i, (n, _) in enumerate(substituicoes) if n == nome), None)

            if indice is None:

                substituicoes.append((nome, prefixo))

                indice = len(substituicoes) - 1

            marcador = f"__ent_{indice}__"

            ini, fim = m.span()

            pergunta_segura = pergunta_segura[:ini] + marcador + pergunta_segura[fim:]

            pn = pn[:ini] + marcador + pn[fim:]

    # Filtros de privacidade adicionais

    pergunta_segura = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[EMAIL_REMOVIDO]", pergunta_segura)

    pergunta_segura = re.sub(r"(?<!\d)\d{3}\.?\d{3}\.?\d{3}-?\d{2}(?!\d)", "[DOCUMENTO_REMOVIDO]", pergunta_segura)

    pergunta_segura = re.sub(r"(?<!\d)\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}(?!\d)", "[DOCUMENTO_REMOVIDO]", pergunta_segura)

    # Telefone: exige DDD (parênteses ou espaço) ou hífen, para não apagar "2025 2026".

    pergunta_segura = re.sub(

        r"(?<!\d)(?:\+?55\s*)?(?:\(\d{2}\)\s*|\d{2}\s+)9?\d{4}[-.\s]?\d{4}(?!\d)",

        "[TELEFONE_REMOVIDO]", pergunta_segura,

    )

    pergunta_segura = re.sub(r"(?<!\d)9?\d{4}-\d{4}(?!\d)", "[TELEFONE_REMOVIDO]", pergunta_segura)

    encontrados = []

    for indice, (nome, prefixo) in enumerate(substituicoes):

        identificador = _pseudonimo(mapa, prefixo, nome)

        pergunta_segura = pergunta_segura.replace(f"__ent_{indice}__", identificador)

        encontrados.append((nome, identificador))

    return pergunta_segura, encontrados

# ============================================================

# QUADRO DE DADOS LOCAL

# ============================================================

class Quadro:

    """Tabela enxuta (valor, quantidade, data) com colunas de dimensão criadas sob demanda."""

    def __init__(self, df, cols):

        self._df = df

        self.cols = cols

        self.tab = pd.DataFrame({"v": _valores(df[cols["valor"]]).to_numpy()})

        if cols.get("qtd"):

            self.tab["q"] = _valores(df[cols["qtd"]]).to_numpy()

        for chave, origem in cols.get("_numericas", {}).items():
            self.tab[f"n_{chave}"] = _valores(df[origem]).to_numpy()

        self.tem_data = False

        if cols.get("data"):

            datas = pd.to_datetime(df[cols["data"]], errors="coerce", dayfirst=True).dt.normalize()

            if datas.notna().any():

                self.tab["d"] = datas.to_numpy()

                self.tem_data = True

    @property

    def tem_quantidade(self):

        return "q" in self.tab.columns

    def coluna(self, dim):

        """Garante e devolve o nome da coluna de uma dimensão dentro de self.tab."""

        chave = f"x_{dim}"

        if chave in self.tab.columns:

            return chave

        if dim in DIMENSOES_TEXTO:

            origem = self._df[self.cols[dim]]

            self.tab[chave] = origem.fillna("(sem informação)").astype(str).str.strip().to_numpy()

        elif dim in self.cols.get("_dinamicas", {}):

            origem = self._df[self.cols["_dinamicas"][dim]]
            valores = origem.fillna("(sem informação)").astype(str).str.strip()
            self.tab[chave] = valores.to_numpy()

        elif dim == "ano":

            self.tab[chave] = self.tab["d"].dt.year.astype("Int64")

        elif dim == "trimestre":

            d = self.tab["d"]

            self.tab[chave] = (d.dt.year.astype("Int64").astype(str) + "-T" + d.dt.quarter.astype("Int64").astype(str)).where(d.notna())

        elif dim == "mes":

            self.tab[chave] = self.tab["d"].dt.strftime("%Y-%m")

        elif dim == "dia":

            self.tab[chave] = self.tab["d"].dt.strftime("%Y-%m-%d")

        else:

            raise ValueError(f"dimensão desconhecida: {dim}")

        return chave

def nomes_para_anonimizar(quadro):

    """{'Cliente': {nomes}, 'Vendedor': {...}} com os nomes reais presentes nos dados."""

    saida = {}

    # Colunas bloqueadas são usadas exclusivamente para substituir valores reais na pergunta.
    # Nenhum desses campos é incluído no esquema de consulta enviado ao Gemini.
    for prefixo, colunas in quadro.cols.get("_colunas_anonimizacao", {}).items():
        for coluna in colunas:
            if coluna in quadro._df.columns:
                valores = quadro._df[coluna].dropna().astype(str).str.strip().unique()
                saida.setdefault(prefixo, set()).update(valores)

    for dim, cfg in DIMENSOES_TEXTO.items():

        prefixo = cfg["prefixo"]

        if prefixo and quadro.cols.get(dim):

            valores = quadro._df[quadro.cols[dim]].dropna().astype(str).str.strip().unique()

            saida.setdefault(prefixo, set()).update(valores)

    for dim, prefixo in quadro.cols.get("_prefixos_sensiveis", {}).items():
        origem = quadro.cols.get("_dinamicas", {}).get(dim)
        if origem:
            valores = quadro._df[origem].dropna().astype(str).str.strip().unique()
            saida.setdefault(prefixo, set()).update(valores)
    return saida

# Cache local de candidatos normalizados. A lógica de seleção abaixo preserva os critérios
# da versão completa; o cache apenas evita reconstruir/normalizar os nomes em cada pergunta.
_CACHE_CANDIDATOS_FUZZY = {}
_LIMITE_CACHE_CANDIDATOS_FUZZY = 8


def corrigir_termos_similares(pergunta, quadro, limite=0.83, margem=0.07):
    """Valida somente nomes/termos da base, sem corrigir gramática da pergunta.

    Retorno: (pergunta_corrigida, correcoes, sugestao, metadados_sugestao).
    Correspondências exatas são preservadas; erros fortes são corrigidos localmente;
    casos plausíveis, porém incertos, pedem confirmação antes de consultar o Gemini.
    """
    texto = str(pergunta or "")
    if not texto.strip():
        return texto, [], None, None

    def compacto(valor):
        # Ignora caixa, acentos, espaços e pontuação para conferir nomes/códigos.
        return re.sub(r"[^a-z0-9]", "", normalizar(str(valor)))

    # Palavras funcionais da pergunta não devem ser interpretadas como nomes.
    ignorar = {normalizar(x) for x in (
        "qual", "quais", "foi", "foram", "e", "o", "a", "os", "as", "um", "uma",
        "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas", "por",
        "para", "com", "quanto", "quantos", "quantas", "me", "mostre", "mostrar",
        "traga", "ver", "veja", "analise", "analisar", "faturamento", "venda", "vendas",
        "quantidade", "vendida", "vendido", "total", "valor", "media", "médio", "medio",
        "pedido", "pedidos", "cliente", "clientes", "produto", "produtos", "sku", "vendedor",
        "vendedores", "representante", "representantes", "empresa", "segmento", "mes", "mês",
        "ano", "evolucao", "evolução", "maiores", "maior", "menores", "menor", "top",
        "ultimos", "últimos", "ultimo", "último", "ultimas", "últimas", "última", "ultima",
        "entre", "sobre", "dos", "das", "crescimento", "queda", "comparar", "comparacao",
    )}

    # Esses termos são rótulos da consulta, não o início de um nome, a menos que
    # o usuário os repita no próprio nome (ex.: "cliente Cliente Exato").
    rotulos_dimensao = {normalizar(x) for x in (
        "cliente", "clientes", "produto", "produtos", "sku", "vendedor", "vendedores",
        "representante", "representantes", "empresa", "empresas", "segmento", "segmentos",
        "gerente", "gerentes", "departamento", "departamentos",
    )}

    # Monta um catálogo local de termos. O texto completo da base nunca sai daqui.
    valores_por_dim = {}

    def adicionar(dim, valor):
        valor = str(valor).strip()
        nv = compacto(valor)
        if len(nv) >= 3:
            valores_por_dim.setdefault(dim, set()).add(valor)

    for dim in DIMENSOES_TEXTO:
        coluna = quadro.cols.get(dim)
        if coluna and coluna in quadro._df.columns:
            for valor in quadro._df[coluna].dropna().astype(str).unique():
                adicionar(dim, valor)

    # Campos adicionais permitidos pelo assistente também podem ser validados localmente.
    for dim, coluna in quadro.cols.get("_dinamicas", {}).items():
        if coluna in quadro._df.columns:
            for valor in quadro._df[coluna].dropna().astype(str).unique():
                adicionar(dim, valor)

    # Colunas bloqueadas continuam fora da consulta à IA, mas seus nomes podem ser
    # usados somente na validação local, para que a pergunta seja pseudonimizada depois.
    dimensoes_por_prefixo = {
        "Cliente": {"cliente"},
        "Vendedor": {"vendedor", "vendedor_cadastro"},
        "Representante": {"representante"},
        "Gerente": {"gerente_regiao"},
        "Dado": {"__geral__"},
    }
    for prefixo, colunas in quadro.cols.get("_colunas_anonimizacao", {}).items():
        dims_destino = dimensoes_por_prefixo.get(prefixo, {"__geral__"})
        for coluna in colunas:
            if coluna in quadro._df.columns:
                for valor in quadro._df[coluna].dropna().astype(str).unique():
                    for dim in dims_destino:
                        adicionar(dim, valor)

    if not valores_por_dim:
        return texto, [], None, None

    # A dica de dimensão mais próxima do termo limita candidatos: um nome de cliente
    # não deve ser corrigido para um produto parecido quando a pergunta diz "cliente".
    pistas_contexto = [
        (r"\bclientes?\b|\bcompradores?\b", {"cliente"}),
        (r"\bvendedores?\b", {"vendedor", "vendedor_cadastro"}),
        (r"\brepresentantes?\b", {"representante", "vendedor"}),
        (r"\b(?:sku|codigo|referencia)\b", {"sku", "produto"}),
        (r"\b(?:produtos?|itens?|modelos?)\b", {"produto", "sku"}),
        (r"\bsegmentos?\b", {"segmento"}),
        (r"\bempresas?\b", {"empresa"}),
        (r"\bdepartamentos?\b", {"departamento"}),
        (r"\bgerentes?\b", {"gerente_regiao"}),
        (r"\bstatus\s+(?:de\s+)?faturamento\b", {"status_faturamento"}),
    ]
    # Usa também o nome dos campos adicionais como pista contextual local.
    for dim, coluna in quadro.cols.get("_dinamicas", {}).items():
        nome_coluna = normalizar(coluna)
        if len(nome_coluna) >= 4:
            pistas_contexto.append((r"\b" + re.escape(nome_coluna) + r"\b", {dim}))

    def dimensoes_contextuais(posicao):
        trecho_anterior = normalizar(texto[max(0, posicao - 120):posicao])
        encontradas = []
        for padrao, dims in pistas_contexto:
            for m in re.finditer(padrao, trecho_anterior):
                encontradas.append((m.end(), dims))
        if not encontradas:
            return None
        return max(encontradas, key=lambda item: item[0])[1]

    # Um cache só é reutilizado se o catálogo de valores realmente não mudou.
    assinatura_catalogo = hash(tuple(
        (dim, frozenset(valores)) for dim, valores in sorted(valores_por_dim.items())
    ))
    chave_cache = (
        id(quadro._df), len(quadro._df), tuple(quadro._df.columns), assinatura_catalogo
    )
    normalizados = _CACHE_CANDIDATOS_FUZZY.get(chave_cache)
    if normalizados is None:
        catalogo = {}
        for dim, valores in sorted(valores_por_dim.items()):
            for valor in sorted(valores, key=lambda item: (normalizar(item), item.casefold())):
                nv = compacto(valor)
                if not nv:
                    continue
                registro = catalogo.setdefault(nv, {"original": valor, "dimensoes": set()})
                registro["dimensoes"].add(dim)
                # Preferir o rótulo mais legível/canônico em colisões de caixa/acentos.
                if len(valor) < len(registro["original"]):
                    registro["original"] = valor

        registros = tuple(
            (nv, item["original"], frozenset(item["dimensoes"]), frozenset(nv), bool(re.search(r"\d", nv)))
            for nv, item in catalogo.items()
        )
        indice_caracteres = {}
        indices_por_dimensao = {}
        indice_exato = {}
        for idx, (nv, _original, dims, caracteres, _tem_digito) in enumerate(registros):
            indice_exato[nv] = idx
            for dim in dims:
                indices_por_dimensao.setdefault(dim, set()).add(idx)
            for caractere in caracteres:
                indice_caracteres.setdefault(caractere, set()).add(idx)
        normalizados = (registros, indice_caracteres, indices_por_dimensao, indice_exato)
        if len(_CACHE_CANDIDATOS_FUZZY) >= _LIMITE_CACHE_CANDIDATOS_FUZZY:
            _CACHE_CANDIDATOS_FUZZY.clear()
        _CACHE_CANDIDATOS_FUZZY[chave_cache] = normalizados

    registros, indice_caracteres, indices_por_dimensao, indice_exato = normalizados
    if not registros:
        return texto, [], None, None

    padrao = re.compile(r"[\wÀ-ÖØ-öø-ÿ]+(?:[-./][\wÀ-ÖØ-öø-ÿ]+)*", re.UNICODE)
    tokens = list(padrao.finditer(texto))
    if not tokens:
        return texto, [], None, None

    spans = []
    vistos_spans = set()
    for i, token in enumerate(tokens):
        for quantidade in range(1, min(6, len(tokens) - i) + 1):
            fim = i + quantidade - 1
            partes = [normalizar(x.group()) for x in tokens[i:fim + 1]]
            if not any(parte not in ignorar for parte in partes):
                continue
            trecho = texto[token.start():tokens[fim].end()]
            alvo = compacto(trecho)
            if len(alvo) < 4:
                continue
            chave_span = (token.start(), tokens[fim].end())
            if chave_span not in vistos_spans:
                spans.append((token.start(), tokens[fim].end(), trecho, alvo, partes, i))
                vistos_spans.add(chave_span)

    if not spans:
        return texto, [], None, None

    # Antes da busca aproximada, identifica trechos que já correspondem a um nome
    # completo da base. Isso é importante para nomes compostos com pontuação ou
    # conectores, por exemplo "Basso & Pancote": sem essa proteção, o token
    # "Basso" poderia ser comparado isoladamente a outro nome como "PASSOS".
    spans_exatos = []
    for a, b, _trecho, alvo, _partes, _indice_inicio in spans:
        indice_nome_exato = indice_exato.get(alvo)
        if indice_nome_exato is None:
            continue
        dims_ativas_exatas = dimensoes_contextuais(a)
        dims_nome_exato = registros[indice_nome_exato][2]
        if not dims_ativas_exatas or (dims_nome_exato & dims_ativas_exatas):
            spans_exatos.append((a, b))

    def contido_em_nome_exato(a, b):
        return any(
            inicio_exato <= a and b <= fim_exato
            and (inicio_exato, fim_exato) != (a, b)
            for inicio_exato, fim_exato in spans_exatos
        )

    candidatos_por_span = []
    for a, b, trecho, alvo, partes, indice_inicio in spans:
        # Nunca proponha uma correção para parte de um nome composto que já foi
        # reconhecido integralmente na base.
        if contido_em_nome_exato(a, b):
            continue
        dims_ativas = dimensoes_contextuais(a)
        indices_dimensao = set()
        if dims_ativas:
            for dim in dims_ativas:
                indices_dimensao.update(indices_por_dimensao.get(dim, set()))
        # Se a dica não corresponde a nenhuma dimensão disponível, usa o catálogo todo.
        usar_catalogo_todo = not indices_dimensao

        # Um valor já existente exatamente nessa dimensão não deve ser proposto como erro.
        indice_alvo = indice_exato.get(alvo)
        if indice_alvo is not None:
            dims_candidato = registros[indice_alvo][2]
            if usar_catalogo_todo or (dims_candidato & dims_ativas):
                continue

        # Não usar o rótulo da dimensão como parte do nome candidato (ex.: "cliente Mercado").
        # Se a palavra se repetir depois do rótulo ("cliente Cliente Exato"), ela pode
        # fazer parte do nome real. Preposições/conjunções só serão aceitas se alinhadas
        # com os mesmos tokens do nome candidato, verificado abaixo.
        if partes[0] in rotulos_dimensao:
            rotulo_repetido = (
                indice_inicio > 0
                and normalizar(tokens[indice_inicio - 1].group()) == partes[0]
            )
            if not rotulo_repetido:
                continue

        alvo_tem_digito = bool(re.search(r"\d", alvo))
        alvo_alfanum = bool(re.search(r"[a-z]", alvo) and alvo_tem_digito)
        indices_candidatos = set()
        for caractere in set(alvo):
            indices_candidatos.update(indice_caracteres.get(caractere, set()))
        pontuados = []
        for idx in indices_candidatos:
            candidato_norm, candidato, dims_candidato, _caracteres, candidato_tem_digito = registros[idx]
            if not usar_catalogo_todo and not (dims_candidato & dims_ativas):
                continue
            if candidato_norm == alvo:
                continue
            # Se a expressão contém conectivos/preposições, só se compara com um nome
            # que possua exatamente esses tokens nessas posições (ex.: "Máquina de Lavar").
            if any(parte in ignorar for parte in partes):
                partes_candidato = [normalizar(t.group()) for t in padrao.finditer(candidato)]
                if len(partes_candidato) != len(partes) or any(
                    parte in ignorar and partes_candidato[pos] != parte
                    for pos, parte in enumerate(partes)
                ):
                    continue
            # Não transformar palavras em códigos nem aceitar pequenas diferenças de SKU.
            if alvo_alfanum and (not candidato_tem_digito or len(candidato_norm) != len(alvo)):
                continue
            if candidato_tem_digito != alvo_tem_digito:
                continue
            maior = max(len(alvo), len(candidato_norm))
            if abs(len(alvo) - len(candidato_norm)) > max(2, int(maior * 0.25)):
                continue
            comparador = difflib.SequenceMatcher(None, alvo, candidato_norm, autojunk=False)
            ratio = comparador.ratio()
            bloco = comparador.find_longest_match(0, len(alvo), 0, len(candidato_norm))
            cobertura = bloco.size / max(len(alvo), len(candidato_norm), 1)
            if cobertura < 0.50:
                continue
            # Curto patamar por tamanho: um erro em uma palavra pequena deve pedir
            # confirmação, enquanto um erro inequívoco em um nome maior pode ser corrigido.
            tamanho = max(len(alvo), len(candidato_norm))
            limite_confirmacao = 0.64 if tamanho <= 6 else (0.68 if tamanho <= 9 else 0.70)
            limite_auto = max(limite, 0.82 if tamanho <= 7 else 0.84)
            if alvo_alfanum:
                limite_confirmacao = max(limite_confirmacao, 0.76)
                limite_auto = max(limite_auto, 0.88)
            if ratio < limite_confirmacao:
                continue
            pontuados.append({
                "a": a, "b": b, "trecho": trecho, "alvo": alvo,
                "candidato_norm": candidato_norm, "candidato": candidato,
                "ratio": ratio, "cobertura": cobertura,
                "limite_auto": limite_auto, "limite_confirmacao": limite_confirmacao,
            })

        if not pontuados:
            continue
        # Ambiguidade é medida entre candidatos do MESMO trecho, não comparando
        # acidentalmente o nome de um cliente com o nome de um produto de outro trecho.
        por_candidato = {}
        for item in pontuados:
            existente = por_candidato.get(item["candidato_norm"])
            if existente is None or item["ratio"] > existente["ratio"]:
                por_candidato[item["candidato_norm"]] = item
        ranking = sorted(por_candidato.values(), key=lambda x: x["ratio"], reverse=True)
        melhor = ranking[0]
        empatados = [x for x in ranking[1:] if melhor["ratio"] - x["ratio"] < margem]
        opcoes = [melhor["candidato"]] + [x["candidato"] for x in empatados[:2]]
        candidatos_por_span.append({
            **melhor, "opcoes": opcoes, "ambigua": bool(empatados),
        })

    if not candidatos_por_span:
        return texto, [], None, None

    # Escolhe o trecho mais provável; em empate, prioriza o trecho mais específico/longer.
    candidatos_por_span.sort(
        key=lambda x: (x["ratio"], len(x["alvo"]), len(x["trecho"])), reverse=True
    )
    melhor = candidatos_por_span[0]
    if melhor["ratio"] >= melhor["limite_auto"] and not melhor["ambigua"]:
        corrigida = texto[:melhor["a"]] + melhor["candidato"] + texto[melhor["b"]:]
        return corrigida, [(melhor["trecho"], melhor["candidato"])], None, None

    # Só interrompe para confirmação quando existe semelhança razoável. O antigo
    # limiar de 50% gerava perguntas sobre palavras sem relação com um nome da base.
    if melhor["ambigua"]:
        opcoes_txt = " ou ".join(f'"{x}"' for x in melhor["opcoes"])
        sugestao = (
            f'Encontrei mais de um nome parecido com "{melhor["trecho"]}": {opcoes_txt}. '
            "Digite o nome correto para confirmar."
        )
    else:
        sugestao = (
            f'Em "{melhor["trecho"]}", você quis dizer "{melhor["candidato"]}"? '
            "Responda sim ou não."
        )
    meta = {
        "termo": melhor["trecho"],
        "candidato": melhor["candidato"],
        "opcoes": melhor["opcoes"],
        "ambigua": melhor["ambigua"],
    }
    return texto, [], sugestao, meta

def montar_esquema(quadro):

    """Descrição dos dados enviada ao planejador. Não contém linhas nem nomes de pessoas/clientes."""

    cols = quadro.cols

    esquema = {

        "hoje": datetime.now().strftime("%Y-%m-%d"),

        "total_de_registros": int(len(quadro.tab)),

    }

    if quadro.tem_data:

        if _data_pedido_sensivel(cols):
            esquema["periodo_dos_dados"] = {
                "primeiro_mes": quadro.tab["d"].min().strftime("%Y-%m"),
                "ultimo_mes": quadro.tab["d"].max().strftime("%Y-%m"),
                "observacao": "Data Pedido é sensível; exibir datas somente por mês, nunca dia exato.",
            }
        else:
            esquema["periodo_dos_dados"] = {
                "primeira_data": quadro.tab["d"].min().strftime("%Y-%m-%d"),
                "ultima_data": quadro.tab["d"].max().strftime("%Y-%m-%d"),
            }

    else:

        esquema["periodo_dos_dados"] = "sem coluna de data válida: não há dimensões nem filtros de tempo"

    dims = {}

    valores_possiveis = {}

    for dim, cfg in DIMENSOES_TEXTO.items():

        if not cols.get(dim):

            continue

        descricao = cfg["rotulo"]

        if cfg["prefixo"]:

            descricao += f' (valores aparecem como "{cfg["prefixo"]} N")'

        dims[dim] = descricao

        if not cfg["prefixo"] and dim != "produto":

            unicos = quadro._df[cols[dim]].dropna().astype(str).str.strip().unique().tolist()

            valores_possiveis[dim] = sorted(unicos)[:MAX_VALORES_LISTADOS] if len(unicos) <= MAX_VALORES_LISTADOS else f"{len(unicos)} valores distintos"

    for dim, origem in cols.get("_dinamicas", {}).items():
        prefixo = cols.get("_prefixos_sensiveis", {}).get(dim)
        dims[dim] = f"coluna {origem!r} (use o ID {dim})" + (f' (valores pseudonimizados como "{prefixo} N")' if prefixo else "")
        serie = quadro._df[origem].dropna().astype(str).str.strip()
        unicos = serie.unique().tolist()
        if prefixo:
            valores_possiveis[dim] = f"{len(unicos)} valores distintos, pseudonimizados; use o identificador citado na pergunta"
        elif len(unicos) <= MAX_VALORES_LISTADOS:
            valores_possiveis[dim] = [sanitizar_valor_textual(v) for v in unicos]
        else:
            valores_possiveis[dim] = f"{len(unicos)} valores distintos"

    esquema["dimensoes_texto_para_agrupar_e_filtrar"] = dims

    if quadro.tem_data:

        tempos_disponiveis = [t for t in TEMPO_AGRUPAR if not (_data_pedido_sensivel(cols) and t == "dia")]
        esquema["dimensoes_de_tempo_para_agrupar"] = tempos_disponiveis

        esquema["filtros_de_tempo"] = {"ano": "ex.: 2026", "trimestre_do_ano": "1 a 4", "mes_do_ano": "1 a 12"}
        if _data_pedido_sensivel(cols):
            esquema["regra_privacidade_data"] = "Data Pedido é sensível: não retorne dias exatos nem datas de primeira/última compra; use mês, trimestre ou ano."

    esquema["valores_possiveis"] = valores_possiveis

    metricas = {"faturamento": "soma do valor faturado", "registros": "número de linhas"}

    if quadro.tem_quantidade:

        metricas["quantidade"] = "soma das quantidades"

        metricas["preco_medio"] = "faturamento ÷ quantidade (média ponderada)"

    if cols.get("cliente"):

        metricas["clientes_distintos"] = "número de clientes diferentes"

    if cols.get("produto"):

        metricas["produtos_distintos"] = "número de produtos diferentes"

    if quadro.tem_data and not _data_pedido_sensivel(cols):

        metricas["data_primeira"] = "primeira data"

        metricas["data_ultima"] = "última data"

    for dim, origem in cols.get("_numericas", {}).items():
        # Defesa adicional: uma coluna bloqueada nunca vira métrica disponível à IA.
        if coluna_bloqueada(origem):
            continue
        metricas[f"soma_{dim}"] = f"soma da coluna numérica autorizada (ID {dim})"
        metricas[f"media_{dim}"] = f"média da coluna numérica autorizada (ID {dim})"
    esquema["metricas"] = metricas

    esquema["operadores_de_filtro"] = ["igual", "contem", "diferente"]

    return esquema

# ============================================================

# VALIDAÇÃO DO PLANO (nada vindo da IA é executado sem passar por aqui)

# ============================================================

def _lista(x):

    if x is None:

        return []

    return list(x) if isinstance(x, (list, tuple)) else [x]

def _chave(x):

    return normalizar(x).replace(" ", "_")

def _validar_consulta(c, dims_agrupar, filtros_validos, metricas_validas, tem_data):

    agrupar = []

    for d in _lista(c.get("agrupar_por")):

        d = _chave(d)

        if d not in dims_agrupar:

            raise ValueError(f"agrupamento por '{d}' indisponível nos dados")

        if d not in agrupar:

            agrupar.append(d)

    if len(agrupar) > 3:

        raise ValueError("agrupamentos demais (máximo 3)")

    metricas = []

    for m in _lista(c.get("metricas")):

        m = _chave(m)

        m = METRICA_SINONIMOS.get(m, m)

        if m not in metricas_validas:

            raise ValueError(f"métrica '{m}' indisponível nos dados")

        if m not in metricas:

            metricas.append(m)

    if not metricas:

        metricas = ["faturamento"]

    filtros = []

    for f in _lista(c.get("filtros")):

        if not isinstance(f, dict):

            raise ValueError("filtro inválido")

        dim = _chave(f.get("dimensao") or "")

        if dim not in filtros_validos:

            raise ValueError(f"filtro por '{dim}' indisponível nos dados")

        op = OPERADORES.get(_chave(f.get("operador") or "igual"))

        if op is None:

            raise ValueError("operador de filtro inválido")

        valores = [v for v in _lista(f.get("valores", f.get("valor"))) if isinstance(v, (str, int, float)) and str(v).strip() != ""][:20]

        if not valores:

            raise ValueError(f"filtro de '{dim}' sem valores")

        if dim in FILTROS_TEMPO:

            try:

                valores = [int(float(v)) for v in valores]

            except (TypeError, ValueError):

                raise ValueError(f"valores inválidos para o filtro '{dim}'")

        filtros.append({"dimensao": dim, "operador": op, "valores": valores})

    periodo = None

    p = c.get("periodo")

    if p:

        if not tem_data:

            raise ValueError("a base não tem coluna de data para filtrar período")

        if not isinstance(p, dict):

            raise ValueError("período inválido")

        periodo = {}

        for campo in ("inicio", "fim"):

            if p.get(campo):

                data = pd.to_datetime(p.get(campo), errors="coerce")

                if pd.isna(data):

                    raise ValueError(f"data de {campo} do período inválida")

                periodo[campo] = data.strftime("%Y-%m-%d")

        periodo = periodo or None

    ultima = bool(c.get("somente_ultima_data"))

    if ultima:

        if not tem_data:

            raise ValueError("última data exige coluna de data")

        if "data_ultima" not in metricas:

            metricas.append("data_ultima")

    ordenar = _chave(c["ordenar_por"]) if c.get("ordenar_por") else None

    if ordenar not in set(agrupar) | set(metricas):

        ordenar = None

    direcao = _chave(c.get("direcao") or "")

    direcao = "asc" if direcao in {"asc", "crescente"} else "desc" if direcao in {"desc", "decrescente"} else None

    try:

        limite = int(c.get("limite") or LIMITE_PADRAO)

    except (TypeError, ValueError):

        limite = LIMITE_PADRAO

    limite = max(1, min(limite, LIMITE_MAXIMO))

    return {

        "descricao": str(c.get("descricao") or "")[:200],

        "agrupar_por": agrupar, "metricas": metricas, "filtros": filtros, "periodo": periodo,

        "somente_ultima_data": ultima, "ordenar_por": ordenar, "direcao": direcao, "limite": limite,

    }

def validar_plano(bruto, cols, tem_data):

    """Valida o JSON do planejador. Retorna (consultas_validas, avisos, esclarecimento)."""

    if not isinstance(bruto, dict):

        raise ValueError("o planejador não devolveu um objeto JSON")

    esclarecimento = bruto.get("esclarecimento")

    esclarecimento = esclarecimento.strip() if isinstance(esclarecimento, str) and esclarecimento.strip() else None

    dims_texto = {d for d in DIMENSOES_TEXTO if cols.get(d)} | set(cols.get("_dinamicas", {}))

    tempos_permitidos = set(TEMPO_AGRUPAR)
    if _data_pedido_sensivel(cols):
        tempos_permitidos.discard("dia")
    dims_agrupar = dims_texto | (tempos_permitidos if tem_data else set())

    filtros_validos = dims_texto | (set(FILTROS_TEMPO) if tem_data else set())

    metricas_validas = {"faturamento", "registros"}

    if cols.get("qtd"):

        metricas_validas |= {"quantidade", "preco_medio"}

    if cols.get("cliente"):

        metricas_validas.add("clientes_distintos")

    if cols.get("produto"):

        metricas_validas.add("produtos_distintos")

    if tem_data and not _data_pedido_sensivel(cols):

        metricas_validas |= {"data_primeira", "data_ultima"}
    for dim in cols.get("_numericas", {}):
        metricas_validas |= {f"soma_{dim}", f"media_{dim}"}

    brutas = _lista(bruto.get("consultas"))

    avisos, consultas = [], []

    for i, c in enumerate(brutas[:MAX_CONSULTAS], start=1):

        if not isinstance(c, dict):

            continue

        try:

            consulta_validada = _validar_consulta(c, dims_agrupar, filtros_validos, metricas_validas, tem_data)
            if _data_pedido_sensivel(cols):
                # Defesa adicional: não permitir retorno de datas exatas mesmo se o modelo
                # tentar solicitar explicitamente métricas de primeira/última data.
                if "dia" in consulta_validada["agrupar_por"]:
                    raise ValueError("agrupamento por dia bloqueado por privacidade de Data Pedido")
                consulta_validada["metricas"] = [m for m in consulta_validada["metricas"] if m not in {"data_primeira", "data_ultima"}]
            consultas.append(consulta_validada)

        except ValueError as exc:

            avisos.append(f"Consulta {i} ignorada: {exc}.")

    if len(brutas) > MAX_CONSULTAS:

        avisos.append(f"Só as {MAX_CONSULTAS} primeiras consultas foram executadas.")

    return consultas, avisos, esclarecimento

# ============================================================

# EXECUÇÃO LOCAL DAS CONSULTAS

# ============================================================

def _mascara_filtro(quadro, filtro, reverso, avisos):

    dim, op, valores = filtro["dimensao"], filtro["operador"], filtro["valores"]

    tab = quadro.tab

    if dim in FILTROS_TEMPO:

        parte = {"ano": tab["d"].dt.year, "trimestre_do_ano": tab["d"].dt.quarter, "mes_do_ano": tab["d"].dt.month}[dim]

        mascara = parte.isin(set(valores))

        return (~mascara & tab["d"].notna()) if op == "diferente" else mascara

    serie = tab[quadro.coluna(dim)]

    prefixo = DIMENSOES_TEXTO.get(dim, {}).get("prefixo") or quadro.cols.get("_prefixos_sensiveis", {}).get(dim)

    alvo = set()

    for v in valores:

        texto = normalizar(v)

        if not texto:

            continue

        if prefixo:

            real = reverso.get(texto)

            if real is not None:

                alvo.add(normalizar(real))

                continue

            if REGEX_PSEUDONIMO.match(texto):

                avisos.append(f"Identificador desconhecido no filtro de {dim}: {v}.")

                continue

        alvo.add(texto)

    normalizados = {u: normalizar(u) for u in serie.unique()}

    if op == "contem":

        ok = {u for u, n in normalizados.items() if any(a in n for a in alvo)}

    else:

        ok = {u for u, n in normalizados.items() if n in alvo}

    mascara = serie.isin(ok)

    return ~mascara if op == "diferente" else mascara

def _json_valor(v):

    if v is None or (not isinstance(v, str) and pd.isna(v)):

        return None

    if isinstance(v, pd.Timestamp):

        return v.strftime("%d/%m/%Y")

    if hasattr(v, "item"):

        v = v.item()

    if isinstance(v, float):

        return round(v, 2)

    return v

def executar_consulta(quadro, consulta, mapa, limite_maximo=LIMITE_MAXIMO):

    """Executa uma consulta validada no DataFrame local e devolve linhas agregadas (com pseudônimos)."""

    tab = quadro.tab

    agrupar, metricas, filtros = consulta["agrupar_por"], consulta["metricas"], consulta["filtros"]

    avisos = []

    reverso = _mapa_reverso(mapa)

    # Cria antes todas as colunas que serão usadas (o recorte abaixo precisa delas).

    for d in agrupar:

        quadro.coluna(d)

    if "clientes_distintos" in metricas:

        quadro.coluna("cliente")

    if "produtos_distintos" in metricas:

        quadro.coluna("produto")

    mascara = pd.Series(True, index=tab.index)

    for f in filtros:

        mascara &= _mascara_filtro(quadro, f, reverso, avisos)

    usa_tempo = bool(

        consulta["periodo"] or consulta["somente_ultima_data"]

        or any(d in TEMPO_AGRUPAR for d in agrupar) or any(f["dimensao"] in FILTROS_TEMPO for f in filtros)

    )

    if consulta["periodo"]:

        if consulta["periodo"].get("inicio"):

            mascara &= tab["d"] >= pd.Timestamp(consulta["periodo"]["inicio"])

        if consulta["periodo"].get("fim"):

            mascara &= tab["d"] <= pd.Timestamp(consulta["periodo"]["fim"])

    if usa_tempo:

        mascara &= tab["d"].notna()

    dados = tab[mascara]

    totais = {"registros": int(len(dados)), "faturamento": round(float(dados["v"].sum()), 2)}

    if quadro.tem_quantidade:

        totais["quantidade"] = round(float(dados["q"].sum()), 2)

    resultado = {

        "descricao": consulta["descricao"], "agrupar_por": agrupar, "metricas": metricas,

        "filtros": filtros, "periodo": consulta["periodo"], "somente_ultima_data": consulta["somente_ultima_data"],

        "totais_do_filtro": totais, "avisos": avisos,

    }

    if dados.empty:
        resultado.update({"linhas_totais": 0, "linhas_retornadas": 0, "truncado": False, "linhas": []})
        return resultado

    # Se a pergunta filtra uma dimensão comercial específica, não exponha totais
    # de grupos com menos de cinco registros. Diferenciamos esse bloqueio de zero
    # resultados reais para não exibir uma mensagem enganosa de ausência de vendas.
    filtros_comerciais = [f for f in filtros if f["dimensao"] not in FILTROS_TEMPO]
    if len(dados) < MIN_REGISTROS_GRUPO_PRIVACIDADE and filtros_comerciais:
        resultado["totais_do_filtro"] = {
            "registros": None, "faturamento": None,
            **({"quantidade": None} if quadro.tem_quantidade else {}),
        }
        resultado["avisos"] = avisos + [
            f"Resultado ocultado por privacidade: menos de {MIN_REGISTROS_GRUPO_PRIVACIDADE} registros correspondem aos critérios."
        ]
        resultado.update({
            "linhas_totais": 0, "linhas_retornadas": 0, "truncado": False,
            "linhas": [], "oculto_por_privacidade": True,
        })
        return resultado

    chaves = [quadro.coluna(d) for d in agrupar]

    if consulta["somente_ultima_data"]:

        nao_tempo = [quadro.coluna(d) for d in agrupar if d in DIMENSOES_TEXTO]

        if nao_tempo:

            dados = dados[dados["d"] == dados.groupby(nao_tempo, dropna=False)["d"].transform("max")]

        else:

            dados = dados[dados["d"] == dados["d"].max()]

    dados = dados.copy()

    spec = {"faturamento": ("v", "sum"), "registros": ("v", "size")}

    if quadro.tem_quantidade:

        spec["quantidade"] = ("q", "sum")

    if "clientes_distintos" in metricas:

        dados["_cli_n"] = dados[quadro.coluna("cliente")]

        spec["clientes_distintos"] = ("_cli_n", "nunique")

    if "produtos_distintos" in metricas:

        dados["_prod_n"] = dados[quadro.coluna("produto")]

        spec["produtos_distintos"] = ("_prod_n", "nunique")

    if "data_primeira" in metricas:

        spec["data_primeira"] = ("d", "min")

    if "data_ultima" in metricas:

        spec["data_ultima"] = ("d", "max")

    # Contagem de suporte calculada localmente para impedir a divulgação de grupos muito pequenos.
    if chaves:
        spec["_registros_privacidade"] = ("v", "size")

    for dim, origem in quadro.cols.get("_numericas", {}).items():
        coluna_n = f"n_{dim}"
        if f"soma_{dim}" in metricas:
            spec[f"soma_{dim}"] = (coluna_n, "sum")
        if f"media_{dim}" in metricas:
            spec[f"media_{dim}"] = (coluna_n, "mean")

    if not chaves:

        dados["_tudo"] = 0

        chaves_agrupamento = ["_tudo"]

    else:

        chaves_agrupamento = chaves

    agg = dados.groupby(chaves_agrupamento, dropna=False, sort=False).agg(**spec).reset_index()

    if "preco_medio" in metricas:

        agg["preco_medio"] = agg["faturamento"] / agg["quantidade"].where(agg["quantidade"] > 0)

    if chaves:
        agg = agg.rename(columns=dict(zip(chaves, agrupar)))
        # A regra de privacidade deve ser executada, não apenas descrita no prompt.
        linhas_antes_privacidade = len(agg)
        agg = agg[agg["_registros_privacidade"] >= MIN_REGISTROS_GRUPO_PRIVACIDADE].copy()
        grupos_ocultados = linhas_antes_privacidade - len(agg)
        agg = agg.drop(columns=["_registros_privacidade"])
        if grupos_ocultados:
            avisos.append(
                f"{grupos_ocultados} grupo(s) com menos de "
                f"{MIN_REGISTROS_GRUPO_PRIVACIDADE} registros foram ocultados por privacidade."
            )
    else:
        agg = agg.drop(columns=["_tudo"])

    # Ordenação: pedida pelo plano; senão cronológica (se agrupa por tempo) ou pela primeira métrica (maior primeiro).

    if consulta["ordenar_por"] and consulta["ordenar_por"] in agg.columns:

        agg = agg.sort_values(consulta["ordenar_por"], ascending=(consulta["direcao"] == "asc"), na_position="last", kind="stable")

    elif any(d in TEMPO_AGRUPAR for d in agrupar):

        agg = agg.sort_values(agrupar, ascending=True, kind="stable")

    elif agrupar:

        agg = agg.sort_values(metricas[0] if metricas[0] in agg.columns else "faturamento", ascending=False, na_position="last", kind="stable")

    linhas_totais = int(len(agg))

    limite = min(consulta["limite"], limite_maximo)

    agg = agg.head(limite).copy()

    for d in agrupar:  # pseudônimos só para o que realmente será devolvido

        prefixo = DIMENSOES_TEXTO.get(d, {}).get("prefixo") or quadro.cols.get("_prefixos_sensiveis", {}).get(d)

        if prefixo:

            agg[d] = agg[d].map(lambda nome: _pseudonimo(mapa, prefixo, str(nome)))
        else:
            agg[d] = agg[d].map(sanitizar_valor_textual)

    colunas = agrupar + [m for m in metricas if m in agg.columns]

    agg = agg[colunas].astype(object)

    resultado.update({

        "linhas_totais": linhas_totais,

        "linhas_retornadas": int(len(agg)),

        "truncado": linhas_totais > len(agg),

        "linhas": [{k: _json_valor(v) for k, v in linha.items()} for linha in agg.to_dict("records")],

    })

    return resultado

# ============================================================

# PROMPTS

# ============================================================

def _extrair_json(texto):

    t = str(texto).strip()

    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.MULTILINE).strip()

    try:

        return json.loads(t)

    except json.JSONDecodeError:

        ini, fim = t.find("{"), t.rfind("}")

        if ini != -1 and fim > ini:

            return json.loads(t[ini:fim + 1])

        raise ValueError("a resposta do planejador não é um JSON válido")

def montar_prompt_planejador(pergunta_segura, esquema, historico):

    partes = ["Esquema dos dados disponíveis (JSON):\n" + json.dumps(esquema, ensure_ascii=False, indent=2)]

    if historico:

        partes.append(

            "Histórico recente da conversa (anonimizado; use para resolver referências):\n"

            + json.dumps(historico[-TURNOS_HISTORICO:], ensure_ascii=False, indent=2)

        )

    partes.append("Pergunta atual (anonimizada):\n" + pergunta_segura)

    partes.append("Devolva apenas o JSON do plano de consultas.")

    return "\n\n".join(partes)

def montar_prompt_resposta(pergunta_segura, resultados, avisos, esquema):

    pacote = {

        "periodo_dos_dados_nos_filtros": esquema.get("periodo_dos_dados"),

        "consultas_executadas": resultados,

        "avisos_gerais": avisos,

    }

    return (

        "Pergunta do usuário (anonimizada):\n" + pergunta_segura + "\n\n"

        "Resultados calculados localmente (JSON):\n" + json.dumps(pacote, ensure_ascii=False, indent=2) + "\n\n"

        "Responda usando somente esses resultados e siga as regras obrigatórias."

    )

# ============================================================

# API (Gemini)

# ============================================================

def obter_chave_api():

    try:

        chave = st.secrets["GEMINI_API_KEY"]

    except Exception:

        chave = None

    return chave or os.environ.get("GEMINI_API_KEY")

@st.cache_resource(show_spinner=False)

def _criar_cliente(chave_api):

    return genai.Client(api_key=chave_api)

def consultar_gemini(prompt, chave_api, sistema=INSTRUCOES_IA, json_mode=False, temperatura=0.2):

    cliente = _criar_cliente(chave_api)

    config = types.GenerateContentConfig(

        system_instruction=sistema,

        temperature=temperatura,

        response_mime_type="application/json" if json_mode else None,

    )

    resposta = None

    for tentativa in range(3):

        try:

            resposta = cliente.models.generate_content(model=MODELO_GEMINI, contents=prompt, config=config)

            break

        except Exception as exc:

            mensagem = str(exc).upper()

            temporario = any(t in mensagem for t in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"))

            if not temporario or tentativa == 2:

                raise

            time.sleep(2 ** (tentativa + 1))

    texto = getattr(resposta, "text", None)

    if not texto:

        raise RuntimeError("A API não retornou texto (resposta vazia ou bloqueada).")

    return texto


# ============================================================
# EXPLICAÇÃO ANÔNIMA DA PROJEÇÃO MENSAL
# ============================================================

INSTRUCOES_EXPLICACAO_PROJECAO = """
Você é um analista experiente de séries temporais e negócios, escrevendo em português do Brasil.
Sua tarefa NÃO é repetir números nem declarar fatos: os fatos já foram calculados e serão exibidos
separadamente pela aplicação. Sua tarefa é sugerir hipóteses plausíveis, verificações e ações práticas.

Você pode explorar hipóteses relacionadas a sazonalidade, mudança de ritmo, demanda irregular,
eventos pontuais, estoque, preço, mix comercial ou comportamento de mercado, mas nunca afirme que
algum desses fatores realmente aconteceu. Escreva como possibilidades a investigar, ligadas aos
indicadores recebidos, e evite listas genéricas de causas desconectadas dos dados.

REGRAS:
- Não invente fatos, métricas, percentuais, números, datas, eventos, nomes, códigos, clientes,
  vendedores, produtos específicos ou identificadores. Não use algarismos no texto das respostas.
- Não repita as métricas numéricas; a aplicação apresenta os números calculados localmente.
- Não apresente hipóteses como causas confirmadas. Evite expressões categóricas como "a causa é",
  "foi causado por", "com certeza" ou "isso comprova".
- Se os indicadores forem fracos ou insuficientes, diga o que falta verificar e não force uma conclusão.
- Não peça dados identificáveis. Sugira verificações que a equipe pode fazer internamente.
- Responda SOMENTE com um objeto JSON válido, sem Markdown ou texto antes/depois, usando exatamente
  este esquema e as três chaves indicadas:
  {"hipoteses": ["...", "..."], "verificacoes": ["...", "..."], "acoes": ["..."]}
- Forneça de duas a três hipóteses, duas a três verificações e de uma a três ações. Cada item deve
  ser uma string curta e completa em português. A lista de hipóteses deve deixar claro que são
  hipóteses, por exemplo começando com "Hipótese:".
"""

_CHAVES_INDICADORES_PROJECAO = {
    "tipo_analise", "metrica", "unidade", "horizonte_meses", "meses_historicos", "simulacoes",
    "modelo", "wape_percentual", "erro_medio_absoluto", "vies_medio",
    "mediana_previsao_central", "previsao_central_primeiro_periodo",
    "previsao_central_ultimo_periodo", "variacao_previsao_inicio_fim_percentual",
    "mediana_limite_inferior_80", "mediana_limite_superior_80",
    "largura_mediana_intervalo_80_relativa_percentual",
}

_MODELOS_PROJECAO_SEGUROS = {
    normalizar("Último mês"): "Último mês",
    normalizar("Média móvel de 3 meses"): "Média móvel de 3 meses",
    normalizar("Média móvel de 6 meses"): "Média móvel de 6 meses",
    normalizar("Mesmo mês do ano anterior"): "Mesmo mês do ano anterior",
    normalizar("Tendência + sazonalidade mensal"): "Tendência + sazonalidade mensal",
    normalizar("Último trimestre"): "Último trimestre",
    normalizar("Média móvel de 4 trimestres"): "Média móvel de 4 trimestres",
    normalizar("Sazonal ingênuo (mesmo trimestre do ano anterior)"): "Modelo sazonal trimestral",
    normalizar("Tendência + sazonalidade trimestral"): "Tendência + sazonalidade trimestral",
    normalizar("Seleção individual por SKU + restrição de estoque"): "Seleção agregada de modelos com restrição de estoque",
    normalizar("Melhor modelo selecionado individualmente por SKU"): "Seleção agregada de modelos",
}


def _numero_finito_anonimo(valor, casas=2):
    """Converte para número arredondado; NaN, infinito e booleanos viram None."""
    import math
    try:
        if valor is None or isinstance(valor, bool):
            return None
        numero = float(valor)
        if not math.isfinite(numero):
            return None
        return round(numero, casas)
    except (TypeError, ValueError, OverflowError):
        return None


def _inteiro_validado(valor, nome, minimo, maximo):
    """Converte e valida um inteiro dentro dos limites explícitos do produto."""
    import math
    if isinstance(valor, bool) or valor is None:
        raise ValueError(f"O campo '{nome}' é obrigatório e deve ser inteiro.")
    try:
        n = float(valor)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"O campo '{nome}' deve ser numérico.")
    if not math.isfinite(n) or not n.is_integer() or not minimo <= n <= maximo:
        raise ValueError(f"O campo '{nome}' deve ser um inteiro entre {minimo} e {maximo}.")
    return int(n)


def _serie_numerica_finita(df, nome):
    """Converte a coluna sem perder o alinhamento entre períodos e remove NaN/inf."""
    import math
    if nome not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    serie = pd.to_numeric(df[nome], errors="coerce").astype(float)
    return serie.map(lambda x: x if math.isfinite(float(x)) else float("nan"))


def validar_indicadores_projecao_anonimos(indicadores):
    """Fail-closed: aceita apenas o esquema e os intervalos de indicadores autorizados."""
    import math
    if not isinstance(indicadores, dict) or set(indicadores) != _CHAVES_INDICADORES_PROJECAO:
        raise ValueError("O pacote de indicadores da projeção não corresponde à lista permitida.")
    if indicadores.get("tipo_analise") != "projecao_probabilistica_mensal":
        raise ValueError("Tipo de análise não autorizado.")
    if indicadores.get("metrica") not in {"faturamento", "quantidade_vendida"}:
        raise ValueError("Métrica de projeção inválida.")
    if indicadores.get("unidade") not in {"moeda", "unidades"}:
        raise ValueError("Unidade de projeção inválida.")
    _inteiro_validado(indicadores.get("horizonte_meses"), "horizonte_meses", 1, 24)
    _inteiro_validado(indicadores.get("meses_historicos"), "meses_historicos", 0, 1200)
    if indicadores.get("modelo") not in set(_MODELOS_PROJECAO_SEGUROS.values()) | {"modelo estatístico selecionado"}:
        raise ValueError("O rótulo do modelo não pertence à lista permitida.")
    simulacoes = indicadores.get("simulacoes")
    if simulacoes is not None:
        _inteiro_validado(simulacoes, "simulacoes", 0, 10000)
    for chave, valor in indicadores.items():
        if isinstance(valor, bool):
            raise ValueError(f"Indicador '{chave}' não pode ser booleano.")
        if isinstance(valor, (int, float)) and not math.isfinite(float(valor)):
            raise ValueError(f"Indicador '{chave}' contém valor não finito.")
    for chave in ("wape_percentual", "erro_medio_absoluto", "simulacoes"):
        valor = indicadores.get(chave)
        if valor is not None and valor < 0:
            raise ValueError(f"Indicador '{chave}' não pode ser negativo.")
    inf = indicadores.get("mediana_limite_inferior_80")
    sup = indicadores.get("mediana_limite_superior_80")
    if inf is not None and sup is not None and inf > sup:
        raise ValueError("O limite inferior mediano está acima do limite superior.")
    largura = indicadores.get("largura_mediana_intervalo_80_relativa_percentual")
    if largura is not None and largura < 0:
        raise ValueError("A largura do intervalo não pode ser negativa.")
    # Garante que o pacote seja serializável sem NaN/Infinity e não contenha campos extras.
    json.dumps(indicadores, ensure_ascii=False, allow_nan=False)
    return indicadores


def preparar_indicadores_projecao_anonimos(resultado, horizonte, metrica, unidade):
    """Monta e valida uma allowlist mínima; nunca serializa o resultado completo da engine."""
    if not isinstance(resultado, dict) or resultado.get("ok") is not True:
        raise ValueError("Não há uma projeção válida para explicar.")
    previsoes = resultado.get("previsoes")
    if not isinstance(previsoes, pd.DataFrame) or previsoes.empty:
        raise ValueError("A projeção não contém previsões para resumir.")

    horizonte_ok = _inteiro_validado(horizonte, "horizonte", 1, 24)
    if len(previsoes) != horizonte_ok:
        raise ValueError("O horizonte informado não corresponde à quantidade de previsões calculadas.")
    if "Previsão central (P50)" not in previsoes.columns:
        raise ValueError("A projeção não contém a previsão central necessária para análise.")

    metrica_norm = normalizar(metrica)
    if metrica_norm in {"faturamento", "valor", "vendas"}:
        metrica_segura = "faturamento"
        unidade_segura = "moeda"
    elif metrica_norm in {"quantidade", "quantidade vendida", "unidades", "qtde"}:
        metrica_segura = "quantidade_vendida"
        unidade_segura = "unidades"
    else:
        raise ValueError("A métrica selecionada não é permitida para a explicação.")

    centrais = _serie_numerica_finita(previsoes, "Previsão central (P50)")
    if centrais.notna().sum() == 0:
        raise ValueError("Não há valores numéricos válidos nas previsões centrais.")

    inicio_bruto = _numero_finito_anonimo(centrais.iloc[0], 8)
    fim_bruto = _numero_finito_anonimo(centrais.iloc[-1], 8)
    inicio = _numero_finito_anonimo(centrais.iloc[0])
    fim = _numero_finito_anonimo(centrais.iloc[-1])
    mediana_central = _numero_finito_anonimo(centrais.median())
    variacao = None
    if inicio_bruto is not None and fim_bruto is not None and abs(inicio_bruto) > 1e-9:
        variacao = _numero_finito_anonimo((fim_bruto - inicio_bruto) / abs(inicio_bruto) * 100, 1)

    inferiores = _serie_numerica_finita(previsoes, "Limite inferior 80% (P10)")
    superiores = _serie_numerica_finita(previsoes, "Limite superior 80% (P90)")
    intervalos = pd.DataFrame({"central": centrais, "inferior": inferiores, "superior": superiores}).dropna()
    if not intervalos.empty and (intervalos["superior"] < intervalos["inferior"]).any():
        raise ValueError("Há intervalos de previsão inconsistentes: limite superior abaixo do inferior.")

    mediana_inferior = _numero_finito_anonimo(intervalos["inferior"].median()) if not intervalos.empty else None
    mediana_superior = _numero_finito_anonimo(intervalos["superior"].median()) if not intervalos.empty else None
    larguras = []
    for linha in intervalos.itertuples(index=False):
        if abs(linha.central) > 1e-9:
            larguras.append((linha.superior - linha.inferior) / abs(linha.central) * 100)
    largura_mediana = _numero_finito_anonimo(pd.Series(larguras).median(), 1) if larguras else None

    n_historico = resultado.get("n_meses")
    if n_historico is None:
        historico = resultado.get("historico")
        try:
            n_historico = len(historico) if historico is not None else None
        except TypeError:
            n_historico = None
    if n_historico is None:
        raise ValueError("A engine não informou a quantidade de meses históricos.")
    n_historico = _inteiro_validado(n_historico, "meses_historicos", 0, 1200)

    rotulo_bruto = normalizar(resultado.get("modelo_rotulo") or "")
    modelo_seguro = _MODELOS_PROJECAO_SEGUROS.get(rotulo_bruto, "modelo estatístico selecionado")

    indicadores = {
        "tipo_analise": "projecao_probabilistica_mensal",
        "metrica": metrica_segura,
        "unidade": unidade_segura,
        "horizonte_meses": horizonte_ok,
        "meses_historicos": n_historico,
        "simulacoes": (
            _inteiro_validado(resultado.get("simulacoes"), "simulacoes", 0, 10000)
            if resultado.get("simulacoes") is not None else None
        ),
        "modelo": modelo_seguro,
        "wape_percentual": _numero_finito_anonimo(resultado.get("wape"), 1),
        "erro_medio_absoluto": _numero_finito_anonimo(resultado.get("mae"), 2),
        "vies_medio": _numero_finito_anonimo(resultado.get("bias"), 2),
        "mediana_previsao_central": mediana_central,
        "previsao_central_primeiro_periodo": inicio,
        "previsao_central_ultimo_periodo": fim,
        "variacao_previsao_inicio_fim_percentual": variacao,
        "mediana_limite_inferior_80": mediana_inferior,
        "mediana_limite_superior_80": mediana_superior,
        "largura_mediana_intervalo_80_relativa_percentual": largura_mediana,
    }
    return validar_indicadores_projecao_anonimos(indicadores)


def _formatar_numero_projecao(valor, casas=2):
    if valor is None:
        return "indisponível"
    return f"{float(valor):,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _fatos_projecao_calculados(indicadores):
    """Texto factual produzido localmente; não depende da narrativa da IA."""
    unidade = "R$" if indicadores["unidade"] == "moeda" else "unidades"
    fatos = [
        f"Métrica analisada: {'faturamento' if indicadores['metrica'] == 'faturamento' else 'quantidade vendida'}.",
        f"Horizonte: {indicadores['horizonte_meses']} períodos; histórico disponível: {indicadores['meses_historicos']} meses.",
        f"Modelo selecionado: {indicadores['modelo']}.",
    ]
    wape = indicadores["wape_percentual"]
    if wape is None:
        fatos.append("WAPE retrospectivo: indisponível; não é possível avaliar a precisão por esta métrica.")
    else:
        fatos.append(f"WAPE retrospectivo: {_formatar_numero_projecao(wape, 1)}%. Ele resume o erro relativo no backtest e não garante acerto futuro.")
    mae = indicadores["erro_medio_absoluto"]
    if mae is None:
        fatos.append("Erro médio absoluto: indisponível.")
    else:
        fatos.append(f"Erro médio absoluto no backtest: {unidade} {_formatar_numero_projecao(mae, 2)}.")
    vies = indicadores["vies_medio"]
    if vies is None:
        fatos.append("Viés médio no backtest: indisponível.")
    elif vies > 0:
        fatos.append(f"Viés médio: {unidade} {_formatar_numero_projecao(vies, 2)}; pela convenção do modelo, indica tendência histórica de superestimar.")
    elif vies < 0:
        fatos.append(f"Viés médio: {unidade} {_formatar_numero_projecao(vies, 2)}; pela convenção do modelo, indica tendência histórica de subestimar.")
    else:
        fatos.append("Viés médio no backtest: zero após arredondamento; isso não significa ausência de erro.")

    variacao = indicadores["variacao_previsao_inicio_fim_percentual"]
    if variacao is None:
        fatos.append("Mudança entre a primeira e a última previsão: indisponível ou sem base inicial adequada.")
    else:
        direcao = "aumenta" if variacao > 1 else "diminui" if variacao < -1 else "permanece aproximadamente estável"
        fatos.append(f"A previsão central {direcao} entre o primeiro e o último período: variação calculada de {_formatar_numero_projecao(variacao, 1)}%.")
    central = indicadores["mediana_previsao_central"]
    if central is not None:
        fatos.append(f"Mediana da previsão central ao longo do horizonte: {unidade} {_formatar_numero_projecao(central, 2)}.")
    inferior, superior = indicadores["mediana_limite_inferior_80"], indicadores["mediana_limite_superior_80"]
    if inferior is not None and superior is not None:
        fatos.append(f"Medianas dos limites do intervalo de 80%: {unidade} {_formatar_numero_projecao(inferior, 2)} a {unidade} {_formatar_numero_projecao(superior, 2)}.")
    largura = indicadores["largura_mediana_intervalo_80_relativa_percentual"]
    if largura is not None:
        fatos.append(f"Largura mediana do intervalo de 80% em relação à previsão central: {_formatar_numero_projecao(largura, 1)}%.")
    if indicadores["meses_historicos"] < 24:
        fatos.append("O histórico tem menos de dois ciclos anuais completos; conclusões sobre sazonalidade anual exigem cautela.")
    return fatos


def _validar_resposta_ia_projecao(texto):
    """Valida forma, campos, comprimento e sinais óbvios de invenção/identificação no texto da IA."""
    dados = _extrair_json(texto)
    chaves_esperadas = {"hipoteses", "verificacoes", "acoes"}
    if not isinstance(dados, dict) or set(dados) != chaves_esperadas:
        raise ValueError("A resposta da IA não respeitou o esquema JSON obrigatório.")
    limites = {"hipoteses": (2, 3), "verificacoes": (2, 3), "acoes": (1, 3)}
    frases_categoricas = re.compile(
        r"\b(a causa (?:é|foi)|(?:é|foi) causado por|foi provocad[oa] por|com certeza|"
        r"comprovadamente|isso comprova|garante que|deve-se a|decorre diretamente de|"
        r"explica a (?:alta|queda|varia[cç][aã]o|tend[eê]ncia)|"
        r"explicam a (?:alta|queda|varia[cç][aã]o|tend[eê]ncia)|"
        r"é consequência direta de|foi consequência de)\b", re.IGNORECASE
    )
    alegacoes_numericas_por_extenso = re.compile(
        r"\b(por cento|percentuais?|dobro|triplo|qu[aá]druplo|metade|"
        r"duas? vezes|tr[eê]s vezes|quatro vezes|milh(?:[aã]o|[õo]es)|"
        r"bilh(?:[aã]o|[õo]es))\b", re.IGNORECASE
    )
    possivel_nome_proprio = re.compile(r"\b[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]{2,}\b")
    termos_tecnicos_bloqueados = re.compile(r"\b(?:WAPE|MAE|SKU|CPF|CNPJ|API|URL|ID)\b", re.IGNORECASE)
    saida = {}
    for campo, (min_itens, max_itens) in limites.items():
        itens = dados.get(campo)
        if not isinstance(itens, list) or not min_itens <= len(itens) <= max_itens:
            raise ValueError(f"A seção '{campo}' deve conter entre {min_itens} e {max_itens} itens.")
        limpos = []
        for item in itens:
            if not isinstance(item, str):
                raise ValueError(f"A seção '{campo}' contém item que não é texto.")
            item = re.sub(r"\s+", " ", item).strip()
            if not 25 <= len(item) <= 400:
                raise ValueError(f"A seção '{campo}' contém texto muito curto ou longo.")
            if re.search(r"\d", item) or alegacoes_numericas_por_extenso.search(item):
                raise ValueError("A resposta gerada contém números ou alegações quantitativas não validadas; os fatos numéricos são exibidos localmente.")
            if sanitizar_valor_textual(item) != item:
                raise ValueError("A resposta gerada parece conter dado pessoal ou identificador.")
            if frases_categoricas.search(item):
                raise ValueError("Uma hipótese foi apresentada como causa confirmada.")
            if termos_tecnicos_bloqueados.search(item):
                raise ValueError("A resposta gerada contém sigla técnica ou identificador não permitido.")
            # Evita nomes de pessoas, marcas e locais no corpo do texto. A primeira palavra
            # é permitida por estar no início da frase; não há entidades nomeadas no pacote.
            texto_para_nomes = re.sub(r"^Hip[oó]tese:\s*", "", item, flags=re.IGNORECASE) if campo == "hipoteses" else item
            primeira_palavra = re.search(r"[A-Za-zÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç]+", texto_para_nomes)
            for nome_candidato in possivel_nome_proprio.finditer(texto_para_nomes):
                if primeira_palavra and nome_candidato.start() == primeira_palavra.start():
                    continue
                raise ValueError("A resposta gerada parece conter nome próprio não autorizado.")
            if campo == "hipoteses" and not re.search(r"\bhip[oó]tese\b", item, re.IGNORECASE):
                item = "Hipótese: " + item[0].lower() + item[1:]
            if len(item) > 420:
                raise ValueError(f"A seção '{campo}' contém texto muito longo após a normalização.")
            limpos.append(item)
        saida[campo] = limpos
    return saida


def _montar_explicacao_projecao(fatos, resposta):
    partes = ["### Leitura objetiva dos indicadores", "\n".join(f"- {fato}" for fato in fatos)]
    partes.append("### Hipóteses para investigar")
    partes.append("\n".join(f"- {item}" for item in resposta["hipoteses"]))
    partes.append("### O que verificar")
    partes.append("\n".join(f"- {item}" for item in resposta["verificacoes"]))
    partes.append("### Ações sugeridas")
    partes.append("\n".join(f"- {item}" for item in resposta["acoes"]))
    partes.append("*As hipóteses são possibilidades, não causas confirmadas. A confirmação depende de verificações internas que não são enviadas à IA.*")
    return "\n\n".join(partes)


def gerar_explicacao_projecao_anonima(resultado, horizonte, metrica, unidade):
    """Gera hipóteses com IA, mantendo os fatos numéricos sob validação determinística local."""
    chave_api = obter_chave_api()
    if not chave_api:
        raise RuntimeError("Chave da API Gemini não configurada. Defina GEMINI_API_KEY nos secrets ou no ambiente.")

    indicadores = preparar_indicadores_projecao_anonimos(resultado, horizonte, metrica, unidade)
    fatos = _fatos_projecao_calculados(indicadores)
    prompt = (
        "Analise os indicadores anônimos e produza somente hipóteses, verificações e ações. "
        "Não repita números e não acrescente fatos numéricos: a aplicação os apresenta separadamente. "
        "As hipóteses devem relacionar-se aos sinais observados, mas sempre como possibilidades.\n\n"
        "INDICADORES PERMITIDOS (JSON):\n"
        + json.dumps(indicadores, ensure_ascii=False, allow_nan=False)
        + "\n\nRetorne exatamente JSON com as chaves hipoteses, verificacoes e acoes."
    )

    # Uma segunda tentativa é reservada apenas para resposta fora do contrato.
    for tentativa in range(2):
        texto = consultar_gemini(
            prompt if tentativa == 0 else (
                prompt + "\n\nCORREÇÃO OBRIGATÓRIA: a resposta anterior falhou na validação. "
                "Gere novamente JSON válido com exatamente as três chaves exigidas; "
                "não inclua algarismos, métricas, datas, identificadores ou causas como fatos."
            ),
            chave_api,
            sistema=INSTRUCOES_EXPLICACAO_PROJECAO,
            json_mode=True,
            temperatura=0.15,
        )
        try:
            resposta = _validar_resposta_ia_projecao(texto)
            return _montar_explicacao_projecao(fatos, resposta)
        except (ValueError, TypeError, json.JSONDecodeError):
            if tentativa == 1:
                raise RuntimeError(
                    "A IA não retornou uma explicação no formato seguro após duas tentativas. "
                    "Tente gerar novamente."
                )
    raise RuntimeError("Não foi possível validar a explicação da projeção.")

# ============================================================

# FLUXO EM DUAS REQUISIÇÕES (sem Streamlit: fácil de testar)

# ============================================================

def _total_registros_seguro(resultado):
    """Retorna a contagem de registros como inteiro, ou None quando foi ocultada/inválida."""
    if not isinstance(resultado, dict):
        return None
    totais = resultado.get("totais_do_filtro")
    if not isinstance(totais, dict):
        return None
    valor = totais.get("registros")
    try:
        if valor is None or isinstance(valor, bool):
            return None
        return int(valor)
    except (TypeError, ValueError, OverflowError):
        return None


def _todos_resultados_vazios(resultados):
    """True somente quando existem resultados estruturados e todos têm zero registros reais."""
    validos = [r for r in resultados if isinstance(r, dict) and "totais_do_filtro" in r and "linhas" in r]
    return bool(validos) and all(_total_registros_seguro(r) == 0 for r in validos)


def _termo_pesquisa_nome_nao_encontrado(pergunta, quadro):
    """Detecta nome explicitamente pesquisado entre aspas que não existe na base.

    Faz a validação localmente antes de enviar a pergunta ao Gemini, evitando que um
    nome não reconhecido seja enviado sem anonimização ou que uma consulta ampla seja
    confundida com uma pesquisa válida pelo nome.
    """
    texto = str(pergunta or "")
    p = normalizar(texto)
    if not any(t in p for t in ("pesquisa por", "pesquisar por", "busca por", "buscar por", "procura por", "procurar por")):
        return None

    # Exige aspas para não confundir palavras comuns da pergunta com nomes próprios.
    m = re.search(r'["“”\']([^"“”\']{2,100})["“”\']', texto)
    if not m:
        return None
    termo = m.group(1).strip()
    nomes_por_prefixo = nomes_para_anonimizar(quadro)
    alvo = normalizar(termo)
    if not alvo:
        return None
    # Se o termo existir como qualquer dimensão pseudonimizada (por exemplo,
    # produto/código ou vendedor), não o trate automaticamente como cliente ausente.
    if any(normalizar(nome) == alvo for nomes in nomes_por_prefixo.values() for nome in nomes):
        return None
    for dim in ("produto", "sku"):
        coluna = quadro.cols.get(dim)
        if coluna and coluna in quadro._df.columns:
            if any(normalizar(v) == alvo for v in quadro._df[coluna].dropna().astype(str).unique()):
                return None
    return termo


def _consulta_rapida_sem_filtros(pergunta, quadro):
    """Reconhece apenas perguntas globais inequívocas; não usa base estática nem histórico."""
    p = normalizar(pergunta)
    # Se houver pedido de segmentação, período ou identificador, usar o planejador completo.
    termos_que_exigem_planejamento = (
        " por ", " por mes", " por ano", " em ", " no mes", " no ano", " trimestre",
        " cliente", " produto", " sku", " vendedor", " representante", " empresa",
        " segmento", " departamento", " pedido", " ultima compra", " ultima data",
        " entre ", " desde ", " ate ", " de ", " para ", " compar", " maior", " menor",
    )
    if any(t in f" {p} " for t in termos_que_exigem_planejamento):
        return None

    menciona_faturamento = any(t in p for t in ("faturamento", "valor vendido", "vendas totais", "total de vendas", "total vendido", "vendeu em dinheiro"))
    menciona_quantidade = any(t in p for t in ("quantidade", "unidades vendidas", "total de unidades", "quantas unidades", "total em unidades"))
    if menciona_faturamento == menciona_quantidade:  # nenhuma ou ambígua
        return None

    metricas = ["faturamento"] if menciona_faturamento else ["quantidade"]
    if menciona_quantidade and not quadro.tem_quantidade:
        return None
    consulta = {
        "descricao": "Total geral calculado localmente pela rota rápida",
        "agrupar_por": [], "metricas": metricas, "filtros": [],
        "periodo": None, "somente_ultima_data": False,
        "ordenar_por": "faturamento", "direcao": "desc", "limite": 1,
    }
    return consulta


def _detectar_consulta_clientes_sem_compra_mes(pergunta):
    """Reconhece pedido genérico pela lista de clientes sem compra no mês atual.

    Não intercepta perguntas sobre produto/SKU ou cliente específico: essas seguem
    para o planejador geral.
    """
    p = normalizar(str(pergunta or ""))
    if not re.search(r"\bclientes\b", p):
        return False
    if re.search(r"\b(?:produto|produtos|sku|item|itens|codigo|referencia)\b", p):
        return False
    pede_lista = bool(re.search(r"\b(?:quais|liste|listar|indique|mostre|mostra|informe|informa|diga|diga-me|me diga)\b", p))
    sem_compra = bool(
        re.search(r"\bnao\s+(?:compr\w*|fizeram\s+compra|realizaram\s+compra|tiveram\s+compra|efetuaram\s+compra)\b", p)
        or re.search(r"\bsem\s+compras?\b", p)
        or re.search(r"\bnao\s+tiveram\s+vendas\b", p)
    )
    menciona_mes = bool(re.search(
        r"\b(?:(?:nesse|neste|esse|este|no|do|durante o)\s+)?mes(?:\s+(?:atual|corrente))?\b", p
    ))
    return pede_lista and sem_compra and menciona_mes


def _saida_clientes_sem_compra_mes(quadro, pergunta):
    """Calcula localmente clientes históricos sem compra registrada no mês corrente.

    A consulta só conclui uma lista quando há ao menos um registro de vendas no mês,
    usado como evidência mínima de que o período está presente na base recebida.
    Não envia nomes de clientes para o modelo generativo.
    """
    if not _detectar_consulta_clientes_sem_compra_mes(pergunta):
        return None

    agora = pd.Timestamp(datetime.now().date())
    inicio_mes = pd.Timestamp(agora.year, agora.month, 1)
    mes_rotulo = inicio_mes.strftime("%m/%Y")
    limite_rotulo = agora.strftime("%d/%m/%Y")
    detalhes = {
        "rota": "clientes_sem_compra_mes",
        "mes_considerado": mes_rotulo,
        "data_limite": limite_rotulo,
        "calculo_local": True,
        "nomes_enviados_a_ia": False,
    }

    def retorno(mensagem, status):
        detalhes_retorno = {**detalhes, "status": status}
        return {
            "resposta": mensagem,
            "resposta_segura": "Consulta de clientes sem compra calculada localmente.",
            "detalhes": detalhes_retorno,
            # Memória deliberadamente sem nomes nem lista de clientes.
            "memoria": {
                "pergunta": f"Clientes sem compra registrada no mês {mes_rotulo}",
                "consultas": [],
                "resposta": "A lista foi calculada localmente; identidades não foram guardadas na memória da IA.",
            },
        }

    if not quadro.cols.get("cliente"):
        return retorno(
            "Não consigo listar esses clientes porque a base recebida não tem uma coluna de cliente reconhecida.",
            "coluna_cliente_ausente",
        )
    if not quadro.tem_data:
        return retorno(
            "Não consigo confirmar quem não comprou neste mês porque a base recebida não tem uma coluna de data de venda válida.",
            "coluna_data_ausente",
        )

    coluna_cliente = quadro.coluna("cliente")
    datas = pd.to_datetime(quadro.tab["d"], errors="coerce")
    ate_hoje = datas.notna() & (datas <= agora)
    if not ate_hoje.any():
        return retorno(
            f"Não consigo verificar compras em {mes_rotulo}: a base não contém datas de venda válidas até {limite_rotulo}.",
            "datas_insuficientes",
        )

    ultima_data = datas.loc[ate_hoje].max()
    if ultima_data < inicio_mes:
        return retorno(
            f"Não consigo identificar os clientes sem compra em {mes_rotulo}, porque os dados de vendas recebidos vão somente até {ultima_data.strftime('%d/%m/%Y')}. Atualize a base com os registros deste mês e tente novamente.",
            "base_desatualizada",
        )

    no_mes = datas.ge(inicio_mes) & datas.le(agora)
    if not no_mes.any():
        return retorno(
            f"A base não contém registros de venda entre {inicio_mes.strftime('%d/%m/%Y')} e {limite_rotulo}. Sem registros deste mês, não é seguro concluir que os clientes não compraram; confirme se a base está atualizada.",
            "sem_registros_no_mes",
        )

    def nome_valido(valor):
        nome = str(valor or "").strip()
        chave = normalizar(nome)
        return bool(chave) and chave not in {
            "nan", "none", "null", "(sem informacao)", "(nao informado)", "(sem informação)"
        }

    nomes_historicos = {}
    for nome in quadro.tab.loc[ate_hoje, coluna_cliente].tolist():
        if not nome_valido(nome):
            continue
        chave = normalizar(nome)
        existente = nomes_historicos.get(chave)
        nome = str(nome).strip()
        if existente is None or (len(nome), nome.casefold()) < (len(existente), existente.casefold()):
            nomes_historicos[chave] = nome

    if not nomes_historicos:
        return retorno(
            "Não encontrei nomes de clientes identificáveis nas vendas com data válida até o momento.",
            "clientes_nao_identificados",
        )

    compraram_no_mes = {
        normalizar(str(nome).strip())
        for nome in quadro.tab.loc[no_mes, coluna_cliente].tolist()
        if nome_valido(nome)
    }
    sem_compra = sorted(
        (nome for chave, nome in nomes_historicos.items() if chave not in compraram_no_mes),
        key=lambda nome: normalizar(nome),
    )
    data_base = min(ultima_data, agora).strftime("%d/%m/%Y")
    detalhes.update({
        "status": "calculado",
        "clientes_com_historico": len(nomes_historicos),
        "clientes_sem_compra": len(sem_compra),
        "considerar_registros_ate": data_base,
    })

    if not sem_compra:
        return retorno(
            f"Todos os clientes identificados no histórico têm ao menos uma compra registrada em {mes_rotulo}, considerando os dados até {data_base}.",
            "todos_compraram_no_mes",
        )

    limite_lista = 200
    exibidos = sem_compra[:limite_lista]
    linhas = "\n".join(f"- {nome}" for nome in exibidos)
    resposta = (
        f"Considerando os registros de vendas em {mes_rotulo} até {data_base}, "
        f"estes {len(sem_compra)} clientes do histórico não têm compra registrada neste mês:\n\n{linhas}"
    )
    if len(sem_compra) > limite_lista:
        resposta += (
            f"\n\nExibi os primeiros {limite_lista} de {len(sem_compra)} clientes para manter a resposta legível."
        )
    resposta += (
        "\n\nObservação: a lista considera os clientes presentes no histórico de vendas recebido e "
        "os registros de compra disponíveis até a data indicada."
    )
    return retorno(resposta, "calculado")


def processar_pergunta(df, cols, pergunta, mapa, historico, gerar, etapa=None):

    """Planeja -> executa localmente -> responde.

    `gerar(prompt, sistema, json_mode)` faz a chamada ao modelo (injetada para permitir testes).

    `etapa(texto)` é opcional e informa o andamento à interface.

    Retorna dict com: resposta (nomes reais), resposta_segura, detalhes, memoria.

    """

    aviso_etapa = etapa or (lambda _t: None)

    quadro = Quadro(df, cols)
    # Esta pergunta exige comparar o universo de clientes com as vendas do mês;
    # não cabe em um filtro simples de consulta. Resolva localmente antes do planejador.
    consulta_clientes_mes = _saida_clientes_sem_compra_mes(quadro, pergunta)
    if consulta_clientes_mes is not None:
        return consulta_clientes_mes

    # Correspondência aproximada acontece antes das validações e da anonimização,
    # sempre na máquina local e somente contra valores categóricos da base.
    pergunta_corrigida, correcoes_fuzzy, sugestao_fuzzy, meta_sugestao_fuzzy = corrigir_termos_similares(pergunta, quadro)
    if sugestao_fuzzy:
        return {
            "resposta": sugestao_fuzzy,
            "resposta_segura": sugestao_fuzzy,
            "detalhes": {
                "rota": "validacao_local_similaridade",
                "plano": [], "avisos": [], "resultados": [],
                "correcoes_automaticas": 0,
                "termo_sugerido": (meta_sugestao_fuzzy or {}).get("termo"),
                "candidato_sugerido": (meta_sugestao_fuzzy or {}).get("candidato"),
                "opcoes_sugeridas": (meta_sugestao_fuzzy or {}).get("opcoes", []),
                "sugestao_ambigua": bool((meta_sugestao_fuzzy or {}).get("ambigua")),
                "pergunta_original": pergunta,
            },
            "memoria": None,
        }

    # Se o usuário pesquisou explicitamente um nome entre aspas e ele não existe
    # exatamente na dimensão de clientes, interromper antes de chamar o Gemini.
    # A interface reconhece plano vazio/resultado inválido e limpa memória e pseudônimos.
    nome_nao_encontrado = _termo_pesquisa_nome_nao_encontrado(pergunta_corrigida, quadro)
    if nome_nao_encontrado is not None:
        return {
            "resposta": "nao foram encontrados registros",
            "resposta_segura": "nao foram encontrados registros",
            "detalhes": {"rota": "validacao_local", "plano": [], "avisos": [], "resultados": []},
            "memoria": None,
        }

    # Os valores reais encontrados nunca são enviados ao Gemini: a pergunta corrigida
    # passa em seguida pela rotina normal de pseudonimização.
    pergunta_segura, _encontrados = anonimizar_pergunta(
        pergunta_corrigida, nomes_para_anonimizar(quadro), mapa
    )

    esquema = montar_esquema(quadro)

    # Rota rápida: perguntas inequívocas de total geral são calculadas localmente.
    # A IA continua sendo consultada para redigir a resposta, mas economizamos a chamada
    # de planejamento. Não há base estática, cache de resultados nem reutilização de histórico.
    consulta_rapida = _consulta_rapida_sem_filtros(pergunta, quadro)
    if consulta_rapida is not None:
        aviso_etapa("Calculando diretamente os dados...")
        consultas = [consulta_rapida]
        avisos = []
        esclarecimento = None
        detalhes = {
            "rota": "rapida_local", "pergunta_enviada": pergunta_segura,
            "plano": consultas, "avisos": avisos,
            "correcoes_automaticas": len(correcoes_fuzzy),
        }
    else:
        # Rota completa: a IA planeja a consulta com base na pergunta atual.
        aviso_etapa("Entendendo a pergunta...")
        texto_plano = gerar(montar_prompt_planejador(pergunta_segura, esquema, historico), INSTRUCOES_PLANEJADOR, True)
        consultas, avisos, esclarecimento = validar_plano(_extrair_json(texto_plano), cols, quadro.tem_data)
        detalhes = {
            "rota": "planejamento_ia", "pergunta_enviada": pergunta_segura,
            "plano": consultas, "avisos": avisos,
            "correcoes_automaticas": len(correcoes_fuzzy),
        }

    detalhes = {
        "pergunta_enviada": pergunta_segura, "plano": consultas, "avisos": avisos,
        "correcoes_automaticas": len(correcoes_fuzzy),
        "termos_corrigidos": list(correcoes_fuzzy),
    }

    if not consultas:

        texto = esclarecimento or (

            "Não consegui identificar quais dados consultar para essa pergunta com as colunas disponíveis. "

            + " ".join(avisos)

        ).strip()

        return {

            "resposta": desanonimizar(texto, mapa), "resposta_segura": texto, "detalhes": detalhes,

            "memoria": {"pergunta": pergunta_segura, "consultas": [], "resposta": texto[:700]},

        }

    # Execução local (o código fornece os dados).

    aviso_etapa("Consultando os dados...")

    resultados, restante = [], MAX_LINHAS_TOTAIS

    for consulta in consultas:

        if restante <= 0:

            avisos.append("Limite total de linhas atingido; algumas consultas não foram executadas.")

            break

        resultado = executar_consulta(quadro, consulta, mapa, limite_maximo=restante)

        restante -= resultado.get("linhas_retornadas", 0)

        resultados.append(resultado)

    detalhes["resultados"] = resultados

    # Se todas as consultas retornarem zero registros, não chamar o Gemini
    # para evitar uma resposta especulativa; a interface limpará o contexto.
    consultas_sem_registros = _todos_resultados_vazios(resultados)
    if consultas_sem_registros:
        texto = "nao foram encontrados registros"
        return {
            "resposta": texto,
            "resposta_segura": texto,
            "detalhes": detalhes,
            "memoria": None,
        }

    # Requisição 2: a IA responde com os dados fornecidos.

    aviso_etapa("Escrevendo a resposta...")

    texto = gerar(montar_prompt_resposta(pergunta_segura, resultados, avisos, esquema), INSTRUCOES_IA, False)

    return {

        "resposta": desanonimizar(texto, mapa), "resposta_segura": texto, "detalhes": detalhes,

        "memoria": {

            "pergunta": pergunta_segura,

            "consultas": [

                {"descricao": c["descricao"], "agrupar_por": c["agrupar_por"], "metricas": c["metricas"],

                 "filtros": c["filtros"], "periodo": c["periodo"], "somente_ultima_data": c["somente_ultima_data"]}

                for c in consultas

            ],

            "resposta": texto[:700],

        },

    }

def resolver_confirmacao_validacao(resposta, pendente):
    """Resolve a resposta a uma sugestão do corretor, sem chamar o modelo.

    Retorna ação ``confirmada``, ``recusada`` ou ``nova_pergunta``.
    """
    texto = str(resposta or "").strip()
    normalizada = normalizar(texto).strip(" .!?;:,")
    if not isinstance(pendente, dict):
        return {"acao": "nova_pergunta", "pergunta": texto, "correcoes": []}

    opcoes = [str(x) for x in (pendente.get("opcoes") or [pendente.get("candidato")]) if x]
    escolha = None
    afirmativas = {
        "sim", "isso", "isso mesmo", "correto", "correta", "exato", "exata",
        "exatamente", "confirmo", "pode ser", "pode", "sim pode", "e isso",
    }
    negativas = {
        "nao", "nao e esse", "nao e essa", "nao e isso", "negativo", "errado",
        "errada", "nenhum", "nenhuma", "outro", "outra",
    }

    if not pendente.get("ambigua") and len(opcoes) == 1 and (
        normalizada in afirmativas
        or normalizada == normalizar(opcoes[0]).strip(" .!?;:,")
        or normalizada.startswith("sim ")
    ):
        escolha = opcoes[0]
    elif pendente.get("ambigua"):
        for opcao in opcoes:
            if normalizada == normalizar(opcao).strip(" .!?;:,"):
                escolha = opcao
                break
        if escolha is None:
            numero_opcao = re.fullmatch(r"(?:opcao\s*)?([1-3])", normalizada)
            if numero_opcao and int(numero_opcao.group(1)) <= len(opcoes):
                escolha = opcoes[int(numero_opcao.group(1)) - 1]

    if escolha:
        termo = str(pendente.get("termo", ""))
        pergunta_original = str(pendente.get("pergunta_original", ""))
        pergunta_corrigida = re.sub(
            re.escape(termo), lambda _m: escolha, pergunta_original,
            count=1, flags=re.IGNORECASE,
        ) if termo else pergunta_original
        return {
            "acao": "confirmada",
            "pergunta": pergunta_corrigida,
            "correcoes": [(termo, escolha)] if termo else [],
        }

    if normalizada in negativas or normalizada.startswith("nao "):
        return {"acao": "recusada", "pergunta": None, "correcoes": []}

    return {"acao": "nova_pergunta", "pergunta": texto, "correcoes": []}


# ============================================================

# INTERFACE DO STREAMLIT

# ============================================================

def exibir_assistente_ia(

    df_vendas=None,

    coluna_cliente=None,

    coluna_valor=None,

    coluna_data=None,

    coluna_produto=None,

    coluna_quantidade=None,

    coluna_vendedor=None,

    coluna_segmento=None,

    colunas_extras=None,

    sufixo="",

    **kwargs,

):

    """Desenha o assistente no contêiner atual.

    Passe a base COMPLETA de vendas, sem aplicar os filtros visuais do painel.

    O assistente cria filtros próprios a partir da pergunta do usuário e os aplica localmente.
    `colunas_extras` aceita {"empresa": "...", "status_faturamento": "...", "departamento": "...",

    "gerente_regiao": "...", "representante": "...", "vendedor_cadastro": "..."}; o que faltar é

    procurado pelos nomes padrão do relatório. Apelidos antigos (df, col_valor, ...) continuam aceitos.

    """

    df_vendas = df_vendas if df_vendas is not None else kwargs.pop("df", None)

    coluna_cliente = coluna_cliente or kwargs.pop("col_cliente", None)

    coluna_valor = coluna_valor or kwargs.pop("col_valor", None)

    coluna_data = coluna_data or kwargs.pop("col_data", None)

    coluna_produto = coluna_produto or kwargs.pop("col_produto", None)

    coluna_quantidade = coluna_quantidade or kwargs.pop("col_qtd", kwargs.pop("col_quantidade", None))

    if df_vendas is None:

        st.error("O assistente não recebeu o DataFrame de vendas.")

        return

    chave_msgs, chave_mem, chave_map = f"ia_mensagens_{sufixo}", f"ia_memoria_{sufixo}", f"ia_pseudonimos_{sufixo}"
    chave_pendente_validacao = f"ia_validacao_pendente_{sufixo}"

    st.session_state.setdefault(chave_msgs, [])

    st.session_state.setdefault(chave_mem, [])

    st.session_state.setdefault(chave_map, {})

    # Assistente flutuante: fica sobre o painel, fixo no canto inferior direito.
    # O estado aberto/fechado e a conversa permanecem no session_state entre reruns.
    chave_aberto = f"ia_aberto_{sufixo}"
    st.session_state.setdefault(chave_aberto, True)
    aberto = bool(st.session_state[chave_aberto])
    pergunta = ""

    st.markdown(
        """
        <style>
        /* Contêiner dedicado: o posicionamento é relativo à janela, não à coluna do dashboard. */
        .st-key-ia_floating_widget {
            position: fixed !important;
            box-sizing: border-box !important;
            display: block !important;
            right: 20px !important;
            bottom: 18px !important;
            top: auto !important;
            left: auto !important;
            width: min(355px, calc(100vw - 28px)) !important;
            max-width: calc(100vw - 28px) !important;
            max-height: min(78vh, 720px) !important;
            overflow: hidden !important;
            z-index: 999999 !important;
            background: linear-gradient(145deg, rgba(26, 8, 11, .985) 0%, rgba(7, 9, 11, .985) 66%, rgba(16, 17, 19, .99) 100%) !important;
            border: 1px solid #403238 !important;
            border-radius: 15px !important;
            box-shadow: 0 18px 55px rgba(0,0,0,.62), 0 0 0 1px rgba(237,28,36,.08), inset 0 1px rgba(255,255,255,.035) !important;
            padding: 12px 13px 11px !important;
            backdrop-filter: blur(16px);
        }
        .st-key-ia_floating_widget [data-testid="stMarkdownContainer"] {
            color: #edf0f2;
            line-height: 1.35;
        }
        .st-key-ia_floating_widget [data-testid="stVerticalBlock"] { gap: .48rem; }
        .st-key-ia_floating_widget [data-testid="stTextInput"] input {
            background: #0b0e10 !important; color: #f5f6f7 !important;
            border: 1px solid #3a3e43 !important; border-radius: 8px !important;
            min-height: 38px; font-size: .82rem;
        }
        .st-key-ia_floating_widget [data-testid="stTextInput"] input:focus {
            border-color: #ed1c24 !important; box-shadow: 0 0 0 1px #ed1c24 !important;
        }
        .st-key-ia_floating_widget [data-testid="stForm"] {
            border: 0 !important; padding: 0 !important; background: transparent !important;
        }
        .st-key-ia_floating_widget [data-testid="stFormSubmitButton"] button {
            min-height: 38px; padding: 0 10px;
            background: linear-gradient(135deg, #f21b24, #b80f18) !important;
            color: #fff !important; border: 1px solid #f22a32 !important;
            border-radius: 8px !important; font-size: 1.1rem; font-weight: 700;
        }
        .st-key-ia_floating_widget [data-testid="stFormSubmitButton"] button:hover {
            background: #ff3038 !important; border-color: #ff6268 !important;
        }
        .st-key-ia_floating_widget [data-testid="stButton"] button {
            background: #111417 !important; border: 1px solid #353a40 !important;
            color: #cdd1d5 !important; border-radius: 8px !important;
            padding: 0 8px !important; min-height: 30px;
        }
        .st-key-ia_floating_widget [data-testid="stButton"] button:hover {
            border-color: #ed1c24 !important; color: #fff !important;
        }
        .st-key-ia_floating_widget [data-testid="stChatMessage"] {
            background: #0c0f11 !important; border: 1px solid #282d32 !important;
            border-radius: 9px !important; padding: 8px 10px !important;
        }
        .st-key-ia_floating_widget [data-testid="stChatMessageContent"] p {
            font-size: .79rem; line-height: 1.35;
        }
        .st-key-ia_floating_widget [data-testid="stExpander"] {
            border-color: #2c3035 !important; border-radius: 8px !important;
        }
        .st-key-ia_floating_widget > div { min-width: 0 !important; }
        .ia-assistant-marker { height: 0; margin: 0; padding: 0; overflow: hidden; }
        .ia-card-title { display:flex; align-items:center; gap:9px; color:#fff; font-weight:700; font-size:1rem; letter-spacing:-.02em; min-height:30px; }
        .ia-card-icon { display:inline-flex; align-items:center; justify-content:center; color:#fff; background:linear-gradient(145deg,#f21b24,#a90c14); width:27px; height:27px; border-radius:9px; font-size:15px; box-shadow:0 3px 12px rgba(237,28,36,.24); }
        .ia-greeting { display:flex; align-items:flex-start; gap:9px; background:linear-gradient(135deg,#121518,#0b0d0f); border:1px solid #292e33; border-radius:10px; padding:10px; margin:1px 0 3px; }
        .ia-robot { flex:0 0 32px; width:32px; height:32px; display:flex; align-items:center; justify-content:center; border-radius:50%; background:#25090c; border:1px solid #5a171c; font-size:20px; }
        .ia-greeting-text { color:#e8eaec; font-size:.78rem; line-height:1.45; }
        .ia-greeting-text strong { color:#fff; }
        .ia-examples { color:#b9bec4; font-size:.72rem; line-height:1.48; padding-top:1px; }
        .ia-examples strong { color:#f3a431; font-weight:600; }
        .ia-muted { color:#9da4ab; font-size:.69rem; line-height:1.35; }
        /* Minimizado: mantém apenas a barra do título e o botão para reabrir. */
        .st-key-ia_floating_widget:has(.ia-assistant-marker.ia-minimized) {
            width: min(260px, calc(100vw - 28px)) !important;
            max-height: 72px !important;
            padding: 9px 11px !important;
            overflow: hidden !important;
        }
        @media (max-width: 600px) {
            .st-key-ia_floating_widget {
                right: 10px !important; bottom: 10px !important;
                width: calc(100vw - 20px) !important; max-width: calc(100vw - 20px) !important;
                max-height: 72vh !important; padding: 10px !important;
            }
            .st-key-ia_floating_widget:has(.ia-assistant-marker.ia-minimized) {
                width: min(245px, calc(100vw - 20px)) !important;
                max-width: min(245px, calc(100vw - 20px)) !important;
                max-height: 68px !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # st.container(key=...) só está disponível em versões mais recentes do Streamlit.
    # Em versões antigas, o assistente continua funcional, mas aparece no fluxo da página
    # em vez de flutuar sobre o painel.
    try:
        import inspect
        suporta_chave_container = "key" in inspect.signature(st.container).parameters
    except (TypeError, ValueError):
        versao_streamlit = tuple(
            int(parte) for parte in re.match(r"^(\d+)\.(\d+)", getattr(st, "__version__", "0.0")).groups()
        ) if re.match(r"^(\d+)\.(\d+)", getattr(st, "__version__", "0.0")) else (0, 0)
        suporta_chave_container = versao_streamlit >= (1, 40)

    container_assistente = (
        st.container(key="ia_floating_widget", border=False)
        if suporta_chave_container else st.container(border=False)
    )
    with container_assistente:
        classe_marcador = "ia-assistant-marker" if aberto else "ia-assistant-marker ia-minimized"
        st.markdown(f'<div class="{classe_marcador}"></div>', unsafe_allow_html=True)

        if aberto:
            cab_esq, cab_limpar, cab_minimizar = st.columns([6, 1, 1], gap="small")
            with cab_esq:
                st.markdown(
                    '<div class="ia-card-title"><span class="ia-card-icon">✦</span>'
                    '<span>Assistente de IA</span></div>',
                    unsafe_allow_html=True,
                )
            with cab_limpar:
                limpar = st.button(
                    "↻", key=f"limpar_ia_{sufixo}",
                    help="Limpar conversa e memória contextual", use_container_width=True,
                )
            with cab_minimizar:
                minimizar = st.button(
                    "−", key=f"minimizar_ia_{sufixo}",
                    help="Minimizar assistente", use_container_width=True,
                )
            if limpar:
                st.session_state[chave_msgs] = []
                st.session_state[chave_mem] = []
                st.session_state[chave_map] = {}
                st.session_state.pop(chave_pendente_validacao, None)
                st.rerun()
            if minimizar:
                st.session_state[chave_aberto] = False
                st.rerun()

            with st.container(height=220, border=False):
                if not st.session_state[chave_msgs]:
                    st.markdown(
                        '<div class="ia-greeting"><div class="ia-robot">🤖</div>'
                        '<div class="ia-greeting-text">Olá! Eu sou o assistente de vendas da '
                        '<strong>Intech Machine</strong>. Pergunte sobre faturamento, clientes, '
                        'produtos e muito mais!</div></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    for mensagem in st.session_state[chave_msgs][-8:]:
                        with st.chat_message(mensagem["role"]):
                            st.markdown(mensagem["content"])
                            if mensagem.get("correcoes_termos"):
                                resumo_termos = " · ".join(
                                    f'"{origem}" → "{destino}"'
                                    for origem, destino in mensagem["correcoes_termos"]
                                )
                                st.caption(f"Termos validados: {resumo_termos}")
                            if mensagem.get("detalhes"):
                                with st.expander("Ver consulta e dados utilizados"):
                                    st.caption(
                                        "Plano e agregações calculados localmente; nomes comerciais "
                                        "são pseudonimizados antes do envio à IA."
                                    )
                                    st.json(mensagem["detalhes"], expanded=False)

            with st.form(key=f"form_pergunta_ia_{sufixo}", clear_on_submit=True):
                entrada_col, enviar_col = st.columns([5, 1], gap="small")
                with entrada_col:
                    pergunta_digitada = st.text_input(
                        "Pergunta ao assistente", placeholder="Digite sua pergunta...",
                        label_visibility="collapsed", key=f"entrada_texto_ia_{sufixo}",
                    )
                with enviar_col:
                    enviar = st.form_submit_button("➤", use_container_width=True)

            if not st.session_state[chave_msgs]:
                st.markdown(
                    '<div class="ia-examples"><strong>✦ Exemplos de perguntas</strong><br>'
                    '• Qual o faturamento do último trimestre?<br>'
                    '• Quais são os 5 maiores clientes?<br>'
                    '• Qual produto mais vendeu?<br>'
                    '• Mostre a evolução de um cliente.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="ia-muted">O assistente consulta a base completa e aplica seus próprios filtros.</div>',
                    unsafe_allow_html=True,
                )

            with st.expander("Privacidade e funcionamento", expanded=False):
                st.caption(
                    "A IA planeja a consulta usando campos autorizados. Os cálculos são executados "
                    "localmente e somente resultados agregados são enviados para gerar a resposta. "
                    "Campos pessoais e nomes comerciais são protegidos por pseudonimização. "
                    "O assistente consulta a base completa e aplica os filtros definidos na pergunta, independentemente dos filtros visuais do painel."
                )
            pergunta = pergunta_digitada.strip() if enviar and pergunta_digitada else ""
        else:
            cab_esq, cab_abrir = st.columns([6, 1], gap="small")
            with cab_esq:
                st.markdown(
                    '<div class="ia-card-title"><span class="ia-card-icon">✦</span>'
                    '<span>Assistente de IA</span></div>',
                    unsafe_allow_html=True,
                )
            with cab_abrir:
                abrir = st.button(
                    "+", key=f"abrir_ia_{sufixo}",
                    help="Abrir assistente", use_container_width=True,
                )
            if abrir:
                st.session_state[chave_aberto] = True
                st.rerun()

    if not pergunta:
        return

    # O corretor valida somente nomes/termos da base; não corrige gramática.
    pergunta_exibida = pergunta
    correcoes_confirmadas = []
    pendente = st.session_state.get(chave_pendente_validacao)
    if pendente:
        confirmacao = resolver_confirmacao_validacao(pergunta, pendente)
        if confirmacao["acao"] == "confirmada":
            pergunta = confirmacao["pergunta"]
            correcoes_confirmadas = confirmacao["correcoes"]
            st.session_state.pop(chave_pendente_validacao, None)
        elif confirmacao["acao"] == "recusada":
            st.session_state.pop(chave_pendente_validacao, None)
            st.session_state[chave_msgs] = (
                st.session_state[chave_msgs]
                + [
                    {"role": "user", "content": pergunta_exibida},
                    {"role": "assistant", "content": "Tudo bem. Reescreva a pergunta com o nome correto e envie novamente."},
                ]
            )[-40:]
            st.rerun()
            return
        else:
            # Outra pergunta foi digitada: abandona a confirmação anterior.
            st.session_state.pop(chave_pendente_validacao, None)

    if df_vendas.empty:

        st.warning("A base completa não contém registros para consulta.")

        return

    chave_api = obter_chave_api()

    if not chave_api:

        st.warning("Configure GEMINI_API_KEY (em .streamlit/secrets.toml ou como variável de ambiente).")

        return

    explicitas = {

        "cliente": coluna_cliente, "produto": coluna_produto, "vendedor": coluna_vendedor,

        "segmento": coluna_segmento, "valor": coluna_valor, "qtd": coluna_quantidade, "data": coluna_data,

        **(colunas_extras or {}),

    }

    cols = resolver_colunas(df_vendas, {k: v for k, v in explicitas.items() if v})

    if not cols.get("valor"):

        st.error("Não localizei a coluna de valor. Informe coluna_valor ao chamar o assistente.")

        return

    def gerar(prompt, sistema, json_mode):

        return consultar_gemini(prompt, chave_api, sistema=sistema, json_mode=json_mode, temperatura=0.0 if json_mode else 0.2)

    
    try:
        with st.spinner("Analisando..."):
            saida = processar_pergunta(
                df_vendas,
                cols,
                pergunta,
                st.session_state[chave_map],
                st.session_state[chave_mem],
                gerar,
            )

    except Exception as exc:
        # Limpar somente a memória contextual.
        st.session_state[chave_mem] = []
        st.session_state[chave_map] = {}

        # Registrar o aviso na conversa para que ele não desapareça no rerun.
        st.session_state[chave_msgs] = (
            st.session_state[chave_msgs]
            + [
                {"role": "user", "content": pergunta_exibida, "correcoes_termos": correcoes_confirmadas},
                {
                    "role": "assistant",
                    "content": (
                        f"Não foi possível obter a resposta do assistente: {exc}. "
                        "A memória contextual foi limpa, mas as mensagens anteriores foram mantidas."
                    ),
                },
            ]
        )[-40:]
        st.rerun()
        return

    # Algumas consultas comparativas são calculadas por uma rota local determinística;
    # seu resultado direto não passa pelo validador de planos do Gemini.
    detalhes = saida.get("detalhes") or {}
    if detalhes.get("rota") == "clientes_sem_compra_mes":
        st.session_state[chave_msgs] = (
            st.session_state[chave_msgs]
            + [
                {"role": "user", "content": pergunta_exibida, "correcoes_termos": correcoes_confirmadas},
                {"role": "assistant", "content": saida.get("resposta", "Não foi possível calcular a lista."), "detalhes": detalhes},
            ]
        )[-40:]
        if saida.get("memoria"):
            st.session_state[chave_mem] = (
                st.session_state[chave_mem] + [saida["memoria"]]
            )[-TURNOS_HISTORICO * 2:]
        st.rerun()
        return

    if detalhes.get("rota") == "validacao_local_similaridade":
        st.session_state[chave_pendente_validacao] = {
            "pergunta_original": detalhes.get("pergunta_original", pergunta),
            "termo": detalhes.get("termo_sugerido"),
            "candidato": detalhes.get("candidato_sugerido"),
            "opcoes": detalhes.get("opcoes_sugeridas") or [detalhes.get("candidato_sugerido")],
            "ambigua": bool(detalhes.get("sugestao_ambigua")),
        }
        st.session_state[chave_msgs] = (
            st.session_state[chave_msgs]
            + [
                {"role": "user", "content": pergunta_exibida},
                {"role": "assistant", "content": saida.get("resposta", "Confirme o nome que você quis dizer.")},
            ]
        )[-40:]
        st.rerun()
        return

    plano = detalhes.get("plano") or []
    resultados = detalhes.get("resultados") or []
    avisos = detalhes.get("avisos") or []
    correcoes_para_exibir = (
        detalhes.get("termos_corrigidos") or []
    ) + correcoes_confirmadas

    # Separar resultado vazio de erro de estrutura/plano.
    resultados_estruturados = [
        resultado
        for resultado in resultados
        if isinstance(resultado, dict)
        and "linhas" in resultado
        and "totais_do_filtro" in resultado
    ]

    sem_registros = _todos_resultados_vazios(resultados_estruturados)

    if sem_registros:
        # Resultado vazio: limpar o contexto e o mapa de pseudônimos,
        # preservando as mensagens anteriores visíveis na conversa.
        st.session_state[chave_mem] = []
        st.session_state[chave_map] = {}
        st.session_state[chave_msgs] = (
            st.session_state[chave_msgs]
            + [
                {"role": "user", "content": pergunta_exibida, "correcoes_termos": correcoes_para_exibir},
                {"role": "assistant", "content": "nao foram encontrados registros"},
            ]
        )[-40:]
        st.rerun()
        return

    resultados_ocultos = bool(resultados_estruturados) and all(
        bool(r.get("oculto_por_privacidade"))
        or (
            isinstance(r.get("linhas"), list) and not r.get("linhas")
            and isinstance(r.get("avisos"), list)
            and any("privacidade" in str(a).casefold() for a in r.get("avisos", []))
        )
        for r in resultados_estruturados
    )
    if resultados_ocultos:
        st.session_state[chave_msgs] = (
            st.session_state[chave_msgs]
            + [
                {"role": "user", "content": pergunta_exibida, "correcoes_termos": correcoes_para_exibir},
                {"role": "assistant", "content": "O resultado foi ocultado por privacidade porque corresponde a menos de 5 registros."},
            ]
        )[-40:]
        st.rerun()
        return

    resultado_invalido = (
        not plano
        or not resultados_estruturados
    )

    if resultado_invalido:
        # Um plano vazio pode ser um esclarecimento legítimo do planejador, não uma falha.
        # Mostrar a resposta original evita mascarar a pergunta de esclarecimento com erro genérico.
        if not plano and str(saida.get("resposta") or "").strip():
            st.session_state[chave_msgs] = (
                st.session_state[chave_msgs]
                + [
                    {"role": "user", "content": pergunta_exibida, "correcoes_termos": correcoes_para_exibir},
                    {"role": "assistant", "content": saida["resposta"]},
                ]
            )[-40:]
            if saida.get("memoria"):
                st.session_state[chave_mem] = (
                    st.session_state[chave_mem] + [saida["memoria"]]
                )[-TURNOS_HISTORICO * 2:]
            st.rerun()
            return

        # Manter as mensagens e limpar somente a memória contextual.
        st.session_state[chave_mem] = []
        st.session_state[chave_map] = {}

        # Registrar o aviso no histórico visível: st.warning seguido de
        # st.rerun() desaparece antes que o usuário consiga lê-lo.
        st.session_state[chave_msgs] = (
            st.session_state[chave_msgs]
            + [
                {"role": "user", "content": pergunta_exibida, "correcoes_termos": correcoes_para_exibir},
                {
                    "role": "assistant",
                    "content": (
                        "Não consegui reconhecer o nome informado ou a consulta "
                        "retornou um resultado inválido/insuficiente. A memória "
                        "contextual foi limpa para evitar reutilizar informações "
                        "incorretas, mas as mensagens anteriores foram mantidas. "
                        "Confira o nome do cliente ou produto e tente novamente."
                    ),
                },
            ]
        )[-40:]
        st.rerun()
        return

    # Somente resultados válidos são adicionados à conversa.
    st.session_state[chave_msgs] = (
        st.session_state[chave_msgs]
        + [
            {"role": "user", "content": pergunta_exibida,
             "correcoes_termos": correcoes_para_exibir},
            {
                "role": "assistant",
                "content": saida["resposta"],
                "detalhes": detalhes,
            },
        ]
    )[-40:]

    st.session_state[chave_mem] = (
        st.session_state[chave_mem] + [saida["memoria"]]
    )[-TURNOS_HISTORICO * 2:]

    st.rerun()