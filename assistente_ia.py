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

from difflib import SequenceMatcher

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

11. Perguntas de AUSÊNCIA ("quais clientes NÃO compraram em outubro", "quem parou de comprar",
    "produtos sem venda no trimestre") exigem uma consulta especial, pois quem não comprou não tem linha
    nos dados. Formato (uma única consulta desse tipo, sem metricas e sem agrupar_por):
    {"tipo": "sem_compra", "descricao": "...", "dimensao": "cliente",
     "filtros": [{"dimensao": "mes_do_ano", "operador": "igual", "valores": [10]},
                 {"dimensao": "ano", "operador": "igual", "valores": [2026]}],
     "periodo": null, "universo": {"filtros": [], "periodo": null},
     "ordenar_por": "faturamento", "direcao": "desc", "limite": 100}
    "dimensao" é quem se quer listar (use uma de "dimensoes_para_consulta_sem_compra"). "filtros" e "periodo"
    descrevem a COMPRA que não aconteceu. "universo" define quem é considerado (padrão: todos que já tiveram
    venda na base; restrinja apenas se a pergunta pedir, por exemplo um segmento ou "nos últimos 12 meses").
    "mês 10" = mes_do_ano igual a 10. Se o ano não for citado, use o ano da última data do período dos dados.

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

11. Consultas com "tipo": "sem_compra" listam itens do universo que NÃO tiveram venda no "criterio_de_compra".
   Use "resumo" (itens_no_universo, itens_que_compraram, itens_sem_compra) e informe o critério e o universo
   considerados. "faturamento", "registros" e "ultima_compra" são históricos do item no universo, não do período
   consultado. Se "itens_sem_compra" for 0, diga que todos os itens do universo compraram. Respeite "avisos"
   e "truncado".

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

# LISTA PERMITIDA: exatamente as colunas da base de vendas. Qualquer outro cabeçalho NUNCA é exposto ao
# assistente. (CNPJ consta na base, mas é bloqueado por COLUNAS_SENSIVEIS/TERMOS_BLOQUEADOS.)
COLUNAS_BASE = (
    "Classe Produto", "Cliente Gerencial", "Grupo Empresarial", "Empresa", "Ano-Mês", "Ano Fat.", "Mês Fat.",
    "Trimestre", "Nº Pedido", "Data Pedido", "CNPJ", "Cliente", "Produto", "Voltagem", "Qtde", "Vlr unitário",
    "Vlr total dos produtos", "Data Faturamento", "N.F.", "IPI", "Valor Bruto da NF", "ST", "Total Bruto",
    "Comissão Gerada", "Representante", "Tipo de Operação", "Departamento", "Vendedor atual do cadastro",
    "Gerente região Venda", "Gerente região Cadastro", "Status Pedido", "Status Faturamento", "Segmento",
    "Porte", "SKU", "Cidade", "Estado", "Região", "Frete", "Validação Dev. Terc", "Classe produto Marca",
    "Sub Classe de produtos",
)
COLUNAS_PERMITIDAS = {normalizar(nome) for nome in COLUNAS_BASE if normalizar(nome) != "cnpj"}

def coluna_formula(nome):
    """A base traz uma coluna calculada cujo CABEÇALHO é a própria fórmula (com referências e nomes)."""
    return str(nome).strip().startswith("=")

def rotulo_coluna(nome):
    """Rótulo seguro do cabeçalho para o esquema: fórmulas/cabeçalhos estranhos nunca vão ao Gemini."""
    texto = str(nome).strip()
    if coluna_formula(texto) or re.search(r"[=;()<>\"]", texto):
        return "Classificação calculada (fórmula)"
    return texto

COLUNAS_TEMPO_PRESERVADAS = {normalizar(nome) for nome in (
    "Ano-Mês", "Ano Fat.", "Mês Fat.", "Trimestre", "Data Pedido", "Data Faturamento",
)}
COLUNAS_VALOR_PRESERVADAS = {normalizar(nome) for nome in (
    "Qtde", "Vlr unitário", "Vlr total dos produtos", "IPI", "Valor Bruto da NF", "ST", "Total Bruto",
    "Comissão Gerada", "Frete",
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
    "produto": {"rotulo": "produto/descrição comercial", "aliases": ["Produto"], "prefixo": "Dado"},
    "sku": {"rotulo": "código SKU do produto", "aliases": ["SKU"], "prefixo": "Dado"},
    "vendedor": {"rotulo": "vendedor atual do cadastro", "aliases": ["Vendedor atual do cadastro"], "prefixo": "Vendedor"},
    "representante": {"rotulo": "representante", "aliases": ["Representante"], "prefixo": "Representante"},
    "vendedor_cadastro": {"rotulo": "vendedor atual do cadastro", "aliases": ["Vendedor atual do cadastro"], "prefixo": "Vendedor"},
    "gerente_regiao": {"rotulo": "gerente de região (cadastro)", "aliases": ["Gerente região Cadastro"], "prefixo": "Gerente"},
    "gerente_venda": {"rotulo": "gerente de região (venda)", "aliases": ["Gerente região Venda"], "prefixo": "Gerente"},
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

# ------------------------------------------------------------
# CAMADAS EXTRAS DE ANONIMIZAÇÃO (nomes parciais, colunas ocultas e trava final)
# ------------------------------------------------------------
PREFIXOS_NOME_PROPRIO = ("Cliente", "Vendedor", "Representante", "Gerente")
COLUNAS_NOMES_EMPRESA = {normalizar(n) for n in ("Cliente", "Grupo Empresarial", "Empresa")}
LIMITE_AMBIGUIDADE_NOME = 5   # token de nome presente em mais cadastros que isso = palavra genérica
MIN_LETRAS_TOKEN_NOME = 4

# Palavras comuns de perguntas/análises e de razões sociais: nunca tratadas como "nome próprio".
PALAVRAS_GENERICAS = {normalizar(p) for p in """
faturamento faturado faturada faturados faturadas faturar vendas venda vendeu vendido vendida vendidos
vender valor valores total totais soma somar quantidade quantidades unidade unidades preco precos
medio media medias maior maiores menor menores ultimo ultima ultimos ultimas primeiro primeira
compra compras comprou comprado compraram pedido pedidos cliente clientes produto produtos
vendedor vendedores representante representantes gerente gerentes empresa empresas segmento
segmentos regiao regioes estado estados cidade cidades trimestre trimestres semestre semestres
mensal mensais anual anuais ano anos mes meses periodo periodos evolucao comparar comparacao
comparativo participacao ranking lista listar liste mostre mostra mostrar quais qual quanto
quantos quantas como onde quem para entre desde sobre todos todas cada mais menos passado
passada atual atuais este esta esse essa isso hoje ontem janeiro fevereiro marco abril maio
junho julho agosto setembro outubro novembro dezembro segunda terca quarta quinta sexta sabado
domingo dado dados item itens codigo sku classe departamento status voltagem porte grupo
comissao imposto impostos bruto liquido nota fiscal tipo operacao data datas dia dias semana
semanas diario diaria acumulado crescimento queda variacao percentual porcentagem resultado
resultados consulta base geral nesse nessa desse dessa dele dela deles delas foram sao teve
tinha houve fizeram temos ainda tambem apenas somente sempre nunca outro outra outros outras
mesmo mesma entao porque quando enquanto durante depois antes apos cerca removido email
telefone documento informacao informado
ltda eireli epp sociedade anonima comercio comercial industria industrial servicos servico
participacoes distribuidora distribuicao importacao exportacao brasil brasileira holding matriz
filial empreendimentos representacoes equipamentos maquinas ferramentas materiais solucoes
sistemas tecnologia transportes transportadora construcoes construtora engenharia agropecuaria
agricola agro
""".split()}

def _tokens_nome(nome):
    """Palavras (só letras, 4+) de um nome que ajudam a identificá-lo; ignora termos genéricos."""
    saida = []
    for tok in re.findall(r"[^\W\d_]{%d,}" % MIN_LETRAS_TOKEN_NOME, str(nome)):
        tok = normalizar(tok)
        if len(tok) >= MIN_LETRAS_TOKEN_NOME and tok not in PALAVRAS_GENERICAS:
            saida.append(tok)
    return saida

def _parece_nome(valor):
    """Valor textual com cara de nome (evita tratar 'Sim', números ou datas como nomes)."""
    v = str(valor).strip()
    if not (MIN_LETRAS_TOKEN_NOME <= len(v) <= 120) or not any(c.isalpha() for c in v):
        return False
    n = normalizar(v)
    return n not in PLACEHOLDERS and n not in PALAVRAS_GENERICAS

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

        df, ["Vlr total dos produtos", "Valor Bruto da NF"]

    )

    eq = explicitas.get("qtd")

    cols["qtd"] = (localizar_coluna_opcional(df, [eq]) if eq else None) or localizar_coluna_opcional(df, ["Qtde"])

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
        if (
            (cabecalho not in COLUNAS_PERMITIDAS and not coluna_formula(nome))
            or nome in reservadas or coluna_bloqueada(nome)
        ):
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

    # Colunas de texto NÃO expostas ao assistente (fora da lista permitida, como "Razão Social",
    # "Nome Fantasia", "Vendedor"...) são lidas só LOCALMENTE para anonimizar a pergunta: se o usuário
    # digitar um nome que existe nelas, ele vira "Dado N" antes do envio. Nunca entram no esquema.
    expostas = set(dinamicas.values()) | reservadas
    ja_cobertas = {c for lista in cols["_colunas_anonimizacao"].values() for c in lista}
    ocultas = []
    for nome in df.columns:
        if nome in expostas or nome in ja_cobertas:
            continue
        serie = df[nome]
        if isinstance(serie, pd.DataFrame):
            serie = serie.iloc[:, 0]
        if (pd.api.types.is_numeric_dtype(serie) or pd.api.types.is_bool_dtype(serie)
                or pd.api.types.is_datetime64_any_dtype(serie)):
            continue
        ocultas.append(nome)
    if ocultas:
        cols["_colunas_anonimizacao"]["Dado"] = sorted(
            set(cols["_colunas_anonimizacao"].get("Dado", [])) | set(ocultas)
        )
    cols["_colunas_ocultas"] = set(ocultas)
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

            if (
                len(nome) >= 2 and nome not in vistos and normalizar(nome) not in PLACEHOLDERS
                # números curtos (anos, quantidades) não são identificadores: "2026" não pode virar "Dado 1"
                and not re.fullmatch(r"\d{1,4}", nome)
                # categorias que são palavras comuns de pergunta ("Médio" em "preço médio") ficam como digitadas
                and not (prefixo == "Dado" and normalizar(nome) in PALAVRAS_GENERICAS)
            ):
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

def _valores_coluna_anonimizacao(quadro, coluna):
    """Valores reais de uma coluna usada só para anonimizar (bloqueada ou oculta)."""
    serie = quadro._df[coluna]
    if isinstance(serie, pd.DataFrame):
        serie = serie.iloc[:, 0]
    valores = serie.dropna().astype(str).str.strip().unique()
    if coluna in quadro.cols.get("_colunas_ocultas", ()):
        valores = [v for v in valores if _parece_nome(v)]
    return valores

def _nomes_proprios(quadro, nomes_por_prefixo):
    """[(prefixo, nome)] de clientes, vendedores, representantes, gerentes e colunas ocultas/bloqueadas."""
    saida, vistos = [], set()

    def adicionar(prefixo, nome):
        nome = str(nome).strip()
        if len(nome) >= 2 and nome not in vistos and normalizar(nome) not in PLACEHOLDERS:
            vistos.add(nome)
            saida.append((prefixo, nome))

    for prefixo in PREFIXOS_NOME_PROPRIO:
        for nome in nomes_por_prefixo.get(prefixo, ()):
            adicionar(prefixo, nome)
    for coluna in quadro.cols.get("_colunas_anonimizacao", {}).get("Dado", ()):
        if coluna in quadro._df.columns:
            for nome in _valores_coluna_anonimizacao(quadro, coluna):
                adicionar("Dado", nome)
    # Colunas expostas (como "Dado N") que guardam nomes de empresas: também protegidas por palavra parcial.
    origens = list(quadro.cols.get("_dinamicas", {}).values()) + [quadro.cols.get("empresa")]
    for coluna in dict.fromkeys(o for o in origens if o):
        if normalizar(coluna) in COLUNAS_NOMES_EMPRESA and coluna in quadro._df.columns:
            for nome in _valores_coluna_anonimizacao(quadro, coluna):
                adicionar("Dado", nome)
    return saida

def anonimizar_nomes_parciais(pergunta_segura, nomes_proprios, mapa):
    """2ª camada: nomes digitados de forma PARCIAL (ex.: só "Bazzo" para "Bazzo Variedades Ltda").

    - palavra que identifica UM único cadastro -> vira o pseudônimo desse cadastro;
    - palavra que identifica vários cadastros -> nada é enviado; devolve a ambiguidade p/ o usuário;
    - palavras genéricas (ltda, comércio, termos de pergunta) ou presentes em muitos cadastros são ignoradas.
    Retorna (texto, [(nome, identificador)], [(termo, [nomes])]).
    """
    texto = str(pergunta_segura)
    por_nome = {}
    for prefixo, nome in nomes_proprios:        # mesmo nome em 2 colunas = mesma pessoa/empresa
        por_nome.setdefault(nome, prefixo)
    indice = {}
    for nome, prefixo in por_nome.items():
        if "@" in nome:   # e-mails já são removidos por regex; não gerar tokens a partir deles
            continue
        for tok in set(_tokens_nome(nome)):
            indice.setdefault(tok, {})[nome] = prefixo
    if not indice:
        return texto, [], []

    trocas, ambiguos = [], []
    for m in re.finditer(r"[^\W\d_]{%d,}" % MIN_LETRAS_TOKEN_NOME, texto):
        entidades = indice.get(normalizar(m.group(0)))
        if entidades:
            # Colunas ocultas (ex.: Razão Social) costumam ser outra grafia do mesmo cadastro visível:
            # havendo cadastro visível, ele decide; a grafia oculta é trocada junto, sem ir à IA.
            visiveis = {n: p for n, p in entidades.items() if p != "Dado"}
            entidades = visiveis or entidades
        if not entidades or len(entidades) > LIMITE_AMBIGUIDADE_NOME:
            continue
        if len(entidades) == 1:
            nome, prefixo = next(iter(entidades.items()))
            trocas.append((m.start(), m.end(), prefixo, nome))
        else:
            ambiguos.append((m.group(0), sorted(entidades)))
    if ambiguos:
        return texto, [], ambiguos

    encontrados = []
    for ini, fim, prefixo, nome in sorted(trocas, reverse=True):
        identificador = _pseudonimo(mapa, prefixo, nome)
        texto = texto[:ini] + identificador + texto[fim:]
        encontrados.append((nome, identificador))
    # "Bazzo Silva" -> "Cliente 1 Cliente 1" -> "Cliente 1"
    texto = re.sub(r"((?:Cliente|Vendedor|Representante|Gerente|Dado) \d+)(?:\s+\1)+", r"\1", texto)
    return texto, encontrados, []

def _preparar_guarda(nomes_proprios):
    return sorted({_normalizar_posicional(n) for _p, n in nomes_proprios if _parece_nome(n)}, key=len, reverse=True)

def _nome_real_no_texto(texto, guarda):
    """Trava final: True se algum nome real conhecido aparece no texto que iria ao Gemini."""
    pn = _normalizar_posicional(texto)
    for nn in guarda:
        if nn in pn and re.search(r"(?<!\w)" + re.escape(nn) + r"(?!\w)", pn):
            return True
    return False

def nomes_para_anonimizar(quadro):

    """{'Cliente': {nomes}, 'Vendedor': {...}} com os nomes reais presentes nos dados."""

    saida = {}

    # Colunas bloqueadas são usadas exclusivamente para substituir valores reais na pergunta.
    # Nenhum desses campos é incluído no esquema de consulta enviado ao Gemini.
    for prefixo, colunas in quadro.cols.get("_colunas_anonimizacao", {}).items():
        for coluna in colunas:
            if coluna in quadro._df.columns:
                valores = _valores_coluna_anonimizacao(quadro, coluna)
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

def montar_esquema(quadro):

    """Esquema agregado enviado ao planejador; nunca inclui linhas nem valores categóricos reais.

    Os dados completos e os valores originais permanecem no DataFrame local. Ao Gemini
    são enviados apenas metadados estruturais, contagens e identificadores pseudonimizados.
    """

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

        # Nunca enviar valores categóricos reais ao modelo. Mesmo quando uma dimensão
        # não tiver prefixo configurado por algum motivo, o esquema contém apenas a
        # contagem de valores distintos; os dados completos e os nomes reais ficam locais.
        unicos = quadro._df[cols[dim]].dropna().astype(str).str.strip().unique()
        valores_possiveis[dim] = f"{len(unicos)} valores distintos; valores anonimizados e não listados"

    for dim, origem in cols.get("_dinamicas", {}).items():
        prefixo = cols.get("_prefixos_sensiveis", {}).get(dim)
        dims[dim] = f"coluna {rotulo_coluna(origem)!r} (use o ID {dim})" + (f' (valores pseudonimizados como "{prefixo} N")' if prefixo else "")
        serie = quadro._df[origem].dropna().astype(str).str.strip()
        unicos = serie.unique().tolist()
        if prefixo:
            valores_possiveis[dim] = f"{len(unicos)} valores distintos, pseudonimizados; use o identificador citado na pergunta"
        elif normalizar(origem) in COLUNAS_VALOR_PRESERVADAS or "data" in normalizar(origem):
            # Valores monetários/quantidades e datas exatas nunca são listados individualmente.
            valores_possiveis[dim] = f"{len(unicos)} valores distintos"
        elif len(unicos) <= MAX_VALORES_LISTADOS:
            valores_possiveis[dim] = [sanitizar_valor_textual(v) for v in unicos]
        else:
            valores_possiveis[dim] = f"{len(unicos)} valores distintos"

    esquema["dimensoes_texto_para_agrupar_e_filtrar"] = dims
    esquema["tipos_de_consulta"] = {
        "agregada": "padrão: agrupa e soma o que existe nas linhas",
        "sem_compra": "ausência: itens do universo que NÃO tiveram venda no critério (ver regra 11)",
    }
    esquema["dimensoes_para_consulta_sem_compra"] = list(dims)

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

TIPOS_SEM_COMPRA = {"sem_compra", "nao_compraram", "nao_comprou", "nao_compra", "sem_venda", "sem_vendas", "ausencia"}
LIMITE_PADRAO_SEM_COMPRA = 100

def _validar_filtros_lista(brutos, filtros_validos):
    filtros = []
    for f in _lista(brutos):
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
    return filtros

def _validar_periodo_dict(p, tem_data):
    if not p:
        return None
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
    return periodo or None

def _validar_sem_compra(c, dims_agrupar, filtros_validos, tem_data):
    """Consulta de AUSÊNCIA: itens do universo que NÃO tiveram venda no critério informado."""
    candidatos = _lista(c.get("agrupar_por"))
    dim = _chave(c.get("dimensao") or (candidatos[0] if candidatos else ""))
    if dim not in dims_agrupar or dim in TEMPO_AGRUPAR:
        raise ValueError(f"consulta de ausência por '{dim}' indisponível nos dados")
    filtros = _validar_filtros_lista(c.get("filtros"), filtros_validos)
    periodo = _validar_periodo_dict(c.get("periodo"), tem_data)
    if not filtros and not periodo:
        raise ValueError("a consulta de ausência precisa de período ou filtro que defina a compra")
    u = c.get("universo") if isinstance(c.get("universo"), dict) else {}
    universo = {
        "filtros": _validar_filtros_lista(u.get("filtros"), filtros_validos),
        "periodo": _validar_periodo_dict(u.get("periodo"), tem_data),
    }
    ordenar = _chave(c["ordenar_por"]) if c.get("ordenar_por") else None
    if ordenar not in {dim, "faturamento", "registros", "ultima_compra"}:
        ordenar = None
    direcao = _chave(c.get("direcao") or "")
    direcao = "asc" if direcao in {"asc", "crescente"} else "desc" if direcao in {"desc", "decrescente"} else None
    try:
        limite = int(c.get("limite") or LIMITE_PADRAO_SEM_COMPRA)
    except (TypeError, ValueError):
        limite = LIMITE_PADRAO_SEM_COMPRA
    return {
        "tipo": "sem_compra", "dimensao": dim, "descricao": str(c.get("descricao") or "")[:200],
        "agrupar_por": [dim], "metricas": [], "filtros": filtros, "periodo": periodo, "universo": universo,
        "somente_ultima_data": False, "ordenar_por": ordenar, "direcao": direcao,
        "limite": max(1, min(limite, LIMITE_MAXIMO)),
    }

def _validar_consulta(c, dims_agrupar, filtros_validos, metricas_validas, tem_data):
    if _chave(c.get("tipo") or "") in TIPOS_SEM_COMPRA:
        return _validar_sem_compra(c, dims_agrupar, filtros_validos, tem_data)
    return _validar_consulta_agregada(c, dims_agrupar, filtros_validos, metricas_validas, tem_data)

def _validar_consulta_agregada(c, dims_agrupar, filtros_validos, metricas_validas, tem_data):

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

def _mascara_criterio(quadro, filtros, periodo, reverso, avisos):
    """Máscara de linhas que atendem a filtros + período (sem herdar filtros visuais do painel)."""
    tab = quadro.tab
    mascara = pd.Series(True, index=tab.index)
    for f in filtros:
        mascara &= _mascara_filtro(quadro, f, reverso, avisos)
    usa_tempo = bool(periodo or any(f["dimensao"] in FILTROS_TEMPO for f in filtros))
    if periodo:
        if periodo.get("inicio"):
            mascara &= tab["d"] >= pd.Timestamp(periodo["inicio"])
        if periodo.get("fim"):
            mascara &= tab["d"] <= pd.Timestamp(periodo["fim"])
    if usa_tempo:
        mascara &= tab["d"].notna()
    return mascara

def _executar_sem_compra(quadro, consulta, mapa, limite_maximo=LIMITE_MAXIMO):
    """Quem NÃO comprou: itens do universo (padrão: todos que já tiveram venda na base) menos os que
    tiveram venda no critério. Calculado localmente; só pseudônimos e agregados voltam."""
    tab = quadro.tab
    dim = consulta["dimensao"]
    chave = quadro.coluna(dim)
    reverso = _mapa_reverso(mapa)
    avisos = []
    universo = consulta.get("universo") or {"filtros": [], "periodo": None}

    m_univ = _mascara_criterio(quadro, universo.get("filtros", []), universo.get("periodo"), reverso, avisos)
    m_comp = _mascara_criterio(quadro, consulta["filtros"], consulta["periodo"], reverso, avisos)

    base = tab[m_univ]
    base = base[~base[chave].map(lambda v: normalizar(v) in PLACEHOLDERS)]
    compraram = set(tab.loc[m_comp, chave].unique())

    resultado = {
        "descricao": consulta["descricao"], "tipo": "sem_compra", "dimensao": dim,
        "explicacao": (
            f"Lista de '{dim}' do universo que NÃO tiveram nenhuma venda no critério de compra. "
            "'faturamento', 'registros' e 'ultima_compra' são históricos, dentro do universo."
        ),
        "criterio_de_compra": {"filtros": consulta["filtros"], "periodo": consulta["periodo"]},
        "universo": universo,
    }

    if base.empty:
        resultado.update({
            "totais_do_filtro": {"registros": 0, "faturamento": 0.0},
            "resumo": {"itens_no_universo": 0, "itens_que_compraram": 0, "itens_sem_compra": 0},
            "avisos": avisos, "linhas_totais": 0, "linhas_retornadas": 0, "truncado": False, "linhas": [],
        })
        return resultado

    mostra_data = quadro.tem_data and not _data_pedido_sensivel(quadro.cols)
    spec = {"faturamento": ("v", "sum"), "registros": ("v", "size")}
    if mostra_data:
        spec["ultima_compra"] = ("d", "max")
    grupos = base.groupby(chave, sort=False).agg(**spec)

    n_univ = int(len(grupos))
    sem = grupos[~grupos.index.isin(compraram)].reset_index().rename(columns={chave: dim})
    n_sem = int(len(sem))

    if not (m_comp.any()):
        avisos.append(
            "Nenhuma venda encontrada para o critério de compra informado: todos os itens aparecem como "
            "'sem compra'. Confira se a base cobre esse período."
        )
    elif quadro.tem_data and consulta["periodo"] and consulta["periodo"].get("inicio") \
            and pd.Timestamp(consulta["periodo"]["inicio"]) > tab["d"].max():
        avisos.append("O período consultado começa depois da última data da base.")
    if any(f["dimensao"] == "mes_do_ano" for f in consulta["filtros"]) and not any(
        f["dimensao"] == "ano" for f in consulta["filtros"]
    ) and not consulta["periodo"]:
        avisos.append("O mês foi consultado sem ano definido: considera esse mês em qualquer ano da base.")

    ordenar = consulta["ordenar_por"] or ("faturamento" if "faturamento" in sem.columns else dim)
    if ordenar in sem.columns:
        crescente = (consulta["direcao"] == "asc") if consulta["direcao"] else (ordenar == dim)
        sem = sem.sort_values(ordenar, ascending=crescente, na_position="last", kind="stable")

    limite = min(consulta["limite"], limite_maximo)
    sem = sem.head(limite).copy()

    prefixo = DIMENSOES_TEXTO.get(dim, {}).get("prefixo") or quadro.cols.get("_prefixos_sensiveis", {}).get(dim)
    if prefixo:
        sem[dim] = sem[dim].map(lambda nome: _pseudonimo(mapa, prefixo, str(nome)))
    else:
        sem[dim] = sem[dim].map(sanitizar_valor_textual)

    sem = sem.astype(object)
    resultado.update({
        "totais_do_filtro": {"registros": int(len(base)), "faturamento": round(float(base["v"].sum()), 2)},
        "resumo": {"itens_no_universo": n_univ, "itens_que_compraram": n_univ - n_sem, "itens_sem_compra": n_sem},
        "avisos": avisos,
        "linhas_totais": n_sem,
        "linhas_retornadas": int(len(sem)),
        "truncado": n_sem > len(sem),
        "linhas": [{k: _json_valor(v) for k, v in linha.items()} for linha in sem.to_dict("records")],
    })
    return resultado

def executar_consulta(quadro, consulta, mapa, limite_maximo=LIMITE_MAXIMO):
    if consulta.get("tipo") == "sem_compra":
        return _executar_sem_compra(quadro, consulta, mapa, limite_maximo)
    return _executar_consulta_agregada(quadro, consulta, mapa, limite_maximo)

def _executar_consulta_agregada(quadro, consulta, mapa, limite_maximo=LIMITE_MAXIMO):

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



def aplicar_fuzzy_base(pergunta, quadro):
    """Corrige termos parecidos com valores reais da base, sem alterar as validações existentes.

    A busca ocorre localmente, antes da pseudonimização. Só valores de colunas categóricas
    permitidas são considerados; colunas de datas, métricas e campos bloqueados ficam fora.
    """
    texto = str(pergunta or "")
    if not texto.strip():
        return texto

    # Coleta valores categóricos disponíveis na base local, sem enviar esses valores à IA.
    colunas = []
    for dim in DIMENSOES_TEXTO:
        origem = quadro.cols.get(dim)
        if origem and origem in quadro._df.columns:
            colunas.append(origem)
    for dim, origem in quadro.cols.get("_dinamicas", {}).items():
        cabecalho = normalizar(origem)
        if (
            origem in quadro._df.columns
            and cabecalho not in COLUNAS_TEMPO_PRESERVADAS
            and cabecalho not in COLUNAS_VALOR_PRESERVADAS
            and not coluna_bloqueada(origem)
        ):
            colunas.append(origem)

    valores = {}
    for origem in dict.fromkeys(colunas):
        if coluna_bloqueada(origem):
            continue
        for valor in quadro._df[origem].dropna().astype(str).unique():
            valor = valor.strip()
            chave = normalizar(valor)
            # Ignora valores vazios/curtos comuns, mas permite códigos alfanuméricos curtos.
            if not chave or chave in PLACEHOLDERS or len(chave) < 3:
                continue
            if len(chave) < 4 and not any(c.isalpha() for c in chave):
                continue
            valores.setdefault(chave, valor)

    if not valores:
        return texto

    # Não altera valores já escritos corretamente na pergunta.
    pergunta_norm = normalizar(texto)
    candidatos = [
        (chave, valor)
        for chave, valor in valores.items()
        if chave not in pergunta_norm
    ]
    if not candidatos:
        return texto

    # Gera trechos de 1 a 8 palavras da pergunta para compará-los com valores da base.
    padrao_tokens = re.compile(r"\b[\wÀ-ÿ]+(?:[-./][\wÀ-ÿ]+)*\b", re.UNICODE)
    tokens = list(padrao_tokens.finditer(texto))
    if not tokens:
        return texto

    spans = []
    max_palavras = min(8, len(tokens))
    for inicio in range(len(tokens)):
        for tamanho in range(1, max_palavras + 1):
            fim_indice = inicio + tamanho - 1
            if fim_indice >= len(tokens):
                break
            ini, fim = tokens[inicio].start(), tokens[fim_indice].end()
            trecho = texto[ini:fim]
            chave_trecho = normalizar(trecho)
            # Se o trecho já corresponde exatamente a um valor existente na base,
            # não o envie ao fuzzy: isso impede trocar um cliente correto
            # (ex.: "Bazzo Variedades") por outro nome parecido (ex.: "Agro Variedades").
            if (
                len(chave_trecho) >= 3
                and chave_trecho not in PLACEHOLDERS
                and chave_trecho not in valores
            ):
                spans.append((ini, fim, chave_trecho, tamanho))

    melhores = []
    for ini, fim, trecho_norm, palavras_trecho in spans:
        melhor_valor, melhor_score = None, 0.0
        segundo_score = 0.0
        for chave, valor in candidatos:
            palavras_valor = len(chave.split())
            if abs(palavras_valor - palavras_trecho) > 1:
                continue
            # Evita comparações sem sentido entre strings de comprimentos muito diferentes.
            diferenca = abs(len(chave) - len(trecho_norm))
            if diferenca > max(3, int(max(len(chave), len(trecho_norm)) * 0.28)):
                continue
            score = SequenceMatcher(None, trecho_norm, chave).ratio()
            if score > melhor_score:
                segundo_score = melhor_score
                melhor_score = score
                melhor_valor = valor
            elif score > segundo_score:
                segundo_score = score

        limite = 0.88 if len(trecho_norm) <= 5 else 0.80
        # Exige diferença mínima para não trocar por um valor apenas ligeiramente mais parecido.
        if melhor_valor is not None and melhor_score >= limite and (
            melhor_score - segundo_score >= 0.025 or melhor_score >= 0.96
        ):
            melhores.append({
                "inicio": ini, "fim": fim, "valor": melhor_valor,
                "score": melhor_score, "tamanho": fim - ini,
            })

    if not melhores:
        return texto

    # Prioriza correspondências mais confiáveis e evita substituir trechos sobrepostos.
    selecionados = []
    for item in sorted(melhores, key=lambda x: (x["score"], x["tamanho"]), reverse=True):
        if any(item["inicio"] < escolhido["fim"] and item["fim"] > escolhido["inicio"] for escolhido in selecionados):
            continue
        selecionados.append(item)

    for item in sorted(selecionados, key=lambda x: x["inicio"], reverse=True):
        texto = texto[:item["inicio"]] + item["valor"] + texto[item["fim"]:]

    return texto


def _pergunta_pede_ausencia(pergunta):
    """True para perguntas como "quais clientes não compraram...", "quem parou de comprar..."."""
    p = normalizar(pergunta)
    return bool(re.search(
        r"\bnao (?:compr|vend|fiz|fez|tiver|tem |pedi|faturar)|\bsem (?:compra|venda|pedido|faturamento)"
        r"|\b(?:parou|pararam|deixou|deixaram) de (?:comprar|vender)|\bnunca (?:compr|vend)|\binativ",
        p,
    ))

def processar_pergunta(df, cols, pergunta, mapa, historico, gerar, etapa=None):

    """Planeja -> executa localmente -> responde.

    `gerar(prompt, sistema, json_mode)` faz a chamada ao modelo (injetada para permitir testes).

    `etapa(texto)` é opcional e informa o andamento à interface.

    Retorna dict com: resposta (nomes reais), resposta_segura, detalhes, memoria.

    """

    aviso_etapa = etapa or (lambda _t: None)

    quadro = Quadro(df, cols)

    # Fuzzy local: aproxima termos da pergunta dos valores existentes na base.
    # As validações do plano e os controles de privacidade abaixo permanecem inalterados.
    pergunta = aplicar_fuzzy_base(pergunta, quadro)

    # Se o usuário pesquisou explicitamente um nome entre aspas e ele não existe
    # exatamente na dimensão de clientes, interromper antes de chamar o Gemini.
    # A interface reconhece plano vazio/resultado inválido e limpa memória e pseudônimos.
    nome_nao_encontrado = _termo_pesquisa_nome_nao_encontrado(pergunta, quadro)
    if nome_nao_encontrado is not None:
        return {
            "resposta": "nao foram encontrados registros",
            "resposta_segura": "nao foram encontrados registros",
            "detalhes": {"rota": "validacao_local", "plano": [], "avisos": [], "resultados": []},
            "memoria": None,
        }

    nomes_por_prefixo = nomes_para_anonimizar(quadro)
    pergunta_segura, _encontrados = anonimizar_pergunta(pergunta, nomes_por_prefixo, mapa)

    # 2ª camada: nomes parciais (ex.: só "Bazzo"). Ambíguo -> nada é enviado à IA.
    nomes_proprios = _nomes_proprios(quadro, nomes_por_prefixo)
    pergunta_segura, _parciais, _ambiguos = anonimizar_nomes_parciais(pergunta_segura, nomes_proprios, mapa)
    if _ambiguos:
        termo, candidatos = _ambiguos[0]
        lista = "; ".join(candidatos[:6]) + ("; ..." if len(candidatos) > 6 else "")
        texto = (
            f'O termo "{termo}" corresponde a mais de um cadastro ({lista}). '
            "Por segurança, nada foi enviado à IA. Digite o nome completo para continuar."
        )
        return {
            "resposta": texto, "resposta_segura": texto, "mensagem_usuario": texto,
            "detalhes": {"rota": "validacao_local", "plano": [], "avisos": [], "resultados": []},
            "memoria": None,
        }
    _encontrados = list(_encontrados) + list(_parciais)

    # Trava final: se, por qualquer caminho, um nome real chegar ao texto de saída, o envio é abortado.
    _guarda = _preparar_guarda(nomes_proprios)
    _gerar_original = gerar

    def gerar(prompt, sistema, json_mode):
        if _nome_real_no_texto(prompt, _guarda):
            raise RuntimeError(
                "Envio bloqueado por segurança: um nome real de cliente, vendedor ou cadastro foi "
                "detectado no texto que iria para a IA. Nada foi enviado."
            )
        return _gerar_original(prompt, sistema, json_mode)

    # Mantém localmente quais clientes foram identificados na pergunta. Depois da
    # validação do plano, o filtro do cliente será imposto pelo código, mesmo que a IA
    # se esqueça de incluí-lo. Isso impede que uma pergunta sobre um cliente específico
    # retorne faturamento/quantidade da base inteira.
    clientes_mencionados = list(dict.fromkeys(
        identificador
        for _nome, identificador in _encontrados
        if str(identificador).casefold().startswith("cliente ")
    ))

    esquema = montar_esquema(quadro)

    # Rota rápida: perguntas inequívocas de total geral são calculadas localmente.
    # A IA continua sendo consultada para redigir a resposta, mas economizamos a chamada
    # de planejamento. Não há base estática, cache de resultados nem reutilização de histórico.
    # Pergunta com nome/código anonimizado ("Cliente 1", "Dado 2") exige filtro: nunca usa a rota de total geral.
    consulta_rapida = None if _encontrados else _consulta_rapida_sem_filtros(pergunta, quadro)
    if consulta_rapida is not None:
        aviso_etapa("Calculando diretamente os dados...")
        consultas = [consulta_rapida]
        avisos = []
        esclarecimento = None
        detalhes = {
            "rota": "rapida_local", "pergunta_enviada": pergunta_segura,
            "plano": consultas, "avisos": avisos,
        }
    else:
        # Rota completa: a IA planeja a consulta com base na pergunta atual.
        aviso_etapa("Entendendo a pergunta...")
        prompt_planejador = montar_prompt_planejador(pergunta_segura, esquema, historico)
        # Quando a pergunta contém somente um código alfanumérico (ex.: BP500),
        # instrui o planejador a procurar em todas as dimensões plausíveis disponíveis,
        # pois o identificador pode estar em SKU, código do produto ou descrição.
        identificador_isolado = re.fullmatch(
            r"\s*(?=[A-Za-z0-9._/-]*[A-Za-z])(?=[A-Za-z0-9._/-]*[0-9])[A-Za-z0-9][A-Za-z0-9._/-]{2,}\s*",
            str(pergunta or ""),
        )
        if identificador_isolado:
            prompt_planejador += (
                "\n\nINSTRUÇÃO ESPECIAL PARA CÓDIGO ISOLADO: a pergunta contém apenas um "
                "identificador alfanumérico. Não presuma que ele esteja somente em SKU. "
                "Crie consultas separadas, com operador 'igual', para cada dimensão disponível "
                "que possa guardar esse identificador: priorize 'sku' e 'produto' e inclua "
                "também campos adicionais (campo_N) cujo cabeçalho indique código, referência, "
                "item ou descrição, se existirem no esquema. Use o mesmo identificador em cada "
                "consulta, respeitando o limite de consultas e todas as regras do esquema."
            )
        texto_plano = gerar(prompt_planejador, INSTRUCOES_PLANEJADOR, True)
        consultas, avisos, esclarecimento = validar_plano(_extrair_json(texto_plano), cols, quadro.tem_data)
        # Pergunta de AUSÊNCIA ("quem não comprou"): se o plano veio agregado, pede uma vez o tipo correto.
        if (
            _pergunta_pede_ausencia(pergunta) and not esclarecimento
            and not any(c.get("tipo") == "sem_compra" for c in consultas)
        ):
            texto_plano = gerar(
                prompt_planejador
                + "\n\nATENÇÃO: a pergunta pede itens que NÃO compraram/venderam. Use obrigatoriamente UMA consulta "
                  "do tipo \"sem_compra\" (regra 11), com os filtros da compra que não ocorreu.",
                INSTRUCOES_PLANEJADOR, True,
            )
            consultas, avisos, esclarecimento = validar_plano(_extrair_json(texto_plano), cols, quadro.tem_data)
        detalhes = {"rota": "planejamento_ia", "pergunta_enviada": pergunta_segura, "plano": consultas, "avisos": avisos}

    # Defesa local: se a pergunta identificou um cliente da base, todas as consultas
    # devem ser filtradas por esse cliente. A IA não pode transformar a solicitação em
    # uma consulta geral omitindo o filtro. As demais validações do plano permanecem.
    if clientes_mencionados:
        if not cols.get("cliente"):
            texto = "Não foi possível consultar esse cliente porque a base não disponibiliza uma dimensão de cliente filtrável."
            return {
                "resposta": texto,
                "resposta_segura": texto,
                "detalhes": {"rota": "validacao_local", "pergunta_enviada": pergunta_segura, "plano": [], "avisos": [texto], "resultados": []},
                "memoria": None,
            }
        for consulta in consultas:
            consulta["filtros"] = [
                f for f in consulta.get("filtros", [])
                if f.get("dimensao") != "cliente"
            ]
            consulta["filtros"].append({
                "dimensao": "cliente",
                "operador": "igual",
                "valores": clientes_mencionados,
            })
        for consulta in consultas:
            if consulta.get("tipo") == "sem_compra":
                universo = consulta.setdefault("universo", {"filtros": [], "periodo": None})
                universo["filtros"] = [f for f in universo.get("filtros", []) if f.get("dimensao") != "cliente"]
                universo["filtros"].append({"dimensao": "cliente", "operador": "igual", "valores": clientes_mencionados})
        avisos.append("Filtro de cliente aplicado localmente para garantir que os resultados correspondam ao cliente mencionado.")

    detalhes = {"pergunta_enviada": pergunta_segura, "plano": consultas, "avisos": avisos}

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

                 "filtros": c["filtros"], "periodo": c["periodo"], "somente_ultima_data": c["somente_ultima_data"],
                 "tipo": c.get("tipo"), "universo": c.get("universo")}

                for c in consultas

            ],

            "resposta": texto[:700],

        },

    }

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
                {"role": "user", "content": pergunta},
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

    # Verificar se a resposta e as consultas são válidas.
    detalhes = saida.get("detalhes") or {}
    plano = detalhes.get("plano") or []
    resultados = detalhes.get("resultados") or []
    avisos = detalhes.get("avisos") or []

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
                {"role": "user", "content": pergunta},
                {"role": "assistant", "content": "nao foram encontrados registros"},
            ]
        )[-40:]
        st.rerun()
        return

    mensagem_local = saida.get("mensagem_usuario")
    if mensagem_local:
        st.session_state[chave_msgs] = (
            st.session_state[chave_msgs]
            + [
                {"role": "user", "content": pergunta},
                {"role": "assistant", "content": mensagem_local},
            ]
        )[-40:]
        st.rerun()
        return

    resultado_invalido = (
        not plano
        or not resultados_estruturados
    )

    if resultado_invalido:
        # Manter as mensagens e limpar somente a memória contextual.
        st.session_state[chave_mem] = []
        st.session_state[chave_map] = {}

        # Registrar o aviso no histórico visível: st.warning seguido de
        # st.rerun() desaparece antes que o usuário consiga lê-lo.
        st.session_state[chave_msgs] = (
            st.session_state[chave_msgs]
            + [
                {"role": "user", "content": pergunta},
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
            {"role": "user", "content": pergunta},
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