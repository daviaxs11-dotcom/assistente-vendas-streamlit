# painel_vendas.py
# Painel interativo de vendas: vendedor, cliente gerencial e produtos por segmento.
# Execute com: streamlit run painel_vendas.py
# Requer Streamlit >= 1.35 (seleção por clique em gráficos e tabelas).

import html
import io
import re
import unicodedata

import pandas as pd
import plotly.express as px
import streamlit as st

from assistente_ia import exibir_assistente_ia


st.set_page_config(page_title="Painel de Vendas", page_icon="📊", layout="wide")


# ---------- Tratamento dos cabeçalhos e dados ----------

def normalizar_texto(valor):
    valor = str(valor or "").replace("\xa0", " ").strip()
    valor = re.sub(r"\s+", " ", valor)
    return valor


def chave_coluna(valor):
    valor = normalizar_texto(valor).lower()
    valor = unicodedata.normalize("NFKD", valor)
    valor = "".join(c for c in valor if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", valor)


def localizar_coluna(df, alternativas):
    mapa = {chave_coluna(c): c for c in df.columns}
    for alternativa in alternativas:
        chave = chave_coluna(alternativa)
        if chave in mapa:
            return mapa[chave]
    return None


def converter_numero(serie):
    """Converte números no padrão brasileiro ou números já numéricos."""
    if pd.api.types.is_numeric_dtype(serie):
        return pd.to_numeric(serie, errors="coerce").fillna(0)
    s = serie.astype(str).str.strip()
    s = s.str.replace("R$", "", regex=False).str.replace("\u00a0", "", regex=False)
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace("%", "", regex=False)
    s = s.replace({"-": "0", "—": "0", "": "0", "nan": "0", "None": "0"})
    # Quando existe vírgula, considera-se o padrão brasileiro: 1.234,56.
    mask_br = s.str.contains(",", na=False)
    s.loc[mask_br] = (
        s.loc[mask_br].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(s, errors="coerce").fillna(0)


EXTENSOES_EXCEL = (".xlsx", ".xlsm", ".xls")


@st.cache_data(show_spinner=False)
def listar_abas(chave, _conteudo, nome):
    """Lista as abas do Excel (cache por arquivo; o conteúdo não é re-hasheado a cada rerun)."""
    if nome.lower().endswith(EXTENSOES_EXCEL):
        return pd.ExcelFile(io.BytesIO(_conteudo)).sheet_names
    return []


def ler_arquivo(conteudo, nome, aba=0):
    nome = nome.lower()
    buf = io.BytesIO(conteudo)
    if nome.endswith(EXTENSOES_EXCEL):
        try:
            # python-calamine é bem mais rápido que openpyxl (pip install python-calamine)
            return pd.read_excel(buf, sheet_name=aba, engine="calamine")
        except (ImportError, ValueError):
            buf.seek(0)
            return pd.read_excel(buf, sheet_name=aba)
    if nome.endswith(".csv"):
        try:
            return pd.read_csv(buf, sep=None, engine="python", encoding="utf-8-sig")
        except UnicodeDecodeError:
            buf.seek(0)
            return pd.read_csv(buf, sep=None, engine="python", encoding="latin-1")
    raise ValueError("Formato não suportado. Envie um arquivo Excel ou CSV.")


def moeda(valor):
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def formatar_eixos(fig, eixo_y=None):
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text="",
        hovermode="x unified",
    )
    if eixo_y:
        fig.update_yaxes(title=eixo_y, tickformat=",.2f")
    return fig


# ---------- Comparativo: último trimestre x mesmo trimestre do ano anterior ----------

COL_IND = "Indicador vs ano anterior"
COL_VAR = "Variação vs ano anterior (%)"
COL_ANT = "Fat. mesmo tri. ano anterior (R$)"
COL_ATU = "Fat. último tri. (R$)"

CONFIG_COLUNAS = {
    COL_VAR: st.column_config.NumberColumn(COL_VAR, format="%.2f%%"),
    COL_ANT: st.column_config.NumberColumn(COL_ANT, format="%.2f"),
    COL_ATU: st.column_config.NumberColumn(COL_ATU, format="%.2f"),
}


def rotulo_trimestre(periodo):
    return f"{periodo.year}-T{periodo.quarter}"


def classificar_variacao(anterior, atual):
    """Compara dois faturamentos e devolve (indicador, variação %).

    Quando o período anterior é zero a variação percentual não existe (None).
    """
    if atual > anterior:
        status = "🟢 ▲ Crescimento"
    elif atual < anterior:
        status = "🔴 ▼ Queda"
    else:
        status = "🟡 ▬ Sem alteração"
    variacao = ((atual - anterior) / abs(anterior) * 100) if anterior != 0 else None
    if anterior == 0 and atual == 0:
        variacao = 0.0
    return status, variacao


def icone_status(status):
    """Ícone curto usado nos rótulos dos gráficos."""
    partes = str(status).split()
    if partes and partes[0] in ("🟢", "🔴", "🟡"):
        return " ".join(partes[:2])
    return "⚪"


def comparativo_anual(base_dados, coluna_grupo, grupos):
    """Compara o trimestre mais recente de base_dados com o MESMO trimestre do ano anterior.

    Retorna (tabela, rótulo do trimestre anterior, rótulo do trimestre atual).
    A tabela é indexada pelos itens de `grupos`, com colunas COL_IND, COL_VAR, COL_ANT e COL_ATU.
    Se não houver datas válidas ou o trimestre do ano anterior não existir na base filtrada,
    a tabela volta vazia.
    """
    dados = base_dados.loc[base_dados["_data"].notna(), ["_data", coluna_grupo, "_valor"]].copy()
    if dados.empty:
        return pd.DataFrame(), None, None
    dados["_tri"] = dados["_data"].dt.to_period("Q")
    tri_atual = dados["_tri"].max()
    tri_anterior = tri_atual - 4  # mesmo trimestre, um ano antes
    rot_ant, rot_atu = rotulo_trimestre(tri_anterior), rotulo_trimestre(tri_atual)
    if tri_anterior not in set(dados["_tri"]):
        return pd.DataFrame(), rot_ant, rot_atu

    vendas = (
        dados[dados["_tri"].isin([tri_anterior, tri_atual])]
        .groupby([coluna_grupo, "_tri"])["_valor"].sum().unstack(fill_value=0)
        .reindex(columns=[tri_anterior, tri_atual], fill_value=0)
        .reindex(list(grupos), fill_value=0)
    )
    linhas = {}
    for item, row in vendas.iterrows():
        anterior, atual = float(row[tri_anterior]), float(row[tri_atual])
        status, variacao = classificar_variacao(anterior, atual)
        linhas[item] = {COL_IND: status, COL_VAR: variacao, COL_ANT: anterior, COL_ATU: atual}
    return pd.DataFrame.from_dict(linhas, orient="index"), rot_ant, rot_atu


def anexar_comparativo(tabela, coluna_grupo, comp):
    """Acrescenta à tabela as colunas do comparativo anual (ou valores vazios se não houver comparação)."""
    tabela = tabela.copy()
    if comp is None or comp.empty:
        tabela[COL_IND] = "⚪ Sem comparação"
        tabela[COL_VAR] = None
        tabela[COL_ANT] = None
        tabela[COL_ATU] = None
        return tabela
    for c in (COL_IND, COL_VAR, COL_ANT, COL_ATU):
        tabela[c] = tabela[coluna_grupo].map(comp[c])
    tabela[COL_IND] = tabela[COL_IND].fillna("⚪ Sem comparação")
    tabela[COL_VAR] = pd.to_numeric(tabela[COL_VAR], errors="coerce").round(2)
    return tabela


def legenda_comparativo(rot_ant, rot_atu, sujeito):
    if rot_ant is None:
        return ("Indicadores comparam o último trimestre com o mesmo trimestre do ano anterior, "
                "mas não há datas válidas nos filtros atuais. ⚪ sem comparação.")
    return (
        f"Indicadores comparam {rot_atu} com {rot_ant} (mesmo trimestre do ano anterior) para {sujeito}, "
        "dentro dos filtros ativos: 🟢 ▲ crescimento | 🔴 ▼ queda | 🟡 ▬ sem alteração | "
        f"⚪ sem comparação ({rot_ant} sem dados nos filtros). "
        "Variação % em branco = sem faturamento no trimestre do ano anterior. "
        "Se o último trimestre ainda estiver em andamento, a comparação é parcial."
    )


# ---------- Navegação entre abas por clique ----------
# st.tabs não pode ser trocada por código, por isso a navegação usa um st.radio horizontal.

ABA_VENDEDOR = "Vendas por vendedor"
ABA_CLIENTE = "Vendas por cliente"
ABA_ITENS = "Itens por segmento"
ABA_PRECO = "Preço médio mensal"
ABA_DADOS = "Conferência dos dados"
ABAS = [ABA_VENDEDOR, ABA_CLIENTE, ABA_ITENS, ABA_PRECO, ABA_DADOS]


# Filtros de cada aba que ficam salvos na sessão ao trocar de seção.
# O Streamlit descarta o estado de widgets que não são desenhados em um rerun; por isso
# estas chaves são "re-gravadas" no início de cada execução (ver mais abaixo).
CHAVES_PERSISTENTES = [
    "f_vendedor_principal", "tempo_vendedor", "top_cliente_vendedor",
    "f_vendedor_cliente", "f_cliente", "tempo_cliente", "cliente_produto", "top_prod_cliente",
    "f_vendedor_segmento", "f_cliente_segmento", "f_produto_segmento", "metrica_segmento", "top_segmento",
    "f_vendedor_preco", "f_cliente_preco", "f_segmento_preco", "f_produto_preco", "top_produtos_preco",
    "detalhe_produto_preco", "detalhe_mes_preco",
]

# Quais filtros do destino são preenchidos quando a pessoa chega por um clique.
CHAVES_NAVEGACAO = {
    ABA_CLIENTE: {"vendedores": "f_vendedor_cliente", "clientes": "f_cliente"},
    ABA_ITENS: {"vendedores": "f_vendedor_segmento", "clientes": "f_cliente_segmento", "produtos": "f_produto_segmento"},
}


def ir_para(aba, **filtros):
    """Troca de aba e grava, nos filtros da aba de destino, os valores vindos do clique."""
    for nome, valor in filtros.items():
        chave = CHAVES_NAVEGACAO.get(aba, {}).get(nome)
        if chave:
            st.session_state[chave] = list(valor)
    if aba == ABA_CLIENTE and filtros.get("clientes"):
        st.session_state["cliente_produto"] = filtros["clientes"][0]
    st.session_state["_aba_pendente"] = aba
    st.rerun()


def multiselect_persistente(rotulo, opcoes, chave):
    """Multiselect que lembra a seleção. Valores que deixaram de existir nas opções são descartados;
    se nada sobrar do que estava salvo, volta a selecionar tudo."""
    salvo = st.session_state.get(chave)
    if salvo is None:
        st.session_state[chave] = list(opcoes)
    else:
        validos = [v for v in salvo if v in opcoes]
        if salvo and not validos:
            validos = list(opcoes)
        st.session_state[chave] = validos
    return st.multiselect(rotulo, opcoes, key=chave)


def radio_persistente(rotulo, opcoes, chave):
    if st.session_state.get(chave) not in opcoes:
        st.session_state[chave] = opcoes[0]
    return st.radio(rotulo, opcoes, horizontal=True, key=chave)


def slider_persistente(rotulo, minimo, maximo, padrao, chave):
    st.session_state.setdefault(chave, padrao)
    return st.slider(rotulo, min_value=minimo, max_value=maximo, key=chave)


def ponto_clicado(evento):
    """Primeiro ponto clicado em um gráfico plotly (ou None)."""
    try:
        pontos = evento["selection"]["points"]
    except (KeyError, TypeError):
        return None
    return pontos[0] if pontos else None


def nome_do_ponto(ponto, mapa_rotulos):
    """Recupera o nome real (cliente/produto) do ponto clicado, mesmo com emoji no rótulo do eixo."""
    custom = ponto.get("customdata")
    if custom:
        return custom[0] if isinstance(custom, (list, tuple)) else custom
    return mapa_rotulos.get(ponto.get("y"))


def linha_clicada(evento):
    """Índice da linha selecionada em uma tabela (ou None)."""
    try:
        linhas = evento["selection"]["rows"]
    except (KeyError, TypeError):
        return None
    return linhas[0] if linhas else None


for _chave in CHAVES_PERSISTENTES:
    if _chave in st.session_state:
        st.session_state[_chave] = st.session_state[_chave]


st.title("📊 Painel Interativo de Vendas")
st.caption("Carregue a base de vendas para explorar faturamento, quantidades, share e itens líderes.")

with st.sidebar:
    st.header("1. Carregar dados")
    arquivo = st.file_uploader("Selecione a base Excel ou CSV", type=["xlsx", "xlsm", "xls", "csv"])
    st.divider()
    st.header("2. Configurações")

if arquivo is None:
    st.info("Para começar, carregue a planilha que contém os cabeçalhos de vendas informados na conversa.")
    st.markdown(
        """
        **O painel contém três análises:**
        1. Vendas por vendedor, com evolução temporal e share dos clientes gerenciais.
        2. Vendas por cliente gerencial, com evolução temporal e share dos produtos.
        3. Itens mais vendidos por segmento, por quantidade e faturamento.

        Os valores de share são calculados dentro do filtro e do período selecionados.
        Os indicadores de crescimento/queda comparam o último trimestre com o mesmo trimestre do ano anterior.
        """
    )
    st.stop()

@st.cache_resource(show_spinner="Lendo e preparando a base (só na primeira vez)...")
def preparar_base(chave, _conteudo, nome, aba):
    """Lê o arquivo e cria as colunas de análise. Roda uma vez por arquivo/aba; depois vem do cache.

    O DataFrame devolvido é compartilhado (sem cópia): o restante do script só filtra com .copy(),
    nunca altera `df` diretamente.
    """
    df = ler_arquivo(_conteudo, nome, aba)
    df.columns = [normalizar_texto(c) for c in df.columns]
    if df.empty:
        raise ValueError("O arquivo não contém linhas de dados.")

    # Localiza colunas ignorando acentos, espaços e espaços não separáveis.
    col = {
        "data_faturamento": localizar_coluna(df, ["Data Faturamento"]),
        "data_pedido": localizar_coluna(df, ["Data Pedido"]),
        "ano": localizar_coluna(df, ["Ano Fat."]),
        "mes": localizar_coluna(df, ["Mês Fat.", "Mes Fat."]),
        "vendedor": localizar_coluna(df, ["Vendedor"]),
        "representante": localizar_coluna(df, ["Representante"]),
        "vendedor_cadastro": localizar_coluna(df, ["Vendedor atual do cadastro"]),
        "cliente_gerencial": localizar_coluna(df, ["Cliente Gerencial"]),
        "cliente": localizar_coluna(df, ["Cliente"]),
        "produto": localizar_coluna(df, ["Produto"]),
        "sku": localizar_coluna(df, ["SKU"]),
        "segmento": localizar_coluna(df, ["Segmento"]),
        "quantidade": localizar_coluna(df, ["Qtde", "Quantidade"]),
        "valor_produtos": localizar_coluna(df, ["Vlr total dos produtos"]),
        "valor_sem_ipi": localizar_coluna(df, ["Valor Total (Sem IPI)"]),
        "valor_bruto": localizar_coluna(df, ["Valor Bruto da NF"]),
        "empresa": localizar_coluna(df, ["Empresa"]),
        "status_faturamento": localizar_coluna(df, ["Status Faturamento"]),
        "departamento": localizar_coluna(df, ["Departamento"]),
        "gerente_regiao": localizar_coluna(df, ["Gerente região Cadastro", "Gerente região"]),
    }

    if not col["quantidade"]:
        raise ValueError("Não encontrei a coluna 'Qtde'. Verifique se o cabeçalho está presente.")

    # Preferir valor total dos produtos; se ausente, usar Valor Total (Sem IPI), depois Valor Bruto da NF.
    col_valor = col["valor_produtos"] or col["valor_sem_ipi"] or col["valor_bruto"]
    if not col_valor:
        raise ValueError("Não encontrei uma coluna de valor: 'Vlr total dos produtos', 'Valor Total (Sem IPI)' ou 'Valor Bruto da NF'.")

    col_vendedor = col["vendedor"] or col["representante"] or col["vendedor_cadastro"]
    col_cliente = col["cliente_gerencial"] or col["cliente"]
    col_produto = col["produto"] or col["sku"]

    def texto(coluna):
        if not coluna:
            return "(Não informado)"
        return df[coluna].fillna("(Não informado)").astype(str).str.strip()

    df["_quantidade"] = converter_numero(df[col["quantidade"]])
    df["_valor"] = converter_numero(df[col_valor])
    df["_vendedor"] = texto(col_vendedor)
    df["_cliente"] = texto(col_cliente)
    df["_produto"] = texto(col_produto)
    df["_segmento"] = texto(col["segmento"])

    # A data de análise padrão é Data Faturamento, com Data Pedido como alternativa.
    data_padrao = col["data_faturamento"] or col["data_pedido"]
    if data_padrao:
        if pd.api.types.is_datetime64_any_dtype(df[data_padrao]):
            df["_data"] = df[data_padrao]
        else:
            df["_data"] = pd.to_datetime(df[data_padrao], errors="coerce", dayfirst=True)
    else:
        df["_data"] = pd.NaT

    # Se a data principal não for interpretável, tenta compor a data usando Ano Fat. e Mês Fat.
    if df["_data"].isna().all() and col["ano"] and col["mes"]:
        ano = pd.to_numeric(df[col["ano"]], errors="coerce")
        mes = pd.to_numeric(df[col["mes"]], errors="coerce")
        df["_data"] = pd.to_datetime(
            ano.astype("Int64").astype(str) + "-" + mes.astype("Int64").astype(str) + "-01",
            errors="coerce",
        )

    return df, col, col_valor, col_vendedor, col_cliente, col_produto, data_padrao


conteudo_arquivo = arquivo.getvalue()
chave_arquivo = str(getattr(arquivo, "file_id", None) or (arquivo.name, arquivo.size))
abas = listar_abas(chave_arquivo, conteudo_arquivo, arquivo.name)
if len(abas) > 1:
    aba_escolhida = st.sidebar.selectbox("Aba da planilha", abas)
else:
    aba_escolhida = abas[0] if abas else 0


try:
    df, col, col_valor, col_vendedor, col_cliente, col_produto, data_padrao = preparar_base(
        chave_arquivo, conteudo_arquivo, arquivo.name, aba_escolhida
    )
except Exception as exc:
    st.error(f"Não foi possível carregar o arquivo: {exc}")
    st.stop()

if not col_vendedor:
    st.warning("Não encontrei Vendedor, Representante ou Vendedor atual do cadastro.")
if not col_cliente:
    st.warning("Não encontrei Cliente Gerencial nem Cliente.")
if not col_produto:
    st.warning("Não encontrei Produto nem SKU.")

# Espaço do assistente de IA (preenchido depois, com os dados já filtrados)
slot_assistente = st.sidebar.container()

# Filtros globais
with st.sidebar:
    st.subheader("Filtros globais")
    data_min = df["_data"].min()
    data_max = df["_data"].max()
    if pd.notna(data_min) and pd.notna(data_max):
        intervalo = st.date_input(
            "Período de análise",
            value=(data_min.date(), data_max.date()),
            min_value=data_min.date(),
            max_value=data_max.date(),
        )
        if isinstance(intervalo, (tuple, list)) and len(intervalo) == 2:
            inicio, fim = pd.Timestamp(intervalo[0]), pd.Timestamp(intervalo[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            filtrado = df[df["_data"].between(inicio, fim, inclusive="both")].copy()
        else:
            filtrado = df.copy()
        st.caption("Para o comparativo com o ano anterior, mantenha no período pelo menos o último trimestre e o mesmo trimestre do ano anterior.")
    else:
        filtrado = df.copy()
        st.caption("Não foi possível identificar datas válidas; os gráficos usarão toda a base.")

    if col["empresa"]:
        opcoes_empresa = sorted(filtrado[col["empresa"]].dropna().astype(str).unique())
        selecionadas_empresa = st.multiselect("Empresa", opcoes_empresa, default=opcoes_empresa)
        filtrado = filtrado[filtrado[col["empresa"]].astype(str).isin(selecionadas_empresa)]
    if col["status_faturamento"]:
        opcoes_status = sorted(filtrado[col["status_faturamento"]].dropna().astype(str).unique())
        if opcoes_status:
            status_escolhidos = st.multiselect("Status de faturamento", opcoes_status, default=opcoes_status)
            filtrado = filtrado[filtrado[col["status_faturamento"]].astype(str).isin(status_escolhidos)]
    if col["departamento"]:
        opcoes_departamento = sorted(filtrado[col["departamento"]].dropna().astype(str).unique())
        if opcoes_departamento:
            departamentos_escolhidos = st.multiselect("Departamento", opcoes_departamento, default=opcoes_departamento)
            filtrado = filtrado[filtrado[col["departamento"]].astype(str).isin(departamentos_escolhidos)]
    if col["gerente_regiao"]:
        opcoes_gerente = sorted(filtrado[col["gerente_regiao"]].dropna().astype(str).unique())
        if opcoes_gerente:
            gerentes_escolhidos = st.multiselect("Gerente região", opcoes_gerente, default=opcoes_gerente)
            filtrado = filtrado[filtrado[col["gerente_regiao"]].astype(str).isin(gerentes_escolhidos)]

with slot_assistente:
    exibir_assistente_ia(
        df_vendas=filtrado,
        coluna_cliente="_cliente" if col_cliente else None,
        coluna_valor="_valor",
        coluna_data="_data",
        coluna_produto="_produto" if col_produto else None,
        coluna_quantidade="_quantidade",
        coluna_vendedor="_vendedor" if col_vendedor else None,
        coluna_segmento="_segmento" if col["segmento"] else None,
        colunas_extras={
            "empresa": col["empresa"],
            "status_faturamento": col["status_faturamento"],
            "departamento": col["departamento"],
            "gerente_regiao": col["gerente_regiao"],
            "representante": col["representante"],
            "vendedor_cadastro": col["vendedor_cadastro"],
        },
        sufixo="painel",
    )

if filtrado.empty:
    st.warning("Nenhum registro encontrado com os filtros selecionados.")
    st.stop()

total_vendas = filtrado["_valor"].sum()
total_qtde = filtrado["_quantidade"].sum()
clientes_count = filtrado["_cliente"].nunique()
produtos_count = filtrado["_produto"].nunique()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Faturamento analisado", moeda(total_vendas))
k2.metric("Quantidade vendida", f"{total_qtde:,.0f}".replace(",", "."))
k3.metric("Clientes gerenciais", f"{clientes_count:,}".replace(",", "."))
k4.metric("Produtos vendidos", f"{produtos_count:,}".replace(",", "."))

if st.sidebar.button("🧹 Limpar filtros das abas"):
    for _chave in CHAVES_PERSISTENTES:
        st.session_state.pop(_chave, None)
    st.rerun()

# Navegação entre as seções. Um clique em gráfico/tabela define "_aba_pendente"
# e a seção é trocada aqui, antes de o st.radio ser criado.
if "_aba_pendente" in st.session_state:
    st.session_state["aba_ativa"] = st.session_state.pop("_aba_pendente")

aba_ativa = st.radio(
    "Seção",
    ABAS,
    horizontal=True,
    key="aba_ativa",
    label_visibility="collapsed",
)
st.divider()


def grafico_tempo(base, granularidade, medida, titulo, chave="tempo"):
    base = base[base["_data"].notna()].copy()
    if granularidade == "Dia":
        base["_eixo_tempo"] = base["_data"].dt.strftime("%Y-%m-%d")
    elif granularidade == "Mês":
        base["_eixo_tempo"] = base["_data"].dt.to_period("M").astype(str)
    else:
        base["_eixo_tempo"] = base["_data"].dt.year.astype(str) + "-T" + base["_data"].dt.quarter.astype(str)
    if base.empty:
        st.info("Não há datas válidas para montar a evolução temporal.")
        return
    resumo = base.groupby("_eixo_tempo", as_index=False)[medida].sum().sort_values("_eixo_tempo")
    fig = px.line(resumo, x="_eixo_tempo", y=medida, markers=True, title=titulo)
    fig.update_layout(xaxis_title="Período", yaxis_title="Valor (R$)" if medida == "_valor" else "Quantidade")
    if medida == "_valor":
        fig.update_yaxes(tickprefix="R$ ", tickformat=",.0f")
    st.plotly_chart(formatar_eixos(fig), use_container_width=True, key=f"{chave}_{granularidade}_{medida}")


if aba_ativa == ABA_VENDEDOR:
    st.subheader("Desempenho por vendedor")
    if not col_vendedor:
        st.info("Adicione uma coluna de vendedor, representante ou vendedor atual do cadastro para usar esta análise.")
    else:
        op_vendedores = sorted(filtrado["_vendedor"].dropna().unique().tolist())
        escolhidos = multiselect_persistente("Vendedor(es)", op_vendedores, "f_vendedor_principal")
        base = filtrado[filtrado["_vendedor"].isin(escolhidos)].copy()
        c1, c2 = st.columns(2)
        with c1:
            resumo_v = base.groupby("_vendedor", as_index=False)["_valor"].sum().sort_values("_valor", ascending=False)
            fig = px.bar(resumo_v, x="_valor", y="_vendedor", orientation="h", title="Faturamento por vendedor", text_auto=".2s")
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Valor (R$)", yaxis_title="")
            fig.update_xaxes(tickprefix="R$ ", tickformat=",.0f")
            st.plotly_chart(formatar_eixos(fig), use_container_width=True, key="grafico_unico_01")
        with c2:
            resumo_q = base.groupby("_vendedor", as_index=False)["_quantidade"].sum().sort_values("_quantidade", ascending=False)
            fig = px.bar(resumo_q, x="_quantidade", y="_vendedor", orientation="h", title="Quantidade por vendedor", text_auto=True)
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Quantidade", yaxis_title="")
            st.plotly_chart(formatar_eixos(fig), use_container_width=True, key="grafico_unico_02")

        granularidade = radio_persistente("Detalhamento temporal", ["Dia", "Mês", "Trimestre"], "tempo_vendedor")
        c1, c2 = st.columns(2)
        with c1:
            grafico_tempo(base, granularidade, "_valor", f"Faturamento por {granularidade.lower()}", chave="tempo_vendedor")
        with c2:
            grafico_tempo(base, granularidade, "_quantidade", f"Quantidade por {granularidade.lower()}", chave="tempo_vendedor")

        st.subheader("Composição do faturamento: clientes gerenciais dentro do vendedor")
        if col_cliente:
            top_n = slider_persistente("Número de clientes exibidos", 5, 30, 10, "top_cliente_vendedor")
            part = base.groupby("_cliente", as_index=False)["_valor"].sum().sort_values("_valor", ascending=False)
            total = part["_valor"].sum()
            part["Share (%)"] = (part["_valor"] / total * 100) if total else 0

            # Último trimestre x mesmo trimestre do ano anterior, para cada cliente.
            comp, rot_ant, rot_atu = comparativo_anual(base, "_cliente", part["_cliente"])
            part = anexar_comparativo(part, "_cliente", comp)
            part["_cliente_grafico"] = part[COL_IND].map(icone_status) + "  " + part["_cliente"]

            part_top = part.head(top_n).copy()
            outros = part.iloc[top_n:]["_valor"].sum()
            if outros > 0:
                # "Demais clientes" reúne os clientes fora do Top N; o indicador usa o total agregado deles.
                demais = part.iloc[top_n:]["_cliente"]
                if comp.empty:
                    status_demais, variacao_demais, ant_demais, atu_demais = "⚪ Sem comparação", None, None, None
                else:
                    ant_demais = float(comp.loc[demais, COL_ANT].sum())
                    atu_demais = float(comp.loc[demais, COL_ATU].sum())
                    status_demais, variacao_demais = classificar_variacao(ant_demais, atu_demais)
                    variacao_demais = round(variacao_demais, 2) if variacao_demais is not None else None
                part_top = pd.concat([part_top, pd.DataFrame([{
                    "_cliente": "Demais clientes", "_valor": outros,
                    "Share (%)": outros / total * 100 if total else 0,
                    COL_IND: status_demais, COL_VAR: variacao_demais,
                    COL_ANT: ant_demais, COL_ATU: atu_demais,
                    "_cliente_grafico": f"{icone_status(status_demais)}  Demais clientes",
                }])], ignore_index=True)

            part_top["Share (%)"] = part_top["Share (%)"].round(2)
            fig = px.bar(part_top.sort_values("_valor"), x="_valor", y="_cliente_grafico", orientation="h",
                         color="_cliente_grafico", custom_data=["_cliente"],
                         title="Share dos clientes no faturamento dos vendedores",
                         hover_data={"Share (%)": True, "_valor": ":,.2f", COL_IND: True,
                                     COL_VAR: ":.2f", COL_ANT: ":,.2f", COL_ATU: ":,.2f"})
            fig.update_layout(showlegend=False, xaxis_title="Valor (R$)", yaxis_title="")
            fig.update_xaxes(tickprefix="R$ ", tickformat=",.0f")
            fig = formatar_eixos(fig)
            fig.update_layout(hovermode="closest")  # necessário para o clique identificar a barra correta
            evento_grafico = st.plotly_chart(
                fig, use_container_width=True, key="grafico_unico_03",
                on_select="rerun", selection_mode="points",
            )
            st.caption(legenda_comparativo(rot_ant, rot_atu, "cada cliente"))
            st.caption("👆 Clique em uma barra do gráfico ou selecione uma linha da tabela para abrir o cliente em 'Vendas por cliente'.")

            ponto = ponto_clicado(evento_grafico)
            if ponto:
                mapa_clientes = dict(zip(part_top["_cliente_grafico"], part_top["_cliente"]))
                cliente_clicado = nome_do_ponto(ponto, mapa_clientes)
                if cliente_clicado == "Demais clientes":
                    st.info("'Demais clientes' reúne vários clientes. Aumente o número de clientes exibidos ou selecione o cliente na tabela abaixo.")
                elif cliente_clicado:
                    ir_para(ABA_CLIENTE, vendedores=escolhidos, clientes=[cliente_clicado])

            tabela_part = (
                part.sort_values("_valor", ascending=False)
                .rename(columns={"_cliente": "Cliente gerencial", "_valor": "Vendas (R$)"})
                .reset_index(drop=True)
            )
            tabela_part["Share (%)"] = (tabela_part["Vendas (R$)"] / total * 100).round(2) if total else 0
            evento_tabela = st.dataframe(
                tabela_part[["Cliente gerencial", "Vendas (R$)", "Share (%)", COL_IND, COL_VAR, COL_ANT, COL_ATU]],
                use_container_width=True, hide_index=True, column_config=CONFIG_COLUNAS,
                on_select="rerun", selection_mode="single-row", key="tabela_clientes_vendedor",
            )
            linha = linha_clicada(evento_tabela)
            if linha is not None:
                ir_para(ABA_CLIENTE, vendedores=escolhidos, clientes=[tabela_part.iloc[linha]["Cliente gerencial"]])
        else:
            st.info("A coluna Cliente Gerencial/Cliente não foi encontrada.")


if aba_ativa == ABA_CLIENTE:
    st.subheader("Desempenho por cliente gerencial")

    if not col_cliente:
        st.info("Adicione a coluna Cliente Gerencial ou Cliente para usar esta análise.")
    elif filtrado.empty:
        st.info("Não existem dados disponíveis para os filtros atuais.")
    else:
        # Filtros vindos de um clique em outra aba (vazio quando a pessoa abre a aba manualmente).

        vendedores_disponiveis = sorted(filtrado["_vendedor"].dropna().unique().tolist())
        escolhidos_cliente_vend = multiselect_persistente("Vendedor(es)", vendedores_disponiveis, "f_vendedor_cliente")
        base_v = filtrado[filtrado["_vendedor"].isin(escolhidos_cliente_vend)].copy()

        clientes_disponiveis = sorted(base_v["_cliente"].dropna().unique().tolist())
        escolhidos_clientes = multiselect_persistente("Cliente(s) gerencial(is)", clientes_disponiveis, "f_cliente")
        base_c = base_v[base_v["_cliente"].isin(escolhidos_clientes)].copy()

        if base_c.empty:
            st.warning("Nenhum registro encontrado para os vendedores e clientes selecionados.")
        else:
            total_vendas = base_c["_valor"].sum()
            total_quantidade = base_c["_quantidade"].sum()
            total_clientes = base_c["_cliente"].nunique()

            # Comparativo geral dos clientes selecionados: último trimestre x mesmo trimestre do ano anterior.
            comp_geral, rot_ant_g, rot_atu_g = comparativo_anual(
                base_c.assign(_todos="Total"), "_todos", ["Total"]
            )

            k1, k2, k3 = st.columns(3)
            k1.metric("Faturamento", moeda(total_vendas))
            k2.metric("Quantidade vendida", f"{total_quantidade:,.0f}".replace(",", "."))
            k3.metric("Clientes gerenciais", f"{total_clientes:,}".replace(",", "."))

            if not comp_geral.empty:
                linha_g = comp_geral.loc["Total"]
                var_g = linha_g[COL_VAR]
                st.metric(
                    f"Faturamento {rot_atu_g} vs {rot_ant_g}",
                    moeda(linha_g[COL_ATU]),
                    delta=f"{var_g:+.2f}%".replace(".", ",") if pd.notna(var_g) else "sem base de comparação",
                    delta_color="normal" if pd.notna(var_g) else "off",
                )
                st.caption(f"Anterior ({rot_ant_g}): {moeda(linha_g[COL_ANT])}")

            c1, c2 = st.columns(2)
            with c1:
                resumo_c = (
                    base_c.groupby("_cliente", as_index=False)
                    .agg(_valor=("_valor", "sum"))
                    .sort_values("_valor", ascending=False)
                    .head(25)
                )
                if not resumo_c.empty:
                    fig = px.bar(
                        resumo_c.sort_values("_valor"),
                        x="_valor", y="_cliente", orientation="h",
                        title="Faturamento por cliente (até 25)", text_auto=".2s"
                    )
                    fig.update_layout(
                        yaxis={"categoryorder": "total ascending"},
                        xaxis_title="Valor (R$)", yaxis_title=""
                    )
                    fig.update_xaxes(tickprefix="R$ ", tickformat=",.0f")
                    st.plotly_chart(formatar_eixos(fig), use_container_width=True, key="grafico_unico_04")

            with c2:
                resumo_qc = (
                    base_c.groupby("_cliente", as_index=False)
                    .agg(_quantidade=("_quantidade", "sum"))
                    .sort_values("_quantidade", ascending=False)
                    .head(25)
                )
                if not resumo_qc.empty:
                    fig = px.bar(
                        resumo_qc.sort_values("_quantidade"),
                        x="_quantidade", y="_cliente", orientation="h",
                        title="Quantidade por cliente (até 25)", text_auto=True
                    )
                    fig.update_layout(
                        yaxis={"categoryorder": "total ascending"},
                        xaxis_title="Quantidade", yaxis_title=""
                    )
                    st.plotly_chart(formatar_eixos(fig), use_container_width=True, key="grafico_unico_05")

            # Share do cliente no faturamento total de cada vendedor.
            st.subheader("Share do cliente gerencial no faturamento do vendedor")
            denominador_vendedor = base_v.groupby("_vendedor")["_valor"].sum()
            share_cliente = (
                base_c.groupby(["_vendedor", "_cliente"], as_index=False)
                .agg(_valor=("_valor", "sum"), _quantidade=("_quantidade", "sum"))
            )
            share_cliente["_total_vendedor"] = share_cliente["_vendedor"].map(denominador_vendedor)
            share_cliente["Share do vendedor (%)"] = (
                share_cliente["_valor"]
                .div(share_cliente["_total_vendedor"].where(share_cliente["_total_vendedor"] != 0))
                .mul(100).fillna(0).round(2)
            )
            if not share_cliente.empty:
                fig = px.bar(
                    share_cliente.sort_values("Share do vendedor (%)"),
                    x="Share do vendedor (%)", y="_cliente", color="_vendedor",
                    orientation="h", barmode="group",
                    title="Participação de cada cliente no faturamento do vendedor",
                    hover_data={
                        "_valor": ":,.2f",
                        "_quantidade": True,
                        "_total_vendedor": ":,.2f"
                    }
                )
                fig.update_layout(xaxis_title="Participação (%)", yaxis_title="")
                fig.update_xaxes(ticksuffix="%")
                st.plotly_chart(formatar_eixos(fig), use_container_width=True, key="grafico_unico_06")
                st.dataframe(
                    share_cliente.rename(columns={
                        "_vendedor": "Vendedor",
                        "_cliente": "Cliente gerencial",
                        "_valor": "Faturamento do cliente (R$)",
                        "_quantidade": "Quantidade",
                        "_total_vendedor": "Faturamento total do vendedor"
                    }),
                    use_container_width=True, hide_index=True
                )

            st.subheader("Evolução temporal dos clientes")
            granularidade_c = radio_persistente("Detalhamento temporal", ["Dia", "Mês", "Trimestre"], "tempo_cliente")
            if base_c["_data"].notna().any():
                c1, c2 = st.columns(2)
                with c1:
                    grafico_tempo(base_c, granularidade_c, "_valor",
                                  f"Faturamento por {granularidade_c.lower()}", chave="tempo_cliente")
                with c2:
                    grafico_tempo(base_c, granularidade_c, "_quantidade",
                                  f"Quantidade por {granularidade_c.lower()}", chave="tempo_cliente")
            else:
                st.info("Não há datas válidas para gerar a evolução temporal.")

            st.subheader("Composição do faturamento: produtos dentro do cliente")
            if col_produto:
                clientes_detalhe = sorted(base_c["_cliente"].dropna().unique().tolist())
                if clientes_detalhe:
                    if st.session_state.get("cliente_produto") not in clientes_detalhe:
                        st.session_state.pop("cliente_produto", None)
                    cliente_detalhe = st.selectbox(
                        "Selecione um cliente para detalhar os produtos",
                        clientes_detalhe, key="cliente_produto"
                    )
                    base_prod = base_c[base_c["_cliente"] == cliente_detalhe].copy()
                    part_prod = (
                        base_prod.groupby("_produto", as_index=False)
                        .agg(_valor=("_valor", "sum"), _quantidade=("_quantidade", "sum"))
                        .sort_values("_valor", ascending=False)
                    )
                    total_prod = part_prod["_valor"].sum()
                    part_prod["Share do cliente (%)"] = (
                        part_prod["_valor"]
                        .div(total_prod if total_prod != 0 else float("nan"))
                        .mul(100).fillna(0).round(2)
                    )

                    # Mesmo comparativo da aba de vendedores: último trimestre x mesmo trimestre do ano anterior.
                    comp_prod, rot_ant_p, rot_atu_p = comparativo_anual(base_prod, "_produto", part_prod["_produto"])
                    part_prod = anexar_comparativo(part_prod, "_produto", comp_prod)
                    part_prod["_produto_grafico"] = part_prod[COL_IND].map(icone_status) + "  " + part_prod["_produto"]

                    top_prod = slider_persistente("Número de produtos exibidos no gráfico", 5, 30, 10, "top_prod_cliente")
                    exibicao = part_prod.head(top_prod).copy()
                    if not exibicao.empty:
                        fig = px.bar(
                            exibicao.sort_values("_valor"),
                            x="_valor", y="_produto_grafico", orientation="h",
                            custom_data=["_produto"],
                            title=f"Produtos de {cliente_detalhe}", text_auto=".2s",
                            hover_data={
                                "Share do cliente (%)": True,
                                "_quantidade": True,
                                COL_IND: True,
                                COL_VAR: ":.2f",
                                COL_ANT: ":,.2f",
                                COL_ATU: ":,.2f",
                            }
                        )
                        fig.update_layout(
                            yaxis={"categoryorder": "total ascending"},
                            xaxis_title="Valor (R$)", yaxis_title=""
                        )
                        fig.update_xaxes(tickprefix="R$ ", tickformat=",.0f")
                        fig = formatar_eixos(fig)
                        fig.update_layout(hovermode="closest")  # necessário para o clique identificar a barra correta
                        evento_prod = st.plotly_chart(
                            fig, use_container_width=True, key="grafico_unico_07",
                            on_select="rerun", selection_mode="points",
                        )
                        st.caption(legenda_comparativo(rot_ant_p, rot_atu_p, "cada produto do cliente selecionado"))
                        st.caption("👆 Clique em uma barra do gráfico ou selecione uma linha da tabela para abrir o produto em 'Itens por segmento'.")

                        ponto = ponto_clicado(evento_prod)
                        if ponto:
                            mapa_produtos = dict(zip(exibicao["_produto_grafico"], exibicao["_produto"]))
                            produto_clicado = nome_do_ponto(ponto, mapa_produtos)
                            if produto_clicado:
                                ir_para(
                                    ABA_ITENS,
                                    vendedores=escolhidos_cliente_vend,
                                    clientes=[cliente_detalhe],
                                    produtos=[produto_clicado],
                                )
                    st.caption(
                        "O share dos produtos usa o faturamento total do cliente selecionado, "
                        "considerando todos os produtos, mesmo os que não aparecem no gráfico."
                    )
                    tabela_prod = (
                        part_prod.drop(columns=["_produto_grafico"])
                        .rename(columns={
                            "_produto": "Produto",
                            "_valor": "Vendas (R$)",
                            "_quantidade": "Quantidade"
                        })
                        .reset_index(drop=True)
                    )
                    evento_tab_prod = st.dataframe(
                        tabela_prod,
                        use_container_width=True, hide_index=True, column_config=CONFIG_COLUNAS,
                        on_select="rerun", selection_mode="single-row", key="tabela_produtos_cliente",
                    )
                    linha = linha_clicada(evento_tab_prod)
                    if linha is not None:
                        ir_para(
                            ABA_ITENS,
                            vendedores=escolhidos_cliente_vend,
                            clientes=[cliente_detalhe],
                            produtos=[tabela_prod.iloc[linha]["Produto"]],
                        )
                else:
                    st.info("Não há clientes disponíveis para detalhamento.")
            else:
                st.info("A coluna Produto/SKU não foi encontrada.")


if aba_ativa == ABA_ITENS:
    st.subheader("Itens mais vendidos")

    if not col_produto:
        st.info("A coluna Produto/SKU não foi encontrada.")

    elif filtrado.empty:
        st.info("Não existem dados disponíveis para os filtros atuais.")

    else:
        # Filtros vindos de um clique em outra aba (vazio quando a pessoa abre a aba manualmente).

        # Filtro de vendedores
        vendedores_seg = sorted(filtrado["_vendedor"].dropna().unique().tolist())
        escolhidos_seg = multiselect_persistente("Vendedor(es)", vendedores_seg, "f_vendedor_segmento")
        base_s = filtrado[filtrado["_vendedor"].isin(escolhidos_seg)].copy()

        # Filtro de clientes gerenciais
        clientes_seg = sorted(base_s["_cliente"].dropna().unique().tolist())
        escolhidos_clientes_seg = multiselect_persistente("Cliente(s) gerencial(is)", clientes_seg, "f_cliente_segmento")
        base_s = base_s[base_s["_cliente"].isin(escolhidos_clientes_seg)].copy()

        # Filtro de produtos
        produtos_seg = sorted(base_s["_produto"].dropna().unique().tolist())
        escolhidos_produtos_seg = multiselect_persistente("Produto(s)", produtos_seg, "f_produto_segmento")
        base_s = base_s[base_s["_produto"].isin(escolhidos_produtos_seg)].copy()

        if base_s.empty:
            st.warning("Nenhum registro encontrado para os vendedores, clientes e produtos selecionados.")

        else:
            # Configuração do ranking
            metrica = radio_persistente("Ordenar ranking por", ["Quantidade", "Faturamento"], "metrica_segmento")
            top_itens = slider_persistente("Quantidade de itens no ranking", 5, 50, 15, "top_segmento")
            coluna_metrica = "_quantidade" if metrica == "Quantidade" else "_valor"
            titulo_metrica = "Quantidade" if metrica == "Quantidade" else "Faturamento (R$)"

            # Ranking de produtos (agrupamento sem utilizar segmento)
            ranking_completo = (
                base_s.groupby("_produto", as_index=False)
                .agg(_quantidade=("_quantidade", "sum"), _valor=("_valor", "sum"))
            )
            ranking = (
                ranking_completo.sort_values(coluna_metrica, ascending=False)
                .head(top_itens).copy()
            )

            if ranking.empty:
                st.info("Não há produtos para exibir no ranking.")

            else:
                # Comparativo anual dos produtos do ranking
                comp_rank, rot_ant_r, rot_atu_r = comparativo_anual(base_s, "_produto", ranking["_produto"])
                ranking = anexar_comparativo(ranking, "_produto", comp_rank)

                dados_grafico = ranking.sort_values(coluna_metrica, ascending=True)
                fig = px.bar(
                    dados_grafico,
                    x=coluna_metrica, y="_produto", orientation="h",
                    title=f"Top {top_itens} produtos por {metrica.lower()}",
                    hover_data={
                        "_quantidade": True,
                        "_valor": ":,.2f",
                        COL_IND: True,
                        COL_VAR: ":.2f",
                    }
                )
                fig.update_layout(
                    xaxis_title=titulo_metrica, yaxis_title="",
                    yaxis={"categoryorder": "total ascending"}
                )
                if metrica == "Faturamento":
                    fig.update_xaxes(tickprefix="R$ ", tickformat=",.0f")
                st.plotly_chart(formatar_eixos(fig), use_container_width=True, key="grafico_unico_08")
                st.caption(legenda_comparativo(rot_ant_r, rot_atu_r, "cada produto do ranking (faturamento)"))

                # Evolução trimestral dos produtos
                st.subheader("Evolução trimestral dos itens")
                base_t = base_s[base_s["_data"].notna()].copy()

                if base_t.empty:
                    st.info("A evolução trimestral exige uma coluna de data válida.")
                else:
                    base_t["_trimestre_periodo"] = base_t["_data"].dt.to_period("Q")

                    # Considerar somente os produtos do ranking
                    base_t = base_t.merge(ranking[["_produto"]].drop_duplicates(), on="_produto", how="inner")

                    evolucao = (
                        base_t.groupby(["_trimestre_periodo", "_produto"], as_index=False)
                        .agg(**{coluna_metrica: (coluna_metrica, "sum")})
                    )

                    if evolucao.empty:
                        st.info("Não há dados trimestrais para os itens do ranking.")
                    else:
                        evolucao = evolucao.sort_values("_trimestre_periodo", ascending=True).copy()
                        evolucao["_trimestre_graf"] = evolucao["_trimestre_periodo"].astype(str)
                        ordem_trimestres = (
                            evolucao[["_trimestre_periodo", "_trimestre_graf"]]
                            .drop_duplicates()
                            .sort_values("_trimestre_periodo")["_trimestre_graf"]
                            .tolist()
                        )

                        fig = px.line(
                            evolucao,
                            x="_trimestre_graf", y=coluna_metrica, color="_produto",
                            markers=True,
                            category_orders={"_trimestre_graf": ordem_trimestres},
                            title=f"Evolução trimestral por {metrica.lower()} — produtos",
                            hover_data={"_produto": True}
                        )
                        fig.update_layout(
                            xaxis_title="Trimestre", yaxis_title=titulo_metrica,
                            legend_title="Produto",
                            xaxis={
                                "type": "category",
                                "categoryorder": "array",
                                "categoryarray": ordem_trimestres
                            }
                        )
                        if metrica == "Faturamento":
                            fig.update_yaxes(tickprefix="R$ ", tickformat=",.0f")
                        st.plotly_chart(formatar_eixos(fig), use_container_width=True, key="grafico_unico_09")

                # Tabela de produtos
                st.subheader("Tabela de produtos")
                tabela_ranking = ranking.rename(
                    columns={"_produto": "Produto", "_quantidade": "Quantidade", "_valor": "Vendas (R$)"}
                )
                st.dataframe(
                    tabela_ranking,
                    use_container_width=True, hide_index=True, column_config=CONFIG_COLUNAS
                )


if aba_ativa == ABA_PRECO:
    st.subheader("Preço médio mensal por produto")
    st.caption("Preço médio = faturamento ÷ quantidade vendida no mês, dentro dos filtros desta aba.")

    if not col_produto:
        st.info("A coluna Produto/SKU não foi encontrada.")
    elif not filtrado["_data"].notna().any():
        st.info("Não há datas válidas para calcular o preço médio mensal.")
    else:
        base_p = filtrado[filtrado["_data"].notna()]

        vendedores_preco = sorted(base_p["_vendedor"].dropna().unique().tolist())
        escolhidos_vend_preco = multiselect_persistente("Vendedor(es)", vendedores_preco, "f_vendedor_preco")
        base_p = base_p[base_p["_vendedor"].isin(escolhidos_vend_preco)]

        clientes_preco = sorted(base_p["_cliente"].dropna().unique().tolist())
        escolhidos_cli_preco = multiselect_persistente("Cliente(s) gerencial(is)", clientes_preco, "f_cliente_preco")
        base_p = base_p[base_p["_cliente"].isin(escolhidos_cli_preco)]

        segmentos_preco = sorted(base_p["_segmento"].dropna().unique().tolist())
        escolhidos_seg_preco = multiselect_persistente("Segmento(s)", segmentos_preco, "f_segmento_preco")
        base_p = base_p[base_p["_segmento"].isin(escolhidos_seg_preco)]
        if not col["segmento"]:
            st.caption("A coluna Segmento não foi encontrada na base; todos os itens aparecem como '(Não informado)'.")

        produtos_preco = sorted(base_p["_produto"].dropna().unique().tolist())
        escolhidos_prod_preco = multiselect_persistente("Produto(s)", produtos_preco, "f_produto_preco")
        base_p = base_p[base_p["_produto"].isin(escolhidos_prod_preco)].copy()

        if base_p.empty:
            st.warning("Nenhum registro encontrado para os filtros selecionados.")
        else:
            top_preco = slider_persistente(
                "Produtos exibidos no gráfico (maiores faturamentos)", 1, 30, 5, "top_produtos_preco"
            )

            base_p["_mes"] = base_p["_data"].dt.to_period("M")
            mensal = (
                base_p.groupby(["_mes", "_produto"], as_index=False)
                .agg(_valor=("_valor", "sum"), _quantidade=("_quantidade", "sum"))
            )
            mensal["_preco"] = mensal["_valor"].div(mensal["_quantidade"].where(mensal["_quantidade"] != 0))
            mensal["_mes_rot"] = mensal["_mes"].astype(str)
            ordem_meses = [str(m) for m in sorted(mensal["_mes"].unique())]

            totais = base_p.groupby("_produto").agg(_valor=("_valor", "sum"), _quantidade=("_quantidade", "sum"))
            totais = totais.sort_values("_valor", ascending=False)
            totais["_preco"] = totais["_valor"].div(totais["_quantidade"].where(totais["_quantidade"] != 0))

            valor_total_p, qtde_total_p = totais["_valor"].sum(), totais["_quantidade"].sum()
            k1, k2, k3 = st.columns(3)
            k1.metric("Preço médio no período", moeda(valor_total_p / qtde_total_p) if qtde_total_p else "—")
            k2.metric("Produtos", f"{len(totais):,}".replace(",", "."))
            k3.metric("Meses com vendas", f"{len(ordem_meses)}")

            # Preço médio de cada cliente em cada mês/produto (base do pop-up e do detalhe abaixo).
            base_p["_mes_rot"] = base_p["_mes"].astype(str)
            por_cliente = (
                base_p.groupby(["_mes_rot", "_produto", "_cliente"], as_index=False)
                .agg(_valor=("_valor", "sum"), _quantidade=("_quantidade", "sum"))
            )
            por_cliente["_preco_cli"] = por_cliente["_valor"].div(
                por_cliente["_quantidade"].where(por_cliente["_quantidade"] != 0)
            )
            por_cliente = por_cliente.merge(
                mensal[["_mes_rot", "_produto", "_preco"]], on=["_mes_rot", "_produto"], how="left"
            )

            top_nomes = totais.head(top_preco).index.tolist()
            graf = mensal[mensal["_produto"].isin(top_nomes)].dropna(subset=["_preco"]).sort_values("_mes").copy()
            if graf.empty:
                st.info("Não há preço calculável (quantidade zerada) para os produtos selecionados.")
            else:
                # Texto do pop-up: clientes que compraram no mês a preço igual ou abaixo da média do produto.
                LIMITE_POPUP = 8
                pc = por_cliente[por_cliente["_produto"].isin(top_nomes)]
                total_por_ponto = pc.groupby(["_mes_rot", "_produto"]).size()
                abaixo = pc[pc["_preco_cli"] <= pc["_preco"] + 1e-9].sort_values("_quantidade", ascending=False)
                textos, contagens = {}, {}
                for (mes_k, prod_k), g in abaixo.groupby(["_mes_rot", "_produto"]):
                    linhas = []
                    for cli, preco, qtd in zip(g["_cliente"].head(LIMITE_POPUP), g["_preco_cli"].head(LIMITE_POPUP), g["_quantidade"].head(LIMITE_POPUP)):
                        qtd_txt = f"{qtd:,.0f}".replace(",", ".")
                        linhas.append(f"• {html.escape(str(cli))}: {moeda(preco)} ({qtd_txt} un.)")
                    if len(g) > LIMITE_POPUP:
                        linhas.append(f"… e mais {len(g) - LIMITE_POPUP} cliente(s) (veja o detalhe abaixo)")
                    textos[(mes_k, prod_k)] = "<br>".join(linhas)
                    contagens[(mes_k, prod_k)] = f"{len(g)} de {int(total_por_ponto.get((mes_k, prod_k), len(g)))}"
                chaves_pontos = list(zip(graf["_mes_rot"], graf["_produto"]))
                graf["_clientes_txt"] = [textos.get(k, "—") for k in chaves_pontos]
                graf["_n_clientes"] = [contagens.get(k, "—") for k in chaves_pontos]

                fig = px.line(
                    graf, x="_mes_rot", y="_preco", color="_produto", markers=True,
                    category_orders={"_mes_rot": ordem_meses, "_produto": top_nomes},
                    title=f"Preço médio mensal — {len(top_nomes)} maiores produtos por faturamento",
                    custom_data=["_quantidade", "_valor", "_n_clientes", "_clientes_txt"],
                )
                fig.update_traces(
                    connectgaps=True,
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>Mês: %{x}<br>Preço médio: R$ %{y:,.2f}<br>"
                        "Quantidade: %{customdata[0]:,.0f} | Faturamento: R$ %{customdata[1]:,.2f}<br><br>"
                        "<b>Clientes a este preço ou abaixo (%{customdata[2]}):</b><br>%{customdata[3]}"
                        "<extra></extra>"
                    ),
                )
                fig.update_layout(
                    xaxis_title="Mês", yaxis_title="Preço médio (R$)", legend_title="Produto",
                    xaxis={"type": "category", "categoryorder": "array", "categoryarray": ordem_meses},
                )
                fig.update_yaxes(tickprefix="R$ ", tickformat=",.2f")
                fig = formatar_eixos(fig)
                fig.update_layout(hovermode="closest")  # um pop-up por ponto, com a lista de clientes
                st.plotly_chart(fig, use_container_width=True, key="grafico_preco_medio")
                st.caption(
                    "Passe o mouse sobre um ponto para ver os clientes que compraram naquele mês a preço igual "
                    "ou abaixo da média do produto. O preço de cada cliente é o preço médio dele no mês."
                )

            # Tabela: um produto por linha, um mês por coluna.
            pivo = (
                mensal.pivot(index="_produto", columns="_mes_rot", values="_preco")
                .reindex(index=totais.index, columns=ordem_meses)
            )
            tabela_preco = pd.DataFrame({
                "Produto": totais.index,
                "Faturamento (R$)": totais["_valor"].values,
                "Quantidade": totais["_quantidade"].values,
                "Preço médio no período (R$)": totais["_preco"].round(2).values,
            })
            if len(ordem_meses) >= 2:
                ultimo, penultimo = ordem_meses[-1], ordem_meses[-2]
                var_mes = (pivo[ultimo] / pivo[penultimo] - 1) * 100
                tabela_preco[f"Var. {ultimo} vs {penultimo} (%)"] = var_mes.replace([float("inf"), float("-inf")], float("nan")).round(2).values
            for mes in ordem_meses:
                tabela_preco[mes] = pivo[mes].round(2).values

            st.subheader("Preço médio por produto e mês (R$)")
            config_preco = {}
            for c in tabela_preco.columns:
                if c.startswith("Var. "):
                    config_preco[c] = st.column_config.NumberColumn(c, format="%.2f%%")
            st.dataframe(tabela_preco, use_container_width=True, hide_index=True, column_config=config_preco)
            st.download_button(
                "Baixar tabela de preço médio em CSV",
                tabela_preco.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                file_name="preco_medio_mensal.csv", mime="text/csv",
            )

            # Detalhe completo dos clientes de um produto em um mês.
            st.subheader("Detalhe dos clientes por produto e mês")
            opcoes_det_prod = list(totais.index)
            if st.session_state.get("detalhe_produto_preco") not in opcoes_det_prod:
                st.session_state["detalhe_produto_preco"] = opcoes_det_prod[0]
            d1, d2 = st.columns(2)
            with d1:
                prod_det = st.selectbox("Produto", opcoes_det_prod, key="detalhe_produto_preco")
            meses_do_produto = sorted(mensal.loc[mensal["_produto"] == prod_det, "_mes_rot"].unique().tolist())
            if st.session_state.get("detalhe_mes_preco") not in meses_do_produto:
                st.session_state["detalhe_mes_preco"] = meses_do_produto[-1]
            with d2:
                mes_det = st.selectbox("Mês", meses_do_produto, key="detalhe_mes_preco")

            det = por_cliente[(por_cliente["_produto"] == prod_det) & (por_cliente["_mes_rot"] == mes_det)].copy()
            preco_med = det["_preco"].iloc[0]
            if pd.notna(preco_med):
                igual_ou_abaixo = det["_preco_cli"] <= preco_med + 1e-9
                det["Situação"] = igual_ou_abaixo.map({True: "🟢 Igual ou abaixo da média", False: "🔴 Acima da média"})
                det["Situação"] = det["Situação"].where(det["_preco_cli"].notna(), "—")
                det["Vs. média do mês (%)"] = ((det["_preco_cli"] / preco_med - 1) * 100).round(2)
                st.caption(
                    f"Preço médio de {prod_det} em {mes_det}: {moeda(preco_med)} — "
                    f"{int(igual_ou_abaixo.sum())} de {len(det)} clientes compraram a esse preço ou abaixo."
                )
            tabela_det = (
                det.sort_values("_preco_cli")
                .rename(columns={
                    "_cliente": "Cliente gerencial", "_quantidade": "Quantidade",
                    "_valor": "Faturamento (R$)", "_preco_cli": "Preço do cliente (R$)",
                })
                .assign(**{"Preço do cliente (R$)": lambda d: d["Preço do cliente (R$)"].round(2)})
            )
            colunas_det = ["Cliente gerencial", "Quantidade", "Faturamento (R$)", "Preço do cliente (R$)"]
            colunas_det += [c for c in ("Vs. média do mês (%)", "Situação") if c in tabela_det.columns]
            st.dataframe(
                tabela_det[colunas_det], use_container_width=True, hide_index=True,
                column_config={"Vs. média do mês (%)": st.column_config.NumberColumn(format="%.2f%%")},
            )


if aba_ativa == ABA_DADOS:
    st.subheader("Conferência da base carregada")
    st.write(f"Linhas carregadas: **{len(df):,}**".replace(",", "."))
    st.write(f"Coluna de valor utilizada: **{col_valor}**")
    st.write(f"Coluna de vendedor utilizada: **{col_vendedor or 'não encontrada'}**")
    st.write(f"Coluna de cliente utilizada: **{col_cliente or 'não encontrada'}**")
    st.write(f"Coluna de produto utilizada: **{col_produto or 'não encontrada'}**")
    st.write(f"Coluna de data utilizada: **{data_padrao or 'não encontrada'}**")
    st.caption("O share é calculado sobre o valor da coluna de venda selecionada e respeita os filtros ativos.")
    st.dataframe(filtrado.drop(columns=["_quantidade", "_valor", "_vendedor", "_cliente", "_produto", "_segmento", "_data"], errors="ignore").head(100), use_container_width=True, hide_index=True)
    csv = filtrado.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Baixar dados filtrados em CSV", csv, file_name="vendas_filtradas.csv", mime="text/csv")
