# painel_vendas.py
# Painel interativo de vendas: vendedor, cliente gerencial e produtos por segmento.
# Execute com: streamlit run painel_vendas.py
# Requer Streamlit >= 1.35 (seleção por clique em gráficos e tabelas).

import base64
import html
import io
import re
import unicodedata
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from engine_projecao import projetar_meses

from assistente_ia import exibir_assistente_ia


st.set_page_config(page_title="Intech Machine | Painel Interativo de Vendas", page_icon="📊", layout="wide", initial_sidebar_state="expanded")


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


def num_br(valor):
    """Inteiro no padrão brasileiro: 2853420 -> 2.853.420."""
    try:
        return f"{valor:,.0f}".replace(",", ".")
    except (TypeError, ValueError):
        return str(valor)


def pct_br(valor, casas=1):
    return f"{valor:.{casas}f}%".replace(".", ",")


def rotulo_moeda_curta(v):
    if v >= 1_000_000:
        return f"R$ {v / 1_000_000:.1f} M".replace(".", ",")
    if v >= 1_000:
        return f"R$ {v / 1_000:.0f} mil"
    return "R$ 0" if v == 0 else f"R$ {v:.0f}"


def rotulo_eixo_curto(v):
    """0, 1M, 200 mil… (para eixos cujo título já traz a unidade)."""
    if v == 0:
        return "0"
    if v >= 1_000_000:
        return f"{v / 1_000_000:g}M".replace(".", ",")
    if v >= 1_000:
        return f"{v / 1_000:g} mil".replace(".", ",")
    return f"{v:g}".replace(".", ",")


def ticks_moeda_br(vmax, alvo=5, rotulador=None):
    """(valores, rótulos) de eixo com números 'redondos' e rótulos em pt-BR."""
    if not vmax or vmax <= 0:
        return None, None
    bruto = vmax / alvo
    mag = 10 ** np.floor(np.log10(bruto))
    passo = next(m * mag for m in (1, 2, 2.5, 5, 10) if bruto <= m * mag)
    n = int(np.ceil(vmax / passo))
    vals = [float(i * passo) for i in range(n + 1)]
    return vals, [(rotulador or rotulo_moeda_curta)(v) for v in vals]


PALETA_GRAFICOS = ["#ed1c24", "#8b929c", "#e8505a", "#7d0f17", "#5f6670", "#b85c63"]
COR_TEXTO_GRAFICO = "#e8eaed"


def padronizar_grafico(fig, altura=None, margem=None, legenda=False):
    """Aplica identidade visual Intech também às cores explícitas das séries."""
    fig.update_layout(
        template="plotly_dark",
        colorway=PALETA_GRAFICOS,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COR_TEXTO_GRAFICO, size=10),
        title=dict(font=dict(size=13, color="#f6f7f8"), x=0, xanchor="left"),
        legend=dict(title_text="", font=dict(family="Inter, sans-serif", color=COR_TEXTO_GRAFICO, size=9)),
        showlegend=legenda,
        hoverlabel=dict(bgcolor="#101316", font_color="#f6f6f6", bordercolor="#393f45"),
    )
    if altura:
        fig.update_layout(height=altura)
    if margem:
        fig.update_layout(margin=margem)

    # O colorway do Plotly não substitui cores definidas diretamente nas séries.
    # Sobrescrevemos essas cores para evitar barras azuis ou estilos antigos.
    indice_cor = 0
    for trace in fig.data:
        tipo = getattr(trace, "type", "")
        if tipo == "bar":
            # Barras sempre no vermelho da marca, independentemente da cor original.
            trace.update(marker=dict(color="#ed1c24", line=dict(color="#ed1c24", width=0)))
        elif tipo in ("scatter", "scattergl"):
            modo = getattr(trace, "mode", "") or ""
            cor = PALETA_GRAFICOS[indice_cor % len(PALETA_GRAFICOS)]
            if "lines" in modo:
                trace.update(line=dict(color=cor, width=2))
            if "markers" in modo:
                trace.update(marker=dict(color=cor, line=dict(color=cor, width=0)))
            indice_cor += 1
        elif tipo == "pie":
            trace.update(marker=dict(colors=PALETA_GRAFICOS, line=dict(color="#151719", width=1)))

    # Grade discreta apenas no eixo de valores; rótulos e títulos uniformes.
    horizontal = any(getattr(t, "orientation", None) == "h" for t in fig.data)
    grade = dict(showgrid=True, gridcolor="rgba(140,150,160,.16)", zerolinecolor="rgba(140,150,160,.22)")
    sem_grade = dict(showgrid=False, zeroline=False)
    fig.update_xaxes(**(grade if horizontal else sem_grade), title_font_size=10, tickfont_size=9,
                     tickfont=dict(family="Inter, sans-serif", color=COR_TEXTO_GRAFICO, size=9))
    fig.update_yaxes(**(sem_grade if horizontal else grade), title_font_size=10, tickfont_size=9,
                     tickfont=dict(family="Inter, sans-serif", color=COR_TEXTO_GRAFICO, size=9))
    return fig


def exibir_grafico_padronizado(fig, *args, **kwargs):
    """Aplica a identidade visual Intech a qualquer gráfico antes de exibi-lo."""
    legenda_existente = getattr(fig.layout, "showlegend", None)
    if legenda_existente is None:
        legenda_existente = len(fig.data) > 1 and any(
            getattr(trace, "name", None) for trace in fig.data
        )
    padronizar_grafico(fig, legenda=bool(legenda_existente))
    return st.plotly_chart(fig, *args, **kwargs)


def formatar_eixos(fig, eixo_y=None):
    """Mantida por compatibilidade: aplica o mesmo layout do dashboard."""
    tem_legenda = len(fig.data) > 1 and any(getattr(t, "name", None) for t in fig.data)
    padronizar_grafico(fig, margem=dict(l=8, r=16, t=40, b=24), legenda=tem_legenda)
    fig.update_layout(hovermode="x unified")
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
ABA_PROJECOES = "Projeções mensais"
ABAS = [ABA_VENDEDOR, ABA_CLIENTE, ABA_ITENS, ABA_PRECO, ABA_PROJECOES, ABA_DADOS]


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


# Identidade visual Intech Machine: preto, vermelho e tipografia moderna.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --intech-red: #ed1c24;
  --intech-red-bright: #ff343c;
  --intech-red-deep: #a90d16;
  --intech-bg: #050607;
  --intech-panel: #0b0e10;
  --intech-panel-2: #101417;
  --intech-line: rgba(160, 170, 180, .18);
  --intech-text: #f4f6f8;
  --intech-muted: #aab2ba;
}

/* Base: visual mais limpo, contraste alto e mais área útil para os gráficos */
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
  font-family: 'Inter', sans-serif;
}
.stApp {
  color: var(--intech-text);
  background:
    radial-gradient(ellipse at 78% 0%, rgba(115, 7, 16, .19), transparent 33%),
    radial-gradient(ellipse at 5% 92%, rgba(28, 35, 42, .18), transparent 34%),
    var(--intech-bg);
}
[data-testid="stHeader"] { background: rgba(5, 6, 7, .88); }
[data-testid="stToolbar"] { right: 1rem; }
.block-container {
  max-width: 100% !important;
  padding: .75rem 1rem 1.25rem !important;
}
[data-testid="stMainBlockContainer"] { gap: .85rem; }
[data-testid="stHorizontalBlock"] { gap: 14px; }
[data-testid="stVerticalBlock"] { gap: .75rem; }

/* Tipografia */
h1, h2, h3, h4 {
  color: #fff !important;
  font-weight: 700 !important;
  letter-spacing: -.035em;
}
h1 { font-size: clamp(1.55rem, 2.2vw, 2.05rem) !important; }
h2 { font-size: 1.35rem !important; }
h3 { font-size: 1.05rem !important; }
p, label, [data-testid="stMarkdownContainer"] { color: #e7e9eb; }
[data-testid="stCaptionContainer"], small { color: var(--intech-muted) !important; }
hr { border-color: var(--intech-line) !important; margin: .65rem 0 !important; }

/* Cabeçalho da página */
.intech-hero {
  position: relative;
  padding: 12px 0 17px;
  margin: 0 0 14px;
  border-bottom: 1px solid var(--intech-line);
}
.intech-hero::after {
  content: "";
  position: absolute;
  left: 0; bottom: -1px;
  width: 150px; height: 2px;
  background: linear-gradient(90deg, var(--intech-red), transparent);
}

/* Barra lateral */
[data-testid="stSidebar"] {
  min-width: 280px;
  max-width: 280px;
  background: linear-gradient(180deg, #0c0f11 0%, #070809 72%, #050607 100%) !important;
  border-right: 1px solid var(--intech-line);
}
[data-testid="stSidebarContent"] { padding: 1rem .82rem 1.2rem !important; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .62rem; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  padding-bottom: .42rem;
  border-bottom: 1px solid rgba(237, 28, 36, .26);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { font-size: .9rem; }

/* Indicadores principais: cartões elevados com acento vermelho discreto */
[data-testid="stMetric"] {
  position: relative;
  isolation: isolate;
  overflow: hidden;
  min-height: 116px;
  padding: 17px 19px !important;
  background: linear-gradient(135deg, #14181b 0%, #0a0c0e 70%, #1d090c 100%) !important;
  border: 1px solid var(--intech-line) !important;
  border-radius: 14px !important;
  box-shadow: 0 8px 24px rgba(0,0,0,.20), inset 0 1px rgba(255,255,255,.035);
  transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
}
[data-testid="stMetric"]::before {
  content: "";
  position: absolute;
  z-index: -1;
  left: 0; top: 17px; bottom: 17px;
  width: 3px;
  border-radius: 0 4px 4px 0;
  background: linear-gradient(180deg, var(--intech-red-bright), rgba(237,28,36,.18));
}
[data-testid="stMetric"]::after {
  content: "";
  position: absolute;
  z-index: -1;
  right: -27px; bottom: -40px;
  width: 116px; height: 94px;
  border-radius: 50%;
  background: rgba(237,28,36,.12);
  filter: blur(25px);
  pointer-events: none;
}
[data-testid="stMetric"]:hover {
  transform: translateY(-2px);
  border-color: rgba(237,28,36,.48) !important;
  box-shadow: 0 12px 28px rgba(0,0,0,.27), 0 0 0 1px rgba(237,28,36,.05);
}
[data-testid="stMetricLabel"] { color: #c5cbd1 !important; font-size: .84rem !important; font-weight: 500 !important; }
[data-testid="stMetricValue"] { color: #fff !important; font-size: clamp(1.3rem, 1.8vw, 1.9rem) !important; font-weight: 750 !important; letter-spacing: -.045em; }
[data-testid="stMetricDelta"] { font-size: .78rem !important; font-weight: 650 !important; }

/* Containers, gráficos e tabelas */
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stPlotlyChart"],
[data-testid="stDataFrame"],
[data-testid="stTable"] {
  border: 1px solid var(--intech-line) !important;
  border-radius: 14px !important;
  background: linear-gradient(145deg, rgba(15,19,22,.95), rgba(7,9,10,.98)) !important;
  box-shadow: 0 7px 22px rgba(0,0,0,.15), inset 0 1px rgba(255,255,255,.025);
}
[data-testid="stPlotlyChart"] { padding: 7px 7px 2px; overflow: hidden; }
.js-plotly-plot, .plot-container { border-radius: 10px; }
.js-plotly-plot .plotly .modebar { opacity: .35; transition: opacity .18s ease; }
.js-plotly-plot:hover .plotly .modebar { opacity: .95; }

/* Campos, menus e upload */
.stTextInput input,
.stTextArea textarea,
.stDateInput input,
[data-baseweb="input"] input,
[data-baseweb="select"] > div {
  color: #f5f7f8 !important;
  background: #0e1214 !important;
  border: 1px solid #30373d !important;
  border-radius: 9px !important;
  transition: border-color .18s ease, box-shadow .18s ease;
}
.stTextInput input:focus,
.stTextArea textarea:focus,
.stDateInput input:focus,
[data-baseweb="input"] input:focus {
  border-color: rgba(237,28,36,.86) !important;
  box-shadow: 0 0 0 3px rgba(237,28,36,.12) !important;
}
[data-baseweb="popover"], [data-baseweb="menu"] { background: #101417 !important; }
[data-baseweb="menu"] li { color: #e9ecef !important; }
[data-testid="stFileUploader"] section {
  background: linear-gradient(145deg, #111518, #090b0d) !important;
  border: 1px dashed rgba(180,190,200,.38) !important;
  border-radius: 12px !important;
  transition: border-color .18s ease, background .18s ease;
}
[data-testid="stFileUploader"] section:hover { border-color: var(--intech-red) !important; background: #111214 !important; }

/* Botões */
.stButton > button,
[data-testid="stDownloadButton"] button,
[data-testid="stFormSubmitButton"] button {
  color: #fff !important;
  background: linear-gradient(135deg, #f3222b 0%, #b70e18 100%) !important;
  border: 1px solid rgba(255,91,98,.58) !important;
  border-radius: 9px !important;
  font-weight: 650 !important;
  box-shadow: 0 4px 14px rgba(237,28,36,.13);
  transition: transform .16s ease, filter .16s ease, box-shadow .16s ease;
}
.stButton > button:hover,
[data-testid="stDownloadButton"] button:hover,
[data-testid="stFormSubmitButton"] button:hover {
  filter: brightness(1.1);
  transform: translateY(-1px);
  box-shadow: 0 7px 18px rgba(237,28,36,.22);
}
.stButton > button:active { transform: translateY(0); }


/* ===== NAVEGAÇÃO PRINCIPAL ===== */
div.st-key-aba_ativa [role="radiogroup"] {
    display: flex !important;
    flex-wrap: nowrap !important;
    align-items: stretch !important;
    gap: 5px !important;
    padding: 6px !important;
    background: linear-gradient(180deg, #090c0e, #07090a) !important;
    border: 1px solid rgba(160,170,180,.20) !important;
    border-radius: 12px !important;
    box-shadow:
        inset 0 1px rgba(255,255,255,.025),
        0 6px 18px rgba(0,0,0,.18) !important;
}

/* ===== BOTÕES ===== */
div.st-key-aba_ativa [role="radiogroup"] > label {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 11px 10px !important;
    margin: 0 !important;
    background: #111417 !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: background .2s ease,
                border-color .2s ease,
                box-shadow .2s ease !important;
}

/* Textos */
div.st-key-aba_ativa [role="radiogroup"] > label p {
    color: #d9dde2 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    line-height: 1.3 !important;
    margin: 0 !important;
    white-space: nowrap !important;
}

/* Hover */
div.st-key-aba_ativa [role="radiogroup"] > label:hover {
    background: #202326 !important;
    border-color: rgba(229,9,25,.65) !important;
}

/* Aba selecionada */
div.st-key-aba_ativa [role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(135deg, #ed1728, #b80716) !important;
    border-color: #f52b3b !important;
    box-shadow: 0 3px 10px rgba(229,9,25,.25) !important;
}

/* Texto da aba selecionada */
div.st-key-aba_ativa [role="radiogroup"] > label:has(input:checked) p {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Indicador de seleção */
div.st-key-aba_ativa [role="radiogroup"] input {
    accent-color: #ffffff !important;
}

/* Remove espaçamento interno excessivo */
div.st-key-aba_ativa [role="radiogroup"] > label > div {
    gap: 6px !important;
}

/* Telas menores */
@media (max-width: 1050px) {
    div.st-key-aba_ativa [role="radiogroup"] {
        flex-wrap: wrap !important;
    }

    div.st-key-aba_ativa [role="radiogroup"] > label {
        flex: 1 1 30% !important;
        padding: 10px 8px !important;
    }

    div.st-key-aba_ativa [role="radiogroup"] > label p {
        font-size: 12px !important;
        white-space: normal !important;
        text-align: center !important;
    }
}
div.st-key-aba_ativa [role="radiogroup"] label {
  flex: 1;
  justify-content: center;
  min-height: 44px;
  padding: 10px 9px;
  color: #c6cbd0 !important;
  background: transparent !important;
  border: 0 !important;
  border-radius: 8px !important;
  transition: color .18s ease, background .18s ease, box-shadow .18s ease;
}
div.st-key-aba_ativa [role="radiogroup"] label:has(input:checked) {
  color: #fff !important;
  background: linear-gradient(135deg, #ed1c24, #b5121b) !important;
  box-shadow: 0 3px 12px rgba(237,28,36,.23);
}
div.st-key-aba_ativa [role="radiogroup"] label p { color: inherit !important; font-weight: 650; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--intech-line); }
[data-testid="stTabs"] button[role="tab"] {
  padding: .68rem .95rem;
  color: #c5cbd0;
  background: rgba(14,17,19,.8);
  border-radius: 9px 9px 0 0;
  transition: background .18s ease;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  color: #fff !important;
  background: linear-gradient(180deg, rgba(237,28,36,.18), rgba(237,28,36,.035)) !important;
  border-bottom: 2px solid var(--intech-red) !important;
}

/* Expansores, avisos e conversação da IA */
[data-testid="stExpander"] {
  border: 1px solid var(--intech-line) !important;
  border-radius: 11px !important;
  background: rgba(12,15,17,.76) !important;
}
[data-testid="stAlert"] {
  border: 1px solid var(--intech-line) !important;
  border-radius: 11px !important;
  background: rgba(15,18,20,.92) !important;
}
[data-testid="stChatMessage"] {
  background: linear-gradient(145deg, #101416, #090b0c) !important;
  border: 1px solid var(--intech-line) !important;
  border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
  color: #fff !important;
  background: #0d1113 !important;
  border: 1px solid #343b41 !important;
  border-radius: 10px !important;
}
:focus-visible { outline: 2px solid rgba(237,28,36,.84) !important; outline-offset: 2px; }

@media (max-width: 1100px) {
  [data-testid="stSidebar"] { min-width: 245px; max-width: 245px; }
  .block-container { padding-left: .8rem !important; padding-right: .8rem !important; }
  [data-testid="stMetric"] { padding: 14px !important; }
  [data-testid="stMetricValue"] { font-size: 1.35rem !important; }
}
@media (max-width: 760px) {
  [data-testid="stSidebar"] { min-width: unset; max-width: unset; }
  .block-container { padding: .6rem .6rem 1rem !important; }
  [data-testid="stHorizontalBlock"] { gap: 8px; }
  div.st-key-aba_ativa [role="radiogroup"] { flex-wrap: wrap; }
  div.st-key-aba_ativa [role="radiogroup"] label { min-width: 46%; }
  h1 { font-size: 1.55rem !important; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition: none !important; animation: none !important; }
}


/* ---- Camada de layout inspirada no mockup oficial Intech ---- */
.block-container {
  padding-top: 126px !important;
  padding-bottom: 54px !important;
  padding-left: 1.2rem !important;
  padding-right: 1.2rem !important;
}
[data-testid="stSidebarContent"] { padding-top: 124px !important; }
[data-testid="stHeader"] { height: 0 !important; background: transparent !important; }
[data-testid="stToolbar"] { top: .25rem; }
[data-testid="stMainBlockContainer"] { gap: .55rem !important; }
[data-testid="stHorizontalBlock"] { gap: 12px !important; }

.intech-topbar {
  position: fixed; z-index: 99999; top: 0; left: 0; right: 0;
  height: 106px; width: 100vw; box-sizing: border-box;
  display: grid; grid-template-columns: 250px minmax(360px,1fr) 240px;
  align-items: center; gap: 20px;
  padding: 13px 34px 12px 52px;
  background:
    linear-gradient(112deg, transparent 0 18%, rgba(237,28,36,.035) 18.1%, transparent 18.7%),
    radial-gradient(ellipse at 68% 0%, rgba(115,7,16,.16), transparent 38%),
    linear-gradient(115deg,#080a0b 0%,#030405 52%,#08090a 100%);
  border-bottom: 1px solid rgba(160,170,180,.19);
  box-shadow: 0 8px 24px rgba(0,0,0,.22);
}
.intech-topbar::before, .intech-topbar::after {
  content: ""; position: absolute; top: 0; bottom: 0; width: 2px;
  background: linear-gradient(180deg,#ed1c24,rgba(237,28,36,.08));
  transform: skewX(-28deg); pointer-events: none;
}
.intech-topbar::before { left: 278px; }
.intech-topbar::after { right: 28%; opacity: .7; }
.intech-brand { width: 160px; height: 75px; display:flex; align-items:center; justify-content:center; }
.intech-brand img { width: 160px; max-height: 75px; object-fit: contain; display:block; }
.intech-heading { min-width: 0; position:relative; z-index:1; }
.intech-heading-title { font-size: clamp(1.25rem,2vw,2rem); line-height:1.14; font-weight:800; letter-spacing:-.045em; color:#f7f7f8; }
.intech-heading-title span { color:#ed1c24; }
.intech-heading-subtitle { margin-top:4px; font-size:.89rem; line-height:1.35; color:#e1e3e5; }
.intech-topbar-aside { justify-self:end; display:flex; align-items:center; gap:17px; position:relative; z-index:1; color:#f3f3f3; font-size:.86rem; line-height:1.35; }
.intech-bars { display:flex; align-items:end; gap:5px; height:32px; }
.intech-bars i { display:block; width:7px; border-radius:1px 1px 0 0; background:#ed1c24; }
.intech-bars i:nth-child(1) { height:14px; }
.intech-bars i:nth-child(2) { height:23px; }
.intech-bars i:nth-child(3) { height:31px; }
.intech-footer {
  position:fixed; z-index:800; bottom:0; left:0; right:0; height:42px; box-sizing:border-box;
  display:flex; align-items:center; justify-content:space-between; gap:16px;
  padding:0 24px; background:linear-gradient(90deg,#07090a,#030405);
  border-top:1px solid rgba(160,170,180,.18); color:#c7cbd0; font-size:.72rem;
}
.intech-footer-brand { display:flex; align-items:center; gap:10px; color:#f4f5f6; }
.intech-footer-brand b { color:#ed1c24; font-size:1.1rem; letter-spacing:-.1em; }
.intech-footer-muted { color:#9ea5ac; }

/* Estrutura dos cartões do dashboard */
[data-testid="stVerticalBlockBorderWrapper"] {
  border:1px solid #292f35 !important; border-radius:13px !important;
  background:linear-gradient(145deg,rgba(11,14,16,.97),rgba(4,6,7,.98)) !important;
  box-shadow:0 5px 18px rgba(0,0,0,.16), inset 0 1px rgba(255,255,255,.018) !important;
}
[data-testid="stMetric"] {
  min-height:120px !important; padding:17px 18px !important;
  background:linear-gradient(135deg,#101315 0%,#080a0b 74%,#260c0f 100%) !important;
  border:1px solid #292e33 !important; border-radius:13px !important;
  box-shadow:none !important;
}
[data-testid="stMetric"]::before { display:none !important; }
[data-testid="stMetric"]::after { right:-26px; bottom:-42px; background:rgba(237,28,36,.13); }
[data-testid="stMetricLabel"] { font-size:.81rem !important; color:#dedfe1 !important; }
[data-testid="stMetricValue"] { font-size:clamp(1.25rem,1.65vw,1.8rem) !important; letter-spacing:-.035em !important; }
[data-testid="stMetricDelta"] { font-size:.75rem !important; }

/* Seletor de páginas com aparência de menu de navegação */
div.st-key-aba_ativa [role="radiogroup"] {
  flex-wrap:nowrap !important; gap:0 !important; padding:0 !important;
  overflow:hidden; border-radius:11px !important; background:#070a0c !important;
  border:1px solid #22282d !important; box-shadow:none !important;
}
div.st-key-aba_ativa [role="radiogroup"] label {
  min-width:0 !important; min-height:59px !important; flex:1 1 0 !important;
  border-right:1px solid rgba(255,255,255,.035) !important;
  border-radius:0 !important; padding:10px 8px !important;
  color:#e4e6e8 !important; background:transparent !important;
}
div.st-key-aba_ativa [role="radiogroup"] label:has(input:checked) {
  background:linear-gradient(135deg,#f21b23,#be0c15) !important;
  box-shadow:none !important; color:white !important;
}
div.st-key-aba_ativa [role="radiogroup"] label p { font-size:.82rem !important; white-space:normal !important; text-align:center; }
div.st-key-aba_ativa [role="radiogroup"] label > div:first-child { display:none; }

/* Filtros da lateral e visualizações Plotly */
[data-testid="stSidebar"] { min-width:280px !important; max-width:280px !important; }
[data-testid="stSidebarContent"] { padding-left: .8rem !important; padding-right: .8rem !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { border-bottom:0 !important; padding-bottom:.1rem !important; }
[data-testid="stSidebar"] [data-testid="stHeading"] h2 { font-size:.95rem !important; }
[data-testid="stSidebar"] label { font-size:.78rem !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div { min-height:34px !important; }
[data-testid="stPlotlyChart"] { border:0 !important; border-radius:8px !important; background:transparent !important; box-shadow:none !important; padding:0 !important; }
.js-plotly-plot, .plot-container { background:transparent !important; }
.dash-section-title { margin:0 0 2px; font-size:1.02rem; font-weight:650; color:#f6f7f8; letter-spacing:-.02em; }
.dash-section-title .dash-red { color:#ed1c24; margin-right:8px; }
.dash-chart-caption { font-size:.77rem; color:#bfc4c9; margin-bottom:3px; }

@media (max-width: 1050px) {
  .intech-topbar { grid-template-columns:180px minmax(220px,1fr); padding-left:20px; padding-right:20px; }
  .intech-brand, .intech-brand img { width:145px; }
  .intech-topbar-aside { display:none; }
  .intech-topbar::before { left:195px; }
  [data-testid="stSidebar"] { min-width:245px !important; max-width:245px !important; }
}
@media (max-width: 760px) {
  .intech-topbar { height:86px; grid-template-columns:104px 1fr; gap:10px; padding:8px 12px; }
  .intech-brand, .intech-brand img { width:100px; height:55px; }
  .intech-topbar::before, .intech-topbar::after { display:none; }
  .intech-heading-title { font-size:1.12rem; }
  .intech-heading-subtitle { font-size:.68rem; }
  .block-container { padding:98px .55rem 52px !important; }
  [data-testid="stSidebarContent"] { padding-top:96px !important; }
  [data-testid="stSidebar"] { min-width:unset !important; max-width:unset !important; }
  div.st-key-aba_ativa [role="radiogroup"] { flex-wrap:wrap !important; }
  div.st-key-aba_ativa [role="radiogroup"] label { flex:1 1 45% !important; min-height:42px !important; }
  .intech-footer { height:38px; padding:0 10px; font-size:.62rem; }
  .intech-footer-muted { display:none; }
}


.ai-card-title { font-size:1.02rem; font-weight:700; letter-spacing:-.025em; color:#f4f5f6; margin:0 0 .25rem; white-space:nowrap; }
.ai-examples { margin-top:.1rem; color:#cbd0d5; font-size:.66rem; line-height:1.35; }
.ai-examples b { color:#f0f1f2; font-weight:600; }
[data-testid="stVerticalBlockBorderWrapper"][style*="height: 112px"] {
  height:112px !important; min-height:112px !important; max-height:112px !important; overflow-y:auto !important;
}
[data-testid="stForm"] { border:0 !important; padding:0 !important; background:transparent !important; }
[data-testid="stFormSubmitButton"] button { min-height:36px !important; padding:.25rem .45rem !important; border-radius:8px !important; }
.ai-examples + div { margin-top:0 !important; }


/* Menu lateral fixo: impede recolher/expandir pelo botão do Streamlit */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarHeader"] button,
button[kind="header"] {
  display: none !important;
  visibility: hidden !important;
  pointer-events: none !important;
}
[data-testid="stSidebar"] {
  display: block !important;
  min-width: 280px !important;
  max-width: 280px !important;
  width: 280px !important;
  transform: none !important;
  visibility: visible !important;
}
[data-testid="stSidebar"][aria-expanded="false"] {
  margin-left: 0 !important;
  width: 280px !important;
  min-width: 280px !important;
  transform: none !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <header class="intech-topbar">
      <div class="intech-brand">
        <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABoCAYAAAC0RL9wAACKSUlEQVR42uz9eXSl533fCX6e5d3uBlwAVSigVoJ7FSmKKlGyRCqSZdixlaKcZFKy0+li0tMJFU9OKHefOYfKme6WND3TodLptMVJnBYz3TkSO0ur4vbCcryovMiirLUoSiKLpEiCZC1ALQAugLu827PMH+/FViyi6Mh2pufMrXNPvbjve5/3ue/z/LbvbxPee37Ulxt0H5G15uPrf5vUHtOJOpVm9lgSq1M7fdeTHtvpvCA59aPMLTern9oYS4hsy3EKEMjNef/HeJWuf+Kt5gbgjJ2+3ufr1ytZO7nj2thsdv1Yqvg0///X5t4os9nlIPilOurJFnx267nUc8wJ2vpP40brxJGX2Ww/z47rKJwLCTMbmGlQ2641NpsFH4NrA2hVf/LP8iFEeuQzO53P8nJjA8VRcPqtzl/v3J/GK5A7/371I67Qj0oUeb7JwNaJMgw3x7RmkwCr+W6/n80qBvBWTEqEPxoDtMV2BrOViQAIPRzfZrPe+9h7n3jvYwAN+LVBQlKDIL6GcbsED+JPQ4IALKeDz1tRTu2KRz7qSY8JXBtjp+mZmS38rI3w8cYxgI3P7jiwFzE7i5jsBufTLauTbTmuPhfh9u9vfD681tqpa85n267LBrNveQ5AhQtvOXbFMtjx/m74nK4dd/18WR7ecX5eJNf93vo1Si685djVDpx6y98GkPce3PH7fodzQqRskZDXfz4+3vH5uuH+ut7aAhgzg/cJzrXxPkbKDlrPofUcSi0QbUrgtd7qp1qN7Qz1T41AuungkVog5hR2mnNvfH71xZcJ0pTe0mCrwH8zBzDyR7qvUmpn+try+4QQbzrOjL3u5+vHpTVvOrf1mmALi7/e+SwvdzxfOvuWc6v2NzvOvyh3Ht/4nb9v8Dve33r3lmNXXHjn+eemfMtzwMbzfcvfj9zx9zlxLX+5Zpxr/pZSopRCKYWUkvHJKWp7p2C08cVBIDKfRE/X9aZU/5EJZLE3+NJEo/YxBoPj/tVXvrTy7A84/0dfZeW5F2mUhk6aX5etb6gAvvjRKFPvTCBa650XOIx2XKDymg3ypgW8DgFuuzbYeXyl9Y4L7KXY8bwROzMAi9jx/k6KHefvldyZgILGdee3QcA3IAAZqR3n70Ww4/juLZ7Pxv1NuZ0hW7vxds4xQDIxc4gD993D1H33PMXoro8CDAa9h2u1xhM/sg0y0ah9jOUrX+5969nZc7/ze/TOfJ/g4jwT3R6xKRlrDx+gl7DODbzcIJQoKG8EAex41tidCTyw0Y4L4Ipixwdsldjx+0mttuMGs95uqBobn285lv4aDun9tuM3c1C/ZXwPQwn6llw6DHcm0OQ6z2fr94Ng+9/XMJhcNnZkQD4Odnx+OlTXH3v9WO58f7YymOtdF0cgZfWuVFYwpnpby2uXF7nw+nleXF1GO/fgrgd+7FHqrc/KoSnwo6tY3UvfWfitPzj6yq/9Nu57LzKx2mcy1DQihbU5S/nqkCBERSBbCQVB6HcmAITZ8XQhbjB/v7MKFCXhjgssbrDAyJ03oE6SnTkkekcVhEjv/P1w5/MqvD6DWN9ARu/MAPxbSLiNa+PWdcddP3Z6ZwnorlnfN0kIFe0s4a6V0Nc8Pyeqc0LK6py8RqVv7yG9eJHn33iNotXktg9/gIkfO/o5Rtu/2Cc78SMTiPnqb/vf/+KXWP3Db3EXEfuEJht0SEnJtSOKxRaikG8iFp/diEDszvs/uJENInfcgMa5PzGH3a5ChTeQQMGO39dBuDOBhDuriD7aWUVzKtyRAESyM4Gba2y8a+ehBTt+XzTiHee/DnK+pQqrd2YwhDszMKclzjmccxvq8LodIqUkXehSn7kZt7LG13/4ErVbDnHvX/+rcGC/WDRrX3r7BLKaf4qR6DMZbrabdh7ZFcQn3dlXvvj0f/Z3iY2hXpbUyoLIFkPFyGOFpBQe7SC0EFiHGt7PSocXYES0M33cSECIP2OwXO4MIqhWY+cNIuXORvJQx966gbYel4ne8Xyo9AaX9ABSDDnz8Fq3rstXn3sBXm4ZQ27R8aXYdl31v9p232s3unQe5OY91u+//n0tg2vGExvXXzs+UmwfC/BOb34mt4y/TsDaX/f5bBCwVzvOz6SrRI0WhDUWljr82je/zi/8w/8WDu17IlMVFLzjK4PZGE6j1ZxJy2NxEpxyQXCUtDg6993vE1lD6CzKlSjnAIlwHoSqjvEV1EdFEN4DwoFXN979/wd4lddBkbapOHpnFcjJ7RvQe7/tGOu2qw5bkRkhMLbcWOx1omDLJlRCb2x+1jeW3xzD+S0okxeb14k33+tNf69biM5v2dTXbEBnNu7vthDCxnxx265fJ9KtDGaDyNky/vpY6yjjFqLYOk8n5Y7zc5HGlSXSFTRkwHhQI1teJd43NaWS+PQNCcTip0BAXT9ZpvmxkvxYTQXPc+HiYy//4dfZX+Zoa4msRXlXEQeyIgAk0nukrwjCI7HDh+KF4//45AEmL3bW4W+Ecqnrb/z1Y1+6bRvAX3Nsnd9GIBvnhgQiVbxNglQbcKsRrDY22zbpsiGR/Lb7br23EGIbTOyl2BTp6+iS2/zby+3fdQKkCLdfv4VAhBAovYmief/m8cX65t9KJFvnJ4sd5+dqimJQEmeCpoiYDOr0Li8Re1BBcPZtoFhbIEFNW1ib4XzsX79A/+yrJKVBekfoDNJVNO6ReC8QXqFwCCQOB0PikBv2yFCa/MdUoW7wupEKKozf0QawlDvDnFs2wPXUKI/bkYA2jdHtBLDxfZlvboYt6tM6PCqHEsZv+d7W6wR6RyniFRuS4VoCq1RkuW1D+i2EUs2pGEqCoQrlNu/NkCg2fB3XITK1FfW7hjg84HWw4/xKJwithawEHdKUIbaXglILkrchQQSbnmoZBB1nymmy8nB67hLtXj60K1ylOwmPd+DxVALVbWhRvnoGrCOZb3ffuxuZCP+RJYgUO6NAtrRbHMriTcfO5dsW/FoO6LzY0QbRQxVtqwojtqgwzrgt+v4WNWydYwu/bQNyzXVe2B1tEKuvJZ7hxh6Or4c2jndD1coLnNvcwBsmmhTghkQixMZztWLL/LfMEVn97YYgjL9mjutsyzq34/zMAKJSgpVgcry16CiEKDyTkc3ekEBqcHIrsUgF9Luf6J2/wG40wkmc8AhRiXspAG8QQ9VKoPHCYQEvfEXVoiKqinjk/1erUDeSIFJsMcKv45TbipKt2xfbxrV+Zw5tua4Nsskh/XbC2cKpxfrGfAsj3QFe+jdz+K33kupNKtbW+QizXadf37xi3XOtZEUIW4hObTHGBdfYT1vvLwXClRvjy/WNLcQmgxVqm8TY+nyq+akd56dKiyIEJ8lsxorNqe1qQ6DmDX7mbTkK16NzLX5KCB+TZ6zMLxDnBis0CPBYhALlHNL7IQFUE7KVsohVw83iHcKBFH6L0/A/UMX6j2zIrOvA2yH4TUJYl6DiTQrrumfXvCUCI4TArztCr7FB1m9mjd2uY29FfACF3CCGrQjOupTa9NluSp1tSJHwb0mcQgiU2EKQw83ntvkl/Mb9hNxiW60TiNTbbBJxDdK0Mf5boHAVc95itF8jgddB6reaX+wMTjhyJehKybLPSSbHQYJEdm5MIKk/JlmP27EzIWIBY+mtdKhnGXYYKuCFQXmLwyJFZXesL5HY2OgOJzxqKEX+f8FK987t6MkVN1DBKgLzG17y7RzQ44Xf0QYx1m5Db7zcboRKFWyoFsJX3HkToaqk/YbkWNfzvXhL1Ozavzfh3k3urLYQgHPuTQjb1jluQDVyE1zYqoKKQG+qsnJIZM5vShllN7fR9ebHzvMLjSe3BUWsyaOAvrDQqm8oNjcmkESckghMlh6TWnYS3XjCrOafX1t4nZvkIiteInw1nvBB9R6CFxKYD0ti4wjyEoGBSOLDAOkFovAoUaCspbfWpblrjDXvUWECA0HsQhDZcOOYLRaJHHrIBe4GjsQbbnAByoNyFbP0AgpVvZ2A+g0iYUK3ddP7DctpfYHTxCL9uokmEVZVytjwezKQ1xGK1b9tCqj3GwzFbxGcN5KwhSp2dNS5ON7R004QbdzruqEiPqjWRdlq5lKAkAgCQFIbbdG3Aww5Lu0zHjYoeg7ZGGfFwLhI6VlwYQ28JvACpR0pfYrEErqEwEsCW7FaIRRGSMpAYAVE1u4Iclif72hDLWvHLhlBv0/napfJiTaMtb8IySnvOPEnisWSng5AMegjncd6v6MrwwlolIqajkjCBl44UpNTpCW5sQQOkmYLJS1x5JFOQpZiM4G2kjAMKf8cpIBbV7e3opL+7SFouXbXIRAQQ+NSVUZXxcHWF8Z5hhbp20cr/hRsqWvtoErF2zmaWRi7o4T0KLxwFSgjXSV93FBDAHqra+Sk6FASKY31ntU8x0YZV6xBjDfoGSiExjuBQpBEAaWMyUTGRGoqhlIZsQjhcErg/NCu3jrX6wEZw2DLt/LjeAvGlmgrcUKya2oS8Mn6xnjbBOKlTwQuI8tn+1eWkc7jtoXSXd92CHsGpz3d0EOkcTLB6witBUoq1ix4a3E6QHhFKEICoas4PPHnZ2MM13Vj3mIoVW6EopXK7OwodFusD1cB3iDXPWw3HP+GNtCNno/bHvx4rSHkjd32iLfZP4B3O/txlFA4aRBDJNPbilF4AXiNKksUBiUlWmnS0rFoSoJWnSwKeNYNIIqJaqM0RscYHx1lYnKUeizori3ROfUVAl8RiB86+ZwUWFURiGVnmFwEekcbT3qHKRzeaTLn2HfoJhCyM5TkyduXIEp2pPUxWT7bvbiAsha3BcZ9K2h2cvcUi96yjKEnPFLqijubEmstcbNJaVIikZC7kqbUBF4xMCnW5NcmJP6ZSpGtG0f5t0ebkk0cW2xRijb/WjfWhzq/GzpORaWaFn8OkuNaP83Wz3yxPRx9GyIEOHUNzHoNgVhVVFESyuHlUNqIoILzcZUbQBhIwWroO0dZr7PvPe9ifP80u99xB8gIVAxJApoOIbDWadtnzuCMxTrQTuDd0E2gBFiH0ALv1ZscqGyZq/duZxBEGryPyI1nrci4ee8+ULKTlsUxJYL5t00ggYpPezc4TloeWz1XEYjFvolArlVLXlq+Sj49RnfPHnpxgnQabT3WlmQmRagQlGEs0aycu8B43kdKga8F+FoCa4M/WxTqOqrU+m9SHuwNVCC9RewIsRmQKYbwr5ebKtdGGAngN5SDHw3mvtG3rfNvKT3YKiHeKiJ3CEK8pR/HDwnEOYbQZAUseIsQisEgrSLWjcCFngEStWecsb/wPnjHnSfTIptNdP2LEHYYqVfZfL5/AmG/eGmtR2iqBXLOIlwlnaT3SFXZZZJoRxtjK8x+vfNelugwYmAdvSInGB2FIHg+CcJTb89IX9e1yWal9zFZcWT14gJN67HebRvAXYdI/K4RbvmZWcYe/CmY3ju0fiUEYmjYaSCHtSVe/sK/onf66wyurlCWBZaSNvrPRYKsq1hyXb3ylR1i1Y1CcdRw08nK64wcGqmVMS5FifdV4ObQ84AVDrFuhvsfncD/Q/04Qog3wdDXHntndjTyrVcgLELYKnxIVoa0RzN05aGExhsLwlMIj6nVYO8UNOtfTJj42Nbxrtj0yyNKPBEOMuZfOcdB47HOEjiPdx4vLEL6iljkptr1VqEwFveWTtZ1T74VNQoDIkyg2QRdqVh/IgKJiE9b+ifIDf2rS4y4SoIE/vqSY51YruQZe3e34a7b5nyz/Qk78Il2qk0jegI/OE4pO5DfT2Q/faXIEGnGpNA0Ik2qxZtStv8sVSw/tJ/l0Eh/O4Z6fyMYUQ2JQoHXCKHAS2pYkFWMmhcSIcwWovB/5jaWXEfAuH5CllpXAbeQhbgOcvDWfpxyE2WUbuOBOUQFWgwfpM0NEgtRgKzVodY4swIPkPceGI0an6TnHqYhn0BpIoKTCI3v9CiKAu8cynqEs2g8Xjms8ljphm6at1ahnLr++Q1V0hWUIqU0AbXJ3dBspqjNQhJvW75n5LPCuzbGkPcGDFVTpN+ZKzfbo8TjbagnT62iHrhclr+UeWKADv5+wvh0hmv3i5KVXopykvG4RUMEmO7gz016+MqG3maky7exeQeBYBAoBloO/9fDt2IQKKysYtOckhvje1n5N/484sy22RvXHHvvkcOwjrd8Dz1aFZRf/b9+LKlsAZypxnYeb6vcC+GHKa1lSj9L6fV65HmOUgHNkVFoNT+XI++XfsitlZyvOLaaA6A7YPX8JUxRUpbV2xpDaXJsUWLLHFvk1f87vL0zO75dWZCnGaUxNEdaECenrUmPdW3/kQrGJpsVXL80zNZzKdGxLO08MPris+xafpVCONCjVc62cFtkhh5G8obgFb20QfPo+1lxPhmV4S+OhuHTzpQzAFrU5wBiWz/NQv8TvHiedgndvEcaOIrRBmE2DPbzEuHlhgokh6Equb6Rn2JnEaStphgyfisrm0P5KgJEWdhbCHp1xcV8FS01US5xImZu0GPk8K280h6hvXuSXfsO0tq7l/bePYxOjaNatQpgOHOWP/jcP6d18SqT1uPzgmgkZpClKCFviPLKG543GwS97o/aqnpdmxH4JhVR7AwTKKLrRi1smDTa4oXahMepBInyHuUMGo3Xmn69Rq8WsuBy/uK974Yuxyfj6AMM3TBFMTgaJrVTY8iPs3zpO5yfJ1rsEPW7CMAKT+ZdlTKCQeRDORXpjWeo1gl5KFUEoOzOCWvN0FNaQ5ecqZtvAik6a8bMBkF09k+kYknoRCp4ns7qUeVBXpMsL/32ChOearMFzTooub1e0XrRM6hKzhgzw9qAqLSE1hN6j7UQmy2hGp4txLH98x1h2BskPEm3sx/nct4jFZJSeqQX6DhBj4yze2Q/d//Vv8S7f+bDUG9CrXGSQM2j1AIarC+OFmU2mwz6bYfFKQ9C4ly1mawAiyO6UZiPvBGKK7etwZ+2mW+UvyFI8VZOSy9ABwGZrRbSOYfVoEZboOVcVjIbB5wGCHQ0t7k/ZMeurBE4V4EgogpNUqKKPFBCVrAyEspNP44bEonbQhDO2h1heEqLDR0Da0iaDdBqTkqJlJVke9sEIiCLZXC2d+EioQUtBVpIPG6DgqWvJrnOia2AeHIconCzYJgg85o52BIIWZZH+leXiIwn8B7lPCEeU26Hka+n9tyIw5obsGi/xUBfv9e6PeIBRmp4LK0gwacGU1qW0h6Xd9e4+x03Y/ZOnlZR9LRX0dcKBB4ZJwSnSrqPlBLc4uJxSlvp+lIgtcIJiRcSb90Nf4C78bpsggxik0mtb9boR7ThrNx5gNjIYfwW2K0q63DuYRhiuxk60Fg8LlQwOQ5azvmhqu2yYlbWwyer/ZrNBt7HywsLhKUlGG5wOUyXAo8SvrKUvMN5u40g7TX5JpQ3KGtkLKX29FxJfc9uiOKvBZJ2TeiTfyIC0TCHdUcXz12ssG1dRbLadfF+DeewouJ+9UN7IQw7Qqj59S2pYn06xR9LEKcGcLxm8pmlhQWks1jpMIHHCFvpyH5nR+SNJIh28gYbYEsAi9++Ma2EorSUJqOp21W0i4i4VDq6gYQ9EwgTnkHoOS8hjqINVTWm+XicmZmr5xdIckvdeBKlcF7ijCAwsvIOhzdQEa18WxTihoCCZJPY35aNc4NgUe3cDaIrKk/QcM8y/G8DEUQKvHXEScCqFNBMYGIEtJgTqkqlMMbNhJuwWRtbHF1+4xxxaVAbBCJArEPW674lhd+SUbhhQ4rNHBJxg7JNyhuy2JJGIWpqAuLwdG37vr8BBy4Hx3VQO1mDk6TZo72Fy8SuMsiErzzBWyXI+qJYWRHIyMx+qEVPS6E768BoddtKIOaY+wOTzl6ZPw9+SM3KYXEU0lIvg20EsdUv4d6GBNH2RqEib02ARkDsFc5Iak6iRYzXCWEgae6ahNFRVGv0k0N18RrrPT3OwM6k5xaoZYbEQoTDWInFkTiNd/aGoTTBDRwx6/SveLP0+NN4Be5GKmCl0th1qSGGKvDwfGFKNIJEhyA9zeldUAsgCU/FG2riluqJ3k6R5snKxQvUjIGho8/jt+QRbapUG3XLEJuw+TY/jt/Zj+McA1OgpsZhbPR6guEGIl5uKX25vHbUrKzRcILSOYT3FQPaahcM1RUjK5dH88A+iMIzgqGKFevTGW7WY5OhxpjJrKB7fp6GsQjpMMP8EefcBkF4sd1fsdXjvSODlTcKp98M99gqjeyQGzfjGt57nIXCOgyOMtZM7NkHBKS2PGaknMm8n7X46VjK06OoT1rTP17mqw92F6+irQXvMdZh8JTeo/Q6LGzeFgG81atYL+y2Rf3cemzlj5ZyZm9oo2yXwtfaQ3lZEgg5VL89e2YOQrTpZwDQWm/YH8KaGXpd+lcXmbAWI8phfYMKDaxUJ79xXG5hEBsO3q2qp3PbcnCuPdZWMCgKmvumoLHFDnrbBLLOudP8mO2sILIS7SvnTcUu5DYbYV20r6tZ0dQ4hOEZseVBW5gCByabHdXxJ+kVj/r5JVq5I/a+CnQTvorRktf4KcQW+2BdIO0ET4dvwxPOdgRo/V4WWF5bJcPSo1IpjQjIkwYHb7sFao2nExWcymDWCxILU5IqoNO48nCWDVgedIkDR1BUjrISi8ETBSFBQFXAbIdXN9x5g2aB3Xjugas2pnIV49j6m/5DX90weNsMRjK8d+UPRTmwsiSRCpsXuECy97aZN+06nQSnthgFM3S7lGtrxBYyVS2yowrs9MJXx7JaJCMr5EohUL5aT+HFBoE679/SCSoA5QWZLWnvmwItO2WRzQZbinO/fSPd+XbeGyCdR3tB7jzC+zfVRr3W8RY0G6CSU4LNKuoO2hI6G7s8dYSdPs3CUwNKDRKHsg6jt6tt62Lci7eH3PSCnfWNVvHWY7ihHUUSYXQIPmBgFVmkqY2N4Zxry9QeixN1KobTqbXHEqVOUQyOR/3siO32WbEpo9qjtK/URuGwQuBiiXGKuLdz7e1+cANHZVhtBuUrdTJwVKHhQwJJbpgNsPPz6YVvz0YRQ+JYNxHkNR77Mi8wtYCRyd14JVIBpIZjiebUNX6bxGQprijR3mGEBalww9DY9XwRLyqbpNRiqEWsBzSCcBXEW6n8O5dWVUiM9TTbozjh49KYGe/sbBir05Urgrcujy+IT8eyalmg3erDV3/wLUpKwqBO0S/pjQQE8Qj5YBVPQSQ9UnuSJGZhrcct97wLeejQg5XRutk+oIl8HGJKkx3LXfcR9cocDRkwCByFLYhKsEHC1RrsXaizfMsEL8sUqUJsFBMc3Esy2ubW8Smiw3t5/be/DN/+DlNpivSOZevIkhFIWlye2k9rYow9Nx1g8uB+mpMTVTlKX5WgPH/2Gea++h30i2+wd7DKqO9TupJ+2GDVRDS0x9xzLwu1UUzPM7F7nPPZEndPTNPMxjJG1anMmdkCdxQFCeoUYe3kQERzi7ma6b/3brLlkgk5RpmM0KfApStMN+vM/eEfcXMtpqZDrp4/z2irTb3d5qVLl1nVgmCkhckbJIemCe88hL5tH42D04xP7KYmI0RmodHC9wf0Fi5z9aVXWHn+LP1XX6W20qFpHdpYxsbGmF+aR483WS57uLKgbRSNTLCsq9ANhacmIywR57xnaXKc6I6bOXDzvcQjTRpTYzQmx9HNGgiHKUtsUbKwtEJUeC69NMfCK68S9zusfe87HCpTDgrJUhxRKwQmt/R9A/beg6jt3QuQkAJVZPmltPedPUnj3UHq49VXzlGvhVzqL5JIWMWxWm8g9h5g5Pbb2XPr7YxO7iFOEsqxFkhNd3GF7OU3ePnXTlE7d45xN2CiHiL6HRwVauiGBQuFG/rUPPTKAhe32HXzzcjRqbu2Gug2TY+9LQkiPFC6dtHtg6vwbB0GlIC35aYLf6jbWevwSqJH6jceuzQzveVlTJ6hZEX9UgoINEp5Lu5KuPUjH2Ti4Dhj45PIkTbs3QVJE4ggW2P1hZc4d/Ys/aKP0xIxtZeZ9z7AgXe+G3/4VkQUQj2epx6fJgrODq2zNtZO7b9n70OT77ibpdN/zLmnfh3X66OtpxHVUEmTl2PJB37+Zzl89D5wIYy2vjjTu/JQ++AdG2wolvq0xUx53EYZwKTVeKI9ueuxn/8f/7sOK7ZNHlQCW/sOa0ttXn+NC1fOkT3/fbJel2Z7DHTEfH/AYKRBML2bvUfu5MBH/3KF+kyOwWjjNEnyFEJ2yF2btf5DtBtPCKnmm/Yumun7H2Rp8WH/wgu8+Ed/wPe/8wwzr69x+co8dSXwa32atQAjq7gmGQaMoygbIQumz+s4dt9+iCP330/96L1wYD80xjsE6nkicYZQZEiRemdmhHNt5ez0ITX6OYry6NT73/Mw3bWElSXyL/8ubzz1FP1OB2Uq/5FXgqjZ3DCgM8xsPBQzqc2P1XQFq2Jof/ulV3kj7XP3wf2sTo0zdcdt3PvuH0PfcQeMtk9X7Q1E5gQkxk4RRl9LiuIo77z78IGbD0xf+c3f5oe//Zv4sqCFvC5osa6JWC0hUMS1N+9Vod6miqU8kGZHelcWUdZgnCWKQjKX40qDEgKpFMI5vIWytNgoJp7es61hynWC5TJd5A8sz1/EFxlhLLFWgJIoGSDwzB+c4N6//GEaM9OQtD6HCs4SqnmDyPxS7+FgdM/zl5X89EtacOu77+Xen/wLjN17L+zel/q48UQ38QlSpFYJvJKJQE0FqLMSMSeh4xrB2drkns5Uq/n51155nsHLPyRZXMZbicsNK7tHCe64E266TaRZ95EkUgvh+NRcl/IR5dVCTciTAMr4BSPcTKaK2ZjwtGi0PtvUei6l93DQCp53Pjq7auyxZDQ51cjiqfzVlz/9xuUrTGDoD/oEozXm+wPW6jV23/8exn7sXnb/2FGyO+74tAr0HFJ2HCKL1iV+AiTNx7uYR4AZNPjQzcWNXZ8L97YeuP3uA0d3z8+y+I//Nee+/V0OCIVdW6WtW5RIjDfYUBMPJAuDnKV9o4x/6P3c/BdnCW87DLsmP4ba3r2qT37CYGck6oxEdCQi08QnS1G2g/buGmVrliv6y4ORhFdWltmTprSjChkqlaK5Z6Iqhg1YxBQyeRLAl8VMK25WHZ50OGcmdh3Z81M/zZ333kXzgb8ASe0MjebnqNefTOGYhWkBqYCsNuhOUWs+nprB8UTXTtJuPDyWr36++8w3kKur1EVV6lZ6huWmttvWPozRSYOgPYax6TG9NQ4rTE69PRvE+ZjVAemlJZrWYLwhDhN8niGLAkKBU1VJF+kExnpcKya+af8Nhw6L7OjqpQtoW6LlsNuElzjnMc7RmxiBmw7Ml6Pxae1lxyLSRcynA5Ln0TqtZf0Hx9/zLn7qjluZufUQzOw/Q735ub6MyK1+oGmzkx4I36LT0qvZypf3xuM/Gd9+x6cPfOADn740v4Ba6dEvSpazlJHwLlAjDIryeC1uPk6RHguVT7DFXGBikGaWWJ+OdXA6NfaY9XZqQ62PayeTPjH12pMAKvdHJaKDj+bl5S5TPYUQnqBZ40qZsxRG7Hnve7jjY8fh3Xek7Br5TJqHU0EhktCCtn4aVU5RDzb6VzQG5YKoJScBekX+cFYyFcb1L8p9N50e33tgqvmx3kOp1hTPvkCyvIroZcgAvFAMcCz1cvJ9U0z99Ae45aG/hjo488keQWepl3+p1ah9rp6Vp8K4Uo/rRNfthtVV8oin/EIi7el4kDL32nm0DmiN1QjWLE4L0kAxsn8vDFOMDX6jsVItHrbB6xaPsKv90ff+7M96iaV5aN9DRFWvjrJnHjZ9TiR1ts3BKnVcAX0VHJ/vLX7p5saE0O+59/MH3/teVr9+BlfkCL8OP8st9vGw9G0YEo+NwugIpbHT13bTuCGBDHDHa862ubqMWV5FmqoqIrKqfSWNqzIFqYLftNdVhflaQjCzF+LoLW0cHUen6a+RLS6xW4LyVc6E9VAUDithYnIaguRUlpeHE/ScTupP1jDTGj1XGxk92W/mJ2YeuL8d1JqPY9JjxpZHlvvlQ3FYOz0W8XHUDg2q+vbEzfXJYfZ+k33veR8v/a//FmkdKnaUCUxM74d6i1oYVNw0TE4FgB0UR1HqLDEbvy/R8SmAzGezQlQR0Fs7WAlE6j0xK51PLb/8Kq00pzSeYGSc1dSy6133cddD/wn82L2fpjXymXlWXp2O4puvN/U0M8dy3AOjTs3RKx5Gqk6jVn+CaKgqpCuPlWn+QPjjP/70TK944KVXzlGv1yiLFLTGh4o1UzI/EnHfR3+Gkf/0o+T79jyx6DgunErHk9GPN+AJ4hugWP3BiTAOzqS4Y9K4drmwzMVnzrLbSMpBn5pLKKVkEAmmb9oHQfB8peKIBKCfdX6pHrd/cRgY9ThA+/ZbNpQiuzZ4VIXxmaChn9g2E5PNYoujKmp9FqAhgieypDnbIful9q7JTxw6fNfnvv5738QKvQFYrDsN2YKypoFidGoa6vWTTsj0TwzzGuwMxmGuLKPTHGUtXgpKVyK9QzuHAQpnCZ0gGuZFuFoN9u+5cSjD8jKm26EWKJwtKtHnJEVpIIo5cODAcKJ6TqvoawBB5p6X2IyB+7wYi06bQMyV/dVPqUAt2Cic2x3FPwnAin+MUfHJNM+OWcG0E7S1kHM1HVabva6eJGM2y/P74yA4y4Gb5qySM0iBUxYZK8RNE9AQJ6/ifkNkRRaWg07YjJ7OYpmMBly3gWYs4tOZT48hINO2U/bMI7IWd0zETIT7GiI9srD0BiLIsJkktRK7e5wDP/lheN97P0EzeXyV9FOaaK5vOeFK01beL9SSYON+SaxPJXCKrJzFmfbA5DPFQD0mAp3qQM7Vk9FPBknlxGy94y4v905SrnTIydFRiNGSwjr0B95N8/3vgX0HH8q8n7EDf2xvM9w75I7H01qlysRwXUYn6rUnG1UEURo7ldkLV2h1+uyvjzLI5sFLCgW9SDJyaC8ElQ0oneggt/c7BCjT7FiQxKfWBv1HVRzO11u1jeaamSln8yK9H5cfjULZCSO1IIHeyupjjdGRT46p6OOpSY9hiqO16X2UQmKFQgg3jLqXQx+Xw4mKAadCcnDffpDhmXpYezKFYwkVsubeTuG4EncYa7J0aYXIDBOJtKQsSwQQICvnmTU4pwikQkqNiyPYNX5DAukvXsb2ekRKYvN8iBdKbOHQUUxj9zh4MyOS6GmG+ncSV9le5OJ+mdtOHIWnGWaAdRj8Um6697d18xfR5RyEJFHF2Qe44wY308eeqKOeHFprcSpoW01cj8TTSbs9Ey1fRTqDsIaxveOgfVxDnqzH8ZMMq4DYwJ7oF+5EPZRPXj8I0icoiMPodBwyVFF4EvJjrC6zfP5VxrJVFAmrpWf6Xe+m/d53QzN5fCUfPBZG4ZkR9GcyxayTom2MmclLOxuUZkZmxVEcbXQ4583acTHS/kSt1vxsDRjAcYdpb9oNnKhP7e3svftw+8IPX8THkjhSpNaiazEHPvrTFNOTxE52RqLkMyNNPgOQDsrj3az38O5a+yffxNRcNqtkfDonm41K1y4DkRZraw/X1+yR15/+Fs1egU5ABh5TenKlyEJBsGdiM6zdMoeEQG7q/IUdHDeYGZuWM61aZZNkrpj1XsTCyyzWwelYb6KhOemxCGiMjnxyPbbPadVGiaPs30O4axR/frWqnODXI879thSHDMH41NRGL0hrmV4Pi3Bvy1GIb+N8p+gNCJGIoQeytMWwzFCVZVRai3MCoaocYRcoqEWdG42frqzg8gwtJANjIJBIWXlJozAm3jOKi6BEd2IgL/onbFjP8I7aiPxMDLjV4lMmsGdNTWEQMxue2Ub4BFkxC76KxsWngRTPr6tCQ2P3lEyioxnM1rP+g1ErIVAKVRYECPZN7oJu98FY1k4x8F9G+JTQnq2PjH5yPY6qnw5OeOuSRqPxBEBZDI4jfIyCorv0BZ3RlknzFHl6P6b7IOfnEZcuMWEMGSFehNz53h+Dm2/+2FqWPUoUUkOfLAdrj8a11mdlWSKdndKlPVpeWnys89JrZK/PQ3eAuf82JvcdeCqe3POQU8GCRnTCeFPSOGjTbH5uzx23ffp5SrQGqareiaPNMSbfeYR47/5PE23vNmsSpoJa83n6+Qm8S3C08S7GubbyRbtk5THrXTtPoqdtEJ6pJdHXSC3Lc2880JaKIutCqPBl5Usqlaqy9aToDBlIB6A36D+MgkZUfyLHzDST1mcph51x0+JYnAyZYcYshtmtKm1EcmrBd56bEu27SMtjKQ6fEJfCtQMNV9Mue5G49fKl18TaOaAQDj3aAlc1a3WO9jqBeHx8QwIJMGfJ04dXzz5Prb+GMwPCkVHKNYNPDUUtJA4k9UGXQMFaYFhqJOw9fA/I9icyuo/oUi3ooEJEMmdmnTRt0vxwbMyM/vYcE4XCakgDoCypKU8yPsar2nDTO+9EOjXfRM5ZNzjulezU4BRbSn7KkfAzMrXHisI/IMOwIzEdS/+EGvQf8kqk3TSbjZsjn4lVa7MXdt+fwNGm3k1GZKvKhS6kt1cH1ApFrQzR/YzOr/wRya//ASY0n18YbXBhcpof//uPfLTfKx5OvIiLppmrJ7Unl9L8CyUctvjpCS2+ePkL//ZLtRfmvthYlRSJZFB3Dw5ERraygjl3lSk/waqJaDZG8O88ALP3Qi04KXvlw8SarrePNGutz65mK4+NxKOfZJAe55lnH/vmv/hfyH/4EvbC6+xp1Fj6N1MsfOT9vOe//S8eWgoaqXSyM77F5xT3yozB4DhJRBzWaRQKkWu6UURy5A7q41NPY5m2vfxh1YieAEjzwfFmVHvcZJ1fcmGQyMvLn+fZ51n97ndZPPcGvdUe1ggiFVO7aWpG7G4+dOgdd8DFq4yvXGJw+QLhriY9B+PBCiT7KEcPwb47v7gWkwnSR5pR8jhAo1Z/wmd+lozZchDPLLX4ghXR9G54MvXZkYTwVJavfir2PqE3OM4VN0Nr9LOMNj55ES62bfKLaMAG00mDJ/rYE4EKz9DpHd/lQiJTJa3l2uCdJbKGwEtyJKWCbmMU3nmYtcBOt4BWsNkvPeBtoFgKOY8x7d7qGrF1xEFIWVZ51kGgsNYSSI3WGmcdxjiM8+yZmABvpyVq/tre1RLZCZSeE3lxtLu6UiX4u6qviFRVPfjMOaLmCFLqOVALOJFVVeR8vB1EyI77XjETlGJqNI5O0++33eKlL5oLF8guXOLlF15kUIs59IH7H5t+5zsgGRJJXVQQoxGPbNCaNZgsReKGpWsEP/j9PyKWBaomWRofxf3Y+zELi88FrZHPykbwRODK40hoeH3KQZqm6YMsXn3qhT/6Oq0XXsMvdrGBoKw50B7lBbXMUx946kFIFkjG9++FWnwGIFB6vgSUqFQRG9fb2MHx8sXnv/SN//ULrDz7LHuNpxaPETlBp1tgF3uUFzuzyaHmxxsyfgIgS+0xnyh8SCeI1Nco8iOYklhq1kqDbzU58I67IQ5Po/QCWs1tgUDjda+2vLT0+Rd//RSvnfodosuXaejKyVbkjrSE5R+cZcH3eXFqjEkUwYV5xlt1wkbCSlEQOHCFo91ug/PtSAZz18Z3iVicpvrN949oPgMaivRY4nzMq6/47PU5OktLrFy8TO4ld3zww48Gd955plFTTxgpZ4YbdX5DNljXpt9HuWGbDeGGAQNuW6ya8JKoXgOl8XI92nyr+vQ2bJAYFhhkrC0tEdqSOArJ8wypQAqFcx5E5Ti0g5LClpQeJvcfAC87ApEpvV18x4SnCUNYXv308pUrSGfx3oHz6DAgLy0DW9LaM4mQeg6p5pHxaVw2+2a/iqBeC8/QN4d5/fxTvWef5dIz32Ht+8+Tv34RmcPK5Ahiai/cftvsBoEAmbGzoReZ2IzAJe/38K4qQICGKS3QWYmSmmI1Y6I5hvYk61CrNK5NCFG16KeCwh9ltc/qC69Qe+MKkzWJLyxlZnDOIL1ECUXgY5RUrIaCg3feCrXkKYAoCU85448lWpyin50w9XgGaVK9eAX/4ovcbg0jVtNLPau5ZRCWHKqNEql6GpE8sYEuaYGHJFN2tqZcvLa4SGQtkQjIvaUbCppH74aR5me2RiOn+GPryULe+5hnn6PzR1/DPPc8+8KA0TiiLA2uVGA9LVdjxJRk+RUCoFUUJCrEFgUhEBhNmRv2HTgI3iVRsN230su7jzSi5uM9nz88EkWVJF+6+hv97zzzoLu6zNzv/BaXz53HlYblXhc7tos77n4nrK09GgbtTw8CeXxdVd5QKa2ZsYtLhKWtAhvFpn4lfOU4rDJUBa3duyCQeCWvYw649g0JRFo7xeoa6WoXlxXoZkTqDGEYQulQSlOWJVJKrFYYL7BKkeydhnrjCUF6bBtxSL0h/s1af2blyhV2OY90rioBqgKyUtDDcejgXoSq8i0AlIxPXxtWXvNlO/3BD780+N6LXHj6G1x+5ruM9LpMpCVJd0Agapi6ZiQJIAmeT83guHeKWhidHJTp8UDKaj6D4jhLV7GDPtKVeAlGCRJKAm8JZUgdya0HD0EtPgnQJX+kpuWC2gyiRgufUeSEacaeOEG5ZQLvCJ2ogvgkFEhy7TEClkLJO2+ZAaUWNuFicWqY7XYkgadYGxxf/P7z+PkLJMN4sOUgJj50kOa+XRz60Pth1/i26iBJIE9VDjp/mLz/6flXXyJ2FqzHB4qOEnBgL1uZRazVaQ9JTVe/Tzm48tWn8a+9zkElmanHmGxAr58SBQlaBEQuJ24kFJFElCVBaHFlgRtAK9FIazFCMHXTTXANjJoZP9uIKh9IQ0RPmLVL3+l887tHeeFV5r78FdbeOMchkbF7eZVms8UVAZ1AEu+fhGb9c0nAqT76oa3IEzgoiqOrC/Mk1lyTDr41B6bqndnetxeCcF7o4M0E4u3UjfuDZPkDxeJS5RA0BoxECk8QBKRZShKG9PMUrUGFAd4FBI0GtMfW75JsfyjlbKyD02TlbO/yEnmvS+AcWIumKhVZCk+hJSP796J0PEcUn7ouBJ31T4hvn/n8M//u1+l/5/vIC/M0s5QD4yPV4viQjBBGQuT0GC7Rc8bJqcD4swD1SJ9UcggzWjO1duECoSsQtsRKiVeQFT2cKygKTzcOYGIUpOusYR4dUBwPZPhJBVCrPOr4Yia9PI/JBwRS0ssGRID2gkAorPUU2tHzllRK+vVJ9OQEA2tmasPVyDM7G8XqNNJ3mvA4ncwvXriMSOqsSkUaN3jNS0Zvnebof/5zJO+8+3PEyamuKx7RMphLEKcYpMfLPLt/pN3+RRaufPrKSy8x5UrygUFOTqDH2jDSmrf54LiKaietd1MZaraG2Gx3Udrp5VdfRaZ9YmcYrK0w6HfQYUQY1CkHAy7jUTbE9h2BF4RBgBIBRZGjvcULCVHM6MEDEAZnNhCrIps1OpghE7PE8jS2f6L3veePfuOX/wUTr1+hvrjCiLHUahk1UxAVOYveUIoclKE0gyN5HjwsI9HZ1sMG16FIZzoL80SlqdoCyk0iqSLBJVZIjJRMHjgAUXDWS5Vsd2+kx/TbkiB5efjq+XlCJLGU2KJEqKqkix+2qbdlgdABSmm8UIxO7oGknmYwK7yd2toJeAOms36qc+ES0lpCQBmDDhTOe4xS+CSmOb0H0YifeEvita6tXj9P9p3v0bh4mYNxhA4E3qTMr66iEZg4ZDBSx++aIEV1lNBZ7F0MILfmuhgzMz/3Gs1AQ1bi0JRSMVoP0bWAVS9I6wmMtcmk6pTY+0PUGQftzGezsRgSmoT5i2+gpMFZR9waI3QOX1pyZymdJxWQS08hLNHULmg1Sa2dWieQaBhJSqirDVU4zlvo7ztAbXqKOx+4n/sOH4Z3vONjWVDOELQ+O6A8HsjwrPOuXTVrKWeCKDzDleUvd555juLSFUJryG2KiAIO3HEnJI0nnHPtYS7Fm3wdxWpvdu3KEo0gotVq4dMuVilUFOAjjSssNnJILRCFwJSWNS+HhfEU2ikyKRDtUdgzDiO1z2wYwAIKKSD39xeDteOh69+/9OwPsN9/ibG+Y1+tRjgacbF7nkQqhHGoWDOybxoaEcHI6CeDIYy9dd4CB0VJ59Il2qYkGubgWOk2ugyUoqpVUCpJe+80aD335pQ31wY7feNQE+uTS+fOEUpBLYqh7FbRnMYgJRizWVjMeo9BsHvfPgiCszncH1ibXBdMHuTHVy5fJQk0OjNgHUEoybzHSwlJjBhr7wxBO9dW3VWirE/bl4SDkrXuMr4WYCJNfXSMtKtJWhMko7s6fTQ1oU/iyhM2W3s0jyHY/J3Tl86/zlQcIFctxoMVku7SEo1mky5g6gmMtIkb9Sdi2CDcMus9TLLhJGhfPX+ekTAi6g3odCWlrxiKFFX1c680kdYESiIP3lTFGonrVKiL49Mmy2f13l0PffATf/+LURSmNGpPoEQHXHtxdfELjd27Pwqg+y4O65xOc3MsF/ZYpIBAdvjGc7Mv/e4f0jKOQIGPAjJnuP3IOwA9tx75vREpMIRXyQazg8UO3SsrRFSAhZSaIIwpLKz1+gROY/sDAqmJZYBB0S9LDJIorqKUB0UO7Ra0atvUKxHEp4GHGeEzIS1YWB0svfYG+8OESWvoD9ZYNJ5Ma5IgwFhQUY3pmw7h8RR5eiyKklOVb+maDEhT0l9eZsJaIuuROEqqSjgOiROCQkGuBOHuXRAEz3v0TObcbDxUuT0+8d5O3TifRsLq4jIBkmTdSWYtxpYopTCm6jGnlMI4i3WO9tgYSNmxMO2c27bLsyKfJctmKc2RwWqXMNBIqnq1UlY9rb0UqDCA2g1qfjjXfv2l51hduUqgQAtLPYrYvXuSMG4wf3WZRNYYrY1B0v5knaowACLIvIYSN7Ul3qzd7SwThyHgsL7K3xClJQpidJxQHx2H5uh8ZT8Vj5JVOS5B6doM7HFWB59irXf8yrkLaOtgkBL6OlrU0LqBCmoIFQ0NEQP9nPHJSYji00LrbeBD12WPAOSxnlpU5sH0wN7Psn+6Rnv0F5ct07Tavzgxtb8Wq+g0KcfClAcBlPXtKEpOOSXS5flzT1359veZ++aztFRAqCRxLaJfZExOTUFuH9BbbJ9N77A5wiA9PljtMhI38QWsLq6R9XOiqE6gIqwRKBUyqeuMGEXdahIZoOMY0axTxCFLZUGpQDZq2DA404cT2+EVn17t8xsAvSxPFi7OUw8CIq3o2YyiFVCgcV4z6Jd4Jxmb2IMIIqKtfpsBx0k5NpQgbawjSwco79HObSSRrVdLroqKyKriTb0GsnoGfmvqLz4G196RQGy39wiLzCy/OocslhiszSO8pZa0saVGEBACI0mdwSCjL6HbTojvvAk0cxrmCBtnoOpSBZVnGfL7KZZnll/4LocyRddmLCcwGAxo2giihGJ6F9x2y1ObiYOVsW8zN7sCjw0Kezwo9bR6Y4m7k93YbsEFm5PVQ+ziGkk/Rdc1RS1h9M7boDSHAdbS7qODmsHo1pl2b7QKfc/KWV59cVbNX6K/uIaXTYIyZFeuCestrq4ZfDzJ1M1HIOQMgG6Fn2UYxEer9Vlq6iRSd+hbBi+ep+Y1vlWnzwppnNHXKUZU3nmXG4o4oTc+wti774OeOzomGx9PYQPQaMr48WrB+9MTsXhqlHzO99YeBRhrj3y8ilMqHt1AcCaij62U/jGn5QJpfkx++4XPzX/qn7L8r/8n/qItCbsdruaGshcikkm49WYYa30cbWb6thp3/bXWAiKHeOY54n5JXUhqjQQXB6wJQ65ABRpvHWsiYDUJWQ4dPZmifUoz6zOa9tnlHb6VMDIzjYr02To8OSjK4wPM8a5JH6kjntxV56NV0GUXs3SFPBvQSzNquoHsSqb9CNIZzIigk3haB2+CoP3p7UgNJ7MiPwqgu8XRxWefI0p7mGKZsplQCo+1tirgFwSoep3BIOXI/lvwMwc/jq6dbCN/MVHq1KYPpPm40JMfuLGK1etWZR+3ZgqupwIKV5WxcY5AKjIhqmrlzQZI0VEwr5ALVVqlOrXV9KfXx/R6FEVGEARI51FKkZUFhgbtiV1V+mVWeU8VySmKbNb7IFYwXwvVSfL0l/I0RTuHEhKl1LbsMSklA2+YGB0FqToAodLPO1wy1FsfHqpFU/RSlH1zaqGUEqk1A2MYH930BF8/ci9P6HUJnUV4g8MSCsV6M60AgRZVdlzuqi5cOgw2wi+cp31tgl+NCpa29E/kWsSW8hGPTALE80kr/CxAb6XzS42odno0iT5pX5rzP/zy77H4e1+l94MXOLCRTlz18bBKETZqm6nSXs1LsQlxptYe88LFFHZ66coiElu1zPMOhEV5U7U18A7tJEYKsBInqx+5WVyjqnbSdyW726MwrGpTC4OTGWbWCbGpWRTpMdMboK2vUibY7EjGsN6vk4BShPXGdddAKLmwvreytTWU90gpKYsMa+22FEfnBSKKCK5TpOFPHKzoF6+iTFn1HBQChMKu95xwHqkVeVmidIAVmlQrmBgDJearILdNWDczfjbW4jT4mKVl/EqXMi/QicabEh2E9ApDbh17Dxx4c91a49rOu1hSbXa6q9O9tVVapkQHEqfFZm8KoRFa0Ree9p49eC3nBBCHyak1238UtSVZvSiOpleXCF3VEMgNO64K5wnQZIFmzea0pvew1aH25odVHLZXLhGaAuWrgLhEKJytclFDD6EMsdLiygoJDOrJxpjOuTbq+kJdhfUnJW62xN0vcFnCMI87X3msIV2bF194au4P/4izX/0K/o03mOrl3GpKcjXsAOYlXgSUgaI2OQHhMBxHJqcSILPMxorT3pJ4XEJWsvz6RSZEAaJEiBIww47GBu0E2jm0DUDYYQKSwA8r4lf1AyQ977lzehqnZEduJCu5WErZIXXHSOQpSjudXlkkKit1yMthJfwtPc1LUdmljI8NjeprCWTo6DM+7l9aRNmK4dqswCmHVHrYZhqs9bg4or5/6kcjEO990r94kTDP0cPCS04pzJBYBBYnJdZaVBhjhEK0WjDRBiUWtnbIrVRbc7hCsURmLi8SDjJ8VVmrkkJJvaqV6h279x8AFZxlWFxsQHq8VpVMaSs4S54eK69eJu13aRhDrDReWYbtjapqjlKTJwFyeg9O6QWx4TAvp9ZBiwxm4zx7YO3CPNGwwJuVwzpMbtjIPgzoB5LW/j3XXZwqcG5wPLLF0bULr1MvDYGzWOGIpMB4B96hvSTRmtRXzYdqjRH0FgLx1iVvRSCVg1WelunK8dD5mE7X000ZvPgc5779LK9/5Y9xl66wNwlpCUeQdokGGfnIKNKKoZ9KkkcRI/v2QTT8HVk+SxydttZNoSS1UJ2UhYrJDIMry0gsCrMhSfyw5OuGN3qInQonK/BMAEJiZdUiLY1C6vv2YWQVebtOIkLIdIMDFsXR3rlLJJmrKsAMiULIqvycQ1IKQdgeg7GxlOTNsH+YyFN+kB8Xaf5g9+IlAmtRSoAz6CjA66qYuLeOrCjJ6w30wakb1g6/oQTpnTtHVGRVlx8hEFJXDU0AuW7Ieo8Qklwq6vumYaRx2usqMX+bTSPsNATgzMzg0iVahdkwzI2zyEDjQzBak+yZxMXR0xLI8LPOuTYinB96909T5o8tX3ydssirChrr3V29GFYuVFWH5bE2jI8i43ATCvS2PbTXOmAhS4+unTtPXFqk8Fgl8F6inUf7oTE31oCp3VBPnrx+eZzyMKY4svr6OWqFRdmqA0giBd5YtARvzbDsqKSQnt27dyFq8dNoufCWVUu6+SNirf9oVKudpNv5RHj+NbLX5nj9q9/kyg9eIFtZoZVb9g9SGji0NVXHJ+UJ23XkeqUTJE4qijBiz8F9EIdnAJytkBon2FB54rD+JGtrXwz7KcoblPMbErEKRZEVY5SVL6EiFL/pa1j3MwgJYy3YNY6ubXrQE6JTGcUsyRDOzsr7u69fIMkN2lWdazebgBqchFIJRqamoF4FhG51Dvbz7ISMok5k3ZRY67cHC4s0SoeQECqNCQJSDdoKpJEUxpKFEvZO/sgSJO5fPE+Q55Uzj+GDcZZAeoSxFM6ipcRbSRkodt96C8R6ziqR6TfXqKqgviyb7c4v0DAOBZSuEqmFdchAEzZaMDqGrDUfZ8DxGHGqi3iEJDile76yG7L+iaXzbyAwKFEV8XfDWl0IBUJivKC2bw/U4qc3Ko5UafvZunEXoyAv6F5YIDIOScX5pKjq/ggg95ZkaheMj57ZFolcFscSoyGRp0B2cI7V85eo5b7qeqIUOYYCg5QBBRaNIwPKKKB5YBri8GtE8enMmdlAquc3tMm0PKaT4FSs3MLys89MX/7Bi5+YP/MMy6+9wrgW6JUVxgYlsaoxpiNU4FheXaJjHNHuNk5Kzl1dZKLeQlqBchKvAvKN+wYb90pxx+SWnhi4/on88gKxLQitRHqL8FXfEyOpWn87DSogUxJFVUNXDPO/3UYROUk8PQmNOv3CnqiH6slNGbIFMcryI73zC0RZWeUbKYFQII3HeYeVnlIF7Dt4CLSeG8DxbXaiFh2PSAQOllcpl1YIjEUiiARYLTBYpFUEKJwXFHFc1Tb4kaqGAeniVXRpqrZhAFJht3juc1PBvdZaSu/ZM3MTXri2vk6KaxCoCjXq9WdWFy5RK9e7LnlkoOlnKV5pGqOj0GhsSa9no5iwVKLSNfN8urNwkVBptJRgK9+MGxKyF5Xq15zeA5oFK+R0lppjAIHcDJ7MyWYpMvpXl9DWVy3V5LBV8rCkf2otI3unoLH9N3khE7xMtviMGFxeIjIS6TTOS3JpKKXFKk/uLQNv6AtLHgfUp3ZDUOnOXop4I8SkgtKnB2V6fOHf/Ksv/fEv/1PO/7tfpfHcSxxYXmFXv0eYr6F1ilOepbRLVg9IbttPNjXOBa3oNEYJ9t8ODJOFqGoCl0FINDkBG89zvcmNXwBwef8EJpu9cn6OwFTtKAInEK5qimNFSCECUh0MWzxAqiWFqhiLG5bkcUMjvbFnCrSaN4IqqLBrH1n3M2zRvRlcXiIuK1jWDdNhhfQ4b3F4jIRd+/ZXNgRMedggsERFpxyuLYTP7FIH0csqQvNVzTDnHHlZYrwjEAFKBrh6Ujkvf1QCMd0eyrlhC971ko/rLYrXHYYSZyt0YHT3bnLv45T82JtvpjqGbJY8HVa8qDa0UgqtNVmWYfHEjTpE8fN9705sjSveGpRGWdBfXUHrKn+kqqZS9aSoen17jIPaaAucj70g9r5alHVi2/CYWkfZ76OHVfiqNuHrWTMOYy219gjoajMXQ/+HECJdt5EqCQJFLyMwFcd2aEphKYWnVFWZzGKYm2ICiRxrgbx+UQvvfWydm5ryltqVRRqXrrC722e8LFFmQCFyyhashp7+SMhFWfBi2qF25838+H/5X/Bjf+cXqE8f2qhZVQXpKayS0Kht6t4KPCaJEadN1j9hbDZLmc12rl5GlEWlPjk5XG2JFQIjK7WzkFXriFJVOR/rRbk3Gcjw+SMyP6zvY62drn6fi7dkYFH0eihT1VqDzXpWzrmq6DWSYGwMY+20g/b6eOQV/O8hRoi0GKSIskSulxy1tmLe1mCtrVDJQCPjCFrNzn8QgVwq0+9gVh4LfvDNT/uLc4wKWXVgFCGiLGhqj3RgSBhVEYPcshbH5BO7UDO3Ejd2fTTwPqPMZq+JDD6tiU+bfp/OlXmCvIt3PZpKQc8iaSJHp9hzz70QyLMl5gg1cZKyf6JOFT27lvsKs19JufTqqzS7DpkaMluio5g4qCMKMF5RhAEjf+E9YM1MM08XktrQW+yr1F2f9h52Rqa9V15lzPSpFT3yPGXFOLJAQASFlRgdsu+22zeQn/UiBolSp3qqTABqy/2HWFjiqss435RcDDQ2GOHZPdMM7n4vl/UELmySBAIhCwYAu2bwK+7hSpKpB1bhUxnMAsTN5PGis/Q5O79IQyqcMwSjTWzcYKkvkcE+TDbFS1FI58hd7P67f4/3/ot/yYH/52Pw8x97iJ/6UOfKkVu4EAkuT8SU1tAwno40EAQIKkfbQqAeFVakZdZ9pAhdm1p8mp6d3q2nSbs1PIJeFHOpPsJr9d280d7P8p334H/2L9L4+/8ph3/5c7j3vp+FsiTTOZHK2G0sdq3HUqKY+MCPQ26nRsv8DIAaDT9ZZRvUnqRXPgyw/Pz3UOUKSvRRSuGlxmqLCQbUrGXRenoHb4b9t6Bl2BmBzzTgia7tPbKe6OWQber1J9eeeYbdNiMZa6Fkg5WwrJr2lBEjOsEllqxRcvDWWyGvndwplOktbRAt5BzWx36td0MR5IWowiekIK43thcOfotX3q36rEspcVphfeVPsV6Q24Kp0WbltxCKlPxYtA0BATIzS69HYlzVr06CcpvNaIx3GAEqiQhqMUTBWbZIjfV8B5TsJDo+tdBZrkqdygrmjZSu+n1X16ACTWt05Lq/pREHTwAMeuls3l+ju6dNND7CyPQUU4cOcuSB++Bqj2/9k3+OSwfYssRbC6WFbg8xNF4jeHojInWQHacWn6xHeu610drMG+0GRWeFJW2wtQg5OsmeQ4dotEb5ub/x0wS1FoxNzNGof7FTZrOl8kkwknxx97uPfOLq70bkqcdRMjCCxkgNRhqdshgc77vg6FQc31V1josIhmmsppE8Fd51y4N7PvohXl74PrXRUVp7p7jlpv2MHDxAbWqqQiprtae5sPTAxMG99GtJJQxLg3UOoTTOiwqli4KzW/v+bSoFlbo8WFvF2RIt1UbTUCkE1rrKnlGSKElAy+v6QGzmZn0sE3rdR9aWlzBFhtXhsItWJTW01hUYVJaUYcDExK6qWs+fyEhPy2MkwSmt1ByZnVm9vPg2CGRYolNIWrvGQVX6vfWgg+uU2sn6J3oLlxGlRSlJGWqK0iBVFcuz4g21qYmqPD4i9shEqs3arUr4ebLifhau0sgdUgq0EAipyERVi9XiKYUnHB0haNUhiZ5ma3GAdfEtRarS9Nji+Qt4b1FKIQtLLDS+LCs1REl0FBKOjeGK4qgMN0McNiKTgdqBg6I2NXHiP5vZ+0WMqcJk2s0zpHaKK/3pwUiL+tIqZZYjrKehYrBQKD8frntgJTBwxynNYcr0WNyInr7lb/+tpxrvfNcnxNVl2vUaLgkIxsdQU3ufIgjP2JZaAJGa3Cc6ajwRxLWFQARP0IR9hw9/4lIQsNZLmWhFWO/Al1BkbfBxHAen035xInGqjSehpT4bkZyinpwa+4kf+6Wx22/5xO0TQaWkBcEcSfIUcXy6BFaz9KH+YHD84MTo3OjBfTOvBJrSCiwC40HHCSpJSNqjUKudJLzOXkj0KQbZ8dXLl8GU6EDhrd9oLiqsw0mFV4rWrrEhgajNBpt+mL5bdZZK/aD38PLCPGFa1QF2drOyuw4DiqKksI7CaXYfOABSdv6EKFY1YCjUGXIzu3R+/saxjEpRekGpBRP79m74Cd6yBH9aHl09t0BgLFopZBBSZoZIxTgpGYQCObUbtJxTyLbHbqNyKeiQlw+snTvPSGaQotJbtRJVXzyhyIFcSZLdE4TN+pkNp9i1xK1Exkr/4asXzjHqLVJVtWa1UGSuwEkFsgqrIEkweXE0d71HmnHj8a2RyXnGbEk6kwSkav/+h/B2uhB2KpMiU8pO1VV9Yfpd73q0P7+ItgVNLUmtp7x0ma4tZmsEHeP9DIgqbD5Vx9CubUVwtgxaZ+p330PD2BnRqD+x1QfQs+Zhlw/araj1hE6zx7Acb9SCJ2yaH1POtVtju04fOHjT7KXOWWQ9RKR9spUluHyFYNfu6QKQUdjBsLBuS2XOzObS3j/Sav8ih6tyPC4fHC+Na0dxBbEGwEScnJqIxyjXLn81GB+f6TtLqzBVUCECVa/R3LWLsNWco7Yl1Xnord8I6xikx1cvLiCNRUqNGSY2SSuQw3x2H2naU1MQhh1q4clNTaeKoRKxOD0Cp12//2nT6VB3Bukqm1RogXEWLxUoTYGl1IrGvr3YIDyr/kNskBriJIP8aOf8wo1rykmJVQqjJRP7pzcSfxzy+qG4/fShlXMXCVyV1uqEqIoNe4FTGjU2ArvHQIlUw5zwm0bsgPJ4EshT5Pn9nTcu0CoMOIt0llCKjYISRkrKQDOydxoZBc9f+ysTFZ3aCE5bXD3ev3IF5w3gCbyoij+7qiOUcxX0S2EIdXgmQC6Ug2IbzBjFnG7EyRP9tDhaDvIHsrVs1uSyI4nn8mA0I4xPH3jXu7iU9enZgkYcEeUZV37wHAHlQgKnpGKTmyXJKSNkJ0PNW8eUDupnRdw8vU4cfTjRhxPG6RlZDrlgHJ+u/DrgnGkTigVGoi/ecuSd1TMxJdoagrUuzC+A9XHR7x5NNQ8OYtrr4Ecs9WmZ+k43HTyysUmi2slo6H/YGiDo1tyj1OpPBu1RhI6rNthO4HwVSt6ankLVayffnIdkEoC8HBxnZe14/9JlQu83mqcJUTUHDYbJd0RRxXyj8OltqOg1SKldWSEoS2pSgfNYW0Ur5HlK4SxohVUC2ajD+DiqUX/iTyZB1itIVHowqwtXGLuhBKlEaiGHsfWBnB/gjnsh36Tf+V72sOgP2r2Ll2j7CnnyxqK8qrotSUVr7x6oRSlRUjVYEXpoV2SzBmYQAZRpsnzhPI3S4nyBswKho8oGERKrwIQBzek9GHyiK0V1lmseqDN2moUrmH4f7w3eOwIBZCVaVdJDWo8vDPb8RdRtt56Jw+3pw6ur+afCVnQmEZyKo+hrQVA7GWzA++XsmswftQVz8eE70OMtsiurxNYSFDnduVfZNT//KEn0dD1Inhx4c3y99ZchzCxuuiX5LI3txds2QrwD6IrgEYDSFe1gyF2DuuzQX/2UW7j8gFzuo3KDEpIR4ekPMpa+9xzj9773E+19+8auUv5G7sX9Quh03QaqC312vanNsjGfl1J2tJRzwpNK6zsCsrgmTkokkgY0Wp8fabSI8wxVZjgv6JaW/VOTOBk8/6Ys0DA6WYFXxVGWlihWOiRaYEuLE67qbFA4QjRdCS4IGJveA4HakWOvLMyTOEskhtELSKSEwlYhMlILjJe0J/dArT6fwexb1fvaRiAZdjZGbb9wkDFYXr0hgVTQYUWZ7BrDKrVgkVP4oZetSI8x3FTWlW3dyyiWVomExHuLM56IKm+bOkxM74Fgi0qUuVlieVp4sM5OoQFjWL16lTFnyW2JMYIkCjfaGjsJLgpp7N5FWZaHtbbTvmRYy28rhm2n/OXLFfbuDc47EqnJ85KwEVNIRxSEZL0+zz3zLPfcd99D1FW1M5PKZzEyUuVRr/RXH0sEKWX/BLWqXKaTqj0dBDdXOsnY7P4jt8z6cpF0aYnYJ/jLl5l/6nc59DM/+RT7J0+KKHqauDlE/NRprLpO6yroMXhYDhyJC2k2oirqVxRtcnOccnB88Oqrx9Pvv8DVZ56j/q2z1IwgLAzSOsaBV7/yTWpH3tVOxvcc25UEH90aIFkU2aywfsrhyXr5w2ONxsc3loJs1ngz472PnZVTSqmFiPg0hSWWIYHQCF957PtlSXvvNNbaaQWU/fREMIxCKNPBcaHBGTvF4jIMUpIgoBx0hz3mBbYoiQPNihAQKMT4+EZVxuuU/zyO78cLr72OthZpDMJrEKJSs4RAKEnhPCUwOV2ZAjnc/7YIxOHab1qJ0lAOsrdtpHsB1JtVJCYkgqHX3G86hJyxM5SGMs3QvlJfnLdoEVEM0YuR9ujmd1KObRw7H6+308I5BoN+1QLOmY1KxN4PAxWlAK0Jm3X6xsxg7BT2zf4Gb107W+ttiNF1dcoXBi2q2lz1KKYYrPLS82e5/crV42FbHJZJ4xRJtL0QhVZzUdR4Yh2XT609VgTqaA1O+vOdQfHS88nA5UBJU1oaQcAgz3nx13+LfXum0LXoeDAxMbfuVfEpxyLLdL+enqiL5ElIj1nsdDWCPRwpsSBq4WcZdB8B2dH97JhdWX7wjWef5dIffI3kuTnkK+dxSchIXMN2FyFPGWmP88OzP0R87Vu8Y+a2p6I9u56mET5ZxLIzsOaocnSa9cpmiAD6a4+i9RxR7WTsfIxS8zgz44ps1lo7ver6J0aW13B5iS8MWI+QGoMnnBgjNa5t0/QY1k1vhvqA8D72zszQH4C1BEpXMX1Sr+8VdBgPHYbDvA0lOxnM4kFnNt6IELduyhb50aXLl5gqDRiL8L6yaYxBqAgpJUVpMcIyMjEGSi0UcPRtqVg1Kv/AAI5LXCc89/qXzXe+x55cMT7sU25U1SxM+CrFV1I1tBFXI/RowpF33A2Lb6APNB9pdVaT0nloNI+iRJZdWfhSbBxhavDPn2Ffrc7K61cYq0+xn5SVPOMyhtrkJLt1AqurM9TUV/uBT+omeYrV/MtqMJhtlwaW1j7BV77NHa19zL/RZywosaMtLvYzrI4QKua8XWN3KwEXUE/zhOzqp5FQplce29CrPUQe7HNfY1+Rs9opKWotgrEWUbRGnnapE3K1TNmdl4x//wVe/Af/Fe/8m3/9CO++94h1jfuLev3JROb3E4z/zThqPHG11/2NplqZjZ3sJGsgV9Q0T//uo6/+0ddZeOY5Gr2cMVcjQuHSFF2uMv1Gn+/9k3/MPfM/h/7Jn3yUid6jjEx82qp4QWnmeq78lFBRViM5qYBavzxRKwcZ6dJjRfr6Y+GCw776Aue+/TUWn38WubTCWBnhBopBAcIaClfScJo9o7voFgNmyhz5e7/LD7/zbe4+8fADvOOOB8IDe54O49pprJ5xffsFOa7+Zh9O1Outz5bd1U8Zrv6G9b0H62WB6DnkcxfoPzdH52unefaVF9GrXXq5oTG6m8uFQY+PgIoRa/kJtXf0k0Z2ZheuLvgk1rR09LzsmZmmBb75DW6TNbJOiitqxGFArGr4wNAdeBZHIurTU9Dtg5MZF69+Ia61ThLpr+WL3S9FsXue5dVPq4VFbnrlMn2h0FPT+EEPr0tgnHxthfqIoi9L+mGNkZtvAS3npS3a26K6r5fW7b3fIqk4XoOTLFxYXv6XJ9svf+FXmCqubKtKt/7/ut1b5ppuKHDTu4hvvwk7McZqWVIaCFSIwRMoQSw8qj/AnJ/HvDFPbaVHywjqo5q0NFztp5hmk9aROxEH93E5go53TBUxRjjKMidWjrYTuAuX6P/wdcRKlxprtFqjlLnBSo1ParzRXYaxFu984P2c80HVs11ulrxfzzCTHuo//AG9K1dhdcCo0MQI8nSAU564loCMyJyh6y3n0wEjd91J7Y7b2Pe+9zJ+7z3YIKO0dZSoowZruPmX6bzwAq99+2WuvNFn3F/Grg0I8pK4KBBpSigsrWZCkoTYZpMXzp9jNa4T3Xwb40fu5qb73kP7tluhNQK+X0E5K10GnVVEkSPyAVcuznHutVdYfvoskcupl32a+YARJ2iIhIAayod04pLIO2x/BWdSZKCxtYSiVmOt9FyWCfqmvRz4wPuYuv0OGmN7IE6qBR6tw+JliAPM1Utc+OFLzL/wEvnlRdzagMuvn+d2Z7Blyp5WiyjL8bln1UmWGw0O/th9LA0UzdEWLoC8zJDCQ1EgCkPoJeH8q1x94xxjQY1ICFa7a2Rpn8ALDu6Zptvvs9ZuUT/6LtLRMVbzKtohK1JEKLhsMtresd8JLn3rW8hLV5lMErSrDPSiKOlnA+rjLZa951JS46f+m/8KPvyhT640a+1RoT/5tglkw2hZuvzVl/6b//6B3pf/mF3l4o4EopykDCSDQNFDUDqQBMROE3jJoMgJ4xCnPGUxIBSedhRSdx6XFXRFj1CH5JklLyyyMUIeRCyVJRmOg7aJkY6ez3GBJ44UuiiIS0s7qdFfvUQjqSG8xDqJDzRLRcaaMwTNJsqE23oaCqo+dnqYQ5PnSyQqIEFUleutqVpQBwoCQb62xFhznDiISAtw9RpzK8t0I8lAe0b3HiIQDQarOenaIrJYZiRQ7KpPENgaYecySgu0lgjhMb7AmIKiyCjLElohcZCgRQ2haywXJUs2o2iE5NqxP57ECknuLWVZ5eVErsT1uuTdNcb27cOVAwKTEtmyQgdzsBlVRPKQQWqtq5Z4QUBZlvR6PcIwpGw0WOitIcZHMF7gcoiiOn0gaI+wsPwKE3GT2kCwW4+gckOv2yFuhOSu5JCI6OVrhJFEFjnpygCCGmlzhBUhSbQiTVPyPKee1Gi3Rgh1NQdjDA3XZW1lhSAIhg06HUkYkacZRVEwIqBbi7kqNVfTnEbSIIkUmelTG2uQrkgC4ajhaAhPjCP2krSbUhSGOA6rINYkZlEJFtojfOS/+4fw7nd9vN+opXXEk28bxdowWJyZWZq/SNPesMEd/XyZQNaoCY0qPRhPREhdRgSu8pLn3TWMdMhIVhVRih6lqQIg83yNWmOUKJJYL9G2xDlH23mEDhjJU6wSdH1GXhrIPWWZowWI0BGHAdaUSC+q6t3OMh6GNNH0uz1kHIPf9MMpVxHIeh+/tPRE67aHdxAGqDAhE46sLGi1QoRPyVcLdKloERJaSRQ3SF3GwovnGIsayMITRZpSN0nLlKDfxaQrTLmYdJDRswNc4JBJADiCOKLVHmPg16gjKS8vEZk1DkYhk8pQygE9n1Kf7zMwBXEo0XGEdxayASMqYGJ0lPNLC5S2gqhlHCCDEJRARR6BIix7FAYcijKvfEaxConjGuPjbV6/eol9tqTWXaGmAmzpsHlOJkLWOqvcNh0gBzl0HTVjiJwmKy15uoKKFVJpXFrS6fVp1xMacYiznjqC0Diu2kWa9QZ7xkYQBgadq3QGXaSEuBYjuykjtRA1WicviyrKQkjGmlVURnZ5nom4RTNJ2B0EJCgCk9PL1yguL3IkvIVemrG4epWo3cBJQb90CASRDvCUxGHEoDSUKoJWswrB12pOceOojzeHmvhslm5vur90lQlnbjiAijUyqBx1WsnKW1kMMDZDCk2cxGRl1SkzigK8d6RpH0xJpANqIqxyAJzD2LyK5FSq2tC+oBeHOOGxZY6WHh1IpHU4U1DmgiCIKLIC76twOm8NkRI0wph6KFjxaZU/teloRANKVAQyFtcwxlAYg1ASZNXLziOQUuFDRe5CdHuE3sDRMzAoDKNphqBgsCuiuHKF0b5hRO0ijGvkxqK9ZFezwWuLayRxQBg2qcUS50oWl5cxCFoj4/j+Krl32CDBSU0URlidU5QDrCuZmGjQy1IKV+JUUWXvheDKkpW1DreEo6TOsGotvdzRFR6pNGGgiHTAGj3iKCYQET7LKAc5RivSomT+/Dxi9wijeNzyIlqXeOtYHnRRrd3YwnPl3CVaUYt2rUF/eQ2rE5LROt2lgpiI1wxE7V1Yn1AGgmZd0l/ukffWUElM3UrCEvJBSplbYqGYStrUFNRizaposkxGai21Wg2VWbLVPjIt8UoyQZPVpRxbk9RUhFnr4suM0WYdg2axN0AF0BgbI2lEGFPST7toJymNIVSgCZFWYIUmmdwDoyMQx6dvGGeyTiA2M7MqHqbGmvKwv3IF0ozQ2RvG+wbNBs5WBrxSGjGsO2Tw6FCRUhLEARLoFjkYS6ADdBiD8GhbVa3ITEFmHToEGUisdxjpWFOVV9TZsuIeSiPDkHJYSQWvQKgK6fAeY0t8lg3D3qE5VCG32h9yvWWyAxtZnPSIQOCkpLSWssir0jxxhI/hwmpOsGeCtbGEsDVKQMlqb5G8cwWXZexrNTgwUeNqN+O1sqQ/PcVAG+qxID50kGDQJb/wGrXFZfa1mow0RxlkhjQtCFyTxbTEju8ib46img2iuibvXgWX8tzrbzBaqxGLAJMOwHmipEFZk1zNcs73u8iRUUR7N2J8F9H4LkQgWel2WFm8SpwHyM4q7Sxn/0Sb5ZUu59KU6LabUZO7ueoMynuYe5VubxnVbpA2RxEj08hglEgvszR/kcXLl6kFJe1mnUt5RqdRY+rWOxm0JrCRJbt6nktzLzFZGEbjgAKwNcHLacTk1D5EWMcYS10oOlcuUyycZ7QHS7tnMLv3ULYUTQ8jLDHiYyJgcWWJ1XqL5UiT7t5Nsm8vaS8lX1ul3YxxvqQoNLLMaWCYn7/A7iik2RwhEZqs16cgxZUlysc4IRnZvx/qydO8zZd+U7kTUx5ZunyJAIf2nhu1oe+VlS2hjCMSAUopEqFxWla14csSZT3eOmxuwTlkoPDBer9gjRdhlfwfCsKoRhhq3LA5aDH8vipAe4/CYY2viDJQSCdQQYUEF2WJEX4ocUxFCFR9vP3QSLdDIrHDXuI5Bq+qxvQehvE7sqoSaCWvLKxipw/xvo//Ahy+F/ZOzYHFvPr8jF+5ykv/j39EOX+BNeu55DTJT3yQ+/7u30bMjJ8JJ8IzXCrvZ2npyNV/++944X/7Ep1ul6nmKEYqet2MsutYatS45//080z81E8gbr/1KZLgFK/88POsLPP0//XvkXqPSksCE4DxZB6yKCKvJZzd1+C2+97PLff/OO3bjqB2T56mHnyRtP9gvnT5+NXfPcXVb32by3/0VerpGlnoaRy5i3f8wv+Fwb59vHNiX0q3m/A7/55//fnHaU1N81Mf/zjh1B0weRMUlxic+RZnfvlxmJ9HKOg2Y/b95Ec5+KGf5JZ33Juytphw9TxX//cv8frp3wOT0fM5mbDc+eh/zW3vex9M7nmaIjuaX15Ivv/Ub7Dwe19GKsUDn/p/wd4mNARcuUz35G9x8Vd/CzfICMfG+V69wTuO/2WaH3g/+rZb56OR8c8EXsZoNUev9zDOtrl48QFef4Ov/ON/xNL8PLYoUVFANkhRTYtLS2RYSejR6b34KPpavqUG1tsgkC0uNOva/dWVqkqIdzckEBkmaB8gbUFYeiIrUUpjjCd3BSNhRD7Icc5Rj2q4UJB7i/MCFYdkHlQU4gQ473DG40yJTXOks4wkNSKh0CokRKJUQM9meGUIghrpoKCWxBhnKbIMMYz3x1WRnLmqstxKSVUZY33ebl04OpwD7w2CgCAKSdAIAyYvmTpwK+72I3DXPfzOpUv85u/+1kyoHO8+OMXx2Q8+dftfP/bg6Sf+OZkx7Hvf+5n82b/GWnOSzz/5vx1dXHnjaDG2i1/44Ie57b53sfSd75I/8yx5v0AqjRKa0bFdtO66k10fOcY3/YDPf/6fPejL4sEf23eAj/+Vv3byjve89/jFZ59n7cI5pnVEM26wVBrWjMLXGvylv/V3GD98D4NklJN//F1++4+/OVuYdPbOWw4xc3Cav/GXHmTPnj1879zrXH31ZcKxUQ498F7c3XfwS3/w+1x67XTyt9/3ft5xz3sYHLyZ+jtuI/zAXzj9b37nj2e/9b//NvfN3sdHD99NbfdeePV1+tkKHJpg8sMf5PKtt/Df/7N/ltzSbvB3P/Q+xm67k+f/8A8YZBm13SNM37SH2z7yV9Izr/4w+dKv/doD8WiLh459hFt+5qcp1lYwr79BOXOIJ377S7zROcff/vCHue097yb85rMsvfRDmknC5O5b2P/AX+RiI+Zf/Mq/n35tce3ziYxp6RDTT4n2wPum9/PRW24n2LMPtTLAF0soGRDHISUDjC0qBzJQb49i5PYeiTsSSD83J+o1/eTWYmznf/A84doqtW4XW5cMAkmqNSCISwidxwmDUQ7dLyu1RQd4WRVJ9cINK1soSgwyUihX2RVYT4DEQkXZUmOyHIEgVAnOVWOIqDKPlDcMsFRhPhaHxUfg0KSmQIbQLfoIXyE00oO0DCtiaKyrPPrBsJaD2+Lg9EBgKxsHYYhsSZyX4CWDQJPGCt+FXffdw+pIxD/6L//fdDoRA2H5au0Mt47d8eDRm97Pq+Gvke0d4ba/dhz7vvuf/iu/8A8e6KytYpYu0Zcv8D59iNveey+LKz1GGnW6ZUZjrI4zjmfzK3zo5z8OB3bP/fLf/a9nvvHdVwl2j3FpfIGfv+/njrcP3sUzZ37AaDsgLVMklo6MOBc2+fBDf5fmT/3Vzhd/9Vfb/+rU77C8VtJfSdk1uZfvvfwMPfdNzvV6/IOf+csIf5JCdbioahx5zwNc6mie+qXf5GqU8PDPfIRO/w2W7Qp/9f5Z/Gpj9vH/4SQrPuLmmXtRt0/hFtYQ9SbWJ7TcJLV3Hn3+10791pFvnv4+Z8YUf/mvfJR6GDBpPGMm4MJAM3Hvh0GMZp/+B48naahYXh3wwu+8xr/9p/+I+LaXadyyF6nq6b/+N19OllzB3/rw3wA7x/yVNxiLcyw53XtG4NYD6f/wD7+QfPkHb7C8tsquVkze6xAgccElmg9+hJ++604ua8tEKNilY9xqzkhYZyn3BEnE0lBdn7z3MMTB84FUT74tAlFKbI9vKe2UGVQFl7dW2BBvUf5Ber+h16+DAuuedSsr1GgDX92Kta4b+d5s2AYb2W/XiTBePyfXEamhB3/YT5T1CvcblTbEUJUafr5RQWYYWb7xmZfDCuDVGS+qFFQ7TPvsKNi7Z5IfLi+xmKeIpE57925Wl17l7PwFjt53N+HUHsYO7qNx+C7+5b89+cBqr0uz2aS71qHeiLnl9ltgpQNliS8zQi0Bz6Wly4x84MeJ7riDX/3q78+82rnM6P7drPiChZXLXO4uM3LLzRitqnybICRLDYMg5shPfJDmjz/A//K7v9f+n3/tN3j5jQvc9877+Gt/6xiTYxPML13iN3//d7hprAkX30D4Ep3UEKMNikaDFy/NYxoNIm8YHWmydGGJUnhkHNLprVJLAtb6GUfGdhNeXSPODTGSVDiiyRF8UJ49e+X1I7kr2DOxj1Yt6Vy5fKVdFAWpdfS9pb13H4sZ7YFWXOgsYmTEhd4KLy1d5dB73k258goXFq8mhXXU4pjd7VHsiz0CIZEo+oMUWQ9Bi7mV/tKRfrbEhz70Xn7mQ+8jkZ5Wo8Zu2+PmsSbhWkq0sIhaWaOWJOhAs5rnYKt1dFIRNuoVOuN9sjVVYUcNKdbXxmClR/LOGsoL/FA98VtDBK7hwBKDxCAwgMGL6m2VwUoDYvjm+sfKW5S3CCzSWxS2KjUz/NxdZ4z1+ylv0H7z763zkOv32BbesN2HUxFUUMXteFmpeYCVHicNpTJ0Wwn1227jufPzrJYZ9V11br3nZvrKcubVOZieeiK+9Q4e+OsnWFxe5dd+98ssryzxrne9uyqZWabcfPMMxcJFAl+iygKJo+9K9GiDOz/2n9CZmOCJ3znFfN7lxz86SzQa0TV95hZeh4NT9JXCO4UtPd3CUIy3ueWnZ+kFgv/p3/82c70+H5j9IP/8H/+3/NwHj/KRO/fyf37fEX71sf8bH3vnEbj4GjpPKaQnHN+F2jXJV597kQv9PiPtFiO7dj2lR9rc/p77ifdM871XXqFXDgDDuw8cwr1+nrAsiREMvKF2yxQDUbS/+/pZSmE4tH+ahpDpxZd/CDiKUNNVGnHwJr7/6jl6QHN6DyP793BhsMavfPUrjB+9Q4zc8665sxfOM/CGkVaDiUZ9rnP+PKGt0rALBwfe8Q7KfHDk5XNz9AZLfOAD9/Kzf+noJz74viOn73vnzBfv2TvJ4Ps/4KUn/mdqb5xnt3OYLKXrS8xIHaTACI9VkvrEOAQRCJG+HeLY2CtZkVepsWl6jKU1zNIKIR70sM7ROvdmvTPP0FyRVUFgUZUERg47v3mxnni/XnbebenTsP1YbNns8prjSrps/56geiv/5rdYH3vL+PIagngzkchhv4hKmli53lPPUyqHmtwD04dOvzw3T6RCbj6wh7vvugWEYO6NS3TK/PCdP/2XUO++76Ev/M5pXr80z9133cktt9xC6TRREjLSqp+88MoP0WlKMwpwgeJSNuDAe97NyLs/xK9887t8+8VXuPmOW3jooZ9j19gISsL3n/0+jI4gRkZRMqQsLEUUMnbXnXD4Nv7Jb/46P3jxh0xPTvB3/upHOCBTvvI/fpp/8wt/nd/71C/CV34Df+YbnHvq12DxKr5wtKcOoHbv+8TchSsoVWf/zXdia+HZA/d98HM/8Tf/HtHNhz/+/NU1ukJTGx+jub85v/TaK1ibI3FkztCa2Uvez2YXL13GBY7bbzoA/d704txrBIHC12PyZgMmJzn70mukueGee+7h6L3vQuiAL3/168x3edWPTX/y7IV5rFDs270bvIuv/PAllLEIHSHrTeLxGZ558TyrA8Oe6f3cccutNEMeb8Xh6dEkPtXrXuKNF7/Hq9/4KlMKJpMYU2b0s5TCuqrPi5DYIGB0ei+EIRaR/YlQLONcZbA4O20WriBWeyjvsZotZSArdUn5TTVq/S39Joy6VRXyb2sKfrva5bdVN95I3t9JzbueQrZB2ENVSrJpf2z9THqLwAx/Z1WOopRQyiqvZGzvAZC1zhsvL9ASEXcf2sed+yaJUaxcHjBf5A/c95H7xde/9qr/rW+codZq8PB/foJvfv37eCl59zsOQyCevzz3yvGg36dea7ASSZal5wM/8xEyavzKb34FrWMe/Mmf4PA0Yn+77Xtxk9VOD+q159vT+4+oNy4ighwxUmPX3XdTNkef+ndf+caDt+3ew9HpaX767sPPP/O5f3REffsPmVi4yPjyCM/8s+cowhHsy/PM1CdwYUxrYhrfyx++/MYlYqv47nM/5JP/919+dDQsSdMcG41+/hvP/pCl1HHbXTdD0z9+7txLj0lVkjhPqUIaew9waaGL7HqsMNx2cB8srxKs9QiR9IRET01Da/zpq/PPPiBzw6Fdk9x0++18/fefptfp8Z1vvzjzoQ/dMXP2tXNYITl80wx0e9P9hUvUnaMUCj0xQWPXbfzwa98nSHbTzxy/fvIUv/+lrq+Jkv9Pe2ceJMd13/fPe6+vuXZ39j5xLEAABHgDkgkSPiRDpiQT8kHDdkkmE0cVOq5yTLtySOWkIjmVqlCKS0XZcblE2+VYVsoOYVuRCJNKeNgSRYriJZIASIIAlgR2sYs9Z3bunu5+L390z+7sAhzSpqsisTxVWzszb/p198zv+37H+/1+36yX4pN3fYC+rf34ORdVX6XWqGG7NllcGsU6jgehMASORf/YGLjet30pSb9DgMikk0+h5aCvzsxh1wJcJFquC5o06yuwaANJmDQPi0S8EgsjUVpiRRJLt9bv1ip9+XMjBEYIIiHWNugiIdDJe7RW+OQYaeI/peO/ODwbn7/117qmMOk/26412lNmhAEhmnHIQLSaugm0UITSoSlcerftoLFYOTpzfgHPWFw9McJ4fxcZ16NcCqhpzSp85o/+8mtMLxf58IcPc+v+7Z878eKzhGHI9bu2Q2X1bkqrpKSk5gcUpEP/zYfgR3986o+/8jDPP/8qB67fz513/OQXAXZu2UK9FnD69Rmq2sr3D49Ra4QEQlKxJNbgIBdXq0dWSyFeIJjsGYBSZd/Cc8/QXSpy08ggw5bCrKySXVlmAEPWdgilpGdohAsXL+0rLK/ghBGeEXznyad5+NEn+NpjT/Dg3z3F9NISgwO9HLhuD5G/cPvKzBRN0SRCYPf0IkbGeHNqCR1lUMqwa3QUMz1NPwqaAaWgyeCOXWg79cobb7yBbSm2jQxw8IZr6U97yFDzrW9+hzBk8sz5Cwgt2H/1PpifR9WqKGEoRyHW4BC58eEvTs8uEjQ1xeUizz3zLN94+GGefPJpHvrGI6Sy+VOmqZFhiDYhTRHQNE2U1mTDuFdCKONmhF1jY+A4zweIve9Yg5igcThtp461HIvS3CJ2M0IaTSjiMK9uM7NaAmaS9wNpxfsNyZ6iSvLdhYg1TijblnVzucMeyXUnveXXiA0awVp/3Zq7DbBNta7hTFuAoNW8zA3jwv8NmqTNaRf4a569ASIUkbAIhE0oFPkdOzi3UmRhtUE3komJEfpHe2ZzffnRiysNzp07x8lXpj/7zIuvMLZ1O79w9KdpVBqH5y+eJwwj9oyPwMLCaFQskbFdyuUKzYk+9h65gzOl+uTxb/wtvQOjHDx4kJTH8UqB+0a2bkXle5mpNalU6qNjo9s4ERpqRlMx4OZ7eW1miZROUw0l+659P9QjbCfLwkqVlFasao3oHWVANpC+YKG6ymq+j/zunZwsLiMcSV9vll+84zYcW6McReRluLBc5aEHH4LyEhNeQGNp5pCen8MyEQKLzMgW/OFRvvfYKywJl219GUZ7upl7+RV6EFiBRrsWE7v2sFgPbn/1whTaEVy1dyejY+rIzQcPPHj8bx7jhRdPcOq1wt3LKxUcYbNv+yT+C0+h/AboiKoyDEyMouvhkROvfI8gqvBDB6/jl372w8jaPGO9vRhjYLqyr/baPLmGIO9lkF7IUrFMVKuS1xlCaYiMQVsW2b5+sJznQ8Thdw4QrfNivZlBvbFaxjYClRDibI4pyU1vBVLETKaJUKpEcFumWCivYDq1zRGJGGCRaHUh3wyS9ShBCxRyE1A3AIN1knjdds3tINloZkVrpZ4JxNEojLHRWGTGhlkqNahoTdqSdA/lyXXZv5vu67q3cbbJc889x4WZVRaWVvipjxxlz2S/OHtm2jSqJaTKsHfHJGZukaXZWcZDjbBdusYnyN9wYOp3jv2vyen5BWpDEV998Ov8+X3/7ZHhXJ4VO8v55Tqjua2EvmFobAvpdBrTDOJCsOERKudryDosNjVe3xjgcqFQ49o919Hf08OePVcz2wxYefQrdFspqoGO+3BNjPDak0+zXC3R6zj8syOHyfakPtsImvvIOfc/8cL5R7765T+kO6zwof27EYU57GIZG1AoUv0DNHp6vvjK0vI9l4zkhwb7wXFOTZ9+fV9fM8QSkEpnGRwd5+xqeXS5WcWx64xsGyWSjH7oox/ifz/0CLNLC/yfb32TKDSkLIvh3t7Z+bm5Ud30iXRIIGxyQ4P4UWlkaeU8tmNz441X8dGfvOGu86fOfFnpkEw6x8Vj36RyaprhmqRaLiDzEiflkLMyqBUBSUNtLSUykwZLFjQm/44BIt30sZrRR10qk2p19t6l6ZOkg2qc7RpaRJbAiViz4H27FZ6V5HyJZmNCYyC5HFCJQx9x+Zj7Nule7fDUSQw42jTeCu9e6RHITqnMUBcCYyw8KwUNg1AWvoJlU0b39uIPdfPit56FII0z3M22q+L15Nrh3nvPZIp89eEpshS5ql9yzyd/9sFaxNHnTl6istJkcjDPaDpg7pUn6M1FrDQlcz0T/OSvf5rvvHBu8vmHnmVhoMpEKoU4d4FtvUOcX1pG5TyGc3kkhideOccvXreXomMxUA8Y0QpmFunpGqcx0EWPleWhE09w8AO/8Cs/9IX/8KXS9AKjP3b7o6ZSP7wtk/r2bO3codlvPo0nDSPj3dCf/dyjL734KeF5bO8bIDUUfPqR//6H9775zRP8y//6xaNzL72OyuaprfqMbtn1aPhX3zhsSx+nEjGXSZO7ahd2XRw69for7I4Ee6+agFR4vDH36r6wtIDbnWdBZLAndvHEydeQ/TYTuRG25i1x8eTr5uCeLZ/dsSX/2TPnfY4/8Cw1V7FlcASVVscqC+fvCRoFfN0gGpwgc8N+llb9VLVYwkSGm6/ZQ73EkXq5TGPhTaq6gVmdZq6+BI0q260eRLGGbwtWpI/pchiMUlzMehS3D8PunbPYqeMDa4Sff59UE2M8opCwVkOJuPmBVoL3+sOyLPwwziC2lZV0fw/JdqXoGRvHyWc+fWb67L2eK3nftbHpWi4vfn186xh18zKmGOC5dT7+yz/PQJ/7seUKf/rK6dfoHxlgdGIcN5v/dr3QOBSWAnQqx873vR+Genns/z7ChdmLvO/AXv7k859/dMBECFd9G0nhhfMrX7z7nv/IymqRqflpqG+hd3AAa3We4sICrCxx03W3kAtKrFwo8d1Ggb8e7f3SLTfuZcfQdTz02FOHv/vsU/ziJ37u0PbtO7h08g0ahSbZnknws7dGNQvhG3qzXaiGf+vM1BS53hx0p+ovnD6ZurAwx49cvQsdmL0LM5cQwo5b7jg2O/dczXyxsH+1XMRIye4dO+GNNz8VNkN6e/vQ0o4ZxmzBzOIMfq3Mtp1boGQ+9eTDD3P0n39y/4Frr+H83PM0aRAGPru374DF1XsWLszjhQ624xEaj1RuhJdfu0CocnjZDKsNi5UyR6U3xNhkPxO9XVOq99XJ5vQUVlhH6QARKJTQSBMH/EUQN1ZPpVIg3nn0agNApNEFFeg8lQZ+sYxMTCaTdAp5Lz+UUkR+TLJiKwcTaqIwJOd2MTkyzopfuvulk99FBQ4HrtpC9fyCqfnzbN8yRKWxyLDIcuNV27nrjp86VinU7stk0382ff7cXWFUJ9eTAmNNTb105lA68MiNjLD7ox/hxfkZjj/xOOlMio9/4H0M0qw/8z///EizVDq86+aD3HTgwJcHMvIuvxkxV5yDnMvWHdtYvXCWxUuznHz8Ea65+Yf5jY8d5oFj32PmwhR/8Hv/gz/tztKoBxSWFzl08zXsvHrnz8/+RemB+apPPj3Itr23QlkdOn/mIsIP+PBP3Arn547MvHqK9x+4FfLiN+ZKC19y0y4/fOshTMjo1AuvIkKBkTZ4Ht7OnSxemsOvreK6KT74/gOF+sun8kG1TmGpQN0Yth48CGl79sSp50a9qM4Hbroe5hfvPfutJwkO33bkZz74ozz+1EtMF5dwnQa37JmExWWChSJdxiEwCinTMDrJyaeOE2Axc3Ge+37vfoSOWL10kbGuDFdNjEz+/u/9m2NWX/po1YpZgy1pgzTYCKQRSVNtRVdPLyh79h8EkJSyj1OP7mK5gC5VcEzcZTCUGjt872sRIyDSmogIdJwKo4zEjQSWJ+rXbx+ksWpzTY+LM3eRDDVuGB7ih6+ZYLyW45c+dpisCbzHvva1e2657cjhAQv2jvbx0x88CIuFu/yLS4zbOWrlJuS6eGXqNP09WfbsGOUTt3/wywsPff2us399jC7lMrWwyODk9rs+9v7r+LuXTrB1IAvNCivz8+hKhe0D/Vx89RSNJx7nl48eme3SPaN/+6zizKULFFdLSCw+efTnuPNnP4CcPfPA/KunMLUqVrcE26a6VGB7fzdhL2zb5sBCgQlbMZxPYy5d+JJuFNnWleX6iQmUHyEKVfpTOcJqJa47WV4kXCpw09YRPM8jWyrnL7z8Mllhkct2I3XAUH8v1AqjA5bhxsE8t24ZJTx1guFqnZUXnuOG224v3LprLP/S9Dw33bibH9s9TPO1U9ilBWzRoIkVE/aUC6SFz8Ebd3OpWEFaLoEfMrJljMnhYXZsmYDFS0fDWgVDRGQsjDbYSuEphQ40wrZBKfqGRtd4WOqG21OC43X821O4Hc0tq8UsRBBONi7MYTdCHOKOI4GlsUP1ngaHTrpeGBFnAzsoPMumXm9SmJkjc/q1fV/4tU/i5LdB3Wfxob/iwtxZ9t95F1/9z79VsIsqj2zw8h/9wZGzzz3Dj3/gQ/s+/5u/Ru3iLOm+bi48/Df0mohu245jAIUiHz90iI//6I/PEplRXnr6rhPHvkL+0gzdVprV0grc9kF+9RM/w69+4g6awQpcnKMye5FUGJFzPayVZR79/fu4+fwbo3f86j1Td9x+7SSeBdn++/Wl0t0ybELxDZ7+4/tx3jzPrkwGY6pE50+SufV6vvJf/x11vUx3FzR+5zjOxVlKr7+EaHyEY/d9/kEid5SF4v75Rx7CWlzCSbv4zQpp1Qvn3+Dm627ghn//m3i5NNRqiAszqHqNRr1GsVZFnT7F9tIq93/63xaokAfF4p/8EanZGV76q79kePu2/L133wmD489XFp7fn/VXuPjC43SxykC/xLUUge3TfPYJ/vVdHzuFJg8SbUiZyOSVliAV2PZU+MhXJp1imbzXRZfWBEGArWTcxlRGaM8lVNAXN557PlkQU+9UPkQl1Hd6xp9Ty8t/On3/n4zO/NkDjPsBRgZUHE1PozPHjv4BB4g0QZyDpGx0XZO2XHAclkyA1Z9nbscI41ftpV7TVN+cpv7mOWqVVfqvvwYxOk42sLGbq7z2vadZrdYYvuoaxievJpdN06wWOfvMd8gtLOMuFrB6h1geHqHrfTeR682hK6vMvPgtqqen2e3loRFQsgQ9112DtXWcpuOADcXvnUCePsegjvBDH7u/l0IzAjvF2YkRrrlxL07GJet1Q9Vw4YUXqF88jRMW6b5UYCDXxWKxQrF3kOz7b4GJYfygSHH2LBMnSyxU5imnDJmdu8ht2YvSacJLC7C8Qv8bL+M6hsXSCtVUF3psB8PX7Ufl0mDqnH/mBKWzZxnTIeNph5V6haV0GveafTg7tqO0ReX1MwRvTCH9ClEmRTg4xPDV12O8bpS7Qu3cNCvPPk+fX2Kox6EQGVZED7mxvei9W8GyEY6FshyUsmNfMYho1upY02epvTHHmLFIG0MzqKE8hcJgS0E1cDghND/y2d+CQ7d8CM99tKVBQtM4bAmvY8qJqEbmaFpyjNkL5vnP/Be8p16kt1KhLhpUPU1f3XlPA0ToJtqSCNvBVCPSlotUNpeaNUw2zapfIjUyTCUI0atlho2FFxnKUlD2ko3K6jL9noWX7eL8fJncwChR06damKfLk6RMhGMMg/kJzsysoLM5Is9g3CaWgN6mJFMNqKyskB4aoGjFlYFaKkRT01wqsGtgECk0S6UioYS+XC+FS8sUe8foHUoxe+kNdBAyPriNoFwjqCzR06VQvktPStEMKqwaQcXpYaUW0O+65EyIXyjhdtsUdZnA8fByIzgiS3NpiRQRucoixg5oiBDf62a2pLG7+7AcAVGNrO7GChv0uAIT1GiakCCVpqgFVSR2rhuzuMh42iXlGKK0y9RSAeF2o2SawIlQ5RoDQjKQMaS6odhoYuhD0Esxrak06hihEZYgMhpLKixh4TcaKK1J+ZoRJx23THMN2hX4zTpdaZdK2eNF4/MTX/gc3Hj9Rn/hCs0EL98HaXU6jzSVlSI9lsICjIkIee8/pIwzyIQQYCmUkBhtEAnb1I5AUZi7hEw5qKhJulSlz0qRI6IZOdRTmmatxIjK4C/Ms0WncCoVlhfmuHpiCL9ZpujXcFIOs2++xjh9OGWfJpqF8jyh042NhajUyUqB8quI1Qp9aQ9XOeRIUbdcwsIKNRGxdfcOzpw+i6o3GbU88gWJWZ1nKOMiHIfimxdwpUs230sY1SgaF9FooII6dqgZjTIMlDWjVQWFKhe7wQsCVNjAhBFhcZqs00NUrWDTJOVoGpGPm/MwnoUMBEGlQjZt09eTwi8aKg0fIQVCGSxjcNBkhI2X6mK2XGJiaIBUVGd+YRrju1w9OMLybImBrhwNK490fbptSbUxR8PRiEiQwiYqBYxZDuVqhONZOK5Lw6+hgxClDM1mRNiTI5OSpLEIdIiVcyiLBqthE+W4hJYg0grSGdr9j9j51Pm3N7GMudNauXSPO3dx//F/8a/YXaiRWi3j5lIsh3V6wwxgMDKMExCJOfbAiuu2TfSDDRDhr28SmiQl01hruyxaNtiQrmmsZPMyKbdSwdqmpWont0zeq7n1OBaiXaRRCCOTjOM4vUWExKQzRsa8fCIEqTEyrj8vWCmsyMKJFLZWWFrGSZ0CECEVYbB1vJ/k6JjkKLQEvgWhEpiyiyVCLONjE6BEvGkWCgstFaEKkz1ihYh5EBCoNQIhYfx4n1bGYX+dkHMiBVpANkoneZ5izZdDrJMthW6S7tPq2C4Esm3cSJl0M0kWqYROo3VMJnDjDHEVYlSEViImspMOBouuqE5Ngu852FYqLq6TGl+GVIxPtJTi3HgXH/zCfzrl57t/2yV9jGp4JxnrHdaDwJzrut9mubjflgoJuLaD0RpXORC27VCzMSnxPaRHNttdCQDMRnC0pwKsfWajuZkwDa/t9lutL87ouH4e3bb7KRFKJ0yxEi0ESAcIEUICIXaoUDpmrJI67mCPgEgESba03FCDk5DztjUNiAlJEQqdHGtk7OQaIWO6bJHcn5Bt4IjntYS9BgYhBVIlWRYSLBFflxBiDUTtwm4A5dhrwt7qIiKEoMXmYiy1nifXAkcbQJqOjUnoLbSKYi4aLBAWBgsVanQUYjseaS9NVKlTq9XBk9iuTcNzGJoYAyELLkkj7QQcFT+8O+tanQl0PHgUyxlZvDQfJwMKC0tEBEGI7ciE1vfKdAbCvIfAsfkG1wDQ9r5p5QTLtc9c6TtYS3MR4IYSk6TLtIqw4nQY2ZbqEidnChOv4EZYoCOMdHBDvQYORSuTIIxLCmSEMUmRgYyTO4UQSTVnQiXniIR+zUYbiZEGIxVGKLQQyNatINc1iVAxaIxEWXacQiRjYGgVJ+a0WIVVS9OIpBs+MXBaQi4tZ01biDbN0fqPtFtpTmttZ1raCKBu2wgZLyxCtujHLUBiYcebioVCwiuisJWD6ylMSlDFZ8avsG9yO2hzWeRKSd52XyTZOoZyuYrv+0jbIggqMZLDOEXkspyP99omyGWaRK+DBNkGjg26onPQIpk2HSVfsYjb+0cJMIwQCBQRjUTDxKt5zJWamFsROFGIbGVJCxNnOEuBVnFhl4wspCUScssWVbeIWbekiMdQSea0EydGJxnUApByPUoZC7DaIMRGxZ9rCb+QAtWmKWjTFq1zt2sAib3hdcv8amkZhdpgXrVA0jpGOQpa90XsFypaoJSESiJSKYTloo3AlhZONgcpQbW8zFSjwM3jI8DlZDkp2zr+zgASRJPKcvAjDZZLrRmQ8hxMGBdNrYGjLdGP98wGu+wcmzPOleN1SbWiMPLKWLvS/KI9KXKdpNIgEBhM0qrIyKSJHeAqkdTXC0TSUBuV/HJSYCkn0WKJmZMIv5SCuDG6lQBSJUIuQcZd8GNZjO9PJMlsrVU+BolGC7lBuNcEWCQ6R4kE3+IyQddSoIy9fk8tULWbYokIxqZTm5mVnMcSBoGdFEfE4JVGJ9evKfgNnO4uXC+LqEUQBmDZNMMG8+USzs4JvC0TYMWd4euY21OId56LVW+Et6cEje6+XlQmjV8NsZSIw5dtnUDes4rEWG2Cq9u0h9kk9XKTZtFXBFjr463RUKm1ysr4u4xNKomMU/mlhTAqaVSX1MtLjRBRPIulwJjYZjcSJaykb6oNMsQRziYNsNGMsSUYqYiEIpI2UspEfuMKUKHcNpMnpuSOhVwjhEXTyI0aYrOZpORaUZuQG4UbQGprzT9BxhomrhEXG9bo+JyJZm0z0RyiNZNKYCXgSqpLRYRt51BOmkALpJQx5ZuJmF6+xJvFZa79+MdhbORXSKi/W+BoNBqHLU3eSnvHOod5lUrR1fW5nuGhe/NDA1RPv8lQLkfUbGI7LjUTronG5ozZzZWEP7ggkW0gaXeu9KYxuaHOHcyGSsV2/dOqSfHb/JU1tWtIolkSy3HjWYyFkTJuGiEijIz/a1sgtcRogdAWUiiMcLGVQUqNkmpNINuFt+UEu/hxQzzhYJQDUsUgIIpNNuklwiYScCS+VfIdKGWvRZ+EiP2Qdh9Ct5r/tkWxTNt12JFMtEcMipYJFsnExNIbfZgWMFqgUzJMFjGbSKjEiddJ65qQjNdFU0O10iDnpiCVZnFhjpXQp3/PDnYe3A9t7FgtFmGkwDSbk210629hYrUydnu66Bnoo3ryNKmubsrLBVzXgUZcadcOkveWFkmqtzaDZIOZJTcKOPqKNmZ7PUpr4QjlerWZRCfA0GuHW7ZLq8oyZgsOEUKhZRxxalpR7LxHrdXIjhlgW46z0mv+wbrpIxKTRGCbgEBIlFREUsZ1EcIgMCgMWlkJGBLG4DUzKQaJSppKS7MefWoJvxKCUKUuc65b2kYLsLRZK4RrhYl14tcYKbC0anPYxXqJdQsoVhK1wkELK55DRWgVxmCpR6iUR67LIaNSNCs1zs6cxwykuOUjH0b39X6RevVOUjGpUUPrw5oon5bymN2V+9zbSsfiSvmB/nz256kFR1cee/yBE3/xVRrffZGttZCtTo6COR87c1FMkdwiqYlkfNMe3Z132vX39157qDvv4yjVORdN2BtTcTZHaaK2kgEhrvA8KzseH24qOdg87lg9V563JaRO0HmFTHsdry8w3lvOD4CXfut7A0JXdZzfTRi13mo8cjunOpXFAl1uDosUi5dWeHVunmpPN/s+epgtt33os3R3/fa7kQ/L7kqdamh92Evbx3oPXDu1P/Qnz6Qs3njyWV6bOce2iSzCGFQSvjfGrPNtAC6dfwDzfZ4uH71NtEG+TW9i6chNP6rZ8AMr1+4oANrpLGA4Vsdx25Yd5zeqc16ecO2OAJNuruP5haM7A9xuH7v8+KYKOwPE6fz7ZMIuCsUqUzNTLNYa9O/by4EP30b6uqsfxdKpdysfllCq4aNvjUx9JDOSvyf7E4eO7B7I3M32QS6+eprg1JuxrxGZ9epAzDohjfZ/oAGCFG+bitJZxXQW0GhD8fDlz42vOgoouvMKLJqNzgAUnft36HrYWUNYpY4AUaLacTzu0dxBQ0XqbRaQzhp8uVhnNfBJT4yx/8d+hJEDN2B6ur9dttUstprNvVsDvGrMUYCQ5qTUQSErqaOjUQor9xZm58g8/tJajblMTKy4+wdoDDIMfrCDvJZ6VwBpRuEVV9DW86jNxLxskwzQQWcNEQnTWQB1s/O4tN+xBr3SHM1NJujme7CDesf7b5U8v6UJFnbWQPptFrCGm2Vo107Ye9UxutNfxk4d/8eUD1FvNA97btxlrhbVjiKikZS0pgQ6jw4nCXR+vTNCazfSeGs9QgM19wONEEt13k19uzLNINh72eeEqLchrLBhvH0MIGwcfsvzCVFHtyXUXekc4aaxzfOLsGOTZr0pYU+0nUMIUTciGm1/fdl11Ded87JrTGTmrT4TJcwCb3n9b/P9a7tOLvW7AIVG+b7ARPvSyjuWdbz7KTd/nZzzu+8KIFcygcqVwn0G8DzvUd9SkwIaElMQUDeYlAEvglEDntLuXGf52nTD349BrM7Djb/HVPXNxxjw2l9vph2u0rxz4xxtAsrG76411k7ZXYOjnc/f9DprkDb22bbztc7VmnPzta2b0La38fem0X6fNTi6cY6N30Ur7Pp2dMxvqUHC6PBqrfIZ6TmFjJP6szTE+xq16Chpdexdi8eVAFIt135dKTXnpd1j/NPjnx7fx4/FYuXrAz3Zj7VeVyq1u7PZ9P3/aOtnMah/ptvy1kJh9VLjU47jPK+8zuQiDThsYj7U4//0M/3/e4Q0Dlt4j36/Xt/axlyHO1jXKNY/6D6iKndGfuOQk3IfJBXvlFe1f6eW5HO478rE+n9RjeZMxwbFFQAAAABJRU5ErkJggg==" alt="Intech Machine" />
      </div>
      <div class="intech-heading">
        <div class="intech-heading-title">Painel Interativo de <span>Vendas</span></div>
        <div class="intech-heading-subtitle">Acompanhe o desempenho comercial e descubra novas oportunidades.</div>
      </div>
      <div class="intech-topbar-aside">
        <div class="intech-bars"><i></i><i></i><i></i></div>
        <div>Mais dados.<br/>Melhores decisões.</div>
      </div>
    </header>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    /* Remove o botão Deploy / menu do Streamlit, que sobrepunha o cabeçalho */
    [data-testid="stToolbar"], [data-testid="stDeployButton"], [data-testid="stAppDeployButton"],
    [data-testid="stMainMenu"], [data-testid="stStatusWidget"] { display:none !important; }
    [data-testid="stHeader"] { background:transparent !important; height:0 !important; }

    /* Abas: ocupam a largura toda, sem bolinha de rádio */
    div.st-key-aba_ativa, div.st-key-aba_ativa [data-testid="stRadio"] { width:100% !important; }
    div.st-key-aba_ativa [role="radiogroup"] { width:100% !important; display:flex !important; }
    div.st-key-aba_ativa [role="radiogroup"] label { justify-content:center !important; }
    div.st-key-aba_ativa [role="radiogroup"] label > *:not(input):not([data-testid="stMarkdownContainer"]):not(:has([data-testid="stMarkdownContainer"])) { display:none !important; }
    div.st-key-aba_ativa [role="radiogroup"] label [data-testid="stMarkdownContainer"] { margin:0 !important; }

    /* Ícones dos títulos de seção */
    .dash-section-title { display:flex; align-items:center; margin-bottom:6px !important; }
    .dash-section-title .dash-red {
      display:inline-flex; align-items:center; justify-content:center; width:24px; height:24px; flex:0 0 24px;
      border-radius:7px; background:rgba(237,28,36,.14); border:1px solid rgba(237,28,36,.38);
      font-size:.8rem; margin-right:10px !important;
    }
    /* Botão Limpar filtros: contorno vermelho, largura total */
    div.st-key-btn_limpar_filtros button {
      width:100% !important; background:transparent !important; color:#ff343c !important;
      border:1px solid #ed1c24 !important; border-radius:9px !important; min-height:42px !important; font-weight:650 !important;
    }
    div.st-key-btn_limpar_filtros button:hover { background:rgba(237,28,36,.12) !important; border-color:#ff343c !important; }
    /* Filtros da lateral: campo fechado, uma linha, placeholder Todas/Todos */
    [data-testid="stSidebar"] [data-baseweb="select"] > div:first-child { flex-wrap:nowrap !important; overflow:hidden !important; max-height:38px !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] input::placeholder { color:#e4e6e8 !important; opacity:1 !important; }

    /* Ícones das abas (SVG embutido) */
    div.st-key-aba_ativa [role="radiogroup"] label p { display:flex !important; align-items:center !important; justify-content:center !important; gap:0; }
    div.st-key-aba_ativa [role="radiogroup"] label p::before {
      content:""; display:inline-block; width:24px; height:24px; margin-right:12px; flex:0 0 24px;
      background-repeat:no-repeat; background-position:center; background-size:contain;
    }
    div.st-key-aba_ativa [role="radiogroup"] > label:nth-child(1) p::before { background-image:url(\"data:image/svg+xml,%3Csvg%20xmlns=%22http://www.w3.org/2000/svg%22%20viewBox=%220%200%2024%2024%22%20fill=%22none%22%20stroke=%22white%22%20stroke-width=%221.7%22%20stroke-linecap=%22round%22%20stroke-linejoin=%22round%22%3E%3Ccircle%20cx=%228%22%20cy=%227%22%20r=%223%22/%3E%3Cpath%20d=%22M2.5%2019c0-3.3%202.5-5.5%205.5-5.5s5.5%202.2%205.5%205.5%22/%3E%3Cpath%20d=%22M15%2017v-3M18%2017v-6M21%2017v-9%22/%3E%3C/svg%3E\") !important; }
    div.st-key-aba_ativa [role="radiogroup"] > label:nth-child(2) p::before { background-image:url(\"data:image/svg+xml,%3Csvg%20xmlns=%22http://www.w3.org/2000/svg%22%20viewBox=%220%200%2024%2024%22%20fill=%22none%22%20stroke=%22white%22%20stroke-width=%221.7%22%20stroke-linecap=%22round%22%20stroke-linejoin=%22round%22%3E%3Ccircle%20cx=%2212%22%20cy=%228%22%20r=%223.5%22/%3E%3Cpath%20d=%22M5%2020c0-3.9%203.1-7%207-7s7%203.1%207%207%22/%3E%3C/svg%3E\") !important; }
    div.st-key-aba_ativa [role="radiogroup"] > label:nth-child(3) p::before { background-image:url(\"data:image/svg+xml,%3Csvg%20xmlns=%22http://www.w3.org/2000/svg%22%20viewBox=%220%200%2024%2024%22%20fill=%22none%22%20stroke=%22white%22%20stroke-width=%221.7%22%20stroke-linecap=%22round%22%20stroke-linejoin=%22round%22%3E%3Crect%20x=%223%22%20y=%2212%22%20width=%227%22%20height=%227%22/%3E%3Crect%20x=%2214%22%20y=%2212%22%20width=%227%22%20height=%227%22/%3E%3Crect%20x=%228.5%22%20y=%224%22%20width=%227%22%20height=%227%22/%3E%3C/svg%3E\") !important; }
    div.st-key-aba_ativa [role="radiogroup"] > label:nth-child(4) p::before { background-image:url(\"data:image/svg+xml,%3Csvg%20xmlns=%22http://www.w3.org/2000/svg%22%20viewBox=%220%200%2024%2024%22%20fill=%22none%22%20stroke=%22white%22%20stroke-width=%221.7%22%20stroke-linecap=%22round%22%20stroke-linejoin=%22round%22%3E%3Cpath%20d=%22M3%2012V4h8l9%209-8%208z%22/%3E%3Ccircle%20cx=%227.5%22%20cy=%228.5%22%20r=%221.3%22/%3E%3C/svg%3E\") !important; }
    div.st-key-aba_ativa [role="radiogroup"] > label:nth-child(5) p::before { background-image:url(\"data:image/svg+xml,%3Csvg%20xmlns=%22http://www.w3.org/2000/svg%22%20viewBox=%220%200%2024%2024%22%20fill=%22none%22%20stroke=%22white%22%20stroke-width=%221.7%22%20stroke-linecap=%22round%22%20stroke-linejoin=%22round%22%3E%3Cpath%20d=%22M3%2017l5-5%204%204%208-9%22/%3E%3Cpath%20d=%22M15%207h5v5%22/%3E%3C/svg%3E\") !important; }
    div.st-key-aba_ativa [role="radiogroup"] > label:nth-child(6) p::before { background-image:url(\"data:image/svg+xml,%3Csvg%20xmlns=%22http://www.w3.org/2000/svg%22%20viewBox=%220%200%2024%2024%22%20fill=%22none%22%20stroke=%22white%22%20stroke-width=%221.7%22%20stroke-linecap=%22round%22%20stroke-linejoin=%22round%22%3E%3Cpath%20d=%22M12%203%204%207v10l8%204%208-4V7z%22/%3E%3Cpath%20d=%22M4%207l8%204%208-4M12%2011v10%22/%3E%3C/svg%3E\") !important; }
    div.st-key-aba_ativa [role="radiogroup"] label { min-height:58px !important; }
    div.st-key-aba_ativa [role="radiogroup"] label p { font-size:.9rem !important; font-weight:600 !important; }

    /* Títulos da barra lateral com ícone vermelho */
    .sb-title { display:flex; align-items:center; gap:12px; margin:.35rem 0 .6rem; font-size:.98rem; font-weight:650; color:#f4f6f8; letter-spacing:-.01em; }
    .sb-title svg { width:22px; height:22px; flex:0 0 22px; fill:none; stroke:#ed1c24; stroke-width:1.7; stroke-linecap:round; stroke-linejoin:round; }
    [data-testid="stSidebar"] hr { margin:.9rem 0 !important; }

    /* Upload: área tracejada com ícone de nuvem */
    [data-testid="stFileUploaderDropzone"] {
      flex-direction:column !important; align-items:center !important; justify-content:center !important; gap:8px !important;
      padding:20px 10px 16px !important; border:1.5px dashed rgba(160,170,180,.38) !important; border-radius:12px !important;
      background:rgba(255,255,255,.015) !important; text-align:center;
    }
    [data-testid="stFileUploaderDropzone"]::before {
      content:""; width:34px; height:34px; background:url(\"data:image/svg+xml,%3Csvg%20xmlns=%22http://www.w3.org/2000/svg%22%20viewBox=%220%200%2024%2024%22%20fill=%22none%22%20stroke=%22%23e4e6e8%22%20stroke-width=%221.5%22%20stroke-linecap=%22round%22%20stroke-linejoin=%22round%22%3E%3Cpath%20d=%22M7%2018a4.5%204.5%200%200%201-.7-8.9A6%206%200%200%201%2017.7%208.6%204%204%200%200%201%2017%2018%22/%3E%3Cpath%20d=%22M12%2021v-8M9%2015.5l3-3%203%203%22/%3E%3C/svg%3E\") center/contain no-repeat;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] > * { display:none !important; }
    [data-testid="stFileUploaderDropzoneInstructions"]::after {
      content:"Selecione a base Excel ou CSV\\A .xlsx, .xlsm, .xls ou .csv"; white-space:pre-line; display:block;
      font-size:.8rem; line-height:1.7; color:#c5cbd1; text-align:center;
    }
    [data-testid="stFileUploaderDropzone"] button {
      min-height:30px !important; padding:.1rem .8rem !important; font-size:.75rem !important; border-radius:8px !important;
      background:transparent !important; border:1px solid rgba(237,28,36,.7) !important; color:#ff5058 !important;
    }

    /* Tabela de share dos clientes (bolinha na cor da fatia) */
    .share-tbl { width:100%; border-collapse:separate; border-spacing:0; table-layout:fixed; font-size:.8rem; }
    .share-tbl th { text-align:right; font-weight:500; color:#c5cbd1; padding:7px 8px; background:rgba(255,255,255,.045); }
    .share-tbl th:first-child { text-align:left; border-radius:8px 0 0 8px; background:transparent; }
    .share-tbl th:last-child { border-radius:0 8px 8px 0; }
    .share-tbl td { text-align:right; padding:9px 8px; color:#f1f3f4; white-space:nowrap; }
    .share-tbl td:first-child { text-align:left; overflow:hidden; text-overflow:ellipsis; }
    .share-tbl th:nth-child(1), .share-tbl td:nth-child(1) { width:46%; }
    .share-tbl th:nth-child(2), .share-tbl td:nth-child(2) { width:34%; }
    .share-tbl .dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:10px; vertical-align:middle; }

    /* Dia / Mês / Trimestre: botão segmentado à direita, sem bolinha */
    div.st-key-dashboard_tempo { display:flex; justify-content:flex-end; margin:0 !important; position:static; }
    div.st-key-dashboard_tempo [data-testid="stRadio"] { width:auto !important; }
    div.st-key-dashboard_tempo [role="radiogroup"] {
      display:inline-flex !important; gap:0 !important; padding:0 !important; overflow:hidden;
      border:1px solid #2a3036 !important; border-radius:8px !important; background:#0a0d0f !important;
    }
    div.st-key-dashboard_tempo [role="radiogroup"] label {
      margin:0 !important; padding:5px 15px !important; min-height:0 !important; border-radius:0 !important;
      background:transparent !important; justify-content:center !important;
    }
    div.st-key-dashboard_tempo [role="radiogroup"] label p { font-size:.8rem !important; color:#e4e6e8 !important; margin:0 !important; }
    div.st-key-dashboard_tempo [role="radiogroup"] label > *:not(input):not([data-testid="stMarkdownContainer"]):not(:has([data-testid="stMarkdownContainer"])) { display:none !important; }
    div.st-key-dashboard_tempo [role="radiogroup"] label:has(input:checked) { background:linear-gradient(135deg,#f21b23,#be0c15) !important; }
    div.st-key-dashboard_tempo [role="radiogroup"] label:has(input:checked) p { color:#fff !important; font-weight:650 !important; }

    /* Esconde a bolinha de rádio em qualquer versão do Streamlit (abas e Dia/Mês/Trimestre) */
    div.st-key-aba_ativa [role="radiogroup"] label > div:first-child:not(:has([data-testid="stMarkdownContainer"])),
    div.st-key-dashboard_tempo [role="radiogroup"] label > div:first-child:not(:has([data-testid="stMarkdownContainer"])),
    div.st-key-aba_ativa [role="radiogroup"] label [data-baseweb="radio"] > div:first-child:not(:has([data-testid="stMarkdownContainer"])),
    div.st-key-dashboard_tempo [role="radiogroup"] label [data-baseweb="radio"] > div:first-child:not(:has([data-testid="stMarkdownContainer"])),
    div.st-key-aba_ativa [role="radiogroup"] label > span:empty,
    div.st-key-dashboard_tempo [role="radiogroup"] label > span:empty { display:none !important; }
    div.st-key-dashboard_tempo [role="radiogroup"] label::before,
    div.st-key-dashboard_tempo [role="radiogroup"] label::after { display:none !important; }

    /* Reserva espaço no fim da página: o botão flutuante "Assistente de IA" e o rodapé fixo não cobrem os cartões */
    [data-testid="stMainBlockContainer"], .block-container { padding-bottom: 110px !important; }
    .dash-section-title { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; min-width:0; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="sb-title"><svg viewBox="0 0 24 24"><circle cx="12" cy="8" r="3.5"/><path d="M5 20c0-3.9 3.1-7 7-7s7 3.1 7 7"/></svg><span>1. Carregar dados</span></div>', unsafe_allow_html=True)
    arquivo = st.file_uploader("Selecione a base Excel ou CSV", type=["xlsx", "xlsm", "xls", "csv"], label_visibility="collapsed")
    st.divider()
    st.markdown('<div class="sb-title"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M2 12h3M19 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1"/></svg><span>2. Configurações</span></div>', unsafe_allow_html=True)

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
        "regiao": localizar_coluna(df, ["Região", "Regiao", "Gerente região Cadastro", "Gerente região"]),
        "tipo_operacao": localizar_coluna(df, ["Tipo de Operação", "Tipo Operação"]),
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
    df["_regiao"] = texto(col.get("regiao") or col.get("gerente_regiao"))

    # Mantém as datas originais separadas. Nunca preenche _data_faturamento com Ano/Mês,
    # pois isso faria uma linha sem data de faturamento parecer faturada.
    if col["data_faturamento"]:
        data_faturamento = pd.to_datetime(
            df[col["data_faturamento"]], errors="coerce", dayfirst=True
        )
    else:
        data_faturamento = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    if col["data_pedido"]:
        data_pedido = pd.to_datetime(
            df[col["data_pedido"]], errors="coerce", dayfirst=True
        )
    else:
        data_pedido = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    df["_data_faturamento"] = data_faturamento
    df["_data_pedido"] = data_pedido

    data_ano_mes = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    if col["ano"] and col["mes"]:
        ano = pd.to_numeric(df[col["ano"]], errors="coerce")
        mes = pd.to_numeric(df[col["mes"]], errors="coerce")
        validos = ano.notna() & mes.between(1, 12)
        if validos.any():
            data_ano_mes.loc[validos] = pd.to_datetime(
                ano.loc[validos].astype(int).astype(str) + "-"
                + mes.loc[validos].astype(int).astype(str).str.zfill(2) + "-01",
                errors="coerce",
            )
    df["_data_ano_mes"] = data_ano_mes
    df["_data"] = data_faturamento.combine_first(data_pedido).combine_first(data_ano_mes)
    data_padrao = "Data Faturamento; Data Pedido quando a data de faturamento estiver vazia"

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

# Data automática por registro: cada linha usa Data Faturamento quando preenchida;
# se estiver vazia ou inválida, usa Data Pedido. Ano/Mês é o último recurso.
# Não há seletor manual porque a regra deve ser aplicada linha a linha na base inteira.
df["_data"] = (
    df["_data_faturamento"]
    .combine_first(df["_data_pedido"])
    .combine_first(df["_data_ano_mes"])
)
data_padrao = "Data Faturamento; Data Pedido quando a data de faturamento estiver vazia"

with st.sidebar:
    st.caption("Data utilizada nos filtros e análises")
    st.markdown(
        "<div style=\"padding:9px 11px;border:1px solid #30373d;border-radius:9px;background:#0e1214;color:#f5f7f8;font-size:0.84rem;\">"
        "Automática por pedido: usa <b>Data Faturamento</b> quando existe; caso contrário, usa <b>Data Pedido</b>."
        "</div>",
        unsafe_allow_html=True,
    )

# Filtros globais
# Os filtros (exceto o período) são guardados para reaplicar no período anterior dos KPIs.
filtros_aplicados = []
periodo_atual = None
with st.sidebar:
    st.markdown('<div class="sb-title"><svg viewBox="0 0 24 24"><rect x="3.5" y="5" width="17" height="15" rx="2"/><path d="M3.5 10h17M8 3v4M16 3v4"/></svg><span>Filtros globais</span></div>', unsafe_allow_html=True)
    data_min = df["_data"].min()
    data_max = df["_data"].max()
    if pd.notna(data_min) and pd.notna(data_max):
        intervalo = st.date_input(
            "Período de análise",
            value=(data_min.date(), data_max.date()),
            min_value=data_min.date(),
            max_value=data_max.date(),
            format="DD/MM/YYYY",
        )
        if isinstance(intervalo, (tuple, list)) and len(intervalo) == 2:
            inicio, fim = pd.Timestamp(intervalo[0]), pd.Timestamp(intervalo[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            filtrado = df[df["_data"].between(inicio, fim, inclusive="both")].copy()
            periodo_atual = (inicio, fim)
        else:
            filtrado = df.copy()
        st.caption("Para o comparativo com o ano anterior, mantenha no período pelo menos o último trimestre e o mesmo trimestre do ano anterior.")
    else:
        filtrado = df.copy()
        st.caption("Não foi possível identificar datas válidas; os gráficos usarão toda a base.")

    if col["empresa"]:
        # Inclui registros sem empresa preenchida no filtro, em vez de descartá-los.
        rotulo_empresa_vazia = "Empresa não informada"
        empresa_filtro = filtrado[col["empresa"]].fillna("").astype(str).str.strip()
        tem_empresa_vazia = empresa_filtro.eq("").any()
        opcoes_empresa = sorted(empresa_filtro[empresa_filtro.ne("")].unique().tolist())
        if tem_empresa_vazia:
            opcoes_empresa.append(rotulo_empresa_vazia)
        selecionadas_empresa = st.multiselect(
            "Empresa", opcoes_empresa, default=[], placeholder="Todas"
        )
        if selecionadas_empresa:  # sem seleção = todas (não filtra nada)
            mascara_empresa = empresa_filtro.isin(selecionadas_empresa)
            if rotulo_empresa_vazia in selecionadas_empresa:
                mascara_empresa |= empresa_filtro.eq("")
            filtrado = filtrado[mascara_empresa]

            def _filtro_empresa(b, c=col["empresa"], sel=list(selecionadas_empresa), vazio=rotulo_empresa_vazia):
                serie = b[c].fillna("").astype(str).str.strip()
                m = serie.isin(sel)
                if vazio in sel:
                    m = m | serie.eq("")
                return b[m]

            filtros_aplicados.append(_filtro_empresa)
    if col["status_faturamento"]:
        opcoes_status = sorted(filtrado[col["status_faturamento"]].dropna().astype(str).unique())
        if opcoes_status:
            status_escolhidos = st.multiselect("Status de faturamento", opcoes_status, default=[], placeholder="Todos")
            if status_escolhidos:  # sem seleção = todos (não filtra nada)
                filtrado = filtrado[filtrado[col["status_faturamento"]].astype(str).isin(status_escolhidos)]
                filtros_aplicados.append(lambda b, c=col["status_faturamento"], sel=list(status_escolhidos): b[b[c].astype(str).isin(sel)])
    if col["departamento"]:
        opcoes_departamento = sorted(filtrado[col["departamento"]].dropna().astype(str).unique())
        if opcoes_departamento:
            departamentos_escolhidos = st.multiselect("Departamento", opcoes_departamento, default=[], placeholder="Todos")
            if departamentos_escolhidos:  # sem seleção = todos (não filtra nada)
                filtrado = filtrado[filtrado[col["departamento"]].astype(str).isin(departamentos_escolhidos)]
                filtros_aplicados.append(lambda b, c=col["departamento"], sel=list(departamentos_escolhidos): b[b[c].astype(str).isin(sel)])
    if col["gerente_regiao"]:
        opcoes_gerente = sorted(filtrado[col["gerente_regiao"]].dropna().astype(str).unique())
        if opcoes_gerente:
            gerentes_escolhidos = st.multiselect("Gerente região", opcoes_gerente, default=[], placeholder="Todos")
            if gerentes_escolhidos:  # sem seleção = todos (não filtra nada)
                filtrado = filtrado[filtrado[col["gerente_regiao"]].astype(str).isin(gerentes_escolhidos)]
                filtros_aplicados.append(lambda b, c=col["gerente_regiao"], sel=list(gerentes_escolhidos): b[b[c].astype(str).isin(sel)])
    if col["tipo_operacao"]:
        opcoes_operacao = sorted(filtrado[col["tipo_operacao"]].dropna().astype(str).unique())
        if opcoes_operacao:
            operacoes_escolhidas = st.multiselect(
                "Tipo de Operação", opcoes_operacao, default=[], placeholder="Todos"
            )
            if operacoes_escolhidas:  # sem seleção = todos (não filtra nada)
                filtrado = filtrado[
                    filtrado[col["tipo_operacao"]].astype(str).isin(operacoes_escolhidas)
                ]
                filtros_aplicados.append(lambda b, c=col["tipo_operacao"], sel=list(operacoes_escolhidas): b[b[c].astype(str).isin(sel)])


def renderizar_assistente_ia():
    # O assistente recebe a base completa; os filtros do dashboard afetam apenas
    # os gráficos e tabelas. As consultas da IA aplicam seus próprios filtros
    # conforme a pergunta do usuário.
    exibir_assistente_ia(
        df_vendas=df,
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


# Navegação horizontal principal, imediatamente abaixo do cabeçalho.
if "_aba_pendente" in st.session_state:
    st.session_state["aba_ativa"] = st.session_state.pop("_aba_pendente")

_rotulos_abas = {
    ABA_VENDEDOR: "Vendas por vendedor",
    ABA_CLIENTE: "Vendas por cliente",
    ABA_ITENS: "Itens por segmento",
    ABA_PRECO: "Preço médio mensal",
    ABA_PROJECOES: "Projeções mensais",
    ABA_DADOS: "Conferência dos dados",
}
aba_ativa = st.radio(
    "Seção",
    ABAS,
    horizontal=True,
    key="aba_ativa",
    label_visibility="collapsed",
    format_func=lambda valor: _rotulos_abas.get(valor, valor),
)

# O assistente é renderizado antes do conteúdo específico da tela. Assim ele continua
# disponível ao trocar de seção e não depende da execução dos gráficos daquela aba.
renderizar_assistente_ia()

if filtrado.empty:
    st.warning("Nenhum registro encontrado com os filtros selecionados.")
    st.stop()

total_vendas = filtrado["_valor"].sum()
total_qtde = filtrado["_quantidade"].sum()
clientes_count = filtrado["_cliente"].nunique()
produtos_count = filtrado["_produto"].nunique()


# ---------- KPIs com variação vs. período anterior ----------
# 1) Período anterior = intervalo de mesmo tamanho imediatamente antes do filtrado, com os mesmos filtros.
# 2) Se a base não tem dados nesse intervalo (ex.: o período filtrado já cobre a base inteira), compara
#    a 2ª metade do período com a 1ª metade, e o cartão passa a dizer isso.
def _metricas(base):
    return {
        "valor": base["_valor"].sum(),
        "qtde": base["_quantidade"].sum(),
        "clientes": base["_cliente"].nunique(),
        "produtos": base["_produto"].nunique(),
    }


def comparativo_kpis():
    """Compara o trimestre de referência com o mesmo intervalo do ano anterior.

    O trimestre é determinado pela data final selecionada no filtro global. A comparação
    vai do primeiro dia desse trimestre até a data final selecionada, contra as mesmas
    datas do ano anterior. Os demais filtros (empresa, vendedor etc.) são reaplicados.
    Não existe fallback para comparação entre metades.
    """
    if periodo_atual is None:
        return None

    _, fim_selecionado = periodo_atual
    if pd.isna(fim_selecionado):
        return None

    fim_ref = pd.Timestamp(fim_selecionado)
    # Início do trimestre referente à data final selecionada (jan/abr/jul/out).
    mes_inicio_trimestre = ((fim_ref.month - 1) // 3) * 3 + 1
    inicio_trimestre = pd.Timestamp(year=fim_ref.year, month=mes_inicio_trimestre, day=1)

    # Mesmo período do ano anterior; DateOffset trata corretamente anos bissextos.
    inicio_ant = inicio_trimestre - pd.DateOffset(years=1)
    fim_ant = fim_ref - pd.DateOffset(years=1)

    base_atual = df[df["_data"].between(inicio_trimestre, fim_ref, inclusive="both")].copy()
    base_anterior = df[df["_data"].between(inicio_ant, fim_ant, inclusive="both")].copy()
    for aplicar in filtros_aplicados:
        base_atual = aplicar(base_atual)
        base_anterior = aplicar(base_anterior)

    if base_atual.empty or base_anterior.empty:
        return {
            "legenda": "trimestre atual vs. mesmo período do ano anterior",
            "rotulo": (
                f"Atual: {inicio_trimestre:%d/%m/%Y} a {fim_ref:%d/%m/%Y} | "
                f"Ano anterior: {inicio_ant:%d/%m/%Y} a {fim_ant:%d/%m/%Y}"
            ),
            "atual": _metricas(base_atual),
            "anterior": _metricas(base_anterior) if not base_anterior.empty else {},
        }

    return {
        "legenda": "trimestre atual vs. mesmo período do ano anterior",
        "rotulo": (
            f"Atual: {inicio_trimestre:%d/%m/%Y} a {fim_ref:%d/%m/%Y} | "
            f"Ano anterior: {inicio_ant:%d/%m/%Y} a {fim_ant:%d/%m/%Y}"
        ),
        "atual": _metricas(base_atual),
        "anterior": _metricas(base_anterior) if not base_anterior.empty else {},
    }


def variacao_html(atual, anterior, rotulo=None, legenda="vs. período anterior"):
    dica = f' title="Período comparado: {rotulo}"' if rotulo else ""
    if anterior is None or not anterior:
        return '<div class="kpi-delta kpi-delta-nd" title="Sem dados no período anterior para comparar"><span>—</span><small>vs. período anterior</small></div>'
    var = (atual - anterior) / abs(anterior) * 100
    if var > 0:
        cls, seta = "kpi-up", "▲"
    elif var < 0:
        cls, seta = "kpi-down", "▼"
    else:
        cls, seta = "kpi-flat", "▬"
    txt = f"{var:+.1f}%".replace(".", ",")
    return f'<div class="kpi-delta"{dica}><span class="{cls}">{seta} {txt}</span><small>{legenda}</small></div>'


_ICONES_KPI = {
    "moeda": '<svg viewBox="0 0 24 24"><ellipse cx="12" cy="6" rx="7" ry="3"/><path d="M5 6v6c0 1.7 3.1 3 7 3s7-1.3 7-3V6"/><path d="M5 12v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6"/></svg>',
    "caixa": '<svg viewBox="0 0 24 24"><path d="M12 3 3.5 7.5v9L12 21l8.5-4.5v-9z"/><path d="M3.5 7.5 12 12l8.5-4.5M12 12v9"/><path d="m7.8 5.3 8.5 4.6"/></svg>',
    "clientes": '<svg viewBox="0 0 24 24"><circle cx="9" cy="8" r="3.2"/><path d="M3 20c0-3.3 2.7-6 6-6s6 2.7 6 6"/><circle cx="17" cy="9" r="2.5"/><path d="M16 14.2c2.9.3 5 2.6 5 5.8"/></svg>',
    "cubo": '<svg viewBox="0 0 24 24"><path d="M12 3 4 7.5v9L12 21l8-4.5v-9z"/><path d="M4 7.5 12 12l8-4.5M12 12v9"/></svg>',
}


def cartao_kpi(icone, rotulo, valor, atual, anterior, periodo=None, legenda="vs. período anterior"):
    return (
        '<div class="kpi-card">'
        f'<div class="kpi-icon">{_ICONES_KPI[icone]}</div>'
        '<div class="kpi-body">'
        f'<div class="kpi-label">{rotulo}</div>'
        f'<div class="kpi-value">{valor}</div>'
        f'{variacao_html(atual, anterior, periodo, legenda)}'
        '</div></div>'
    )


_comp = comparativo_kpis() or {}
_ant = _comp.get("anterior", {})
_atu = _comp.get("atual", {})
_per = _comp.get("rotulo")
_leg = _comp.get("legenda", "vs. período anterior")
st.markdown(
    """
    <style>
    .kpi-grid { display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:14px; margin:6px 0 16px; }
    .kpi-card {
      position:relative; overflow:hidden; display:flex; align-items:flex-start; gap:16px;
      padding:20px 22px; border-radius:14px; border:1px solid rgba(160,170,180,.18);
      background:linear-gradient(135deg, #0d1114 0%, #090b0d 62%, rgba(237,28,36,.16) 140%);
    }
    .kpi-icon { flex:0 0 auto; width:46px; height:46px; display:flex; align-items:center; justify-content:center; }
    .kpi-icon svg { width:42px; height:42px; fill:none; stroke:#ed1c24; stroke-width:1.6; stroke-linecap:round; stroke-linejoin:round; }
    .kpi-body { min-width:0; }
    .kpi-label { color:#e1e3e5; font-size:.92rem; font-weight:500; }
    .kpi-value { color:#fff; font-size:clamp(1.35rem,2vw,2.05rem); font-weight:750; letter-spacing:-.04em; line-height:1.2; margin:6px 0 10px; white-space:nowrap; }
    .kpi-delta { display:flex; align-items:center; gap:12px; flex-wrap:wrap; font-size:.8rem; }
    .kpi-delta small { color:#9ea5ac; font-size:.74rem; }
    .kpi-up { color:#ff343c; font-weight:700; }
    .kpi-down { color:#aab2ba; font-weight:700; }
    .kpi-flat, .kpi-delta-nd span { color:#aab2ba; font-weight:700; }
    @media (max-width:1100px) { .kpi-grid { grid-template-columns:repeat(2, minmax(0,1fr)); } }
    @media (max-width:560px) { .kpi-grid { grid-template-columns:1fr; } .kpi-card { padding:14px; } }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="kpi-grid">'
    + cartao_kpi("moeda", "Faturamento analisado", moeda(total_vendas), _atu.get("valor", total_vendas), _ant.get("valor"), _per, _leg)
    + cartao_kpi("caixa", "Quantidade vendida", f"{total_qtde:,.0f}".replace(",", "."), _atu.get("qtde", total_qtde), _ant.get("qtde"), _per, _leg)
    + cartao_kpi("clientes", "Clientes gerenciais", f"{clientes_count:,}".replace(",", "."), _atu.get("clientes", clientes_count), _ant.get("clientes"), _per, _leg)
    + cartao_kpi("cubo", "Produtos vendidos", f"{produtos_count:,}".replace(",", "."), _atu.get("produtos", produtos_count), _ant.get("produtos"), _per, _leg)
    + "</div>",
    unsafe_allow_html=True,
)

if st.sidebar.button("Limpar filtros", key="btn_limpar_filtros", use_container_width=True):
    for _chave in CHAVES_PERSISTENTES:
        st.session_state.pop(_chave, None)
    st.rerun()



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
    exibir_grafico_padronizado(formatar_eixos(fig), use_container_width=True, key=f"{chave}_{granularidade}_{medida}")


if aba_ativa == ABA_PROJECOES:
    st.markdown(
        '<div class="dash-section-title"><span class="dash-red">◌</span>Projeções probabilísticas mensais · Produto / Região / Vendedor</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Previsões calculadas localmente mês a mês, sem dados externos. Você pode analisar o total ou separar por produto, região e vendedor. "
        "As faixas mostram a incerteza estimada a partir dos erros históricos; não são garantias de resultado."
    )

    # Escolha da dimensão permite projetar cada série comercial separadamente.
    dimensoes_proj = {"Visão geral": None}
    if col_produto:
        dimensoes_proj["Produto"] = "_produto"
    if col.get("regiao") or col.get("gerente_regiao"):
        dimensoes_proj["Região"] = "_regiao"
    if col_vendedor:
        dimensoes_proj["Vendedor"] = "_vendedor"

    # Opções em faixa horizontal para deixar as análises segmentadas fáceis de localizar.
    dimensao_proj = st.radio(
        "ANALISAR PROJEÇÃO POR",
        list(dimensoes_proj),
        horizontal=True,
        key="tipo_analise_projecao_mensal",
        label_visibility="visible",
    )
    st.caption("Cada opção calcula uma série mensal independente a partir dos filtros globais ativos.")

    c_metrica, c_item, c_horiz, c_sim = st.columns([1.25, 1.7, 1.0, 1.0], gap="small")
    with c_metrica:
        metrica_proj = st.selectbox(
            "Métrica para validação",
            ["Faturamento", "Quantidade vendida"],
            key="metrica_projecao_mensal",
            help="Compare a qualidade da previsão em reais ou em unidades vendidas."
        )
    coluna_valor_proj = "_valor" if metrica_proj == "Faturamento" else "_quantidade"
    rotulo_metrica_proj = "Faturamento (R$)" if metrica_proj == "Faturamento" else "Quantidade vendida (unidades)"
    rotulo_realizado_proj = "Faturamento realizado (R$)" if metrica_proj == "Faturamento" else "Quantidade vendida (unidades)"
    rotulo_unidade_proj = "R$" if metrica_proj == "Faturamento" else "unidades"
    base_proj = filtrado.copy()
    coluna_grupo_proj = dimensoes_proj[dimensao_proj]
    item_proj = None
    if coluna_grupo_proj:
        opcoes_itens = sorted(base_proj[coluna_grupo_proj].fillna("(Não informado)").astype(str).unique().tolist())
        if not opcoes_itens:
            st.warning(f"Não há valores disponíveis para a dimensão {dimensao_proj.lower()} nos filtros ativos.")
            st.stop()
        with c_item:
            item_proj = st.selectbox(
                f"Selecionar {dimensao_proj.lower()}", opcoes_itens,
                key=f"item_projecao_{dimensao_proj.lower().replace(' ', '_')}"
            )
        base_proj = base_proj[base_proj[coluna_grupo_proj].fillna("(Não informado)").astype(str) == item_proj].copy()
    else:
        with c_item:
            st.text_input("Série projetada", value=f"{metrica_proj} total filtrado", disabled=True)

    with c_horiz:
        horizonte_proj = st.selectbox("Horizonte", [3, 6, 12, 18, 24], index=2,
                                      format_func=lambda n: f"{n} meses", key="horizonte_projecao_mensal")
    with c_sim:
        simulacoes_proj = st.selectbox("Simulações", [1000, 3000, 5000], index=1,
                                       format_func=lambda n: f"{n:,}".replace(",", "."), key="simulacoes_projecao_mensal")

    st.caption(f"Série: **{dimensao_proj}**" + (f" — **{item_proj}**" if item_proj is not None else " — faturamento geral") +
               f" · Métrica: **{metrica_proj}** · {len(base_proj):,} registros após os filtros ativos".replace(",", "."))

    try:
        resultado_proj = projetar_meses(
            base_proj, horizonte=int(horizonte_proj), simulacoes=int(simulacoes_proj),
            coluna_data="_data", coluna_valor=coluna_valor_proj,
        )
    except Exception as exc:
        resultado_proj = {"ok": False, "erro": f"Não foi possível calcular a projeção: {exc}"}

    if not resultado_proj.get("ok"):
        st.warning(resultado_proj.get("erro", "Não há dados suficientes para projetar."))
        st.info(
            "Para uma projeção mensal inicial, selecione uma série com pelo menos 12 meses encerrados. "
            "Quando a análise for por produto, região ou vendedor, escolha um item com histórico suficiente. "
            "Se necessário, amplie o período nos filtros globais."
        )
    else:
        previsoes_proj = resultado_proj["previsoes"].copy()
        historico_proj = resultado_proj["historico_df"].copy()
        # A engine usa nomes genéricos de faturamento internamente; a interface
        # apresenta a unidade correta conforme a métrica selecionada.
        coluna_historico_engine = "Faturamento realizado (R$)"
        historico_proj = historico_proj.rename(columns={coluna_historico_engine: rotulo_realizado_proj})
        comparacao_proj = resultado_proj["comparacao"].copy()
        unidade_erro = "R$" if metrica_proj == "Faturamento" else "unidades"
        comparacao_proj = comparacao_proj.rename(columns={
            "MAE (R$)": f"MAE ({unidade_erro})",
            "Viés médio (R$)": f"Viés médio ({unidade_erro})",
        })
        wape_proj, mae_proj = resultado_proj.get("wape"), resultado_proj.get("mae")

        m1, m2, m3, m4 = st.columns(4, gap="small")
        m1.metric("Modelo selecionado", resultado_proj["modelo_rotulo"])
        m2.metric("Erro WAPE no backtest", f"{wape_proj:.1f}%" if pd.notna(wape_proj) and wape_proj != float("inf") else "N/D")
        m3.metric("Erro médio absoluto", (moeda(mae_proj) if metrica_proj == "Faturamento" else f"{mae_proj:,.1f} unidades".replace(",", ".")) if pd.notna(mae_proj) else "N/D")
        m4.metric("Meses históricos", str(resultado_proj["n_meses"]))

        if resultado_proj["n_meses"] < 24:
            st.warning("O histórico ainda é curto para avaliar com segurança a sazonalidade anual. As projeções e intervalos devem ser interpretados com cautela.")
        if resultado_proj.get("mes_parcial") is not None:
            mes_p = resultado_proj["mes_parcial"]
            st.info(f"O mês atual ({mes_p.strftime('%m/%Y')}) está em andamento. O valor parcial não treina o modelo; a projeção representa o total esperado para o mês completo.")
        ultima_data_proj = resultado_proj.get("ultima_data")
        if ultima_data_proj is not None:
            ultimo_mes_base = pd.Timestamp(ultima_data_proj).to_period("M")
            mes_corrente = pd.Timestamp.today().to_period("M")
            if ultimo_mes_base < mes_corrente - 1:
                st.warning(
                    f"A série selecionada termina em {ultimo_mes_base.strftime('%m/%Y')}, mas o mês atual é "
                    f"{mes_corrente.strftime('%m/%Y')}. A projeção parte do último mês da base; atualize os dados "
                    "ou confira os filtros para que o resultado represente o futuro operacional."
                )

        fig_proj = go.Figure()
        hist_fechado = historico_proj[historico_proj["Tipo"] == "Realizado"]
        hist_parcial = historico_proj[historico_proj["Tipo"] == "Realizado parcial"]
        if not hist_fechado.empty:
            fig_proj.add_trace(go.Scatter(
                x=hist_fechado["Mês"], y=hist_fechado[rotulo_realizado_proj],
                name=f"{metrica_proj} realizado", mode="lines+markers",
                line=dict(color="#9298a1", width=2), marker=dict(size=5),
                hovertemplate=("%{x}<br>Realizado: R$ %{y:,.2f}<extra></extra>" if metrica_proj == "Faturamento" else "%{x}<br>Quantidade realizada: %{y:,.0f} unidades<extra></extra>"),
            ))
        fig_proj.add_trace(go.Scatter(
            x=previsoes_proj["Mês"], y=previsoes_proj["Limite superior 90% (P95)"],
            name="Limite superior 90%", mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig_proj.add_trace(go.Scatter(
            x=previsoes_proj["Mês"], y=previsoes_proj["Limite inferior 90% (P05)"],
            name="Intervalo probabilístico 90%", mode="lines", fill="tonexty",
            fillcolor="rgba(120,128,140,0.18)", line=dict(width=0),
            hovertemplate=("%{x}<br>Limite inferior 90%: R$ %{y:,.2f}<extra></extra>" if metrica_proj == "Faturamento" else "%{x}<br>Limite inferior 90%: %{y:,.0f} unidades<extra></extra>"),
        ))
        fig_proj.add_trace(go.Scatter(
            x=previsoes_proj["Mês"], y=previsoes_proj["Limite superior 80% (P90)"],
            name="Limite superior 80%", mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig_proj.add_trace(go.Scatter(
            x=previsoes_proj["Mês"], y=previsoes_proj["Limite inferior 80% (P10)"],
            name="Intervalo probabilístico 80%", mode="lines", fill="tonexty",
            fillcolor="rgba(237,28,36,0.18)", line=dict(width=0),
            hovertemplate=("%{x}<br>Limite inferior 80%: R$ %{y:,.2f}<extra></extra>" if metrica_proj == "Faturamento" else "%{x}<br>Limite inferior 80%: %{y:,.0f} unidades<extra></extra>"),
        ))
        fig_proj.add_trace(go.Scatter(
            x=previsoes_proj["Mês"], y=previsoes_proj["Previsão central (P50)"],
            name="Projeção central (P50)", mode="lines+markers",
            line=dict(color="#ed1c24", width=3), marker=dict(size=7, color="#ed1c24"),
            hovertemplate=("%{x}<br>Projeção central: R$ %{y:,.2f}<extra></extra>" if metrica_proj == "Faturamento" else "%{x}<br>Projeção central: %{y:,.0f} unidades<extra></extra>"),
        ))
        if not hist_parcial.empty:
            fig_proj.add_trace(go.Scatter(
                x=hist_parcial["Mês"], y=hist_parcial[rotulo_realizado_proj],
                name="Realizado parcial", mode="markers", marker=dict(color="#ffb347", size=9, symbol="diamond"),
                hovertemplate=("%{x}<br>Realizado parcial: R$ %{y:,.2f}<extra></extra>" if metrica_proj == "Faturamento" else "%{x}<br>Quantidade parcial: %{y:,.0f} unidades<extra></extra>"),
            ))
        eixo_meses = hist_fechado["Mês"].tolist() + previsoes_proj["Mês"].tolist()
        if not hist_parcial.empty:
            eixo_meses.insert(len(hist_fechado), hist_parcial.iloc[0]["Mês"])
        padronizar_grafico(fig_proj, altura=430, margem=dict(l=12, r=12, t=40, b=35), legenda=True)
        fig_proj.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="Mês", yaxis_title=rotulo_metrica_proj,
        )
        fig_proj.update_xaxes(showgrid=False, categoryorder="array", categoryarray=eixo_meses)

        # Recalcula explicitamente a escala Y em toda alteração de métrica/série.
        # Isso evita que o gráfico mantenha uma faixa anterior (ex.: milhões de R$
        # ao alternar para quantidade vendida) após a troca dos seletores.
        valores_eixo = []
        if not hist_fechado.empty:
            valores_eixo.extend(pd.to_numeric(
                hist_fechado[rotulo_realizado_proj], errors="coerce"
            ).dropna().tolist())
        if not hist_parcial.empty:
            valores_eixo.extend(pd.to_numeric(
                hist_parcial[rotulo_realizado_proj], errors="coerce"
            ).dropna().tolist())
        for nome_coluna in (
            "Previsão central (P50)",
            "Limite inferior 80% (P10)", "Limite superior 80% (P90)",
            "Limite inferior 90% (P05)", "Limite superior 90% (P95)",
        ):
            if nome_coluna in previsoes_proj:
                valores_eixo.extend(pd.to_numeric(
                    previsoes_proj[nome_coluna], errors="coerce"
                ).dropna().tolist())
        valores_eixo = np.asarray(valores_eixo, dtype=float)
        valores_eixo = valores_eixo[np.isfinite(valores_eixo)]

        if valores_eixo.size:
            minimo = float(np.min(valores_eixo))
            maximo = float(np.max(valores_eixo))
            amplitude = maximo - minimo
            margem = max(amplitude * 0.08, abs(maximo) * 0.03, abs(minimo) * 0.03, 1.0)
            limite_inferior = 0.0 if minimo >= 0 else minimo - margem
            limite_superior = maximo + margem
            if limite_superior <= limite_inferior:
                limite_superior = limite_inferior + 1.0
        else:
            limite_inferior, limite_superior = 0.0, 1.0

        # A chave de revisão muda com a métrica e a série; uma escala usada em
        # faturamento não deve ser reaproveitada ao visualizar unidades vendidas.
        assinatura_grafico = f"{metrica_proj}|{dimensao_proj}|{item_proj or 'geral'}"
        fig_proj.update_layout(
            uirevision=assinatura_grafico,
            yaxis=dict(
                range=[limite_inferior, limite_superior],
                autorange=False,
                nticks=7,
                gridcolor="#252a2f",
                zerolinecolor="#3d4248",
                automargin=True,
            ),
        )
        if metrica_proj == "Faturamento":
            fig_proj.update_yaxes(tickprefix="R$ ", ticksuffix="", tickformat="~s")
        else:
            # Quantidade é exibida como número de unidades, nunca na escala monetária.
            fig_proj.update_yaxes(tickprefix="", ticksuffix="", tickformat=",.0f")
        exibir_grafico_padronizado(
            fig_proj, use_container_width=True, config={"displayModeBar": False},
            key="grafico_projecao_probabilistica_mensal",
        )

        st.subheader("Projeção mês a mês")
        tabela_proj = previsoes_proj[[
            "Mês", "Previsão central (P50)", "Limite inferior 80% (P10)",
            "Limite superior 80% (P90)", "Limite inferior 90% (P05)", "Limite superior 90% (P95)",
        ]].copy()
        formato_proj = "R$ %.2f" if metrica_proj == "Faturamento" else "%.0f unidades"
        st.dataframe(tabela_proj, use_container_width=True, hide_index=True, column_config={
            "Previsão central (P50)": st.column_config.NumberColumn(format=formato_proj),
            "Limite inferior 80% (P10)": st.column_config.NumberColumn(format=formato_proj),
            "Limite superior 80% (P90)": st.column_config.NumberColumn(format=formato_proj),
            "Limite inferior 90% (P05)": st.column_config.NumberColumn(format=formato_proj),
            "Limite superior 90% (P95)": st.column_config.NumberColumn(format=formato_proj),
        })
        st.caption(resultado_proj["observacao"])
        st.download_button(
            "Baixar projeção mensal em CSV", tabela_proj.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
            file_name=f"projecao_mensal_{dimensao_proj.lower().replace(' ', '_')}.csv", mime="text/csv",
        )
        with st.expander("Avaliação dos modelos no histórico", expanded=False):
            st.caption("O backtest prevê um mês à frente usando somente dados anteriores. WAPE menor indica menor erro absoluto relativo ao volume realizado; com poucos testes, a classificação pode ser instável.")
            st.dataframe(comparacao_proj, use_container_width=True, hide_index=True, column_config={
                "WAPE retroativo (%)": st.column_config.NumberColumn(format="%.2f%%"),
                f"MAE ({unidade_erro})": st.column_config.NumberColumn(format=("R$ %.2f" if metrica_proj == "Faturamento" else "%.0f")),
                f"Viés médio ({unidade_erro})": st.column_config.NumberColumn(format=("R$ %.2f" if metrica_proj == "Faturamento" else "%.0f")),
            })

if aba_ativa == ABA_VENDEDOR:
    # Painel executivo (visão geral) — organiza os gráficos como no mockup fornecido.
    base_dash = filtrado.copy()
    RED = "#ed1c24"
    RED2 = "#b5121b"
    GREY = "#8b929c"
    TEXT = "#e8eaed"

    def estilo_dashboard(fig, altura=215):
        return padronizar_grafico(fig, altura=altura, margem=dict(l=4, r=8, t=34, b=20))

    # Linha superior: dois gráficos de vendedor e share dos clientes.
    sup_esq, sup_dir = st.columns([1.04, 1.0], gap="small")
    with sup_esq:
        with st.container(border=True):
            st.markdown('<div class="dash-section-title"><span class="dash-red">◉</span>Desempenho por vendedor</div>', unsafe_allow_html=True)
            if col_vendedor:
                op_vendedores_dash = base_dash.groupby("_vendedor", dropna=False)["_valor"].sum().sort_values(ascending=False).head(5).index.tolist()
                dash_v = base_dash[base_dash["_vendedor"].isin(op_vendedores_dash)]
                resumo_v_dash = dash_v.groupby("_vendedor", as_index=False)["_valor"].sum().sort_values("_valor", ascending=True)
                resumo_q_dash = dash_v.groupby("_vendedor", as_index=False)["_quantidade"].sum().sort_values("_quantidade", ascending=True)
                cv1, cv2 = st.columns(2, gap="small")
                with cv1:
                    st.markdown('<div class="dash-chart-caption">Faturamento por vendedor</div>', unsafe_allow_html=True)
                    fig_vdash = px.bar(resumo_v_dash, x="_valor", y="_vendedor", orientation="h", text="_valor", color_discrete_sequence=[RED])
                    fig_vdash.update_traces(text=[f"R$ {num_br(v)}" for v in resumo_v_dash["_valor"]], texttemplate="%{text}", textposition="outside", cliponaxis=False, marker_line_width=0)
                    fig_vdash.update_layout(xaxis_title="Valor (R$)", yaxis_title="", margin=dict(l=2,r=34,t=8,b=12))
                    _tv, _tt = ticks_moeda_br(max(resumo_v_dash["_valor"].max(), 1) * 1.45, alvo=4, rotulador=rotulo_eixo_curto)
                    fig_vdash.update_xaxes(tickmode="array", tickvals=_tv, ticktext=_tt, range=[0, _tv[-1]])
                    exibir_grafico_padronizado(estilo_dashboard(fig_vdash, 202), use_container_width=True, config={"displayModeBar": False}, key="dashboard_vendedor_valor")
                with cv2:
                    st.markdown('<div class="dash-chart-caption">Quantidade por vendedor</div>', unsafe_allow_html=True)
                    fig_qdash = px.bar(resumo_q_dash, x="_quantidade", y="_vendedor", orientation="h", text="_quantidade", color_discrete_sequence=[GREY])
                    fig_qdash.update_traces(text=[num_br(v) for v in resumo_q_dash["_quantidade"]], texttemplate="%{text}", textposition="outside", cliponaxis=False, marker_line_width=0)
                    fig_qdash.update_layout(xaxis_title="Quantidade", yaxis_title="", margin=dict(l=2,r=32,t=8,b=12))
                    _tv, _tt = ticks_moeda_br(max(resumo_q_dash["_quantidade"].max(), 1) * 1.35, alvo=4, rotulador=rotulo_eixo_curto)
                    fig_qdash.update_xaxes(tickmode="array", tickvals=_tv, ticktext=_tt, range=[0, _tv[-1]])
                    exibir_grafico_padronizado(estilo_dashboard(fig_qdash, 202), use_container_width=True, config={"displayModeBar": False}, key="dashboard_vendedor_qtde")
            else:
                st.info("Adicione uma coluna de vendedor para visualizar este gráfico.")

    with sup_dir:
        with st.container(border=True):
            st.markdown('<div class="dash-section-title"><span class="dash-red">◈</span>Share dos clientes no faturamento</div>', unsafe_allow_html=True)
            if col_cliente and not base_dash.empty:
                part_dash = base_dash.groupby("_cliente", as_index=False)["_valor"].sum().sort_values("_valor", ascending=False)
                grand_total_dash = part_dash["_valor"].sum()
                top_dash = part_dash.head(5).copy()
                resto_dash = part_dash.iloc[5:]["_valor"].sum()
                if resto_dash > 0:
                    top_dash = pd.concat([top_dash, pd.DataFrame([{"_cliente": "Outros clientes", "_valor": resto_dash}])], ignore_index=True)
                top_dash["Share (%)"] = (top_dash["_valor"] / grand_total_dash * 100).fillna(0).round(1) if grand_total_dash else 0
                CORES_SHARE = ["#ed1c24", "#7d0f17", "#e8505a", "#9aa0a8", "#5f6670", "#3a4149"]
                cs1, cs2 = st.columns([.92, 1.18], gap="small")
                with cs1:
                    fig_share = px.pie(top_dash, names="_cliente", values="_valor", hole=.60,
                                       color_discrete_sequence=CORES_SHARE)
                    fig_share.update_traces(textinfo="none", marker=dict(line=dict(color="#090b0d", width=1)),
                                            hovertemplate="%{label}<br>R$ %{value:,.2f}<br>%{percent}<extra></extra>")
                    fig_share.update_layout(showlegend=False, margin=dict(l=0,r=0,t=2,b=2),
                                            annotations=[dict(text=f"Total<br>R$ {grand_total_dash/1_000_000:.2f} M".replace(".", ","), x=.5, y=.5, font=dict(color="white",size=12), showarrow=False)])
                    exibir_grafico_padronizado(estilo_dashboard(fig_share, 218), use_container_width=True, config={"displayModeBar": False}, key="dashboard_share_donut")
                with cs2:
                    _linhas = "".join(
                        f'<tr><td><span class="dot" style="background:{CORES_SHARE[k % len(CORES_SHARE)]}"></span>{html.escape(str(nome))}</td>'
                        f'<td>{num_br(valor)}</td><td>{pct_br(share)}</td></tr>'
                        for k, (nome, valor, share) in enumerate(zip(top_dash["_cliente"], top_dash["_valor"], top_dash["Share (%)"]))
                    )
                    st.markdown(
                        '<table class="share-tbl"><thead><tr><th>Cliente gerencial</th><th>Faturamento (R$)</th><th>Share</th></tr></thead>'
                        f'<tbody>{_linhas}</tbody></table>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Adicione a coluna Cliente ou Cliente Gerencial para visualizar a participação.")

    # Linha inferior: evolução temporal, composição por segmento e assistente.
    inf_esq, inf_meio = st.columns([1.95, 1.05], gap="small")
    with inf_esq:
        with st.container(border=True):
            _tit_col, _seg_col = st.columns([1, 1], gap="small", vertical_alignment="center")
            with _tit_col:
                st.markdown('<div class="dash-section-title"><span class="dash-red">▥</span>Evolução temporal do faturamento</div>', unsafe_allow_html=True)
            with _seg_col:
                granularidade_dash = st.radio("Período", ["Dia", "Mês", "Trimestre"], horizontal=True, key="dashboard_tempo", label_visibility="collapsed")
            serie_dash = base_dash[base_dash["_data"].notna()].copy()
            if granularidade_dash == "Dia":
                serie_dash["_periodo"] = serie_dash["_data"].dt.strftime("%d/%m/%y")
                serie_dash["_ordem"] = serie_dash["_data"].dt.normalize()
            elif granularidade_dash == "Trimestre":
                serie_dash["_periodo"] = serie_dash["_data"].dt.year.astype(str) + "-T" + serie_dash["_data"].dt.quarter.astype(str)
                serie_dash["_ordem"] = serie_dash["_data"].dt.to_period("Q").astype(str)
            else:
                nomes_mes = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
                serie_dash["_periodo"] = serie_dash["_data"].dt.month.map(nomes_mes) + "/" + serie_dash["_data"].dt.strftime("%y")
                serie_dash["_ordem"] = serie_dash["_data"].dt.to_period("M").astype(str)
            if not serie_dash.empty:
                evolucao_dash = serie_dash.groupby(["_ordem", "_periodo"], as_index=False)["_valor"].sum().sort_values("_ordem")
                fig_evolucao = px.line(evolucao_dash, x="_periodo", y="_valor", markers=True)
                fig_evolucao.update_traces(line=dict(color=RED, width=2.5), marker=dict(size=5, color=RED, line=dict(color="#ff8a8f",width=1)), fill="tozeroy", fillcolor="rgba(237,28,36,.16)")
                fig_evolucao.update_layout(xaxis_title="", yaxis_title="", margin=dict(l=6,r=6,t=6,b=22), hovermode="x unified")
                fig_evolucao.update_traces(
                    customdata=[f"R$ {num_br(v)}" for v in evolucao_dash["_valor"]],
                    hovertemplate="%{x}<br><b>%{customdata}</b><extra></extra>",
                )
                _tv, _tt = ticks_moeda_br(evolucao_dash["_valor"].max())
                if _tv:
                    fig_evolucao.update_yaxes(tickmode="array", tickvals=_tv, ticktext=_tt, range=[0, _tv[-1]])
                fig_evolucao.update_xaxes(tickangle=0, nticks=12)
                exibir_grafico_padronizado(estilo_dashboard(fig_evolucao, 224), use_container_width=True, config={"displayModeBar": False}, key="dashboard_evolucao_faturamento")
            else:
                st.info("Não existem datas válidas no período selecionado.")

    with inf_meio:
        with st.container(border=True):
            st.markdown('<div class="dash-section-title"><span class="dash-red">◉</span>Composição do faturamento</div>', unsafe_allow_html=True)
            st.markdown('<div class="dash-chart-caption">Produtos por segmento</div>', unsafe_allow_html=True)
            if col["segmento"]:
                seg_dash = base_dash.groupby("_segmento", as_index=False)["_valor"].sum().sort_values("_valor", ascending=False).head(6).sort_values("_valor")
                if not seg_dash.empty:
                    cores_seg = ["#ed1c24", "#c91620", "#a91019", "#e2444c", "#b85c63", "#686f7a"]
                    _total_seg = base_dash["_valor"].sum()
                    seg_dash = seg_dash.assign(_rotulo=[f"R$ {num_br(v)}   {pct_br(v / _total_seg * 100 if _total_seg else 0)}" for v in seg_dash["_valor"]])
                    fig_seg = px.bar(seg_dash, x="_valor", y="_segmento", orientation="h", text="_rotulo", color="_segmento", color_discrete_sequence=cores_seg)
                    fig_seg.update_traces(texttemplate="%{text}", textposition="outside", cliponaxis=False, marker_line_width=0)
                    fig_seg.update_layout(showlegend=False, xaxis_title="", yaxis_title="", margin=dict(l=0,r=32,t=2,b=4))
                    fig_seg.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, range=[0, max(seg_dash["_valor"].max(), 1) * 1.9])
                    exibir_grafico_padronizado(estilo_dashboard(fig_seg, 222), use_container_width=True, config={"displayModeBar": False}, key="dashboard_composicao_segmento")
                else:
                    st.info("Não há segmentos na seleção atual.")
            else:
                st.info("A coluna Segmento não foi encontrada.")


    st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)

    # Análises completas preservadas para consulta sem ocupar o primeiro quadro do dashboard.
    with st.expander("▸ Análises detalhadas por vendedor e clientes", expanded=False):
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
                    exibir_grafico_padronizado(formatar_eixos(fig), use_container_width=True, key="grafico_unico_01")
                with c2:
                    resumo_q = base.groupby("_vendedor", as_index=False)["_quantidade"].sum().sort_values("_quantidade", ascending=False)
                    fig = px.bar(resumo_q, x="_quantidade", y="_vendedor", orientation="h", title="Quantidade por vendedor", text_auto=True)
                    fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Quantidade", yaxis_title="")
                    exibir_grafico_padronizado(formatar_eixos(fig), use_container_width=True, key="grafico_unico_02")

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
                    evento_grafico = exibir_grafico_padronizado(
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
                    exibir_grafico_padronizado(formatar_eixos(fig), use_container_width=True, key="grafico_unico_04")

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
                    exibir_grafico_padronizado(formatar_eixos(fig), use_container_width=True, key="grafico_unico_05")

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
                exibir_grafico_padronizado(formatar_eixos(fig), use_container_width=True, key="grafico_unico_06")
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
                        evento_prod = exibir_grafico_padronizado(
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
                exibir_grafico_padronizado(formatar_eixos(fig), use_container_width=True, key="grafico_unico_08")
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
                        exibir_grafico_padronizado(formatar_eixos(fig), use_container_width=True, key="grafico_unico_09")

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
                exibir_grafico_padronizado(fig, use_container_width=True, key="grafico_preco_medio")
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

# O assistente já foi renderizado no início da área principal para permanecer
# disponível independentemente da seção selecionada.
_atualizado_em = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%d/%m/%Y %H:%M")
st.markdown(
    f"""
    <footer class="intech-footer">
      <div class="intech-footer-brand"><b>▮▮▮</b><span>Intech Machine</span><span style="color:#60666c">|</span><span>Painel Interativo de Vendas</span></div>
      <div class="intech-footer-muted">Dados atualizados em: {_atualizado_em}</div>
    </footer>
    """,
    unsafe_allow_html=True,
)



# Camada final de padronização visual: mantém a mesma paleta, tipografia,
# bordas, raios e estados de interação em todas as seções do painel.
st.markdown("""
<style>
:root {
  --ui-red: #ed1c24;
  --ui-red-hover: #ff343c;
  --ui-bg: #050607;
  --ui-surface: #0b0e10;
  --ui-surface-hover: #14181b;
  --ui-border: rgba(160, 170, 180, .22);
  --ui-text: #f4f6f8;
  --ui-muted: #aab2ba;
  --ui-radius: 10px;
}
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stSidebar"], button, input, textarea, select,
[data-baseweb], [data-testid="stMarkdownContainer"] {
  font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] {
  width: 280px !important;
  min-width: 280px !important;
  max-width: 280px !important;
  background: linear-gradient(180deg, #0c0f11 0%, #070809 72%, #050607 100%) !important;
  border-right: 1px solid var(--ui-border) !important;
}
/* Campos de entrada */
[data-baseweb="input"] > div,
[data-baseweb="select"] > div,
[data-baseweb="textarea"] > div,
[data-testid="stDateInput"] [data-baseweb="input"] > div,
[data-testid="stNumberInput"] [data-baseweb="input"] > div {
  background: var(--ui-surface) !important;
  border: 1px solid var(--ui-border) !important;
  border-radius: var(--ui-radius) !important;
  color: var(--ui-text) !important;
  box-shadow: none !important;
}
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea,
[data-baseweb="select"] *,
[data-testid="stDateInput"] input,
[data-testid="stNumberInput"] input {
  color: var(--ui-text) !important;
  font-family: 'Inter', sans-serif !important;
}
[data-baseweb="input"] > div:focus-within,
[data-baseweb="select"] > div:focus-within,
[data-baseweb="textarea"] > div:focus-within {
  border-color: var(--ui-red) !important;
  box-shadow: 0 0 0 1px var(--ui-red) !important;
}
/* Botões: mesmo formato e comportamento; vermelho para a ação principal */
.stApp [data-testid="stButton"] button,
.stApp [data-testid="stDownloadButton"] button,
.stApp [data-testid="stFormSubmitButton"] button {
  border-radius: var(--ui-radius) !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  transition: border-color .15s ease, background .15s ease, color .15s ease !important;
}
.stApp [data-testid="stButton"] button[kind="primary"],
.stApp [data-testid="stFormSubmitButton"] button,
.stApp [data-testid="stDownloadButton"] button {
  background: linear-gradient(135deg, var(--ui-red-hover), #be0c15) !important;
  color: #fff !important;
  border: 1px solid var(--ui-red) !important;
}
.stApp [data-testid="stButton"] button[kind="primary"]:hover,
.stApp [data-testid="stFormSubmitButton"] button:hover,
.stApp [data-testid="stDownloadButton"] button:hover {
  background: var(--ui-red) !important;
  border-color: var(--ui-red-hover) !important;
}
.stApp [data-testid="stButton"] button[kind="secondary"] {
  background: var(--ui-surface) !important;
  color: var(--ui-text) !important;
  border: 1px solid var(--ui-border) !important;
}
.stApp [data-testid="stButton"] button[kind="secondary"]:hover {
  background: var(--ui-surface-hover) !important;
  border-color: var(--ui-red) !important;
  color: #fff !important;
}
/* Cartões, métricas, tabelas e áreas de dados */
[data-testid="stMetric"], [data-testid="stDataFrame"],
[data-testid="stTable"], [data-testid="stFileUploader"] section,
[data-testid="stAlert"] {
  border-radius: var(--ui-radius) !important;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 1px solid var(--ui-border) !important;
  overflow: hidden !important;
}
[data-testid="stFileUploader"] section {
  background: var(--ui-surface) !important;
  border: 1px dashed var(--ui-border) !important;
}
[data-testid="stFileUploader"] section:hover {
  border-color: var(--ui-red) !important;
}
/* Checkbox e seleções */
[data-testid="stCheckbox"] label, [data-testid="stRadio"] label {
  color: var(--ui-text) !important;
}
[data-testid="stCheckbox"] input:checked + div,
[data-baseweb="checkbox"] input:checked ~ div {
  background-color: var(--ui-red) !important;
  border-color: var(--ui-red) !important;
}
/* Consistência de títulos, divisórias e textos auxiliares */
h1, h2, h3, h4 { color: #fff !important; font-family: 'Inter', sans-serif !important; }
hr { border-color: var(--ui-border) !important; }
[data-testid="stCaptionContainer"], .intech-footer-muted { color: var(--ui-muted) !important; }
/* Mantém o menu lateral sempre aberto e sem controle de recolher */
[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarHeader"] button,
button[kind="header"] { display: none !important; pointer-events: none !important; }
</style>
""", unsafe_allow_html=True)
