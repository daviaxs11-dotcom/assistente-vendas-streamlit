# engine_projecao.py
# Engine local de previsões probabilísticas mensais e trimestrais.
# Sem chamadas externas: usa pandas/numpy e dados filtrados pelo painel.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


MIN_TRIMESTRES = 6
SIMULACOES_PADRAO = 3000
SEMENTE_PADRAO = 42


@dataclass
class DadosSerie:
    historico: pd.Series
    serie_completa: pd.Series
    trimestre_parcial: pd.Period | None
    valor_parcial: float | None
    ultima_data: pd.Timestamp | None


def _transformar_valores(y: np.ndarray) -> tuple[np.ndarray, str]:
    """Usa log1p quando todos os valores são não negativos; senão, escala aditiva."""
    y = np.asarray(y, dtype=float)
    if np.all(np.isfinite(y)) and np.all(y >= 0):
        return np.log1p(y), "log1p"
    return y.astype(float), "aditiva"


def _destransformar_valores(z: np.ndarray, transformacao: str) -> np.ndarray:
    if transformacao == "log1p":
        return np.maximum(np.expm1(z), 0.0)
    return np.asarray(z, dtype=float)


def preparar_serie_trimestral(
    df: pd.DataFrame,
    coluna_data: str = "_data",
    coluna_valor: str = "_valor",
    hoje: Any | None = None,
) -> DadosSerie:
    """Agrega vendas por trimestre, mantendo separado o trimestre corrente parcial.

    Um trimestre é considerado parcial quando corresponde ao trimestre de hoje e ainda
    não terminou. Ele não entra no treinamento; pode ser exibido como realizado parcial.
    Trimestres sem registros dentro da faixa filtrada recebem zero.
    """
    if coluna_data not in df.columns or coluna_valor not in df.columns:
        raise ValueError(f"A base precisa conter as colunas {coluna_data!r} e {coluna_valor!r}.")

    base = df[[coluna_data, coluna_valor]].copy()
    base[coluna_data] = pd.to_datetime(base[coluna_data], errors="coerce")
    base[coluna_valor] = pd.to_numeric(base[coluna_valor], errors="coerce")
    base = base.dropna(subset=[coluna_data, coluna_valor])
    if base.empty:
        return DadosSerie(pd.Series(dtype=float), pd.Series(dtype=float), None, None, None)

    base["_trimestre"] = base[coluna_data].dt.to_period("Q-DEC")
    somas = base.groupby("_trimestre")[coluna_valor].sum().sort_index()
    inicio, fim = somas.index.min(), somas.index.max()
    indice = pd.period_range(inicio, fim, freq="Q-DEC")
    completa = somas.reindex(indice, fill_value=0.0).astype(float)

    data_max = pd.Timestamp(base[coluna_data].max())
    hoje_ts = pd.Timestamp(hoje).normalize() if hoje is not None else pd.Timestamp.today().normalize()
    trimestre_atual = hoje_ts.to_period("Q-DEC")

    trimestre_parcial = None
    valor_parcial = None
    # Só exclui o trimestre em curso; trimestres históricos são tratados como encerrados.
    if completa.index.max() == trimestre_atual and trimestre_atual.end_time.normalize() > hoje_ts:
        trimestre_parcial = completa.index.max()
        valor_parcial = float(completa.iloc[-1])
        historico = completa.iloc[:-1].copy()
    else:
        historico = completa.copy()

    return DadosSerie(
        historico=historico,
        serie_completa=completa,
        trimestre_parcial=trimestre_parcial,
        valor_parcial=valor_parcial,
        ultima_data=data_max,
    )


def _matriz_tendencia_sazonal(periodos: pd.PeriodIndex) -> np.ndarray:
    t = np.arange(len(periodos), dtype=float)
    # Q1 como categoria-base; variáveis indicadoras para Q2, Q3 e Q4.
    q = np.asarray([p.quarter for p in periodos], dtype=int)
    return np.column_stack([np.ones(len(periodos)), t, (q == 2).astype(float), (q == 3).astype(float), (q == 4).astype(float)])


def _prever_transformado(
    y_transformado: np.ndarray,
    periodos: pd.PeriodIndex,
    periodos_futuros: pd.PeriodIndex,
    modelo: str,
) -> np.ndarray:
    """Faz previsão determinística na escala transformada."""
    y = np.asarray(y_transformado, dtype=float)
    n = len(y)
    h = len(periodos_futuros)
    if n == 0:
        return np.zeros(h, dtype=float)

    if modelo == "ultimo_trimestre":
        return np.repeat(y[-1], h)

    if modelo == "media_4_trimestres":
        return np.repeat(float(np.mean(y[-min(4, n):])), h)

    if modelo == "sazonal_ingenua":
        if n < 4:
            return np.repeat(y[-1], h)
        return np.asarray([y[n - 4 + (i % 4)] for i in range(h)], dtype=float)

    if modelo == "tendencia_sazonal":
        X = _matriz_tendencia_sazonal(periodos)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        futuro_t = np.arange(n, n + h, dtype=float)
        q = np.asarray([p.quarter for p in periodos_futuros], dtype=int)
        Xf = np.column_stack([
            np.ones(h), futuro_t,
            (q == 2).astype(float), (q == 3).astype(float), (q == 4).astype(float),
        ])
        return Xf @ coef

    raise ValueError(f"Modelo desconhecido: {modelo}")


def _modelos_disponiveis(n: int) -> list[str]:
    modelos = ["ultimo_trimestre", "media_4_trimestres"]
    if n >= 8:
        modelos.append("sazonal_ingenua")
    # São necessários mais dados para que tendência + sazonalidade não fiquem sobreajustadas.
    if n >= 16:
        modelos.append("tendencia_sazonal")
    return modelos


def rotulo_trimestre(periodo: pd.Period) -> str:
    """Rótulo amigável do trimestre, por exemplo 2026-T1."""
    return f"{periodo.year}-T{periodo.quarter}"


def _rotulo_modelo(modelo: str) -> str:
    return {
        "ultimo_trimestre": "Último trimestre",
        "media_4_trimestres": "Média móvel de 4 trimestres",
        "sazonal_ingenua": "Sazonal ingênuo (mesmo trimestre do ano anterior)",
        "tendencia_sazonal": "Tendência + sazonalidade trimestral",
    }.get(modelo, modelo)


def _backtest_modelo(
    y_original: np.ndarray,
    periodos: pd.PeriodIndex,
    modelo: str,
    transformacao: str,
) -> dict[str, Any]:
    """Walk-forward de um trimestre à frente; cada previsão usa apenas passado."""
    n = len(y_original)
    inicio_teste = max(4, min(12, n - 3), n - 8)
    reais, previstos, erros_transformados = [], [], []

    for corte in range(inicio_teste, n):
        treino_original = y_original[:corte]
        treino_z = np.log1p(np.maximum(treino_original, 0.0)) if transformacao == "log1p" else treino_original.astype(float)
        futuro_periodo = pd.PeriodIndex([periodos[corte]], freq="Q-DEC")
        pz = _prever_transformado(treino_z, periodos[:corte], futuro_periodo, modelo)[0]
        pv = float(_destransformar_valores(np.asarray([pz]), transformacao)[0])
        reais.append(float(y_original[corte]))
        previstos.append(pv)
        real_z = float(np.log1p(max(y_original[corte], 0.0))) if transformacao == "log1p" else float(y_original[corte])
        erros_transformados.append(real_z - pz)

    if not reais:
        return {"wape": np.inf, "mae": np.inf, "bias": np.nan, "erros_transformados": []}

    r = np.asarray(reais, dtype=float)
    p = np.asarray(previstos, dtype=float)
    denom = float(np.sum(np.abs(r)))
    wape = float(np.sum(np.abs(r - p)) / denom * 100) if denom > 1e-9 else float(np.mean(np.abs(r - p)))
    mae = float(np.mean(np.abs(r - p)))
    bias = float(np.mean(p - r))
    return {
        "wape": wape, "mae": mae, "bias": bias,
        "erros_transformados": [float(e) for e in erros_transformados if np.isfinite(e)],
        "n_testes": len(reais),
    }


def projetar_trimestres(
    df: pd.DataFrame,
    horizonte: int = 4,
    simulacoes: int = SIMULACOES_PADRAO,
    coluna_data: str = "_data",
    coluna_valor: str = "_valor",
    semente: int = SEMENTE_PADRAO,
    hoje: Any | None = None,
) -> dict[str, Any]:
    """Retorna histórico, previsão probabilística, comparação de modelos e diagnóstico.

    Os percentis 5/50/95 e 10/50/90 vêm de simulações por bootstrap dos erros de
    previsões históricas de um trimestre à frente. São intervalos empíricos aproximados,
    não garantias de cobertura.
    """
    horizonte = int(max(1, min(12, horizonte)))
    simulacoes = int(max(500, min(10000, simulacoes)))
    dados = preparar_serie_trimestral(df, coluna_data, coluna_valor, hoje=hoje)
    historico = dados.historico

    if len(historico) < MIN_TRIMESTRES:
        return {
            "ok": False,
            "erro": f"São necessários pelo menos {MIN_TRIMESTRES} trimestres encerrados para gerar uma projeção probabilística inicial. Há {len(historico)} trimestre(s) disponíveis.",
            "historico": historico,
            "trimestre_parcial": dados.trimestre_parcial,
            "valor_parcial": dados.valor_parcial,
            "ultima_data": dados.ultima_data,
        }

    y_original = historico.to_numpy(dtype=float)
    z, transformacao = _transformar_valores(y_original)
    periodos = pd.PeriodIndex(historico.index, freq="Q-DEC")
    modelos = _modelos_disponiveis(len(historico))
    comparacao = []
    detalhes_backtest = {}
    for modelo in modelos:
        resultado_bt = _backtest_modelo(y_original, periodos, modelo, transformacao)
        detalhes_backtest[modelo] = resultado_bt
        comparacao.append({
            "Modelo": _rotulo_modelo(modelo),
            "WAPE retroativo (%)": resultado_bt["wape"],
            "MAE (R$)": resultado_bt["mae"],
            "Viés médio (R$)": resultado_bt["bias"],
            "Testes retroativos": resultado_bt.get("n_testes", 0),
            "_modelo": modelo,
        })

    ranking = sorted(comparacao, key=lambda x: (x["WAPE retroativo (%)"], x["MAE (R$)"]))
    melhor_modelo = ranking[0]["_modelo"]
    ultimo_periodo = periodos[-1]

    if dados.trimestre_parcial is not None:
        primeiro_futuro = dados.trimestre_parcial
        valor_parcial = dados.valor_parcial
    else:
        primeiro_futuro = ultimo_periodo + 1
        valor_parcial = None
    periodos_futuros = pd.period_range(primeiro_futuro, periods=horizonte, freq="Q-DEC")
    prev_z = _prever_transformado(z, periodos, periodos_futuros, melhor_modelo)
    prev_point = _destransformar_valores(prev_z, transformacao)

    erros = np.asarray(detalhes_backtest[melhor_modelo]["erros_transformados"], dtype=float)
    erros = erros[np.isfinite(erros)]
    if len(erros) < 3:
        # Reserva de variabilidade: diferenças históricas na escala do modelo.
        if len(z) > 1:
            erros = np.diff(z)
    erros = erros[np.isfinite(erros)]
    if len(erros) == 0:
        erros = np.asarray([0.0], dtype=float)
    # Limita a influência de erros extremos isolados quando a amostra permite.
    if len(erros) >= 6:
        lo, hi = np.quantile(erros, [0.05, 0.95])
        erros = np.clip(erros, lo, hi)
    erros = erros - float(np.median(erros))

    rng = np.random.default_rng(semente)
    choques = rng.choice(erros, size=(simulacoes, horizonte), replace=True)
    # Cada trimestre é um nível previsto. Acumular erros transformados como passeio
    # aleatório pode fazer as faixas crescerem exponencialmente com log1p e gerar
    # escalas absurdas em quantidade vendida. Aplicamos um erro por período, sem cumsum.
    caminhos_z = prev_z[None, :] + choques
    caminhos = _destransformar_valores(caminhos_z, transformacao)
    if transformacao == "aditiva":
        caminhos = caminhos.astype(float)

    percentis = {p: np.quantile(caminhos, p / 100.0, axis=0) for p in (5, 10, 50, 90, 95)}
    # O percentil 50 é a estimativa central da distribuição simulada.
    previsoes = pd.DataFrame({
        "Trimestre": [rotulo_trimestre(p) for p in periodos_futuros],
        "Período": periodos_futuros,
        "Previsão central (P50)": percentis[50],
        "Limite inferior 80% (P10)": percentis[10],
        "Limite superior 80% (P90)": percentis[90],
        "Limite inferior 90% (P05)": percentis[5],
        "Limite superior 90% (P95)": percentis[95],
    })

    hist_df = pd.DataFrame({
        "Período": historico.index,
        "Trimestre": [rotulo_trimestre(p) for p in historico.index],
        "Faturamento realizado (R$)": historico.to_numpy(dtype=float),
        "Tipo": "Realizado",
    })
    if dados.trimestre_parcial is not None:
        parcial_df = pd.DataFrame([{
            "Período": dados.trimestre_parcial,
            "Trimestre": rotulo_trimestre(dados.trimestre_parcial),
            "Faturamento realizado (R$)": float(dados.valor_parcial or 0),
            "Tipo": "Realizado parcial",
        }])
        hist_df = pd.concat([hist_df, parcial_df], ignore_index=True)

    comparacao_df = pd.DataFrame(comparacao).drop(columns=["_modelo"])
    comparacao_df = comparacao_df.sort_values(["WAPE retroativo (%)", "MAE (R$)"]).reset_index(drop=True)
    return {
        "ok": True,
        "historico": historico,
        "historico_df": hist_df,
        "previsoes": previsoes,
        "modelo": melhor_modelo,
        "modelo_rotulo": _rotulo_modelo(melhor_modelo),
        "comparacao": comparacao_df,
        "wape": detalhes_backtest[melhor_modelo]["wape"],
        "mae": detalhes_backtest[melhor_modelo]["mae"],
        "bias": detalhes_backtest[melhor_modelo]["bias"],
        "n_trimestres": len(historico),
        "transformacao": transformacao,
        "simulacoes": simulacoes,
        "trimestre_parcial": dados.trimestre_parcial,
        "valor_parcial": valor_parcial,
        "ultima_data": dados.ultima_data,
        "observacao": (
            "A projeção usa apenas o histórico filtrado localmente. Os limites são percentis empíricos "
            "de simulações de erros históricos e não representam garantia; a confiabilidade depende do volume "
            "e da estabilidade do histórico."
        ),
    }


# ============================================================
# PROJEÇÕES PROBABILÍSTICAS MENSAIS
# ============================================================

MIN_MESES = 12


@dataclass
class DadosSerieMensal:
    historico: pd.Series
    serie_completa: pd.Series
    mes_parcial: pd.Period | None
    valor_parcial: float | None
    ultima_data: pd.Timestamp | None


def preparar_serie_mensal(
    df: pd.DataFrame,
    coluna_data: str = "_data",
    coluna_valor: str = "_valor",
    hoje: Any | None = None,
) -> DadosSerieMensal:
    """Agrega por mês e mantém separado o mês corrente ainda incompleto."""
    if coluna_data not in df.columns or coluna_valor not in df.columns:
        raise ValueError(f"A base precisa conter as colunas {coluna_data!r} e {coluna_valor!r}.")
    base = df[[coluna_data, coluna_valor]].copy()
    base[coluna_data] = pd.to_datetime(base[coluna_data], errors="coerce", dayfirst=True)
    base[coluna_valor] = pd.to_numeric(base[coluna_valor], errors="coerce")
    base = base.dropna(subset=[coluna_data, coluna_valor])
    if base.empty:
        vazio = pd.Series(dtype=float)
        return DadosSerieMensal(vazio, vazio, None, None, None)

    base["_mes"] = base[coluna_data].dt.to_period("M")
    somas = base.groupby("_mes")[coluna_valor].sum().sort_index()
    indice = pd.period_range(somas.index.min(), somas.index.max(), freq="M")
    completa = somas.reindex(indice, fill_value=0.0).astype(float)
    data_max = pd.Timestamp(base[coluna_data].max())
    hoje_ts = pd.Timestamp(hoje).normalize() if hoje is not None else pd.Timestamp.today().normalize()
    mes_atual = hoje_ts.to_period("M")

    mes_parcial, valor_parcial = None, None
    if completa.index.max() == mes_atual and mes_atual.end_time.normalize() > hoje_ts:
        mes_parcial = completa.index.max()
        valor_parcial = float(completa.iloc[-1])
        historico = completa.iloc[:-1].copy()
    else:
        historico = completa.copy()
    return DadosSerieMensal(historico, completa, mes_parcial, valor_parcial, data_max)


def _matriz_tendencia_sazonal_mensal(periodos: pd.PeriodIndex) -> np.ndarray:
    t = np.arange(len(periodos), dtype=float)
    m = np.asarray([p.month for p in periodos], dtype=int)
    # Janeiro é a categoria-base; de fevereiro a dezembro há variáveis indicadoras.
    return np.column_stack([np.ones(len(periodos)), t] + [(m == k).astype(float) for k in range(2, 13)])


def _prever_mensal(y: np.ndarray, periodos: pd.PeriodIndex, futuros: pd.PeriodIndex, modelo: str) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n, h = len(y), len(futuros)
    if n == 0:
        return np.zeros(h, dtype=float)
    if modelo == "ultimo_mes":
        return np.repeat(y[-1], h)
    if modelo == "media_3_meses":
        return np.repeat(float(np.mean(y[-min(3, n):])), h)
    if modelo == "media_6_meses":
        return np.repeat(float(np.mean(y[-min(6, n):])), h)
    if modelo == "sazonal_ingenua_mensal":
        if n < 12:
            return np.repeat(y[-1], h)
        return np.asarray([y[n - 12 + (i % 12)] for i in range(h)], dtype=float)
    if modelo == "tendencia_sazonal_mensal":
        X = _matriz_tendencia_sazonal_mensal(periodos)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        futuro_t = np.arange(n, n + h, dtype=float)
        m = np.asarray([p.month for p in futuros], dtype=int)
        Xf = np.column_stack([np.ones(h), futuro_t] + [(m == k).astype(float) for k in range(2, 13)])
        return Xf @ coef
    raise ValueError(f"Modelo mensal desconhecido: {modelo}")


def _modelos_mensais_disponiveis(n: int) -> list[str]:
    modelos = ["ultimo_mes", "media_3_meses", "media_6_meses"]
    if n >= 24:
        modelos.append("sazonal_ingenua_mensal")
    if n >= 36:
        modelos.append("tendencia_sazonal_mensal")
    return modelos


def _rotulo_modelo_mensal(modelo: str) -> str:
    return {
        "ultimo_mes": "Último mês",
        "media_3_meses": "Média móvel de 3 meses",
        "media_6_meses": "Média móvel de 6 meses",
        "sazonal_ingenua_mensal": "Mesmo mês do ano anterior",
        "tendencia_sazonal_mensal": "Tendência + sazonalidade mensal",
    }.get(modelo, modelo)


def _backtest_mensal(y_original: np.ndarray, periodos: pd.PeriodIndex, modelo: str, transformacao: str) -> dict[str, Any]:
    """Validação walk-forward de um mês à frente, sem usar dados do mês previsto."""
    n = len(y_original)
    inicio_teste = max(4, min(12, n - 3), n - 8)
    reais, previstos, erros_transformados = [], [], []
    for corte in range(inicio_teste, n):
        treino = y_original[:corte]
        treino_z = np.log1p(np.maximum(treino, 0.0)) if transformacao == "log1p" else treino.astype(float)
        futuro = pd.PeriodIndex([periodos[corte]], freq="M")
        pz = _prever_mensal(treino_z, periodos[:corte], futuro, modelo)[0]
        pv = float(_destransformar_valores(np.asarray([pz]), transformacao)[0])
        reais.append(float(y_original[corte])); previstos.append(pv)
        real_z = float(np.log1p(max(y_original[corte], 0.0))) if transformacao == "log1p" else float(y_original[corte])
        erros_transformados.append(real_z - pz)
    if not reais:
        return {"wape": np.inf, "mae": np.inf, "bias": np.nan, "erros_transformados": [], "n_testes": 0}
    r, p = np.asarray(reais), np.asarray(previstos)
    denom = float(np.sum(np.abs(r)))
    return {
        "wape": float(np.sum(np.abs(r-p))/denom*100) if denom > 1e-9 else float(np.mean(np.abs(r-p))),
        "mae": float(np.mean(np.abs(r-p))),
        "bias": float(np.mean(p-r)),
        "erros_transformados": [float(e) for e in erros_transformados if np.isfinite(e)],
        "n_testes": len(reais),
    }


def projetar_meses(
    df: pd.DataFrame,
    horizonte: int = 6,
    simulacoes: int = SIMULACOES_PADRAO,
    coluna_data: str = "_data",
    coluna_valor: str = "_valor",
    semente: int = SEMENTE_PADRAO,
    hoje: Any | None = None,
) -> dict[str, Any]:
    """Projeção probabilística mensal com seleção por backtesting e bootstrap de erros."""
    horizonte = int(max(1, min(36, horizonte)))
    simulacoes = int(max(500, min(10000, simulacoes)))
    dados = preparar_serie_mensal(df, coluna_data, coluna_valor, hoje=hoje)
    historico = dados.historico
    if len(historico) < MIN_MESES:
        return {
            "ok": False,
            "erro": f"São necessários pelo menos {MIN_MESES} meses encerrados para a projeção mensal probabilística. Há {len(historico)} mês(es) disponíveis.",
            "historico": historico, "mes_parcial": dados.mes_parcial,
            "valor_parcial": dados.valor_parcial, "ultima_data": dados.ultima_data,
        }

    y = historico.to_numpy(dtype=float)
    z, transformacao = _transformar_valores(y)
    periodos = pd.PeriodIndex(historico.index, freq="M")
    comparacao, detalhes = [], {}
    for modelo in _modelos_mensais_disponiveis(len(historico)):
        bt = _backtest_mensal(y, periodos, modelo, transformacao)
        detalhes[modelo] = bt
        comparacao.append({
            "Modelo": _rotulo_modelo_mensal(modelo), "WAPE retroativo (%)": bt["wape"],
            "MAE (R$)": bt["mae"], "Viés médio (R$)": bt["bias"],
            "Testes retroativos": bt.get("n_testes", 0), "_modelo": modelo,
        })
    ranking = sorted(comparacao, key=lambda x: (x["WAPE retroativo (%)"], x["MAE (R$)"]))
    melhor = ranking[0]["_modelo"]
    if dados.mes_parcial is not None:
        primeiro_futuro = dados.mes_parcial
    else:
        primeiro_futuro = periodos[-1] + 1
    futuros = pd.period_range(primeiro_futuro, periods=horizonte, freq="M")
    prev_z = _prever_mensal(z, periodos, futuros, melhor)

    erros = np.asarray(detalhes[melhor]["erros_transformados"], dtype=float)
    erros = erros[np.isfinite(erros)]
    if len(erros) < 3:
        erros = np.diff(z) if len(z) > 1 else np.asarray([0.0])
    erros = erros[np.isfinite(erros)]
    if len(erros) == 0:
        erros = np.asarray([0.0], dtype=float)

    # Limita a influência de erros extremos isolados quando há amostra suficiente.
    # Isso é especialmente importante para séries com muitas quantidades zero/baixas
    # e poucas vendas extraordinárias.
    if len(erros) >= 6:
        lo, hi = np.quantile(erros, [0.05, 0.95])
        erros = np.clip(erros, lo, hi)

    # Centraliza os choques: o intervalo expressa a dispersão, e não a tendência
    # sistemática de erro do modelo (que é mostrada separadamente no viés do backtest).
    erros = erros - float(np.median(erros))

    rng = np.random.default_rng(semente)
    choques = rng.choice(erros, size=(simulacoes, horizonte), replace=True)

    # Cada previsão é um nível mensal projetado. Somar cumulativamente os erros
    # transformados cria um passeio aleatório artificial e faz as faixas explodirem
    # exponencialmente com log1p (ex.: milhões de unidades em séries pequenas).
    # Sorteamos um erro histórico por mês, sem acumulá-lo sobre os meses anteriores.
    caminhos_z = prev_z[None, :] + choques
    caminhos = _destransformar_valores(caminhos_z, transformacao)
    perc = {p: np.quantile(caminhos, p/100.0, axis=0) for p in (5, 10, 50, 90, 95)}
    previsoes = pd.DataFrame({
        "Mês": [p.strftime("%m/%Y") for p in futuros], "Período": futuros,
        "Previsão central (P50)": perc[50],
        "Limite inferior 80% (P10)": perc[10], "Limite superior 80% (P90)": perc[90],
        "Limite inferior 90% (P05)": perc[5], "Limite superior 90% (P95)": perc[95],
    })
    hist_df = pd.DataFrame({"Período": historico.index, "Mês": [p.strftime("%m/%Y") for p in historico.index],
                            "Faturamento realizado (R$)": y, "Tipo": "Realizado"})
    if dados.mes_parcial is not None:
        hist_df = pd.concat([hist_df, pd.DataFrame([{
            "Período": dados.mes_parcial, "Mês": dados.mes_parcial.strftime("%m/%Y"),
            "Faturamento realizado (R$)": float(dados.valor_parcial or 0), "Tipo": "Realizado parcial",
        }])], ignore_index=True)
    comparacao_df = pd.DataFrame(comparacao).drop(columns=["_modelo"]).sort_values(
        ["WAPE retroativo (%)", "MAE (R$)"]
    ).reset_index(drop=True)
    bt_melhor = detalhes[melhor]
    return {
        "ok": True, "historico": historico, "historico_df": hist_df, "previsoes": previsoes,
        "modelo": melhor, "modelo_rotulo": _rotulo_modelo_mensal(melhor), "comparacao": comparacao_df,
        "wape": bt_melhor["wape"], "mae": bt_melhor["mae"], "bias": bt_melhor["bias"],
        "n_meses": len(historico), "transformacao": transformacao, "simulacoes": simulacoes,
        "mes_parcial": dados.mes_parcial, "valor_parcial": dados.valor_parcial,
        "ultima_data": dados.ultima_data,
        "observacao": "Projeção calculada localmente por mês. Os intervalos são percentis empíricos de simulações de erros históricos e não garantem cobertura estatística.",
    }
