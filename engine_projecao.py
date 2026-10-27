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
        "_caminhos_simulados": caminhos,
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
    fim: pd.Period | None = None,
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
    # `fim` estende a série com zeros até o último mês da base inteira. Sem isso, um SKU
    # descontinuado termina no seu último mês de venda e é projetado como se ainda vendesse.
    ultimo = somas.index.max()
    if fim is not None and pd.Period(fim, freq="M") > ultimo:
        ultimo = pd.Period(fim, freq="M")
    indice = pd.period_range(somas.index.min(), ultimo, freq="M")
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


def _meses_do_ano(periodos) -> np.ndarray:
    """Vetor de meses (1-12). Aceita PeriodIndex ou um vetor já convertido."""
    if isinstance(periodos, np.ndarray):
        return periodos.astype(int)
    return np.asarray(pd.PeriodIndex(periodos, freq="M").month, dtype=int)


def _prever_mensal_arr(y: np.ndarray, meses_hist: np.ndarray, meses_fut: np.ndarray, modelo: str) -> np.ndarray:
    """Núcleo numérico da previsão mensal; recebe só arrays (sem pandas), por performance."""
    y = np.asarray(y, dtype=float)
    n, h = len(y), len(meses_fut)
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
        return y[n - 12 + (np.arange(h) % 12)].astype(float)
    if modelo == "tendencia_sazonal_mensal":
        ks = np.arange(2, 13)
        X = np.column_stack([np.ones(n), np.arange(n, dtype=float)] + [(meses_hist == k).astype(float) for k in ks])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        Xf = np.column_stack([np.ones(h), np.arange(n, n + h, dtype=float)] + [(meses_fut == k).astype(float) for k in ks])
        return Xf @ coef
    raise ValueError(f"Modelo mensal desconhecido: {modelo}")


def _prever_mensal(y: np.ndarray, periodos: pd.PeriodIndex, futuros: pd.PeriodIndex, modelo: str) -> np.ndarray:
    return _prever_mensal_arr(y, _meses_do_ano(periodos), _meses_do_ano(futuros), modelo)


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
    meses = _meses_do_ano(periodos)
    nao_negativa = bool(np.all(y_original >= 0))
    reais, previstos, erros_transformados = [], [], []
    for corte in range(inicio_teste, n):
        treino = y_original[:corte]
        treino_z = np.log1p(np.maximum(treino, 0.0)) if transformacao == "log1p" else treino.astype(float)
        pz = _prever_mensal_arr(treino_z, meses[:corte], meses[corte:corte + 1], modelo)[0]
        pv = float(_destransformar_valores(np.asarray([pz]), transformacao)[0])
        if nao_negativa:
            pv = max(pv, 0.0)
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
    fim_serie: pd.Period | None = None,
) -> dict[str, Any]:
    """Projeção probabilística mensal com seleção por backtesting e bootstrap de erros."""
    horizonte = int(max(1, min(36, horizonte)))
    simulacoes = int(max(500, min(10000, simulacoes)))
    dados = preparar_serie_mensal(df, coluna_data, coluna_valor, hoje=hoje, fim=fim_serie)
    historico = dados.historico
    if len(historico) < MIN_MESES:
        return {
            "ok": False,
            "erro": f"São necessários pelo menos {MIN_MESES} meses encerrados para a projeção mensal probabilística. Há {len(historico)} mês(es) disponíveis.",
            "historico": historico, "mes_parcial": dados.mes_parcial,
            "valor_parcial": dados.valor_parcial, "ultima_data": dados.ultima_data,
        }

    y = historico.to_numpy(dtype=float)
    # Escala original (R$/unidades), não log1p: vendas por SKU são intermitentes (muitos zeros) e
    # erros em log1p chegam a ±10; ao aplicar expm1 e somar centenas de SKUs, as faixas explodem
    # (bilhões). Na escala original a soma dos SKUs é aditiva e o piso em zero evita negativos.
    z, transformacao = y.astype(float), "aditiva"
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

    # Preserva os erros históricos observados. Não aplicamos clipping por MAD ou
    # fator multiplicativo: os extremos precisam ser diagnosticados, não escondidos.
    # A validação de qualidade do intervalo deve ocorrer no backtest e no diagnóstico.

    # Centraliza os choques: o intervalo expressa a dispersão, e não a tendência
    # sistemática de erro do modelo (que é mostrada separadamente no viés do backtest).
    # Sem centralização por mediana: em séries intermitentes a mediana do erro é negativa
    # (muitos meses zerados) e centralizar inflaria a média simulada. Os erros já carregam o viés.

    # Não limitar artificialmente os choques. Se o intervalo ficar extremo,
    # isso deve refletir os erros históricos amostrados e ser investigado com
    # diagnóstico por SKU, em vez de impor um teto arbitrário.

    rng = np.random.default_rng(semente)
    choques = rng.choice(erros, size=(simulacoes, horizonte), replace=True)

    # Cada previsão é um nível mensal projetado. Somar cumulativamente os erros
    # transformados cria um passeio aleatório artificial e faz as faixas explodirem
    # exponencialmente com log1p (ex.: milhões de unidades em séries pequenas).
    # Sorteamos um erro histórico por mês, sem acumulá-lo sobre os meses anteriores.
    caminhos_z = prev_z[None, :] + choques
    caminhos = _destransformar_valores(caminhos_z, transformacao)
    if np.all(y >= 0):
        caminhos = np.maximum(caminhos, 0.0)
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
        "ultima_data": dados.ultima_data, "_caminhos_simulados": caminhos,
        "observacao": "Projeção calculada localmente por mês. Os intervalos são percentis empíricos dos erros históricos amostrados, sem limites artificiais. Intervalos extremos devem ser investigados na distribuição dos erros e no backtest; não há garantia de cobertura estatística.",
    }


# ============================================================
# PROJEÇÃO CONSOLIDADA COM SELEÇÃO DE MODELO POR SKU
# ============================================================

def _projetar_consolidado_por_sku(
    df: pd.DataFrame,
    funcao_projecao,
    coluna_sku: str,
    horizonte: int,
    simulacoes: int,
    coluna_data: str,
    coluna_valor: str,
    semente: int,
    hoje: Any | None,
    frequencia: str,
) -> dict[str, Any]:
    """Seleciona o melhor modelo individualmente por SKU e soma as distribuições previstas.

    Os intervalos consolidados são calculados somando as simulações de cada SKU
    caminho a caminho e só depois calculando os percentis. Isso evita somar
    percentis individuais, operação que não representa corretamente a distribuição
    da soma.
    """
    obrigatorias = {coluna_sku, coluna_data, coluna_valor}
    ausentes = obrigatorias.difference(df.columns)
    if ausentes:
        return {
            "ok": False,
            "erro": "Colunas obrigatórias ausentes: " + ", ".join(sorted(ausentes)),
        }

    base = df[[coluna_sku, coluna_data, coluna_valor]].copy()
    base[coluna_sku] = base[coluna_sku].astype("string").str.strip()
    base[coluna_data] = pd.to_datetime(base[coluna_data], errors="coerce", dayfirst=True)
    base[coluna_valor] = pd.to_numeric(base[coluna_valor], errors="coerce")
    base = base.dropna(subset=[coluna_sku, coluna_data, coluna_valor])
    base = base[base[coluna_sku].ne("")]
    if base.empty:
        return {"ok": False, "erro": "Não há registros válidos com SKU, data e valor."}

    resultados = []
    skus_sem_projecao = []
    extra = {"fim_serie": base[coluna_data].max().to_period("M")} if frequencia == "M" else {}
    for i, (sku, grupo) in enumerate(base.groupby(coluna_sku, sort=True, observed=True)):
        resultado = funcao_projecao(
            grupo,
            horizonte=horizonte,
            simulacoes=simulacoes,
            coluna_data=coluna_data,
            coluna_valor=coluna_valor,
            semente=semente + i,
            hoje=hoje,
            **extra,
        )
        if resultado.get("ok"):
            resultados.append((str(sku), resultado))
        else:
            skus_sem_projecao.append({
                "SKU": str(sku),
                "Motivo": resultado.get("erro", "Não foi possível gerar a projeção."),
            })

    if not resultados:
        return {
            "ok": False,
            "erro": "Nenhum SKU possui histórico suficiente para gerar projeção.",
            "skus_sem_projecao": pd.DataFrame(skus_sem_projecao),
        }

    # Soma os caminhos simulados alinhados por número de simulação e horizonte.
    caminhos_soma = np.sum(
        [np.asarray(r["_caminhos_simulados"], dtype=float) for _, r in resultados],
        axis=0,
    )
    percentis = {
        p: np.quantile(caminhos_soma, p / 100.0, axis=0)
        for p in (5, 10, 50, 90, 95)
    }

    previsoes_base = resultados[0][1]["previsoes"].copy()
    if frequencia == "M":
        previsoes = pd.DataFrame({
            "Mês": previsoes_base["Mês"],
            "Período": previsoes_base["Período"],
            "Previsão central (P50)": percentis[50],
            "Limite inferior 80% (P10)": percentis[10],
            "Limite superior 80% (P90)": percentis[90],
            "Limite inferior 90% (P05)": percentis[5],
            "Limite superior 90% (P95)": percentis[95],
        })
    else:
        previsoes = pd.DataFrame({
            "Trimestre": previsoes_base["Trimestre"],
            "Período": previsoes_base["Período"],
            "Previsão central (P50)": percentis[50],
            "Limite inferior 80% (P10)": percentis[10],
            "Limite superior 80% (P90)": percentis[90],
            "Limite inferior 90% (P05)": percentis[5],
            "Limite superior 90% (P95)": percentis[95],
        })

    # Consolida o histórico dos SKUs projetáveis, sem duplicar períodos ausentes.
    series_historicas = [r["historico"] for _, r in resultados]
    historico_consolidado = pd.concat(series_historicas, axis=1).fillna(0.0).sum(axis=1).sort_index()
    hist_df = pd.DataFrame({
        "Período": historico_consolidado.index,
        "Valor realizado": historico_consolidado.to_numpy(dtype=float),
        "Tipo": "Realizado",
    })
    if frequencia == "M":
        hist_df["Mês"] = [p.strftime("%m/%Y") for p in historico_consolidado.index]
        hist_df = hist_df[["Período", "Mês", "Valor realizado", "Tipo"]]
    else:
        hist_df["Trimestre"] = [rotulo_trimestre(p) for p in historico_consolidado.index]
        hist_df = hist_df[["Período", "Trimestre", "Valor realizado", "Tipo"]]

    modelos_por_sku = pd.DataFrame([
        {
            "SKU": sku,
            "Modelo selecionado": r["modelo_rotulo"],
            "WAPE retroativo (%)": r["wape"],
            "MAE (R$)": r["mae"],
            "Viés médio (R$)": r["bias"],
            "Períodos históricos": r.get("n_meses", r.get("n_trimestres")),
        }
        for sku, r in resultados
    ]).sort_values("SKU").reset_index(drop=True)

    # Métricas consolidadas calculadas sobre a soma das séries reais históricas.
    # A seleção do modelo continua sendo por SKU; esta métrica é apenas diagnóstico.
    reais = historico_consolidado.to_numpy(dtype=float)
    if len(reais) > 1:
        previos = reais[:-1]
        observados = reais[1:]
        mae_consolidado = float(np.mean(np.abs(observados - previos)))
        denom = float(np.sum(np.abs(observados)))
        wape_consolidado = float(np.sum(np.abs(observados - previos)) / denom * 100) if denom > 1e-9 else mae_consolidado
    else:
        mae_consolidado, wape_consolidado = np.nan, np.nan

    return {
        "ok": True,
        "previsoes": previsoes,
        "historico": historico_consolidado,
        "historico_df": hist_df,
        "modelo": "seleção_individual_por_sku",
        "modelo_rotulo": "Melhor modelo selecionado individualmente por SKU",
        "modelos_por_sku": modelos_por_sku,
        "skus_projetados": len(resultados),
        "skus_sem_projecao": pd.DataFrame(skus_sem_projecao),
        "wape": wape_consolidado,
        "mae": mae_consolidado,
        "simulacoes": int(caminhos_soma.shape[0]),
        "observacao": (
            "Cada SKU elegível foi projetado com o modelo de menor erro retroativo. "
            "A projeção consolidada é a soma das previsões por SKU; os intervalos foram "
            "calculados somando as simulações individuais antes de obter os percentis. "
            "SKUs sem histórico mínimo estão listados em 'skus_sem_projecao' e não entram na soma."
        ),
    }


def projetar_meses_consolidado(
    df: pd.DataFrame,
    horizonte: int = 6,
    simulacoes: int = SIMULACOES_PADRAO,
    coluna_data: str = "_data",
    coluna_valor: str = "_valor",
    coluna_sku: str = "SKU",
    semente: int = SEMENTE_PADRAO,
    hoje: Any | None = None,
) -> dict[str, Any]:
    """Projeção mensal consolidada: escolhe o melhor modelo para cada SKU e soma os resultados."""
    return _projetar_consolidado_por_sku(
        df, projetar_meses, coluna_sku, horizonte, simulacoes,
        coluna_data, coluna_valor, semente, hoje, "M",
    )


def projetar_trimestres_consolidado(
    df: pd.DataFrame,
    horizonte: int = 4,
    simulacoes: int = SIMULACOES_PADRAO,
    coluna_data: str = "_data",
    coluna_valor: str = "_valor",
    coluna_sku: str = "SKU",
    semente: int = SEMENTE_PADRAO,
    hoje: Any | None = None,
) -> dict[str, Any]:
    """Projeção trimestral consolidada: escolhe o melhor modelo para cada SKU e soma os resultados."""
    return _projetar_consolidado_por_sku(
        df, projetar_trimestres, coluna_sku, horizonte, simulacoes,
        coluna_data, coluna_valor, semente, hoje, "Q",
    )


# ============================================================
# PROJEÇÃO MENSAL POR SKU COM RESTRIÇÃO DE ESTOQUE E BACKTEST
# ============================================================

def projetar_meses_com_estoque(
    df: pd.DataFrame,
    estoque: pd.DataFrame,
    horizonte: int = 6,
    simulacoes: int = SIMULACOES_PADRAO,
    coluna_data: str = "_data",
    coluna_valor: str = "_valor",
    coluna_sku: str = "SKU",
    coluna_estoque_sku: str = "SKU",
    coluna_estoque_quantidade: str = "Quantidade",
    coluna_estoque_data: str = "Data",
    semente: int = SEMENTE_PADRAO,
    hoje: Any | None = None,
) -> dict[str, Any]:
    """
    Projeta a demanda individualmente por SKU, aplica o saldo de estoque em cada
    caminho probabilístico e soma os caminhos já limitados. Também compara WAPE
    baseline vs. previsão limitada por estoque em backtest walk-forward.

    Estoque futuro não é conhecido: usa o último saldo disponível e assume zero
    reposição durante o horizonte. Para faturamento, converte saldo em unidades
    usando receita média histórica por unidade do SKU.
    """
    required_sales = {coluna_sku, coluna_data, coluna_valor}
    required_stock = {coluna_estoque_sku, coluna_estoque_quantidade, coluna_estoque_data}
    if not required_sales.issubset(df.columns):
        return {"ok": False, "erro": "A base de vendas precisa conter: " + ", ".join(sorted(required_sales - set(df.columns)))}
    if not required_stock.issubset(estoque.columns):
        return {"ok": False, "erro": "O estoque precisa conter: " + ", ".join(sorted(required_stock - set(estoque.columns)))}

    tem_q = coluna_valor != "_quantidade" and "_quantidade" in df.columns
    vendas = df[[coluna_sku, coluna_data, coluna_valor] + (["_quantidade"] if tem_q else [])].copy()
    vendas[coluna_sku] = vendas[coluna_sku].astype("string").str.strip()
    vendas[coluna_data] = pd.to_datetime(vendas[coluna_data], errors="coerce", dayfirst=True)
    vendas[coluna_valor] = pd.to_numeric(vendas[coluna_valor], errors="coerce")
    if tem_q:
        vendas["_quantidade"] = pd.to_numeric(vendas["_quantidade"], errors="coerce")
    vendas = vendas.dropna(subset=[coluna_sku, coluna_data, coluna_valor])
    vendas = vendas[vendas[coluna_sku].ne("")]
    if vendas.empty:
        return {"ok": False, "erro": "Não há vendas válidas para projetar."}
    vendas["_mes_p"] = vendas[coluna_data].dt.to_period("M")
    vendas[coluna_sku] = vendas[coluna_sku].astype(str)

    stk = estoque[[coluna_estoque_sku, coluna_estoque_quantidade, coluna_estoque_data]].copy()
    stk.columns = ["_sku_stock", "_qty_stock", "_date_stock"]
    stk["_sku_stock"] = stk["_sku_stock"].astype("string").str.strip()
    stk["_qty_stock"] = pd.to_numeric(stk["_qty_stock"], errors="coerce")
    stk["_date_stock"] = pd.to_datetime(stk["_date_stock"], errors="coerce", dayfirst=True)
    stk = stk.dropna(subset=["_sku_stock", "_qty_stock", "_date_stock"])
    stk = stk[stk["_sku_stock"].ne("") & stk["_qty_stock"].ge(0)]
    if stk.empty:
        return {"ok": False, "erro": "O histórico de estoque não contém saldos válidos."}
    # Se houver linhas repetidas SKU/data (ex.: depósitos), soma os saldos.
    stk = stk.groupby(["_sku_stock", "_date_stock"], as_index=False)["_qty_stock"].sum()
    stk["_sku_stock"] = stk["_sku_stock"].astype(str)
    # Um único agrupamento por SKU, em vez de filtrar o DataFrame inteiro a cada SKU.
    stk_por_sku = {k: g.sort_values("_date_stock") for k, g in stk.groupby("_sku_stock", sort=False)}
    hoje_ts = pd.Timestamp(hoje).normalize() if hoje is not None else pd.Timestamp.today().normalize()
    mes_corrente = hoje_ts.to_period("M")
    futuros = pd.period_range(mes_corrente + 1, periods=int(max(1, min(36, horizonte))), freq="M")
    simulacoes = int(max(500, min(10000, simulacoes)))

    resultados = []
    diagnostico_skus = []
    skus_sem = []
    # Receita por unidade (valor/quantidade) por SKU, calculada uma vez.
    if tem_q:
        _q = vendas.dropna(subset=["_quantidade"]).groupby(coluna_sku)[[coluna_valor, "_quantidade"]].sum()
        receita_por_sku = (_q[coluna_valor] / _q["_quantidade"].where(_q["_quantidade"] > 0)).fillna(0.0).to_dict()
    else:
        receita_por_sku = {}
    fim_global = vendas[coluna_data].max().to_period("M")
    for i, (sku, grupo) in enumerate(vendas.groupby(coluna_sku, sort=True, observed=True)):
        grupo = grupo.sort_values(coluna_data).copy()
        # A função mensal já escolhe o modelo por WAPE e produz os caminhos simulados.
        r = projetar_meses(
            grupo, horizonte=len(futuros), simulacoes=simulacoes,
            coluna_data=coluna_data, coluna_valor=coluna_valor,
            semente=semente + i, hoje=hoje_ts, fim_serie=fim_global,
        )
        if not r.get("ok"):
            skus_sem.append({"SKU": str(sku), "Motivo": r.get("erro", "Histórico insuficiente")})
            continue
        hist_q = grupo.copy()
        hist_q["_mes"] = hist_q[coluna_data].dt.to_period("M")
        mensal_valor = hist_q.groupby("_mes")[coluna_valor].sum()
        # A quantidade é necessária para converter estoque em faturamento.
        # Se a própria métrica for quantidade, o saldo já está na mesma unidade.
        receita_unidade = np.nan
        if tem_q:
            receita_unidade = float(receita_por_sku.get(str(sku), 0.0))

        snapshots = stk_por_sku.get(str(sku))
        if snapshots is None or snapshots.empty:
            skus_sem.append({"SKU": str(sku), "Motivo": "Sem correspondência no histórico de estoque"})
            # Sem estoque correspondente, não limitar o caminho deste SKU.
            caminhos_aj = np.asarray(r["_caminhos_simulados"], dtype=float)
            saldo_inicial = np.nan
        else:
            snapshots_atuais = snapshots[snapshots["_date_stock"].le(hoje_ts)]
            if snapshots_atuais.empty:
                skus_sem.append({"SKU": str(sku), "Motivo": "Sem saldo de estoque com data até hoje"})
                caminhos_aj = np.asarray(r["_caminhos_simulados"], dtype=float)
                saldo_inicial = np.nan
            else:
                saldo_inicial = float(snapshots_atuais.iloc[-1]["_qty_stock"])
                saldo_por_caminho = np.full(simulacoes, saldo_inicial, dtype=float)
                caminhos_base = np.asarray(r["_caminhos_simulados"], dtype=float)
                caminhos_aj = np.zeros_like(caminhos_base)
                for j in range(caminhos_base.shape[1]):
                    if coluna_valor == "_quantidade":
                        unidades_demandadas = np.maximum(caminhos_base[:, j], 0.0)
                        unidades_atendidas = np.minimum(unidades_demandadas, saldo_por_caminho)
                        caminhos_aj[:, j] = unidades_atendidas
                    else:
                        if not np.isfinite(receita_unidade) or receita_unidade <= 0:
                            # Não inventa conversão se não houver quantidade para calcular preço médio.
                            caminhos_aj[:, j] = caminhos_base[:, j]
                            continue
                        unidades_demandadas = np.maximum(caminhos_base[:, j], 0.0) / receita_unidade
                        unidades_atendidas = np.minimum(unidades_demandadas, saldo_por_caminho)
                        caminhos_aj[:, j] = unidades_atendidas * receita_unidade
                    saldo_por_caminho = np.maximum(saldo_por_caminho - unidades_atendidas, 0.0)
        resultados.append((str(sku), r, caminhos_aj, saldo_inicial))
        diagnostico_skus.append({
            "SKU": str(sku), "Modelo selecionado": r.get("modelo_rotulo", r.get("modelo")),
            "WAPE individual (%)": r.get("wape", np.nan),
            "Estoque inicial considerado (unidades)": saldo_inicial,
            "Restrição aplicada": bool(np.isfinite(saldo_inicial)),
        })

    if not resultados:
        return {"ok": False, "erro": "Nenhum SKU possui histórico suficiente para projeção mensal.", "skus_sem_projecao": pd.DataFrame(skus_sem)}

    # O horizonte operacional começa no próximo mês-calendário. Não reutilize
    # as datas internas de um SKU: se a série individual tiver lacunas ou terminar
    # em mês antigo, isso desloca a faixa probabilística para dentro do histórico.
    # Os caminhos de cada SKU começam no primeiro mês após o histórico (ou no mês parcial);
    # os rótulos precisam vir da própria projeção. Usar `mes_corrente + 1` deslocava tudo
    # um mês e escondia o mês corrente (ex.: outubro).
    futuros = pd.PeriodIndex(resultados[0][1]["previsoes"]["Período"], freq="M")

    caminhos_base_total = np.sum([np.asarray(r["_caminhos_simulados"], dtype=float) for _, r, _, _ in resultados], axis=0)
    caminhos_estoque_total = np.sum([c for _, _, c, _ in resultados], axis=0)

    def montar_previsoes(caminhos):
        perc = {p: np.quantile(caminhos, p / 100.0, axis=0) for p in (5, 10, 50, 90, 95)}
        return pd.DataFrame({
            "Mês": [p.strftime("%m/%Y") for p in futuros],
            "Período": futuros,
            "Previsão central (P50)": perc[50],
            "Limite inferior 80% (P10)": perc[10],
            "Limite superior 80% (P90)": perc[90],
            "Limite inferior 90% (P05)": perc[5],
            "Limite superior 90% (P95)": perc[95],
        })
    previsoes_base = montar_previsoes(caminhos_base_total)
    previsoes_estoque = montar_previsoes(caminhos_estoque_total)

    # Backtest comparável: previsão um mês à frente com/sem limite de estoque conhecido
    # no mês anterior. Só entram SKU-mês com snapshot anterior ao mês avaliado.
    reais_bt, prev_base_bt, prev_estoque_bt, periodos_bt = [], [], [], []
    for sku, g in vendas.groupby(coluna_sku, sort=True):
        serie = g.groupby("_mes_p")[coluna_valor].sum().sort_index()
        if len(serie):
            serie = serie.reindex(pd.period_range(serie.index.min(), max(serie.index.max(), fim_global), freq="M"), fill_value=0.0)
        if len(serie) < MIN_MESES + 1:
            continue
        gs = stk_por_sku.get(sku)
        if gs is None or gs.empty:
            continue
        datas_estoque = gs["_date_stock"].to_numpy()
        saldos_estoque = gs["_qty_stock"].to_numpy(float)
        periodos = pd.PeriodIndex(serie.index, freq="M")
        meses_all = _meses_do_ano(periodos)
        y = serie.to_numpy(float)
        if coluna_valor != "_quantidade":
            if tem_q:
                mv = g.groupby("_mes_p")[[coluna_valor, "_quantidade"]].sum().reindex(periodos, fill_value=0.0)
                cum_v = np.concatenate([[0.0], np.cumsum(mv[coluna_valor].to_numpy(float))])
                cum_q = np.concatenate([[0.0], np.cumsum(mv["_quantidade"].to_numpy(float))])
            else:
                cum_v = cum_q = None
        for corte in range(max(MIN_MESES, len(y) - 8), len(y)):
            periodo_teste = periodos[corte]
            data_corte = np.datetime64((periodo_teste.start_time - pd.Timedelta(days=1)).to_datetime64())
            pos = int(np.searchsorted(datas_estoque, data_corte, side="right")) - 1
            if pos < 0:
                continue
            treino = y[:corte]
            modelos = _modelos_mensais_disponiveis(len(treino))
            # Escolhe o modelo usando apenas o passado disponível naquele corte.
            ranking_bt = []
            for mod in modelos:
                bt = _backtest_mensal(treino, meses_all[:corte], mod, "aditiva")
                ranking_bt.append((bt["wape"], bt["mae"], mod))
            modelo = sorted(ranking_bt, key=lambda x: (x[0], x[1]))[0][2]
            pred_z = _prever_mensal_arr(treino, meses_all[:corte], meses_all[corte:corte + 1], modelo)[0]
            pred = max(float(pred_z), 0.0)
            stock_units = float(saldos_estoque[pos])
            if coluna_valor == "_quantidade":
                pred_stock = min(max(pred, 0.0), stock_units)
            else:
                # Aproximação de valor unitário calculada só com o treino (meses anteriores ao testado).
                vu = float(cum_v[corte] / cum_q[corte]) if cum_q is not None and cum_q[corte] > 0 else 0.0
                pred_stock = min(max(pred, 0.0), stock_units * vu) if vu > 0 else pred
            periodos_bt.append(periodo_teste)
            reais_bt.append(float(y[corte]))
            prev_base_bt.append(float(pred))
            prev_estoque_bt.append(float(pred_stock))

    def wape(real, pred):
        if not real:
            return np.nan
        real_a, pred_a = np.asarray(real, float), np.asarray(pred, float)
        denom = np.sum(np.abs(real_a))
        return float(np.sum(np.abs(real_a - pred_a)) / denom * 100) if denom > 1e-9 else float(np.mean(np.abs(real_a-pred_a)))

    wape_base = wape(reais_bt, prev_base_bt)
    wape_stock = wape(reais_bt, prev_estoque_bt)
    mae_stock = float(np.mean(np.abs(np.asarray(reais_bt) - np.asarray(prev_estoque_bt)))) if reais_bt else np.nan
    # WAPE agregado: soma os SKUs de cada mês antes de medir o erro. É a métrica coerente com a
    # projeção consolidada exibida; o WAPE SKU-mês é mais severo (erros de sinais opostos não se compensam).
    wape_agr_stock = wape_agr_base = mae_agr = np.nan
    if reais_bt:
        bt_df = pd.DataFrame({"p": periodos_bt, "real": reais_bt, "base": prev_base_bt, "est": prev_estoque_bt})
        m = bt_df.groupby("p")[["real", "base", "est"]].sum()
        den = float(m["real"].abs().sum())
        if den > 1e-9:
            wape_agr_stock = float((m["est"] - m["real"]).abs().sum() / den * 100)
            wape_agr_base = float((m["base"] - m["real"]).abs().sum() / den * 100)
        mae_agr = float((m["est"] - m["real"]).abs().mean())
    comparacao_wape = pd.DataFrame([
        {"Cenário": "Baseline — sem restrição de estoque", "WAPE (%)": wape_base, "Observações": len(reais_bt)},
        {"Cenário": "Modelo condicionado ao estoque", "WAPE (%)": wape_stock, "Observações": len(reais_bt)},
    ])
    # Consolida o histórico dos SKUs efetivamente projetados.
    historicos = [r["historico"] for _, r, _, _ in resultados]
    hist_consolidado = pd.concat(historicos, axis=1).fillna(0.0).sum(axis=1).sort_index()
    historico_df_consolidado = pd.DataFrame({
        "Período": hist_consolidado.index,
        "Mês": [p.strftime("%m/%Y") for p in hist_consolidado.index],
        "Valor realizado": hist_consolidado.to_numpy(dtype=float),
        "Tipo": "Realizado",
    })

    return {
        "ok": True,
        "previsoes": previsoes_estoque,
        "previsoes_sem_estoque": previsoes_base,
        "historico_df": historico_df_consolidado,
        "historico": hist_consolidado,
        "modelo": "seleção_individual_por_sku_com_estoque",
        "modelo_rotulo": "Seleção individual por SKU + restrição de estoque",
        "modelos_por_sku": pd.DataFrame(diagnostico_skus),
        "skus_projetados": len(resultados),
        "skus_sem_projecao": pd.DataFrame(skus_sem),
        "wape": wape_agr_stock if np.isfinite(wape_agr_stock) else wape_stock,
        "mae": mae_agr if np.isfinite(mae_agr) else mae_stock,
        "wape_agregado_mensal": wape_agr_stock,
        "wape_agregado_mensal_sem_estoque": wape_agr_base,
        "wape_sku_mes": wape_stock,
        "mae_sku_mes": mae_stock,
        "wape_backtest_sem_estoque": wape_base,
        "wape_backtest_com_estoque": wape_stock,
        "comparacao_wape_estoque": comparacao_wape,
        "n_testes_backtest_estoque": len(reais_bt),
        "simulacoes": simulacoes,
        "_caminhos_simulados": caminhos_estoque_total,
        "_caminhos_sem_estoque": caminhos_base_total,
        "observacao": (
            "Cada SKU foi previsto individualmente e limitado pelo estoque mais recente disponível; "
            "os caminhos limitados foram somados antes do cálculo dos percentis. O cenário assume zero reposição. "
            "A comparação de WAPE usa backtest walk-forward nos SKU-mês com snapshot de estoque anterior ao mês avaliado; "
            "a melhora só pode ser concluída se houver amostra suficiente e o WAPE condicionado for menor."
        ),
    }
