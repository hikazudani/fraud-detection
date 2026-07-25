"""Cálculos usados pela dashboard: sweep de threshold, matriz de custo e curva PR.

Este módulo é intencionalmente livre de Streamlit para poder ser reaproveitado
pelo script `scripts/exportar_artefatos_dashboard.py` e testado por linha de
comando.

A matriz de custo segue a definição da especificação do projeto:

    custo_total = FP * custo_FP + FN * custo_FN

Como o custo é linear em FP e FN, um sweep pré-calculado com as contagens da
matriz de confusão por threshold permite recalcular o custo para qualquer par
de custos instantaneamente, sem reprocessar as predições.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUNAS_SWEEP = ["threshold", "tp", "fp", "fn", "tn", "precision", "recall", "f1"]

ROTULOS_LINHAS_CONFUSAO = ["Real legítima", "Real fraude"]
ROTULOS_COLUNAS_CONFUSAO = ["Prevista legítima", "Prevista fraude"]


def custo_total(fp: float, fn: float, custo_fp: float, custo_fn: float) -> float:
    """Custo total simulado de uma matriz de confusão.

    Espelha `calcular_custo` do notebook de modelagem.
    """
    return float(fp) * float(custo_fp) + float(fn) * float(custo_fn)


def grade_thresholds(
    n_thresholds: int = 200,
    threshold_operacional: float | None = None,
) -> np.ndarray:
    """Grade de thresholds concentrada nos valores baixos.

    Modelos com `scale_pos_weight` costumam operar em thresholds bem pequenos,
    por isso parte da grade é geométrica entre 1e-4 e 1e-2. O threshold
    operacional escolhido na validação é sempre incluído, garantindo que a
    dashboard consiga reproduzir exatamente o ponto de operação do relatório.
    """
    n_baixos = max(int(n_thresholds * 0.2), 10)
    n_altos = max(n_thresholds - n_baixos, 10)

    grade = np.concatenate([
        np.geomspace(1e-4, 1e-2, n_baixos),
        np.linspace(1e-2, 0.99, n_altos),
    ])

    if threshold_operacional is not None and np.isfinite(threshold_operacional):
        grade = np.append(grade, float(threshold_operacional))

    return np.unique(np.round(grade, 8))


def calcular_sweep_threshold(
    y_true,
    y_score,
    n_thresholds: int = 200,
    threshold_operacional: float | None = None,
) -> pd.DataFrame:
    """Contagens da matriz de confusão e métricas para uma grade de thresholds.

    A predição positiva é `y_score >= threshold`, igual ao notebook. O cálculo
    ordena os scores uma única vez e usa busca binária para obter TP e FP em
    toda a grade de forma vetorizada, evitando varrer as predições por
    threshold.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()

    if y_true.shape != y_score.shape:
        raise ValueError(
            f"y_true e y_score têm tamanhos diferentes: {y_true.shape} vs {y_score.shape}."
        )

    scores_positivos = np.sort(y_score[y_true == 1])
    scores_negativos = np.sort(y_score[y_true == 0])

    n_positivos = scores_positivos.size
    n_negativos = scores_negativos.size

    thresholds = grade_thresholds(n_thresholds, threshold_operacional)

    # Quantidade de scores >= threshold em cada classe.
    tp = n_positivos - np.searchsorted(scores_positivos, thresholds, side="left")
    fp = n_negativos - np.searchsorted(scores_negativos, thresholds, side="left")

    fn = n_positivos - tp
    tn = n_negativos - fp

    sweep = pd.DataFrame({
        "threshold": thresholds,
        "tp": tp.astype(int),
        "fp": fp.astype(int),
        "fn": fn.astype(int),
        "tn": tn.astype(int),
    })

    return adicionar_metricas_derivadas(sweep)


def adicionar_metricas_derivadas(sweep: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta precision, recall e F1 a partir das contagens de confusão."""
    tabela = sweep.copy()

    tp = tabela["tp"].to_numpy(dtype=float)
    fp = tabela["fp"].to_numpy(dtype=float)
    fn = tabela["fn"].to_numpy(dtype=float)

    previstos_positivos = tp + fp
    reais_positivos = tp + fn

    # zero_division=0, mesma convenção do notebook.
    tabela["precision"] = np.divide(
        tp, previstos_positivos,
        out=np.zeros_like(tp), where=previstos_positivos > 0,
    )
    tabela["recall"] = np.divide(
        tp, reais_positivos,
        out=np.zeros_like(tp), where=reais_positivos > 0,
    )

    soma = tabela["precision"] + tabela["recall"]
    tabela["f1"] = np.divide(
        2 * tabela["precision"] * tabela["recall"], soma,
        out=np.zeros_like(tp), where=soma > 0,
    )

    return tabela[COLUNAS_SWEEP]


def aplicar_matriz_custo(
    sweep: pd.DataFrame,
    custo_fp: float,
    custo_fn: float,
) -> pd.DataFrame:
    """Recalcula o custo total de cada linha do sweep para os custos informados."""
    tabela = sweep.copy()
    tabela["custo_total"] = tabela["fp"] * float(custo_fp) + tabela["fn"] * float(custo_fn)
    return tabela


def linha_de_menor_custo(sweep_com_custo: pd.DataFrame) -> pd.Series:
    """Linha do sweep com menor custo total, desempatando por F1 maior."""
    ordenado = sweep_com_custo.sort_values(
        by=["custo_total", "f1", "recall"],
        ascending=[True, False, False],
    )
    return ordenado.iloc[0]


def linha_no_threshold(sweep: pd.DataFrame, threshold: float) -> pd.Series:
    """Linha do sweep cujo threshold é o mais próximo do valor informado."""
    distancias = (sweep["threshold"] - float(threshold)).abs()
    return sweep.loc[distancias.idxmin()]


def calcular_curva_pr(y_true, y_score, max_pontos: int = 500) -> pd.DataFrame:
    """Curva Precision-Recall, opcionalmente reduzida a `max_pontos` pontos.

    O último ponto devolvido por `precision_recall_curve` (recall 0, precision 1)
    não tem threshold correspondente e recebe `NaN` na coluna `threshold`.
    """
    from sklearn.metrics import precision_recall_curve

    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    curva = pd.DataFrame({
        "recall": recall,
        "precision": precision,
        "threshold": np.append(thresholds, np.nan),
    })

    return reduzir_pontos(curva, max_pontos)


def reduzir_pontos(curva: pd.DataFrame, max_pontos: int) -> pd.DataFrame:
    """Downsample uniforme preservando o primeiro e o último ponto."""
    if max_pontos is None or len(curva) <= max_pontos:
        return curva.reset_index(drop=True)

    indices = np.unique(
        np.linspace(0, len(curva) - 1, max_pontos).astype(int)
    )

    return curva.iloc[indices].reset_index(drop=True)


def calcular_auprc(y_true, y_score) -> float:
    """AUPRC (average precision) da classe fraude."""
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(np.asarray(y_true).astype(int), np.asarray(y_score, dtype=float)))


def matriz_confusao_df(tp: int, fp: int, fn: int, tn: int) -> pd.DataFrame:
    """Matriz de confusão 2x2 com os mesmos rótulos usados no notebook."""
    return pd.DataFrame(
        [[int(tn), int(fp)], [int(fn), int(tp)]],
        index=ROTULOS_LINHAS_CONFUSAO,
        columns=ROTULOS_COLUNAS_CONFUSAO,
    )


def matriz_confusao_longa(tp: int, fp: int, fn: int, tn: int) -> pd.DataFrame:
    """Versão longa da matriz de confusão, pronta para o heatmap do Altair."""
    return pd.DataFrame([
        {"real": "Real legítima", "previsto": "Prevista legítima", "quantidade": int(tn), "sigla": "VN"},
        {"real": "Real legítima", "previsto": "Prevista fraude", "quantidade": int(fp), "sigla": "FP"},
        {"real": "Real fraude", "previsto": "Prevista legítima", "quantidade": int(fn), "sigla": "FN"},
        {"real": "Real fraude", "previsto": "Prevista fraude", "quantidade": int(tp), "sigla": "VP"},
    ])
