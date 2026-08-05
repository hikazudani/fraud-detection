"""Simulador da matriz de custo: custo_FP, custo_FN e threshold de decisão.

O recálculo é aritmética vetorizada sobre um sweep de no máximo algumas centenas
de linhas, então roda a cada movimento dos sliders sem precisar de cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import charts, loaders, metrics  # noqa: E402

st.set_page_config(page_title="Simulador de custo", page_icon="⚖️", layout="wide")


def main() -> None:
    loaders.render_sidebar_fontes()

    st.title("Simulador da matriz de custo")
    st.caption("custo_total = FP × custo_FP + FN × custo_FN")

    metadata = loaders.carregar_metadata()
    modelo_final = loaders.carregar_modelo_final()
    comparacao = loaders.carregar_comparacao_modelos()

    if modelo_final is None and len(comparacao) == 0:
        loaders.exigir_dados(comparacao)
        return

    custos_padrao = metadata["custos"]
    sweep = loaders.carregar_sweep()

    custo_fp, custo_fn = _sliders_de_custo(custos_padrao)

    st.divider()

    if sweep is not None:
        _simulador_com_threshold(sweep, custo_fp, custo_fn, modelo_final)
    else:
        _simulador_sem_threshold(modelo_final, custo_fp, custo_fn)

    st.divider()
    _custo_por_modelo(comparacao, custo_fp, custo_fn, modelo_final)


def _sliders_de_custo(custos_padrao: dict) -> tuple[float, float]:
    """Sliders de custo unitário de falso positivo e falso negativo."""
    esquerda, direita = st.columns(2)

    custo_fp = esquerda.slider(
        "Custo de um falso positivo (FP)",
        min_value=0.0,
        max_value=200.0,
        value=float(custos_padrao["custo_fp"]),
        step=1.0,
        help="Transação legítima bloqueada: atrito com o cliente.",
    )

    custo_fn = direita.slider(
        "Custo de um falso negativo (FN)",
        min_value=0.0,
        max_value=5000.0,
        value=float(custos_padrao["custo_fn"]),
        step=10.0,
        help="Fraude não detectada: perda financeira.",
    )

    razao = custo_fn / custo_fp if custo_fp > 0 else None
    st.caption(
        f"Um falso negativo custa {razao:,.1f}× um falso positivo."
        if razao is not None
        else "Com custo_FP igual a zero, apenas os falsos negativos pesam no custo."
    )

    return custo_fp, custo_fn


def _simulador_com_threshold(
    sweep: pd.DataFrame,
    custo_fp: float,
    custo_fn: float,
    modelo_final: pd.Series | None,
) -> None:
    """Simulação completa: o threshold também é ajustável."""
    com_custo = metrics.aplicar_matriz_custo(sweep, custo_fp, custo_fn)

    threshold_operacional = loaders.threshold_operacional()
    linha_minima = metrics.linha_de_menor_custo(com_custo)

    minimo = float(com_custo["threshold"].min())
    maximo = float(com_custo["threshold"].max())
    valor_inicial = (
        float(threshold_operacional)
        if threshold_operacional is not None and minimo <= threshold_operacional <= maximo
        else float(linha_minima["threshold"])
    )

    threshold_escolhido = st.slider(
        "Threshold de decisão",
        min_value=minimo,
        max_value=maximo,
        value=valor_inicial,
        step=(maximo - minimo) / 500,
        format="%.4f",
        help="Uma transação é classificada como fraude quando a probabilidade prevista "
             "é maior ou igual ao threshold.",
    )

    linha = metrics.linha_no_threshold(com_custo, threshold_escolhido)

    _cartoes_simulacao(linha, com_custo, linha_minima, threshold_operacional, custo_fp, custo_fn)

    esquerda, direita = st.columns([3, 2])

    with esquerda:
        charts.exibir(
            charts.curva_custo(
                com_custo,
                threshold_escolhido=float(linha["threshold"]),
                threshold_minimo=float(linha_minima["threshold"]),
                threshold_operacional=threshold_operacional,
            )
        )

    with direita:
        st.markdown("**Matriz de confusão no threshold escolhido**")
        st.dataframe(
            metrics.matriz_confusao_df(
                tp=int(linha["tp"]), fp=int(linha["fp"]),
                fn=int(linha["fn"]), tn=int(linha["tn"]),
            ),
            width="stretch",
        )
        st.caption(
            f"Precision {linha['precision']:.4f} · Recall {linha['recall']:.4f} · "
            f"F1 {linha['f1']:.4f}"
        )

    st.caption(
        "O eixo do threshold está em escala logarítmica porque os thresholds úteis "
        "em classe rara se concentram nos valores baixos."
    )


def _cartoes_simulacao(
    linha: pd.Series,
    com_custo: pd.DataFrame,
    linha_minima: pd.Series,
    threshold_operacional: float | None,
    custo_fp: float,
    custo_fn: float,
) -> None:
    """Cartões de custo no threshold escolhido, mínimo possível e comparação."""
    colunas = st.columns(3)

    colunas[0].metric(
        "Custo no threshold escolhido",
        charts.formatar_milhar(linha["custo_total"]),
        help=f"FP = {int(linha['fp'])}, FN = {int(linha['fn'])}",
    )

    delta_minimo = linha["custo_total"] - linha_minima["custo_total"]
    colunas[1].metric(
        "Menor custo possível",
        charts.formatar_milhar(linha_minima["custo_total"]),
        delta=(
            f"{charts.formatar_milhar(-delta_minimo)} economizáveis"
            if delta_minimo > 0
            else "threshold já é o de menor custo"
        ),
        delta_color="inverse" if delta_minimo > 0 else "off",
        help=f"No threshold {linha_minima['threshold']:.4f}.",
    )

    if threshold_operacional is not None:
        linha_operacional = metrics.linha_no_threshold(com_custo, threshold_operacional)
        delta_operacional = linha["custo_total"] - linha_operacional["custo_total"]
        colunas[2].metric(
            "Custo no threshold operacional",
            charts.formatar_milhar(linha_operacional["custo_total"]),
            delta=f"{charts.formatar_milhar(delta_operacional)} no escolhido",
            delta_color="inverse",
            help=f"Threshold escolhido na validação: {threshold_operacional:.4f}.",
        )
    else:
        colunas[2].metric("Custo no threshold operacional", "não publicado")


def _simulador_sem_threshold(
    modelo_final: pd.Series | None, custo_fp: float, custo_fn: float
) -> None:
    """Simulação restrita ao FP/FN fixo do modelo final.

    Sem o sweep não é possível variar o threshold sem inventar contagens, então
    o slider correspondente é omitido e o motivo é explicado.
    """
    st.info(
        "**Slider de threshold indisponível nesta fonte de dados.** Variar o "
        "threshold exige as predições do conjunto de teste "
        "(`predicoes_teste_melhor_modelo.csv`) ou o sweep derivado "
        "(`curva_threshold.csv`). Os custos abaixo usam o FP e o FN do modelo "
        "final no threshold já otimizado."
    )

    if modelo_final is None:
        return

    fp = pd.to_numeric(modelo_final.get("fp"), errors="coerce")
    fn = pd.to_numeric(modelo_final.get("fn"), errors="coerce")

    if pd.isna(fp) or pd.isna(fn):
        st.warning("A fonte de dados atual não traz as contagens de FP e FN.")
        return

    custo = metrics.custo_total(fp, fn, custo_fp, custo_fn)
    custo_publicado = pd.to_numeric(modelo_final.get("custo_total"), errors="coerce")

    colunas = st.columns(3)
    colunas[0].metric(
        "Custo total simulado", charts.formatar_milhar(custo),
        delta=(
            f"{charts.formatar_milhar(custo - custo_publicado)} vs publicado"
            if pd.notna(custo_publicado)
            else None
        ),
        delta_color="inverse",
    )
    colunas[1].metric("Falsos positivos (FP)", charts.formatar_milhar(int(fp)))
    colunas[2].metric("Falsos negativos (FN)", charts.formatar_milhar(int(fn)))

    st.caption(
        f"Composição: {int(fp)} FP × {custo_fp:g} = "
        f"{charts.formatar_milhar(fp * custo_fp)} · "
        f"{int(fn)} FN × {custo_fn:g} = {charts.formatar_milhar(fn * custo_fn)}"
    )

    contagens = loaders.contagens_confusao_modelo_final()
    if contagens is not None:
        with st.expander("Matriz de confusão do modelo final"):
            st.dataframe(metrics.matriz_confusao_df(**contagens), width="stretch")


def _custo_por_modelo(
    comparacao: pd.DataFrame,
    custo_fp: float,
    custo_fn: float,
    modelo_final: pd.Series | None,
) -> None:
    """Custo de todos os modelos recalculado com os custos escolhidos."""
    st.subheader("Custo total por modelo com os custos escolhidos")

    tabela = comparacao.dropna(subset=["fp", "fn"]).copy()

    if len(tabela) == 0:
        st.info("A fonte de dados atual não traz FP e FN por modelo.")
        return

    tabela["custo_total"] = tabela["fp"] * custo_fp + tabela["fn"] * custo_fn
    tabela = tabela.sort_values("custo_total")

    destaque = modelo_final["modelo"] if modelo_final is not None else None

    charts.exibir(charts.barras_custo_por_modelo(tabela, destaque=destaque))

    melhor = tabela.iloc[0]
    frase = (
        f"Com esses custos, o menor custo total é do modelo **{melhor['modelo']}** "
        f"({charts.formatar_milhar(melhor['custo_total'])})"
    )

    if destaque is not None and destaque in set(tabela["modelo"]) and melhor["modelo"] != destaque:
        custo_destaque = tabela.loc[tabela["modelo"] == destaque, "custo_total"].iloc[0]
        frase += (
            f", enquanto o modelo final selecionado ({destaque}) fica em "
            f"{charts.formatar_milhar(custo_destaque)}."
        )
    else:
        frase += "."

    st.markdown(frase)

    st.caption(
        "A ordenação por custo pode divergir da ordenação por F1: quanto mais caro "
        "o falso negativo, mais o critério de custo favorece recall alto."
    )


main()
