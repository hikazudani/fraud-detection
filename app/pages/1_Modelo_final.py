"""Página do modelo final: matriz de confusão e curva Precision-Recall."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import charts, loaders, metrics  # noqa: E402

st.set_page_config(page_title="Modelo final", page_icon="🎯", layout="wide")


def _milhar(valor: int) -> str:
    """Formata um inteiro com ponto como separador de milhar."""
    return f"{int(valor):,}".replace(",", ".")


def main() -> None:
    loaders.render_sidebar_fontes()

    st.title("Modelo final")

    metadata = loaders.carregar_metadata()
    modelo_final = loaders.carregar_modelo_final()

    if modelo_final is None:
        loaders.exigir_dados(pd.DataFrame())
        return

    _cabecalho(modelo_final, metadata)

    st.divider()
    _secao_matriz_confusao(modelo_final)

    st.divider()
    _secao_curva_pr(modelo_final)

    st.divider()
    _nota_cenario()


def _cabecalho(modelo_final: pd.Series, metadata: dict) -> None:
    """Identificação do modelo, cenário e threshold operacional."""
    st.subheader(modelo_final["modelo"])

    threshold = loaders.threshold_operacional()

    colunas = st.columns(3)
    colunas[0].metric("Cenário de features", str(metadata.get("melhor_cenario_features", "—")))
    colunas[1].metric(
        "Threshold operacional",
        f"{threshold:.4f}" if threshold is not None else "não publicado",
    )
    colunas[2].metric("AUPRC no teste", f"{pd.to_numeric(modelo_final.get('auprc'), errors='coerce'):.4f}")

    if threshold is None:
        st.caption(
            "O valor numérico do threshold não consta na fonte de dados atual. "
            "Ele aparece aqui quando o metadata do pipeline estiver disponível."
        )


def _secao_matriz_confusao(modelo_final: pd.Series) -> None:
    """Matriz de confusão do modelo final no conjunto de teste."""
    st.subheader("Matriz de confusão no conjunto de teste")

    contagens = loaders.contagens_confusao_modelo_final()

    if contagens is None:
        st.info(
            "A fonte de dados atual não traz as quatro contagens da matriz de "
            "confusão (TP, FP, FN, TN)."
        )
        return

    tp, fp, fn, tn = contagens["tp"], contagens["fp"], contagens["fn"], contagens["tn"]
    total_fraudes = tp + fn
    total_legitimas = tn + fp

    esquerda, direita = st.columns([3, 2])

    with esquerda:
        charts.exibir(
            charts.heatmap_confusao(metrics.matriz_confusao_longa(tp, fp, fn, tn)),
            altura=220,
        )

    with direita:
        st.metric("Fraudes detectadas (VP)", _milhar(tp))
        st.metric("Fraudes não detectadas (FN)", _milhar(fn))
        st.metric("Alertas falsos (FP)", _milhar(fp))

    st.markdown(
        f"O modelo detectou **{_milhar(tp)}** das **{_milhar(total_fraudes)}** fraudes "
        f"do conjunto de teste, deixando passar **{_milhar(fn)}**. Ao mesmo tempo, "
        f"classificou **{_milhar(fp)}** das **{_milhar(total_legitimas)}** transações "
        "legítimas como fraude."
    )

    with st.expander("Ver a matriz como tabela"):
        st.dataframe(metrics.matriz_confusao_df(tp, fp, fn, tn), width="stretch")


def _secao_curva_pr(modelo_final: pd.Series) -> None:
    """Curva Precision-Recall, ou o ponto de operação quando a curva não existe."""
    st.subheader("Curva Precision-Recall")

    precision = pd.to_numeric(modelo_final.get("precision"), errors="coerce")
    recall = pd.to_numeric(modelo_final.get("recall"), errors="coerce")
    auprc = pd.to_numeric(modelo_final.get("auprc"), errors="coerce")
    prevalencia = loaders.prevalencia_teste()

    ponto_operacao = (
        {"recall": float(recall), "precision": float(precision)}
        if pd.notna(recall) and pd.notna(precision)
        else None
    )

    curva = loaders.carregar_curva_pr()

    if curva is not None:
        charts.exibir(
            charts.curva_pr(curva, ponto_operacao=ponto_operacao, prevalencia=prevalencia)
        )
        st.caption(
            f"AUPRC = {auprc:.4f}. A curva vem das predições salvas do modelo final "
            "no conjunto de teste."
        )
        return

    # Sem as predições salvas, mostra só o que existe: o ponto de operação.
    st.warning(
        "A curva completa exige as predições do conjunto de teste "
        "(`predicoes_teste_melhor_modelo.csv`) ou o derivado leve "
        "(`curva_precision_recall.csv`). Nenhum dos dois está disponível, então o "
        "gráfico abaixo mostra apenas o ponto de operação publicado."
    )

    if ponto_operacao is None:
        return

    curva_minima = pd.DataFrame([{**ponto_operacao, "threshold": None}])

    charts.exibir(
        charts.curva_pr(curva_minima, ponto_operacao=ponto_operacao, prevalencia=prevalencia)
    )

    if pd.notna(auprc):
        st.caption(
            f"AUPRC publicado = {auprc:.4f}"
            + (
                f" · prevalência de fraude no teste = {prevalencia:.4f} "
                "(precision de um classificador aleatório)"
                if prevalencia is not None
                else ""
            )
        )

    st.markdown(
        "Para gerar a curva: execute o notebook de modelagem e, opcionalmente, "
        "`python scripts/exportar_artefatos_dashboard.py`."
    )


def _nota_cenario() -> None:
    st.caption(
        "O cenário `sem_pos_transacao` remove saldos finais e variáveis derivadas "
        "de inconsistência de saldo. O cenário `completo` chega a desempenho quase "
        "perfeito, mas foi tratado apenas como diagnóstico: essas variáveis podem "
        "carregar informação próxima do resultado da transação ou artefatos do "
        "simulador PaySim."
    )


main()
