"""Dashboard de detecção de fraude — visão geral e comparação de modelos.

Ponto de entrada do app. Execute a partir da raiz do repositório:

    streamlit run app/streamlit_app.py

A dashboard é somente de leitura: consome os resultados já salvos pelo pipeline
de modelagem e não treina nem executa modelos.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Permite `from utils import ...` tanto no app principal quanto nas páginas.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import charts, loaders  # noqa: E402

st.set_page_config(
    page_title="Detecção de fraude — resultados",
    page_icon="🛡️",
    layout="wide",
)

METRICAS_DISPONIVEIS = {
    "F1 (fraude)": ("f1", "decimal", "f1_minimo"),
    "Recall (fraude)": ("recall", "decimal", "recall_minimo"),
    "Precision (fraude)": ("precision", "decimal", "precision_minima"),
    "AUPRC": ("auprc", "decimal", None),
    "Custo total simulado": ("custo_total", "moeda", None),
}

CONFIG_COLUNAS = {
    "modelo": st.column_config.TextColumn("Modelo", width="medium"),
    "precision": st.column_config.NumberColumn("Precision", format="%.4f"),
    "recall": st.column_config.NumberColumn("Recall", format="%.4f"),
    "f1": st.column_config.ProgressColumn("F1", format="%.4f", min_value=0.0, max_value=1.0),
    "auprc": st.column_config.NumberColumn("AUPRC", format="%.4f"),
    "fp": st.column_config.NumberColumn("FP", format="%d"),
    "fn": st.column_config.NumberColumn("FN", format="%d"),
    "custo_total": st.column_config.NumberColumn("Custo total", format="%.0f"),
    "tempo_medio_ms": st.column_config.NumberColumn("Tempo médio (ms)", format="%.6f"),
    "tempo_p95_ms": st.column_config.NumberColumn("p95 (ms)", format="%.6f"),
}

COLUNAS_TABELA = [
    "modelo", "precision", "recall", "f1", "auprc",
    "fp", "fn", "custo_total", "tempo_medio_ms", "tempo_p95_ms",
]


def main() -> None:
    loaders.render_sidebar_fontes()

    st.title("Detecção de fraude em transações financeiras")
    st.caption(
        "Resultados salvos do pipeline de modelagem sobre o dataset PaySim. "
        "Fraude corresponde a cerca de 0,13% das transações, então a avaliação "
        "é orientada por métricas da classe minoritária, custo do erro e tempo "
        "de inferência — não por acurácia."
    )

    comparacao = loaders.carregar_comparacao_modelos()

    if not loaders.exigir_dados(comparacao):
        return

    metadata = loaders.carregar_metadata()
    modelo_final = loaders.carregar_modelo_final()

    _secao_modelo_final(modelo_final, metadata)

    st.divider()
    st.subheader("Comparação entre modelos")
    st.caption(
        f"Cenário `{metadata.get('melhor_cenario_features', 'sem_pos_transacao')}`, "
        "conjunto de teste, cada modelo no seu threshold otimizado."
    )

    _tabela_comparacao(comparacao)
    _grafico_comparacao(comparacao, metadata, modelo_final)

    st.divider()
    _rodape_navegacao()


def _secao_modelo_final(modelo_final: pd.Series | None, metadata: dict) -> None:
    """Cartões com as métricas do modelo escolhido e o atendimento às metas."""
    if modelo_final is None:
        return

    st.subheader(f"Modelo final: {modelo_final['modelo']}")

    criterio = metadata.get("criterio_selecao")
    if criterio:
        st.caption(f"Critério de seleção: {criterio}.")

    metas = metadata["metas"]
    colunas = st.columns(4)

    cartoes = [
        ("F1 (fraude)", "f1", metas.get("f1_minimo")),
        ("Recall (fraude)", "recall", metas.get("recall_minimo")),
        ("Precision (fraude)", "precision", metas.get("precision_minima")),
        ("AUPRC", "auprc", None),
    ]

    for coluna, (rotulo, chave, meta) in zip(colunas, cartoes):
        valor = pd.to_numeric(modelo_final.get(chave), errors="coerce")

        if pd.isna(valor):
            coluna.metric(rotulo, "sem dado")
            continue

        if meta is None:
            coluna.metric(rotulo, f"{valor:.4f}")
            continue

        delta = valor - meta
        coluna.metric(
            rotulo,
            f"{valor:.4f}",
            delta=f"{delta:+.4f} vs meta {meta:.2f}",
            delta_color="normal",
        )


def _tabela_comparacao(comparacao: pd.DataFrame) -> None:
    """Tabela ordenada por F1, com as colunas do relatório."""
    tabela = comparacao[COLUNAS_TABELA].copy()

    st.dataframe(
        tabela,
        hide_index=True,
        width="stretch",
        column_config=CONFIG_COLUNAS,
    )

    if tabela["tempo_p95_ms"].isna().all():
        st.caption(
            "Tempo de inferência indisponível para todos os modelos nesta fonte de "
            "dados. A página *Tempo de inferência* detalha o que está disponível."
        )


def _grafico_comparacao(
    comparacao: pd.DataFrame, metadata: dict, modelo_final: pd.Series | None
) -> None:
    """Barras de uma métrica por vez, com a meta correspondente."""
    rotulo_metrica = st.radio(
        "Métrica do gráfico",
        options=list(METRICAS_DISPONIVEIS),
        horizontal=True,
    )

    coluna, formato, chave_meta = METRICAS_DISPONIVEIS[rotulo_metrica]
    meta = metadata["metas"].get(chave_meta) if chave_meta else None

    if comparacao[coluna].isna().all():
        st.info(f"A fonte de dados atual não traz valores de {rotulo_metrica}.")
        return

    grafico = charts.barras_comparacao(
        comparacao,
        coluna=coluna,
        titulo_eixo=rotulo_metrica,
        formato=formato,
        meta=meta,
        rotulo_meta=f"Meta ({rotulo_metrica})",
        destaque=modelo_final["modelo"] if modelo_final is not None else None,
    )

    charts.exibir(grafico)

    if meta is not None:
        st.caption(
            f"A linha tracejada marca a meta de {rotulo_metrica.lower()} "
            f"({meta:.2f}) definida na especificação do projeto."
        )

    if coluna == "custo_total":
        custos = metadata["custos"]
        st.caption(
            f"Custo calculado com custo_FP = {custos['custo_fp']:g} e "
            f"custo_FN = {custos['custo_fn']:g}. Use a página *Simulador de custo* "
            "para variar esses valores."
        )


def _rodape_navegacao() -> None:
    st.markdown(
        """
        **Nas outras páginas:** matriz de confusão e curva Precision-Recall do
        modelo final, simulador da matriz de custo, robustez por imbalance ratio
        e tempo de inferência.
        """
    )
    st.caption(
        "Dataset PaySim (CC BY-SA 4.0), sintético. Os resultados são "
        "experimentais e não equivalem a validação em dados reais."
    )


main()
