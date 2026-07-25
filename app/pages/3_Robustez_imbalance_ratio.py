"""Robustez do modelo final sob diferentes níveis de desbalanceamento."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import charts, loaders  # noqa: E402

st.set_page_config(page_title="Robustez por imbalance ratio", page_icon="📉", layout="wide")

# Ordem fixa: cada métrica mantém sua cor independentemente de quantas
# estiverem selecionadas.
METRICAS = {
    "Precision": "precision",
    "Recall": "recall",
    "F1": "f1",
    "AUPRC": "auprc",
}

CONFIG_COLUNAS = {
    "cenario_ir": st.column_config.TextColumn("Imbalance ratio"),
    "precision": st.column_config.NumberColumn("Precision", format="%.4f"),
    "recall": st.column_config.NumberColumn("Recall", format="%.4f"),
    "f1": st.column_config.NumberColumn("F1", format="%.4f"),
    "auprc": st.column_config.NumberColumn("AUPRC", format="%.4f"),
    "fp": st.column_config.NumberColumn("FP", format="%d"),
    "fn": st.column_config.NumberColumn("FN", format="%d"),
    "total_amostra": st.column_config.NumberColumn("Transações na amostra", format="%d"),
    "fraudes_amostra": st.column_config.NumberColumn("Fraudes na amostra", format="%d"),
}


def main() -> None:
    loaders.render_sidebar_fontes()

    st.title("Robustez por imbalance ratio")
    st.caption(
        "Cada cenário reamostra o conjunto de teste para uma proporção alvo de "
        "legítimas por fraude, mantendo o modelo e o threshold do resultado final."
    )

    robustez = loaders.carregar_robustez_ir()

    if len(robustez) == 0:
        st.info(
            "Nenhum resultado por imbalance ratio encontrado "
            "(`resultados_robustez_imbalance_ratio.csv`)."
        )
        return

    metadata = loaders.carregar_metadata()
    meta_recall = metadata["metas"].get("recall_minimo")

    _grafico(robustez, meta_recall)

    st.divider()
    _leitura(robustez, meta_recall)

    st.divider()
    _tabela(robustez)


def _grafico(robustez: pd.DataFrame, meta_recall: float | None) -> None:
    """Linhas por cenário para as métricas selecionadas."""
    selecionadas = st.multiselect(
        "Métricas exibidas",
        options=list(METRICAS),
        default=list(METRICAS),
    )

    if not selecionadas:
        st.info("Selecione ao menos uma métrica.")
        return

    colunas = [METRICAS[rotulo] for rotulo in selecionadas]
    disponiveis = [c for c in colunas if c in robustez.columns and robustez[c].notna().any()]

    if not disponiveis:
        st.info("A fonte de dados atual não traz as métricas selecionadas.")
        return

    longo = robustez.melt(
        id_vars=["cenario_ir"],
        value_vars=disponiveis,
        var_name="coluna",
        value_name="valor",
    ).dropna(subset=["valor"])

    inverso = {valor: rotulo for rotulo, valor in METRICAS.items()}
    longo["metrica"] = longo["coluna"].map(inverso)

    # Preserva a ordem fixa das métricas para a cor não depender da seleção.
    ordem_metricas = [rotulo for rotulo in METRICAS if METRICAS[rotulo] in disponiveis]

    charts.exibir(
        charts.linhas_por_ir(
            longo,
            ordem_cenarios=list(robustez["cenario_ir"]),
            ordem_metricas=ordem_metricas,
            meta_recall=meta_recall if "recall" in disponiveis else None,
        ),
        altura=380,
    )

    if meta_recall is not None and "recall" in disponiveis:
        st.caption(
            f"A linha tracejada marca a meta de recall ({meta_recall:.2f}). "
            "Os rótulos à direita identificam cada série."
        )


def _leitura(robustez: pd.DataFrame, meta_recall: float | None) -> None:
    """Interpretação calculada a partir dos dados carregados."""
    st.subheader("Leitura dos resultados")

    primeiro = robustez.iloc[0]
    ultimo = robustez.iloc[-1]

    if meta_recall is not None and "recall" in robustez.columns:
        aprovados = robustez[robustez["recall"] >= meta_recall]

        if len(aprovados) > 0:
            pior_aprovado = aprovados.iloc[-1]
            st.markdown(
                f"- O recall se manteve **acima da meta de {meta_recall:.2f}** até o "
                f"cenário **{pior_aprovado['cenario_ir']}** "
                f"(recall {pior_aprovado['recall']:.4f})."
            )
        else:
            st.markdown(
                f"- O recall **não atingiu a meta de {meta_recall:.2f}** em nenhum "
                "dos cenários testados."
            )

    for coluna, rotulo in (("precision", "precision"), ("f1", "F1")):
        if coluna not in robustez.columns:
            continue

        inicial = pd.to_numeric(primeiro.get(coluna), errors="coerce")
        final = pd.to_numeric(ultimo.get(coluna), errors="coerce")

        if pd.isna(inicial) or pd.isna(final):
            continue

        queda = inicial - final
        direcao = "caiu" if queda > 0 else "subiu"

        st.markdown(
            f"- A {rotulo} {direcao} de **{inicial:.4f}** em "
            f"{primeiro['cenario_ir']} para **{final:.4f}** em "
            f"{ultimo['cenario_ir']} ({abs(queda):.4f} de diferença)."
        )

    st.caption(
        "O padrão esperado é este: quanto mais rara a fraude, mais transações "
        "legítimas existem para cada fraude e mais falsos positivos o mesmo "
        "threshold produz. O modelo continua encontrando as fraudes, mas o custo "
        "em alertas falsos cresce — por isso o threshold precisaria ser reajustado "
        "conforme a tolerância da instituição."
    )


def _tabela(robustez: pd.DataFrame) -> None:
    """Tabela equivalente ao gráfico, exigida como alternativa acessível."""
    st.subheader("Tabela por cenário")

    colunas = [
        c for c in (
            "cenario_ir", "precision", "recall", "f1", "auprc",
            "fp", "fn", "total_amostra", "fraudes_amostra",
        )
        if c in robustez.columns
    ]

    st.dataframe(
        robustez[colunas],
        hide_index=True,
        width="stretch",
        column_config=CONFIG_COLUNAS,
    )


main()
