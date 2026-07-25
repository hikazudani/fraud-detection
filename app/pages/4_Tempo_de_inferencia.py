"""Tempo de inferência por transação: médio e p95, comparados às metas."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import charts, loaders  # noqa: E402

st.set_page_config(page_title="Tempo de inferência", page_icon="⏱️", layout="wide")

COLUNAS_TEMPO = ["tempo_medio_ms", "tempo_p95_ms"]

CONFIG_COLUNAS = {
    "modelo": st.column_config.TextColumn("Modelo", width="medium"),
    "tempo_medio_ms": st.column_config.NumberColumn("Tempo médio (ms)", format="%.6f"),
    "tempo_p95_ms": st.column_config.NumberColumn("p95 (ms)", format="%.6f"),
    "margem_media": st.column_config.TextColumn("Margem vs meta média"),
    "status": st.column_config.TextColumn("Status"),
}


def main() -> None:
    loaders.render_sidebar_fontes()

    st.title("Tempo de inferência")
    st.caption(
        "Tempo por transação medido no conjunto de avaliação. O projeto trata "
        "latência como critério de sucesso junto com a detecção: um modelo muito "
        "preciso, mas lento, é inadequado para decisão em tempo de transação."
    )

    comparacao = loaders.carregar_comparacao_modelos()
    metadata = loaders.carregar_metadata()

    if not loaders.exigir_dados(comparacao):
        return

    metas = metadata["metas"]
    meta_media = metas.get("tempo_medio_ms_maximo")
    meta_p95 = metas.get("tempo_p95_ms_maximo")

    com_tempo = comparacao.dropna(subset=COLUNAS_TEMPO, how="all").copy()

    if len(com_tempo) == 0:
        _sem_dados()
        return

    _cartoes(com_tempo, meta_media, meta_p95)

    st.divider()
    _grafico(com_tempo, meta_media, meta_p95)

    st.divider()
    _tabela(com_tempo, meta_media, meta_p95)

    faltando = len(comparacao) - len(com_tempo)
    if faltando > 0:
        st.caption(
            f"{faltando} de {len(comparacao)} modelos não têm tempo registrado nesta "
            "fonte de dados e ficaram fora do gráfico."
        )

    st.divider()
    st.caption(
        "Os tempos vêm do ambiente experimental e não garantem desempenho em "
        "produção, que dependeria da infraestrutura, do volume de requisições "
        "simultâneas e da integração com sistemas externos."
    )


def _cartoes(com_tempo: pd.DataFrame, meta_media: float | None, meta_p95: float | None) -> None:
    """Cartões do modelo mais rápido e do atendimento às metas."""
    mais_rapido = com_tempo.sort_values("tempo_medio_ms").iloc[0]

    colunas = st.columns(3)

    tempo_medio = pd.to_numeric(mais_rapido.get("tempo_medio_ms"), errors="coerce")
    tempo_p95 = pd.to_numeric(mais_rapido.get("tempo_p95_ms"), errors="coerce")

    colunas[0].metric("Modelo mais rápido", str(mais_rapido["modelo"]))
    colunas[1].metric(
        "Tempo médio",
        f"{charts.formatar_milhar(tempo_medio, 6)} ms" if pd.notna(tempo_medio) else "sem dado",
        delta=(
            f"{charts.formatar_milhar(meta_media / tempo_medio, 0)}× abaixo da meta"
            if pd.notna(tempo_medio) and meta_media and tempo_medio > 0
            else None
        ),
        delta_color="off",
    )
    colunas[2].metric(
        "p95",
        f"{charts.formatar_milhar(tempo_p95, 6)} ms" if pd.notna(tempo_p95) else "sem dado",
        delta=(
            f"{charts.formatar_milhar(meta_p95 / tempo_p95, 0)}× abaixo da meta"
            if pd.notna(tempo_p95) and meta_p95 and tempo_p95 > 0
            else None
        ),
        delta_color="off",
    )


def _grafico(com_tempo: pd.DataFrame, meta_media: float | None, meta_p95: float | None) -> None:
    """Barras na escala dos dados, ou pontos em escala log com as metas."""
    escala_log = st.toggle(
        "Escala logarítmica com as metas",
        value=False,
        help="As metas (50 ms e 100 ms) são milhares de vezes maiores que os tempos "
             "medidos. Em escala linear elas achatariam as barras; em escala "
             "logarítmica as duas grandezas cabem no mesmo eixo.",
    )

    longo = com_tempo.melt(
        id_vars=["modelo"],
        value_vars=COLUNAS_TEMPO,
        var_name="coluna",
        value_name="valor",
    ).dropna(subset=["valor"])

    longo["medida"] = longo["coluna"].map(charts.ROTULOS_TEMPO)

    if escala_log:
        metas = {}
        if meta_media:
            metas[f"meta média {meta_media:g} ms"] = float(meta_media)
        if meta_p95:
            metas[f"meta p95 {meta_p95:g} ms"] = float(meta_p95)

        charts.exibir(charts.pontos_tempo_log(longo, metas), altura=380)
        st.caption(
            "As linhas tracejadas são as metas do projeto. Em escala logarítmica as "
            "marcas viram pontos, porque barras precisariam de uma base em zero."
        )
        return

    charts.exibir(charts.barras_tempo(longo), altura=380)
    st.caption(
        "Escala linear, ajustada aos tempos medidos: as metas ficam fora do eixo "
        "por serem ordens de grandeza maiores. Ative a escala logarítmica para "
        "vê-las."
    )


def _tabela(com_tempo: pd.DataFrame, meta_media: float | None, meta_p95: float | None) -> None:
    """Tabela com a margem em relação às metas e o status de cada modelo."""
    st.subheader("Atendimento às metas")

    tabela = com_tempo[["modelo"] + COLUNAS_TEMPO].copy()

    def _status(linha: pd.Series) -> str:
        medio = pd.to_numeric(linha["tempo_medio_ms"], errors="coerce")
        p95 = pd.to_numeric(linha["tempo_p95_ms"], errors="coerce")

        verificacoes = []
        if meta_media and pd.notna(medio):
            verificacoes.append(medio <= meta_media)
        if meta_p95 and pd.notna(p95):
            verificacoes.append(p95 <= meta_p95)

        if not verificacoes:
            return "sem dado"

        return "Atendida" if all(verificacoes) else "Não atendida"

    def _margem(linha: pd.Series) -> str:
        medio = pd.to_numeric(linha["tempo_medio_ms"], errors="coerce")

        if not meta_media or pd.isna(medio) or medio <= 0:
            return "sem dado"

        return f"{charts.formatar_milhar(meta_media / medio, 0)}× abaixo"

    tabela["margem_media"] = tabela.apply(_margem, axis=1)
    tabela["status"] = tabela.apply(_status, axis=1)

    st.dataframe(
        tabela,
        hide_index=True,
        width="stretch",
        column_config=CONFIG_COLUNAS,
    )

    if meta_media and meta_p95:
        st.caption(
            f"Metas do projeto: tempo médio ≤ {meta_media:g} ms e p95 ≤ "
            f"{meta_p95:g} ms por transação."
        )


def _sem_dados() -> None:
    st.info(
        "Nenhum modelo tem tempo de inferência registrado na fonte de dados atual. "
        "O tempo por modelo está em `resultados_teste.csv`, gerado pelo notebook de "
        "modelagem — a tabela de apresentação (`tabela_principal_modelos.csv`) traz "
        "apenas o tempo médio, e o seed de demonstração traz apenas o do modelo final."
    )


main()
