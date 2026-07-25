"""Leitura e normalização dos artefatos do pipeline, com cache do Streamlit.

Toda leitura de disco passa por `_ler_csv`/`_ler_json`, que recebem o `mtime` do
arquivo na assinatura. Assim, quando o notebook regrava um artefato, a chave de
cache muda e o Streamlit relê o arquivo automaticamente.

As funções de carregamento devolvem um schema canônico em snake_case,
independentemente de o dado ter vindo de `resultados_teste.csv` (formato cru do
notebook) ou de `tabela_principal_modelos.csv` (formato de apresentação).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from . import metrics
from .paths import ArtefatoResolvido, esta_em_modo_demo, resolver_artefato, resolver_todos

# Metas da seção 5 da especificação, usadas quando o metadata não as traz.
METAS_PADRAO = {
    "recall_minimo": 0.80,
    "precision_minima": 0.60,
    "f1_minimo": 0.85,
    "tempo_medio_ms_maximo": 50.0,
    "tempo_p95_ms_maximo": 100.0,
}

CUSTOS_PADRAO = {"custo_fp": 10.0, "custo_fn": 500.0}

COLUNAS_CANONICAS = [
    "modelo",
    "precision",
    "recall",
    "f1",
    "auprc",
    "tp",
    "fp",
    "fn",
    "tn",
    "custo_total",
    "tempo_medio_ms",
    "tempo_p95_ms",
    "threshold",
    "cenario",
]

RENOMEIO_RESULTADOS_TESTE = {
    "precision_fraude": "precision",
    "recall_fraude": "recall",
    "f1_fraude": "f1",
    "tempo_medio_ms_por_transacao": "tempo_medio_ms",
    "tempo_p95_ms_por_transacao": "tempo_p95_ms",
    "cenario_features": "cenario",
}

RENOMEIO_TABELA_PRINCIPAL = {
    "Modelo": "modelo",
    "Precision": "precision",
    "Recall": "recall",
    "F1": "f1",
    "AUPRC": "auprc",
    "FP": "fp",
    "FN": "fn",
    "Custo total": "custo_total",
    "Tempo médio (ms/transação)": "tempo_medio_ms",
    "Observação": "observacao",
}

SUFIXO_THRESHOLD_OTIMIZADO = " - threshold otimizado F1"


# --------------------------------------------------------------------------- #
# Leitura bruta
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def _ler_csv(caminho: str, mtime: float) -> pd.DataFrame:
    """Lê um CSV. `mtime` participa da chave de cache e invalida na regravação."""
    return pd.read_csv(caminho)


@st.cache_data(show_spinner=False)
def _ler_json(caminho: str, mtime: float) -> dict:
    """Lê um JSON. `mtime` participa da chave de cache."""
    with open(caminho, encoding="utf-8") as arquivo:
        return json.load(arquivo)


def _carregar(chave: str) -> pd.DataFrame | dict | None:
    """Resolve um artefato e o lê, ou devolve None se não existir em nenhum lugar."""
    artefato = resolver_artefato(chave)

    if artefato is None:
        return None

    if artefato.caminho.suffix == ".json":
        return _ler_json(str(artefato.caminho), artefato.mtime)

    return _ler_csv(str(artefato.caminho), artefato.mtime)


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #

def carregar_metadata() -> dict:
    """Metadata da execução, com custos e metas preenchidos por padrão se ausentes."""
    metadata = _carregar("metadata") or {}

    metadata["custos"] = {**CUSTOS_PADRAO, **(metadata.get("custos") or {})}
    metadata["metas"] = {**METAS_PADRAO, **(metadata.get("metas") or {})}

    return metadata


def nome_curto_modelo(nome: str) -> str:
    """Encurta o nome do modelo, replicando `nome_curto_modelo` do notebook.

    Mantém a nomenclatura da dashboard idêntica à da tabela do relatório.
    """
    substituicoes = [
        (SUFIXO_THRESHOLD_OTIMIZADO, ""),
        (" - threshold 0.5", ""),
        ("Regressão Logística - class_weight", "Regressão Logística"),
        ("Random Forest - class_weight", "Random Forest class_weight"),
        ("XGBoost - scale_pos_weight", "XGBoost scale_pos_weight"),
        ("LightGBM - scale_pos_weight", "LightGBM scale_pos_weight"),
    ]

    for antigo, novo in substituicoes:
        nome = nome.replace(antigo, novo)

    return nome.strip()


# --------------------------------------------------------------------------- #
# Comparação de modelos
# --------------------------------------------------------------------------- #

def carregar_comparacao_modelos() -> pd.DataFrame:
    """Tabela canônica de comparação dos modelos no cenário operacional.

    Prefere `resultados_teste.csv` porque ele traz p95 e as quatro contagens da
    matriz de confusão. Cai para `tabela_principal_modelos.csv` quando o arquivo
    cru não está disponível — nesse caso o p95 fica ausente.
    """
    metadata = carregar_metadata()
    cenario = metadata.get("melhor_cenario_features") or metadata.get(
        "cenario_final_operacional", "sem_pos_transacao"
    )

    resultados_teste = _carregar("resultados_teste")

    if resultados_teste is not None and len(resultados_teste) > 0:
        return _normalizar_resultados_teste(resultados_teste, cenario)

    tabela_principal = _carregar("tabela_principal_modelos")

    if tabela_principal is not None and len(tabela_principal) > 0:
        return _normalizar_tabela_principal(tabela_principal, cenario)

    return pd.DataFrame(columns=COLUNAS_CANONICAS)


def _normalizar_resultados_teste(resultados: pd.DataFrame, cenario: str) -> pd.DataFrame:
    """Filtra o cenário operacional e mantém uma linha por modelo.

    Segue a mesma seleção de `montar_tabela_principal` do notebook: apenas as
    linhas com threshold otimizado, mais as duas baselines.
    """
    tabela = resultados.copy()

    if "conjunto" in tabela.columns:
        tabela = tabela[tabela["conjunto"] == "teste"]

    if "cenario_features" in tabela.columns and cenario in set(tabela["cenario_features"]):
        tabela = tabela[tabela["cenario_features"] == cenario]

    mascara = (
        tabela["modelo"].str.contains(SUFIXO_THRESHOLD_OTIMIZADO, regex=False)
        | tabela["modelo"].str.startswith("Baseline")
    )

    # Se o arquivo não trouxer linhas com threshold otimizado, não descarta tudo.
    if mascara.any():
        tabela = tabela[mascara]

    tabela = tabela.rename(columns=RENOMEIO_RESULTADOS_TESTE)
    tabela["modelo"] = tabela["modelo"].map(nome_curto_modelo)

    tabela = tabela.sort_values(
        by=["f1", "recall", "precision"], ascending=[False, False, False]
    ).drop_duplicates(subset=["modelo"], keep="first")

    return _finalizar_comparacao(tabela)


def _normalizar_tabela_principal(tabela: pd.DataFrame, cenario: str) -> pd.DataFrame:
    """Converte a tabela de apresentação para o schema canônico."""
    tabela = tabela.rename(columns=RENOMEIO_TABELA_PRINCIPAL).copy()
    tabela["cenario"] = cenario
    return _finalizar_comparacao(tabela)


def _finalizar_comparacao(tabela: pd.DataFrame) -> pd.DataFrame:
    """Garante todas as colunas canônicas e ordena por F1."""
    for coluna in COLUNAS_CANONICAS:
        if coluna not in tabela.columns:
            tabela[coluna] = pd.NA

    numericas = [
        "precision", "recall", "f1", "auprc", "tp", "fp", "fn", "tn",
        "custo_total", "tempo_medio_ms", "tempo_p95_ms", "threshold",
    ]
    for coluna in numericas:
        tabela[coluna] = pd.to_numeric(tabela[coluna], errors="coerce")

    extras = [c for c in ("observacao",) if c in tabela.columns]

    return (
        tabela[COLUNAS_CANONICAS + extras]
        .sort_values(by=["f1", "recall"], ascending=[False, False])
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# Modelo final
# --------------------------------------------------------------------------- #

def carregar_modelo_final() -> pd.Series | None:
    """Linha do modelo final operacional, no schema canônico."""
    resultado = _carregar("resultado_modelo_final")

    if resultado is not None and len(resultado) > 0:
        linha = resultado.rename(columns=RENOMEIO_RESULTADOS_TESTE).iloc[0].copy()
        linha["modelo"] = nome_curto_modelo(str(linha["modelo"]))
        return linha

    # Fallback: melhor linha da comparação.
    comparacao = carregar_comparacao_modelos()

    if len(comparacao) == 0:
        return None

    metadata = carregar_metadata()
    nome_esperado = nome_curto_modelo(str(metadata.get("melhor_modelo_base", "")))

    correspondentes = comparacao[comparacao["modelo"] == nome_esperado]

    return (correspondentes.iloc[0] if len(correspondentes) else comparacao.iloc[0]).copy()


def contagens_confusao_modelo_final() -> dict[str, int] | None:
    """TP/FP/FN/TN do modelo final, quando disponíveis."""
    linha = carregar_modelo_final()

    if linha is None:
        return None

    contagens = {}
    for chave in ("tp", "fp", "fn", "tn"):
        valor = pd.to_numeric(linha.get(chave), errors="coerce")
        if pd.isna(valor):
            return None
        contagens[chave] = int(valor)

    return contagens


# --------------------------------------------------------------------------- #
# Robustez por imbalance ratio
# --------------------------------------------------------------------------- #

def carregar_robustez_ir() -> pd.DataFrame:
    """Resultados por imbalance ratio, no schema canônico e ordenados por IR."""
    resultados = _carregar("robustez_ir")

    if resultados is None or len(resultados) == 0:
        return pd.DataFrame()

    tabela = resultados.rename(columns=RENOMEIO_RESULTADOS_TESTE).copy()

    if "imbalance_ratio_alvo" not in tabela.columns and "conjunto" in tabela.columns:
        tabela["imbalance_ratio_alvo"] = (
            tabela["conjunto"].astype(str).str.extract(r"(\d+)$").astype(float)
        )

    tabela["imbalance_ratio_alvo"] = pd.to_numeric(
        tabela["imbalance_ratio_alvo"], errors="coerce"
    )
    tabela["cenario_ir"] = tabela["imbalance_ratio_alvo"].map(
        lambda valor: f"1:{int(valor)}" if pd.notna(valor) else "desconhecido"
    )

    return tabela.sort_values("imbalance_ratio_alvo").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Predições, curva PR e sweep de threshold
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner="Calculando sweep de threshold...")
def _sweep_das_predicoes(
    caminho: str, mtime: float, n_thresholds: int, threshold_operacional: float | None
) -> pd.DataFrame:
    predicoes = _ler_csv(caminho, mtime)
    return metrics.calcular_sweep_threshold(
        predicoes["y_true"],
        predicoes["probabilidade_fraude"],
        n_thresholds=n_thresholds,
        threshold_operacional=threshold_operacional,
    )


@st.cache_data(show_spinner="Calculando curva Precision-Recall...")
def _curva_pr_das_predicoes(caminho: str, mtime: float, max_pontos: int) -> pd.DataFrame:
    predicoes = _ler_csv(caminho, mtime)
    return metrics.calcular_curva_pr(
        predicoes["y_true"], predicoes["probabilidade_fraude"], max_pontos=max_pontos
    )


def threshold_operacional() -> float | None:
    """Threshold escolhido na validação, do metadata ou das predições salvas."""
    metadata = carregar_metadata()
    valor = pd.to_numeric(metadata.get("melhor_threshold"), errors="coerce")

    if pd.notna(valor):
        return float(valor)

    linha_final = carregar_modelo_final()
    if linha_final is not None:
        valor = pd.to_numeric(linha_final.get("threshold"), errors="coerce")
        if pd.notna(valor):
            return float(valor)

    return None


def carregar_sweep(n_thresholds: int = 200) -> pd.DataFrame | None:
    """Sweep de threshold: do derivado leve, das predições, ou None em modo demo."""
    curva_threshold = _carregar("curva_threshold")

    if curva_threshold is not None and len(curva_threshold) > 0:
        tabela = curva_threshold.copy()
        if not {"precision", "recall", "f1"}.issubset(tabela.columns):
            tabela = metrics.adicionar_metricas_derivadas(tabela)
        return tabela.sort_values("threshold").reset_index(drop=True)

    predicoes = resolver_artefato("predicoes_teste")

    if predicoes is not None:
        return _sweep_das_predicoes(
            str(predicoes.caminho), predicoes.mtime, n_thresholds, threshold_operacional()
        )

    return None


def carregar_curva_pr(max_pontos: int = 500) -> pd.DataFrame | None:
    """Curva PR: do derivado leve, das predições, ou None em modo demo."""
    curva = _carregar("curva_pr")

    if curva is not None and len(curva) > 0:
        return curva

    predicoes = resolver_artefato("predicoes_teste")

    if predicoes is not None:
        return _curva_pr_das_predicoes(str(predicoes.caminho), predicoes.mtime, max_pontos)

    return None


def prevalencia_teste() -> float | None:
    """Proporção de fraudes no conjunto de teste, base da curva PR."""
    contagens = contagens_confusao_modelo_final()

    if contagens is not None:
        total = sum(contagens.values())
        if total > 0:
            return (contagens["tp"] + contagens["fn"]) / total

    metadata = carregar_metadata()
    conjunto = metadata.get("conjunto_teste") or {}

    if conjunto.get("total"):
        return conjunto["fraudes"] / conjunto["total"]

    return None


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

def render_sidebar_fontes() -> dict[str, ArtefatoResolvido | None]:
    """Mostra na sidebar a origem de cada artefato e o estado de demonstração."""
    resolucao = resolver_todos()
    metadata = carregar_metadata()
    modo_demo = esta_em_modo_demo(resolucao)

    with st.sidebar:
        st.subheader("Fonte dos dados")

        if modo_demo:
            st.warning(
                "**Modo demonstração.** Os números vêm do seed versionado em "
                "`app/data/exemplo`, transcrito da especificação do projeto. "
                "Execute o notebook de modelagem para usar os artefatos reais."
            )
        else:
            data_execucao = metadata.get("data_execucao", "data não informada")
            st.success(f"Artefatos do pipeline · execução de {data_execucao}")

        modelo = nome_curto_modelo(str(metadata.get("melhor_modelo_base", "não informado")))
        cenario = metadata.get("melhor_cenario_features", "não informado")

        st.caption(f"**Modelo final:** {modelo}")
        st.caption(f"**Cenário:** `{cenario}`")

        threshold = threshold_operacional()
        st.caption(
            f"**Threshold operacional:** {threshold:.4f}" if threshold is not None
            else "**Threshold operacional:** não publicado"
        )

        with st.expander("Arquivos carregados"):
            linhas = [
                {
                    "Artefato": artefato.nome_arquivo,
                    "Origem": artefato.origem_relativa,
                }
                for artefato in resolucao.values()
                if artefato is not None
            ]

            if linhas:
                st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch")

            ausentes = [
                chave for chave, artefato in resolucao.items() if artefato is None
            ]
            if ausentes:
                st.caption("Ausentes: " + ", ".join(sorted(ausentes)))

        st.caption(
            "Dashboard somente de leitura: consome resultados já salvos, sem "
            "retreinar modelos."
        )

    return resolucao


def exigir_dados(comparacao: pd.DataFrame) -> bool:
    """Mostra instruções e devolve False quando não há nenhum dado carregável."""
    if len(comparacao) > 0:
        return True

    st.error("Nenhum artefato de resultados foi encontrado.")
    st.markdown(
        """
        A dashboard consome resultados já salvos pelo pipeline. Para gerá-los:

        1. execute `notebooks/preprocessamento_fraud_detection.ipynb`;
        2. execute `notebooks/modelagem_fraud_detection_refatorado_limpo.ipynb`;
        3. opcionalmente rode `python scripts/exportar_artefatos_dashboard.py`
           para gerar os artefatos leves de curva PR e sweep de threshold.

        Também é possível apontar para outra pasta com a variável de ambiente
        `FRAUD_DASHBOARD_DATA_DIR`.
        """
    )
    return False


def caminho_relativo(caminho: Path) -> str:
    """Caminho legível para exibição na interface."""
    from .paths import RAIZ_PROJETO

    try:
        return str(caminho.relative_to(RAIZ_PROJETO))
    except ValueError:
        return str(caminho)
