"""Gráficos Altair da dashboard.

Decisões de visualização adotadas em todos os gráficos deste módulo:

- **uma escala por eixo**, nunca dois eixos y;
- cores categóricas atribuídas em ordem fixa (azul, laranja, aqua, amarelo),
  nunca cicladas, e sempre pela identidade da série — filtrar séries não
  repinta as que sobraram;
- magnitude contínua (matriz de confusão) usa rampa de um só matiz;
- marcas finas: linhas de 2px, pontos de pelo menos 8px, grade recessiva;
- rótulos diretos nas barras e no último ponto das linhas, de modo que a
  identidade e o valor nunca dependam apenas da cor;
- passos de cor escolhidos por modo (claro/escuro), não invertidos
  automaticamente.

As cores em modo claro de aqua e amarelo ficam abaixo de 3:1 contra a
superfície do app. Onde elas aparecem (gráfico por imbalance ratio), o gráfico
traz rótulos diretos e a página traz a tabela equivalente.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

# Passos validados para cada modo: matiz igual, passo escolhido para a
# superfície em que o gráfico é desenhado.
PALETAS = {
    "light": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
        "sequencial": ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#104281"],
        "ink": "#0b0b0b",
        "ink_secundario": "#52514e",
        "muted": "#898781",
        "grade": "#e1e0d9",
        "eixo": "#c3c2b7",
        "superficie": "#ffffff",
        "bom": "#0ca30c",
        "critico": "#d03b3b",
    },
    "dark": {
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
        "sequencial": ["#0d366b", "#184f95", "#256abf", "#5598e7", "#9ec5f4"],
        "ink": "#ffffff",
        "ink_secundario": "#c3c2b7",
        "muted": "#898781",
        "grade": "#2c2c2a",
        "eixo": "#383835",
        "superficie": "#0e1117",
        "bom": "#0ca30c",
        "critico": "#d03b3b",
    },
}

ALTURA_PADRAO = 320


def modo_tema() -> str:
    """Modo de tema ativo no navegador do usuário ('light' ou 'dark')."""
    try:
        tipo = st.context.theme.type
    except Exception:
        tipo = None

    return "dark" if tipo == "dark" else "light"


def paleta() -> dict:
    """Paleta do modo ativo."""
    return PALETAS[modo_tema()]


def _aplicar_tema(grafico: alt.Chart, altura: int = ALTURA_PADRAO) -> alt.Chart:
    """Aplica cromo recessivo e cores de texto do modo ativo.

    Os gráficos são renderizados com `theme=None` para que estas cores valham,
    então título, eixos e legenda precisam ser configurados explicitamente.
    """
    cores = paleta()

    return (
        grafico.properties(height=altura, background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=cores["muted"],
            titleColor=cores["ink_secundario"],
            gridColor=cores["grade"],
            domainColor=cores["eixo"],
            tickColor=cores["eixo"],
            labelFontSize=12,
            titleFontSize=12,
            titleFontWeight="normal",
        )
        .configure_legend(
            labelColor=cores["ink_secundario"],
            titleColor=cores["ink_secundario"],
            labelFontSize=12,
            titleFontSize=12,
            symbolStrokeWidth=3,
            orient="top",
            direction="horizontal",
            title=None,
        )
        .configure_title(color=cores["ink"], fontSize=14, fontWeight=600, anchor="start")
    )


def exibir(grafico: alt.Chart, altura: int = ALTURA_PADRAO) -> None:
    """Renderiza o gráfico ocupando a largura do container."""
    st.altair_chart(_aplicar_tema(grafico, altura), width="stretch", theme=None)


def _formato_altair(formato: str) -> str:
    return {"decimal": ".4f", "moeda": ",.0f", "inteiro": ",.0f", "ms": ".6f"}.get(formato, ".4f")


def formatar_milhar(valor: float, decimais: int = 0) -> str:
    """Formata um número no padrão brasileiro (ponto de milhar, vírgula decimal).

    Formatar cada número isoladamente evita o erro de aplicar a troca de
    separadores sobre uma frase inteira, que corromperia as vírgulas do texto.
    """
    texto = f"{valor:,.{decimais}f}"
    return texto.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


# --------------------------------------------------------------------------- #
# Seção 1 — comparação de modelos
# --------------------------------------------------------------------------- #

def barras_comparacao(
    dados: pd.DataFrame,
    coluna: str,
    titulo_eixo: str,
    formato: str = "decimal",
    meta: float | None = None,
    rotulo_meta: str | None = None,
    destaque: str | None = None,
) -> alt.Chart:
    """Barras horizontais de uma métrica por modelo, ordenadas por valor.

    Série única: a cor não carrega informação e o título nomeia a métrica. O
    modelo final recebe um passo mais escuro apenas como destaque, com o nome
    do modelo sempre presente no eixo.
    """
    cores = paleta()
    fmt = _formato_altair(formato)

    tabela = dados.dropna(subset=[coluna]).copy()
    tabela["destaque"] = tabela["modelo"] == destaque

    base = alt.Chart(tabela)

    barras = base.mark_bar(cornerRadiusEnd=4, height=alt.RelativeBandSize(0.72)).encode(
        x=alt.X(f"{coluna}:Q", title=titulo_eixo, axis=alt.Axis(format=fmt)),
        y=alt.Y("modelo:N", sort="-x", title=None),
        color=alt.Color(
            "destaque:N",
            scale=alt.Scale(domain=[True, False], range=[cores["series"][0], cores["sequencial"][2]]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("modelo:N", title="Modelo"),
            alt.Tooltip(f"{coluna}:Q", title=titulo_eixo, format=fmt),
        ],
    )

    rotulos = base.mark_text(
        align="left", dx=6, fontSize=12, color=cores["ink_secundario"]
    ).encode(
        x=alt.X(f"{coluna}:Q"),
        y=alt.Y("modelo:N", sort="-x"),
        text=alt.Text(f"{coluna}:Q", format=fmt),
    )

    camadas = [barras, rotulos]

    if meta is not None:
        dados_meta = pd.DataFrame({"meta": [meta]})
        camadas.append(
            alt.Chart(dados_meta).mark_rule(
                color=cores["muted"], strokeDash=[4, 4], strokeWidth=1.5
            ).encode(
                x=alt.X("meta:Q"),
                tooltip=[alt.Tooltip("meta:Q", title=rotulo_meta or "Meta", format=fmt)],
            )
        )

    altura = max(len(tabela) * 34, 160)

    return alt.layer(*camadas).properties(height=altura)


# --------------------------------------------------------------------------- #
# Seção 2 — matriz de confusão
# --------------------------------------------------------------------------- #

def heatmap_confusao(matriz_longa: pd.DataFrame) -> alt.Chart:
    """Matriz de confusão 2x2.

    A cor codifica a taxa dentro da classe real (magnitude contínua, rampa de um
    matiz). As contagens absolutas aparecem como rótulo em todas as células, de
    modo que nenhum valor depende da cor — necessário aqui, porque com fraude a
    0,1% as contagens de acerto e de erro diferem em três ordens de grandeza.
    """
    cores = paleta()

    tabela = matriz_longa.copy()
    total_por_classe = tabela.groupby("real")["quantidade"].transform("sum")
    tabela["taxa"] = tabela["quantidade"] / total_por_classe.where(total_por_classe > 0, 1)

    base = alt.Chart(tabela).encode(
        x=alt.X("previsto:N", title=None, axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y("real:N", title=None),
    )

    celulas = base.mark_rect(stroke=cores["superficie"], strokeWidth=2, cornerRadius=4).encode(
        color=alt.Color(
            "taxa:Q",
            scale=alt.Scale(range=cores["sequencial"], domain=[0, 1]),
            legend=alt.Legend(title="Taxa na classe real", format=".0%", gradientLength=140),
        ),
        tooltip=[
            alt.Tooltip("sigla:N", title="Célula"),
            alt.Tooltip("quantidade:Q", title="Transações", format=","),
            alt.Tooltip("taxa:Q", title="Taxa na classe real", format=".2%"),
        ],
    )

    contagem = base.mark_text(fontSize=20, fontWeight=600, dy=-8).encode(
        text=alt.Text("quantidade:Q", format=","),
        color=alt.condition(
            alt.datum.taxa > 0.5, alt.value(cores["superficie"]), alt.value(cores["ink"])
        ),
    )

    sigla = base.mark_text(fontSize=11, dy=14).encode(
        text=alt.Text("sigla:N"),
        color=alt.condition(
            alt.datum.taxa > 0.5, alt.value(cores["superficie"]), alt.value(cores["muted"])
        ),
    )

    return alt.layer(celulas, contagem, sigla).properties(height=220)


# --------------------------------------------------------------------------- #
# Seção 3 — curva Precision-Recall
# --------------------------------------------------------------------------- #

def curva_pr(
    curva: pd.DataFrame,
    ponto_operacao: dict | None = None,
    prevalencia: float | None = None,
) -> alt.Chart:
    """Curva Precision-Recall do modelo final, com ponto de operação.

    A linha de prevalência é a referência honesta em dados desbalanceados: é a
    precision que um classificador aleatório alcançaria.
    """
    cores = paleta()

    linha = alt.Chart(curva).mark_line(
        color=cores["series"][0], strokeWidth=2, interpolate="step-after"
    ).encode(
        x=alt.X("recall:Q", title="Recall (fraude)", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("precision:Q", title="Precision (fraude)", scale=alt.Scale(domain=[0, 1])),
    )

    foco = alt.selection_point(
        nearest=True, on="pointerover", fields=["recall"], empty=False
    )

    pontos_invisiveis = alt.Chart(curva).mark_point(size=80, opacity=0).encode(
        x=alt.X("recall:Q"),
        y=alt.Y("precision:Q"),
        tooltip=[
            alt.Tooltip("recall:Q", title="Recall", format=".4f"),
            alt.Tooltip("precision:Q", title="Precision", format=".4f"),
            alt.Tooltip("threshold:Q", title="Threshold", format=".4f"),
        ],
    ).add_params(foco)

    destaque = alt.Chart(curva).mark_point(
        size=90, filled=True, color=cores["series"][0],
        stroke=cores["superficie"], strokeWidth=2,
    ).encode(
        x=alt.X("recall:Q"),
        y=alt.Y("precision:Q"),
        opacity=alt.condition(foco, alt.value(1), alt.value(0)),
    )

    camadas = [linha, pontos_invisiveis, destaque]

    if prevalencia is not None:
        dados_prev = pd.DataFrame({"prevalencia": [prevalencia]})
        camadas.append(
            alt.Chart(dados_prev).mark_rule(
                color=cores["muted"], strokeDash=[4, 4], strokeWidth=1.5
            ).encode(
                y=alt.Y("prevalencia:Q"),
                tooltip=[
                    alt.Tooltip("prevalencia:Q", title="Prevalência (aleatório)", format=".4f")
                ],
            )
        )
        camadas.append(
            alt.Chart(dados_prev).mark_text(
                text="prevalência (classificador aleatório)",
                align="left", dx=6, dy=-8, fontSize=11, color=cores["muted"],
            ).encode(x=alt.value(8), y=alt.Y("prevalencia:Q"))
        )

    if ponto_operacao is not None:
        dados_ponto = pd.DataFrame([ponto_operacao])
        camadas.append(
            alt.Chart(dados_ponto).mark_point(
                size=180, filled=True, color=cores["series"][1],
                stroke=cores["superficie"], strokeWidth=2,
            ).encode(
                x=alt.X("recall:Q"),
                y=alt.Y("precision:Q"),
                tooltip=[
                    alt.Tooltip("recall:Q", title="Recall no ponto de operação", format=".4f"),
                    alt.Tooltip("precision:Q", title="Precision no ponto de operação", format=".4f"),
                ],
            )
        )
        camadas.append(
            alt.Chart(dados_ponto).mark_text(
                text="ponto de operação", align="right", dx=-14, fontSize=11,
                color=cores["ink_secundario"],
            ).encode(x=alt.X("recall:Q"), y=alt.Y("precision:Q"))
        )

    return alt.layer(*camadas)


# --------------------------------------------------------------------------- #
# Seção 4 — custo por threshold
# --------------------------------------------------------------------------- #

def curva_custo(
    sweep: pd.DataFrame,
    threshold_escolhido: float | None = None,
    threshold_minimo: float | None = None,
    threshold_operacional: float | None = None,
) -> alt.Chart:
    """Custo total simulado em função do threshold de decisão."""
    cores = paleta()

    linha = alt.Chart(sweep).mark_line(color=cores["series"][0], strokeWidth=2).encode(
        x=alt.X(
            "threshold:Q",
            title="Threshold de decisão",
            scale=alt.Scale(type="log", domain=[max(sweep["threshold"].min(), 1e-4), 1]),
        ),
        y=alt.Y("custo_total:Q", title="Custo total simulado", axis=alt.Axis(format=",.0f")),
        tooltip=[
            alt.Tooltip("threshold:Q", title="Threshold", format=".4f"),
            alt.Tooltip("custo_total:Q", title="Custo total", format=",.0f"),
            alt.Tooltip("fp:Q", title="FP", format=","),
            alt.Tooltip("fn:Q", title="FN", format=","),
            alt.Tooltip("recall:Q", title="Recall", format=".4f"),
            alt.Tooltip("precision:Q", title="Precision", format=".4f"),
        ],
    )

    camadas = [linha]

    marcadores = [
        (threshold_escolhido, cores["series"][1], "escolhido"),
        (threshold_minimo, cores["bom"], "custo mínimo"),
        (threshold_operacional, cores["muted"], "operacional"),
    ]

    for valor, cor, rotulo in marcadores:
        if valor is None:
            continue

        dados = pd.DataFrame({"threshold": [float(valor)], "rotulo": [rotulo]})

        camadas.append(
            alt.Chart(dados).mark_rule(color=cor, strokeWidth=2, strokeDash=[3, 3]).encode(
                x=alt.X("threshold:Q"),
                tooltip=[
                    alt.Tooltip("rotulo:N", title="Marcador"),
                    alt.Tooltip("threshold:Q", title="Threshold", format=".4f"),
                ],
            )
        )
        camadas.append(
            alt.Chart(dados).mark_text(
                align="left", dx=5, dy=-6, fontSize=11, color=cor,
            ).encode(x=alt.X("threshold:Q"), y=alt.value(10), text=alt.Text("rotulo:N"))
        )

    return alt.layer(*camadas)


def barras_custo_por_modelo(dados: pd.DataFrame, destaque: str | None = None) -> alt.Chart:
    """Custo total recalculado por modelo, ordenado do menor para o maior."""
    return barras_comparacao(
        dados,
        coluna="custo_total",
        titulo_eixo="Custo total simulado",
        formato="moeda",
        destaque=destaque,
    )


# --------------------------------------------------------------------------- #
# Seção 5 — robustez por imbalance ratio
# --------------------------------------------------------------------------- #

def linhas_por_ir(
    dados_longos: pd.DataFrame,
    ordem_cenarios: list[str],
    ordem_metricas: list[str],
    meta_recall: float | None = None,
) -> alt.Chart:
    """Métricas por cenário de imbalance ratio.

    Até quatro séries, cada uma com legenda e rótulo direto no último ponto —
    a identidade nunca depende só da cor. O eixo x é ordinal porque os cenários
    são pontos de teste discretos, não uma variável contínua.
    """
    cores = paleta()
    escala_cor = alt.Scale(domain=ordem_metricas, range=cores["series"][: len(ordem_metricas)])

    base = alt.Chart(dados_longos).encode(
        x=alt.X("cenario_ir:N", title="Imbalance ratio (legítimas por fraude)", sort=ordem_cenarios),
        y=alt.Y("valor:Q", title="Valor da métrica", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("metrica:N", scale=escala_cor, legend=alt.Legend(title=None)),
    )

    linhas = base.mark_line(strokeWidth=2, point=False)

    pontos = base.mark_point(size=90, filled=True, stroke=cores["superficie"], strokeWidth=2).encode(
        tooltip=[
            alt.Tooltip("cenario_ir:N", title="Cenário"),
            alt.Tooltip("metrica:N", title="Métrica"),
            alt.Tooltip("valor:Q", title="Valor", format=".4f"),
        ]
    )

    ultimo_cenario = ordem_cenarios[-1]
    rotulos = (
        alt.Chart(dados_longos[dados_longos["cenario_ir"] == ultimo_cenario])
        .mark_text(align="left", dx=8, fontSize=11)
        .encode(
            x=alt.X("cenario_ir:N", sort=ordem_cenarios),
            y=alt.Y("valor:Q"),
            text=alt.Text("metrica:N"),
            color=alt.Color("metrica:N", scale=escala_cor, legend=None),
        )
    )

    camadas = [linhas, pontos, rotulos]

    if meta_recall is not None:
        dados_meta = pd.DataFrame({"meta": [meta_recall]})
        camadas.append(
            alt.Chart(dados_meta).mark_rule(
                color=cores["muted"], strokeDash=[4, 4], strokeWidth=1.5
            ).encode(
                y=alt.Y("meta:Q"),
                tooltip=[alt.Tooltip("meta:Q", title="Meta de recall", format=".2f")],
            )
        )

    return alt.layer(*camadas).properties(width="container")


# --------------------------------------------------------------------------- #
# Seção 6 — tempo de inferência
# --------------------------------------------------------------------------- #

ROTULOS_TEMPO = {"tempo_medio_ms": "Tempo médio", "tempo_p95_ms": "p95"}


def barras_tempo(dados_longos: pd.DataFrame) -> alt.Chart:
    """Tempo médio e p95 por modelo, na mesma escala de milissegundos.

    As duas medidas compartilham a unidade e o eixo — nunca dois eixos y.
    """
    cores = paleta()
    ordem = list(ROTULOS_TEMPO.values())
    escala_cor = alt.Scale(domain=ordem, range=cores["series"][:2])

    base = alt.Chart(dados_longos)

    barras = base.mark_bar(cornerRadiusEnd=4, height=alt.RelativeBandSize(0.8)).encode(
        x=alt.X("valor:Q", title="Milissegundos por transação", axis=alt.Axis(format=".6f")),
        y=alt.Y("modelo:N", title=None, sort="-x"),
        yOffset=alt.YOffset("medida:N", sort=ordem),
        color=alt.Color("medida:N", scale=escala_cor, legend=alt.Legend(title=None)),
        tooltip=[
            alt.Tooltip("modelo:N", title="Modelo"),
            alt.Tooltip("medida:N", title="Medida"),
            alt.Tooltip("valor:Q", title="ms por transação", format=".6f"),
        ],
    )

    rotulos = base.mark_text(align="left", dx=5, fontSize=11, color=cores["ink_secundario"]).encode(
        x=alt.X("valor:Q"),
        y=alt.Y("modelo:N", sort="-x"),
        yOffset=alt.YOffset("medida:N", sort=ordem),
        text=alt.Text("valor:Q", format=".6f"),
    )

    altura = max(dados_longos["modelo"].nunique() * 56, 200)

    return alt.layer(barras, rotulos).properties(height=altura)


def pontos_tempo_log(dados_longos: pd.DataFrame, metas: dict[str, float]) -> alt.Chart:
    """Mesmos tempos em escala logarítmica, com as metas visíveis.

    Em escala log as barras perderiam a base no zero, então as marcas viram
    pontos. Esta é a única visão em que os tempos medidos (~0,004 ms) e as metas
    (50 ms e 100 ms) cabem no mesmo eixo.
    """
    cores = paleta()
    ordem = list(ROTULOS_TEMPO.values())
    escala_cor = alt.Scale(domain=ordem, range=cores["series"][:2])

    minimo = max(dados_longos["valor"].min() / 3, 1e-6)
    maximo = max(max(metas.values()) * 3, dados_longos["valor"].max() * 3)

    pontos = alt.Chart(dados_longos).mark_point(
        size=140, filled=True, stroke=cores["superficie"], strokeWidth=2
    ).encode(
        x=alt.X(
            "valor:Q",
            title="Milissegundos por transação (escala log)",
            scale=alt.Scale(type="log", domain=[minimo, maximo]),
        ),
        y=alt.Y("modelo:N", title=None),
        color=alt.Color("medida:N", scale=escala_cor, legend=alt.Legend(title=None)),
        tooltip=[
            alt.Tooltip("modelo:N", title="Modelo"),
            alt.Tooltip("medida:N", title="Medida"),
            alt.Tooltip("valor:Q", title="ms por transação", format=".6f"),
        ],
    )

    camadas = [pontos]

    for rotulo, valor in metas.items():
        dados_meta = pd.DataFrame({"meta": [valor], "rotulo": [rotulo]})
        camadas.append(
            alt.Chart(dados_meta).mark_rule(
                color=cores["critico"], strokeDash=[4, 4], strokeWidth=1.5
            ).encode(
                x=alt.X("meta:Q"),
                tooltip=[
                    alt.Tooltip("rotulo:N", title="Meta"),
                    alt.Tooltip("meta:Q", title="ms", format=".0f"),
                ],
            )
        )
        camadas.append(
            alt.Chart(dados_meta).mark_text(
                align="right", dx=-6, dy=-6, fontSize=11, color=cores["critico"],
            ).encode(x=alt.X("meta:Q"), y=alt.value(10), text=alt.Text("rotulo:N"))
        )

    altura = max(dados_longos["modelo"].nunique() * 40, 200)

    return alt.layer(*camadas).properties(height=altura)
