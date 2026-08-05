#!/usr/bin/env python3
"""Gera os artefatos leves que a dashboard usa para curva PR e sweep de threshold.

O notebook de modelagem salva `predicoes_teste_melhor_modelo.csv` com uma linha
por transação do conjunto de teste (~89 mil linhas). A dashboard não precisa
desse arquivo inteiro: a curva Precision-Recall cabe em algumas centenas de
pontos, e o custo total para qualquer par (custo_FP, custo_FN) pode ser
recalculado a partir das contagens da matriz de confusão por threshold.

Este script deriva esses dois arquivos, somando menos de 50 KB, para que a
dashboard funcione sem depender do dataset completo.

Com `--incluir-resultados`, também copia os arquivos de resultados (poucos KB
cada) para a mesma pasta. Como `app/data/dashboard` não é ignorada pelo Git,
isso permite versionar tudo o que a dashboard precisa e publicá-la na web com
os números reais.

Uso:

    python scripts/exportar_artefatos_dashboard.py
    python scripts/exportar_artefatos_dashboard.py --entrada notebooks/data/modelagem
    python scripts/exportar_artefatos_dashboard.py --incluir-resultados
    python scripts/exportar_artefatos_dashboard.py --n-thresholds 400 --max-pontos-pr 800
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

RAIZ_PROJETO = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(RAIZ_PROJETO / "app"))

from utils import metrics  # noqa: E402

ENTRADA_PADRAO = RAIZ_PROJETO / "data" / "modelagem"
SAIDA_PADRAO = RAIZ_PROJETO / "app" / "data" / "dashboard"

ARQUIVO_PREDICOES = "predicoes_teste_melhor_modelo.csv"
ARQUIVO_METADATA = "metadata_modelagem.json"

# Arquivos de resultados, todos de poucos KB, copiáveis com --incluir-resultados.
ARQUIVOS_RESULTADOS = (
    "resultados_teste.csv",
    "resultado_modelo_final.csv",
    "resultados_robustez_imbalance_ratio.csv",
    "tabela_principal_modelos.csv",
    ARQUIVO_METADATA,
)


def caminho_legivel(caminho: Path) -> str:
    """Caminho relativo à raiz do projeto quando possível, absoluto caso contrário."""
    try:
        return str(caminho.relative_to(RAIZ_PROJETO))
    except ValueError:
        return str(caminho)


def analisar_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deriva curva PR e sweep de threshold das predições salvas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--entrada", type=Path, default=ENTRADA_PADRAO,
        help="Pasta com os artefatos do notebook de modelagem.",
    )
    parser.add_argument(
        "--saida", type=Path, default=SAIDA_PADRAO,
        help="Pasta onde gravar os artefatos leves.",
    )
    parser.add_argument(
        "--n-thresholds", type=int, default=200,
        help="Quantidade de thresholds no sweep.",
    )
    parser.add_argument(
        "--max-pontos-pr", type=int, default=500,
        help="Máximo de pontos na curva Precision-Recall.",
    )
    parser.add_argument(
        "--incluir-resultados", action="store_true",
        help="Copia também os arquivos de resultados (poucos KB cada) para a pasta "
             "de saída, permitindo versioná-los e publicar a dashboard com os "
             "números reais.",
    )
    return parser.parse_args()


def carregar_predicoes(entrada: Path) -> pd.DataFrame:
    """Lê as predições do conjunto de teste, com erro explícito se faltarem."""
    caminho = entrada / ARQUIVO_PREDICOES

    if not caminho.is_file():
        raise SystemExit(
            f"Arquivo não encontrado: {caminho}\n\n"
            "Este script consome as predições salvas pelo notebook de modelagem.\n"
            "Execute, na ordem:\n"
            "  1. notebooks/preprocessamento_fraud_detection.ipynb\n"
            "  2. notebooks/modelagem_fraud_detection_refatorado_limpo.ipynb\n\n"
            "Se o notebook foi executado a partir da pasta notebooks, tente:\n"
            "  python scripts/exportar_artefatos_dashboard.py "
            "--entrada notebooks/data/modelagem"
        )

    predicoes = pd.read_csv(caminho)

    faltando = {"y_true", "probabilidade_fraude"} - set(predicoes.columns)
    if faltando:
        raise SystemExit(
            f"{caminho} não tem as colunas esperadas: {sorted(faltando)}. "
            f"Colunas encontradas: {list(predicoes.columns)}."
        )

    return predicoes


def carregar_metadata(entrada: Path) -> dict:
    """Lê o metadata da execução, se existir."""
    caminho = entrada / ARQUIVO_METADATA

    if not caminho.is_file():
        print(f"[aviso] {caminho} não encontrado; seguindo sem metadata.")
        return {}

    with open(caminho, encoding="utf-8") as arquivo:
        return json.load(arquivo)


def copiar_resultados(entrada: Path, saida: Path) -> list[Path]:
    """Copia os arquivos de resultados para a pasta de saída.

    São arquivos de poucos kilobytes, ao contrário das predições e do modelo
    serializado. Copiá-los para `app/data/dashboard` permite versioná-los e
    publicar a dashboard com os números reais, sem carregar o dataset.
    """
    copiados = []

    for nome in ARQUIVOS_RESULTADOS:
        origem = entrada / nome

        if not origem.is_file():
            print(f"[aviso] {nome} não encontrado em {caminho_legivel(entrada)}; pulando.")
            continue

        destino = saida / nome
        shutil.copyfile(origem, destino)
        copiados.append(destino)

    return copiados


def obter_threshold_operacional(predicoes: pd.DataFrame, metadata: dict) -> float | None:
    """Threshold usado no resultado final, do metadata ou das próprias predições."""
    valor = pd.to_numeric(metadata.get("melhor_threshold"), errors="coerce")

    if pd.notna(valor):
        return float(valor)

    if "threshold_usado" in predicoes.columns:
        valor = pd.to_numeric(predicoes["threshold_usado"], errors="coerce").dropna()
        if len(valor) > 0:
            return float(valor.iloc[0])

    return None


def main() -> None:
    argumentos = analisar_argumentos()

    predicoes = carregar_predicoes(argumentos.entrada)
    metadata = carregar_metadata(argumentos.entrada)

    y_true = predicoes["y_true"]
    y_score = predicoes["probabilidade_fraude"]
    threshold_operacional = obter_threshold_operacional(predicoes, metadata)

    print(f"Predições carregadas: {len(predicoes):,} transações")
    print(f"Fraudes: {int(y_true.sum()):,} · Legítimas: {int((y_true == 0).sum()):,}")
    if threshold_operacional is not None:
        print(f"Threshold operacional: {threshold_operacional:.6f}")

    sweep = metrics.calcular_sweep_threshold(
        y_true, y_score,
        n_thresholds=argumentos.n_thresholds,
        threshold_operacional=threshold_operacional,
    )
    curva_pr = metrics.calcular_curva_pr(y_true, y_score, max_pontos=argumentos.max_pontos_pr)
    auprc = metrics.calcular_auprc(y_true, y_score)

    argumentos.saida.mkdir(parents=True, exist_ok=True)

    caminho_sweep = argumentos.saida / "curva_threshold.csv"
    caminho_pr = argumentos.saida / "curva_precision_recall.csv"
    caminho_manifest = argumentos.saida / "manifest.json"

    sweep.to_csv(caminho_sweep, index=False)
    curva_pr.to_csv(caminho_pr, index=False)

    manifest = {
        "fonte": str(argumentos.entrada / ARQUIVO_PREDICOES),
        "data_export": datetime.now().isoformat(timespec="seconds"),
        "modelo": metadata.get("melhor_modelo_base"),
        "cenario": metadata.get("melhor_cenario_features"),
        "threshold_operacional": threshold_operacional,
        "n_teste": int(len(predicoes)),
        "fraudes_teste": int(y_true.sum()),
        "prevalencia": float(y_true.mean()),
        "auprc": auprc,
        "n_thresholds": int(len(sweep)),
        "n_pontos_curva_pr": int(len(curva_pr)),
    }

    with open(caminho_manifest, "w", encoding="utf-8") as arquivo:
        json.dump(manifest, arquivo, ensure_ascii=False, indent=4)
        arquivo.write("\n")

    gerados = [caminho_sweep, caminho_pr, caminho_manifest]

    if argumentos.incluir_resultados:
        gerados.extend(copiar_resultados(argumentos.entrada, argumentos.saida))

    print("\nArquivos gerados:")
    total_kb = 0.0
    for caminho in gerados:
        tamanho_kb = caminho.stat().st_size / 1024
        total_kb += tamanho_kb
        print(f"  {caminho_legivel(caminho)} ({tamanho_kb:.1f} KB)")
    print(f"  total: {total_kb:.1f} KB")

    print(f"\nAUPRC: {auprc:.4f} · prevalência: {manifest['prevalencia']:.6f}")
    print(
        "\nA dashboard passa a usar estes arquivos automaticamente. "
        "Eles são pequenos o suficiente para serem versionados."
    )

    if argumentos.incluir_resultados:
        print(
            "\nPara publicar a dashboard com estes números, versione a pasta:\n"
            f"  git add {caminho_legivel(argumentos.saida)}"
        )
    else:
        print(
            "\nUse --incluir-resultados para copiar também os arquivos de "
            "resultados e poder versionar tudo o que a dashboard precisa."
        )


if __name__ == "__main__":
    main()
