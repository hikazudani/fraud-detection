"""Resolução dos caminhos dos artefatos consumidos pela dashboard.

O pipeline de modelagem grava seus artefatos em `data/modelagem` ou, quando o
notebook é executado a partir da própria pasta `notebooks`, em
`notebooks/data/modelagem`. Essas pastas não são versionadas.

Para que a dashboard funcione mesmo em um clone limpo, cada artefato é
procurado em uma lista ordenada de diretórios candidatos. O último candidato é
o seed de demonstração versionado em `app/data/exemplo`, transcrito das tabelas
publicadas em `docs/00-project-specification.md`. Quando algum artefato vem do
seed, a dashboard sinaliza que está em modo demonstração.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

RAIZ_PROJETO = Path(__file__).resolve().parents[2]

DIR_DADOS_APP = RAIZ_PROJETO / "app" / "data"
DIR_DERIVADOS = DIR_DADOS_APP / "dashboard"
DIR_EXEMPLO = DIR_DADOS_APP / "exemplo"

VARIAVEL_AMBIENTE_DADOS = "FRAUD_DASHBOARD_DATA_DIR"

# Nomes lógicos dos artefatos. As chaves são usadas em todo o app.
ARTEFATOS = {
    "resultados_teste": "resultados_teste.csv",
    "resultado_modelo_final": "resultado_modelo_final.csv",
    "tabela_principal_modelos": "tabela_principal_modelos.csv",
    "robustez_ir": "resultados_robustez_imbalance_ratio.csv",
    "metadata": "metadata_modelagem.json",
    "predicoes_teste": "predicoes_teste_melhor_modelo.csv",
    "curva_pr": "curva_precision_recall.csv",
    "curva_threshold": "curva_threshold.csv",
    "manifest_derivados": "manifest.json",
}


@dataclass(frozen=True)
class ArtefatoResolvido:
    """Um artefato encontrado em disco."""

    chave: str
    nome_arquivo: str
    caminho: Path
    origem: str
    is_demo: bool
    mtime: float

    @property
    def origem_relativa(self) -> str:
        """Caminho da pasta de origem relativo à raiz do projeto, quando possível."""
        try:
            return str(self.caminho.parent.relative_to(RAIZ_PROJETO))
        except ValueError:
            return str(self.caminho.parent)


def diretorios_candidatos() -> list[Path]:
    """Diretórios onde os artefatos são procurados, em ordem de precedência."""
    candidatos: list[Path] = []

    diretorio_env = os.environ.get(VARIAVEL_AMBIENTE_DADOS)
    if diretorio_env:
        candidatos.append(Path(diretorio_env).expanduser())

    candidatos.extend([
        RAIZ_PROJETO / "data" / "modelagem",
        RAIZ_PROJETO / "notebooks" / "data" / "modelagem",
        DIR_DERIVADOS,
        DIR_EXEMPLO,
    ])

    return candidatos


def resolver_artefato(chave: str) -> ArtefatoResolvido | None:
    """Procura um artefato nos diretórios candidatos e devolve o primeiro achado."""
    if chave not in ARTEFATOS:
        raise KeyError(f"Artefato desconhecido: {chave}. Opções: {sorted(ARTEFATOS)}")

    nome_arquivo = ARTEFATOS[chave]

    for diretorio in diretorios_candidatos():
        caminho = diretorio / nome_arquivo

        if not caminho.is_file():
            continue

        return ArtefatoResolvido(
            chave=chave,
            nome_arquivo=nome_arquivo,
            caminho=caminho,
            origem=str(diretorio),
            is_demo=diretorio == DIR_EXEMPLO,
            mtime=caminho.stat().st_mtime,
        )

    return None


def resolver_todos() -> dict[str, ArtefatoResolvido | None]:
    """Resolve todos os artefatos conhecidos de uma vez."""
    return {chave: resolver_artefato(chave) for chave in ARTEFATOS}


def esta_em_modo_demo(resolucao: dict[str, ArtefatoResolvido | None] | None = None) -> bool:
    """Indica se algum artefato essencial veio do seed de demonstração."""
    if resolucao is None:
        resolucao = resolver_todos()

    essenciais = ("resultados_teste", "resultado_modelo_final", "metadata", "robustez_ir")

    return any(
        resolucao[chave] is not None and resolucao[chave].is_demo
        for chave in essenciais
    )
