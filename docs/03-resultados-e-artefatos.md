# Resultados e artefatos

## Artefatos principais

- `resultado_modelo_final.csv`: registro do melhor modelo no cenário operacional.
- `tabela_principal_modelos.csv`: comparação de modelos por F1, recall, precision, AUPRC, custo e tempo.
- `resultados_calibracao.csv`: avaliação de calibração probabilística.
- `metadata_modelagem.json`: metadados da execução.
- `melhor_modelo.joblib`: modelo persistido para uso posterior.

## Interpretação

O notebook executado mostrou que:

- o cenário operacional final foi `sem_pos_transacao`;
- modelos de gradient boosting tiveram desempenho forte;
- LightGBM apresentou bom compromisso entre desempenho e tempo;
- a calibração melhorou a qualidade das probabilidades.

## Limitações

A análise é experimental e usa um dataset sintético. Ou seja, os resultados são úteis para estudo e comparação metodológica, mas não substituem validação em dados reais.
