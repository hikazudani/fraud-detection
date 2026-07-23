# Detecção de Fraude em Transações Financeiras

Projeto acadêmico da disciplina de Projetos de IA para detecção de fraude em transações financeiras sintéticas. O foco do repositório é avaliar modelos supervisionados para classificação binária em um cenário fortemente desbalanceado, com atenção especial a:

- recall e precision da classe fraude;
- F1-score e AUPRC;
- custo operacional simulado de falsos positivos e falsos negativos;
- tempo de inferência por transação;
- qualidade probabilística após calibração.

O dataset principal é o [PaySim Synthetic Financial Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1), disponibilizado sob licença CC BY-SA 4.0 e usado como base experimental para estudar fraude financeira em um contexto sintético.

## Visão geral do projeto

Este repositório organiza o fluxo experimental em três notebooks principais:

1. [notebooks/eda_fraud_detection_refatorado.ipynb](notebooks/eda_fraud_detection_refatorado.ipynb)
   - baixa o dataset via `kagglehub`;
   - explora distribuição das classes;
   - identifica padrões gerais de fraude e inconsistências de saldo.

2. [notebooks/preprocessamento_fraud_detection.ipynb](notebooks/preprocessamento_fraud_detection.ipynb)
   - remove colunas de alto risco de leakage/cardinalidade;
   - cria features derivadas como `log_amount` e indicadores de erro de saldo;
   - aplica divisão temporal por `step`;
   - salva os conjuntos processados em `data/processado`.

3. [notebooks/modelagem_fraud_detection_refatorado_limpo.ipynb](notebooks/modelagem_fraud_detection_refatorado_limpo.ipynb)
   - carrega os datasets processados;
   - compara baselines e modelos como Regressão Logística, Random Forest, XGBoost e LightGBM;
   - compara estratégias com e sem SMOTE;
   - ajusta threshold operacional;
   - executa análise de calibração de probabilidade;
   - salva resultados e artefatos finais em `data/modelagem`.

## Resultado atual observado

O fluxo de modelagem foi executado com sucesso no ambiente configurado do projeto. Os resultados salvos em disco indicam que:

- o cenário operacional final é `sem_pos_transacao`;
- o melhor modelo por equilíbrio de custo total foi `LightGBM + SMOTE`;
- o melhor modelo em ranking de F1 do cenário operacional foi `LightGBM scale_pos_weight`;
- a calibração de probabilidade melhorou a qualidade das probabilidades para os modelos de gradient boosting.

### Resumo dos principais artefatos de saída

| Artefato | Descrição |
|---|---|
| `resultado_modelo_final.csv` | linha final do melhor modelo escolhido pelo critério operacional |
| `tabela_principal_modelos.csv` | comparação detalhada entre modelos |
| `resultados_calibracao.csv` | análise de calibração de probabilidades |
| `metadata_modelagem.json` | metadados de execução e caminhos dos outputs |
| `melhor_modelo.joblib` | modelo treinado persistido para uso posterior |

## Estrutura da pasta

```text
fraud-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── especificacao-formal-projeto.md
├── notebooks/
│   ├── eda_fraud_detection_refatorado.ipynb
│   ├── preprocessamento_fraud_detection.ipynb
│   └── modelagem_fraud_detection_refatorado_limpo.ipynb
├── docs/
│   ├── README.md
│   ├── 01-visao-geral.md
│   ├── 02-pipeline-execucao.md
│   └── 03-resultados-e-artefatos.md
```

## Pré-requisitos

- Python 3.11
- dependências listadas em [requirements.txt](requirements.txt)
- credenciais válidas do Kaggle para baixar o dataset via `kagglehub`

## Como executar

### 1. Executar os notebooks na ordem

1. `eda_fraud_detection_refatorado.ipynb`
2. `preprocessamento_fraud_detection.ipynb`
3. `modelagem_fraud_detection_refatorado_limpo.ipynb`

### 3. Verificar os artefatos gerados

Os resultados esperados são gerados em pastas de saída temporárias e não devem ser versionadas:

- `data/processado/`
- `data/modelagem/`
- `notebooks/data/modelagem/` quando a execução é disparada a partir do contexto do notebook

## Stack técnico

- Python
- pandas
- numpy
- scikit-learn
- XGBoost
- LightGBM
- imbalanced-learn
- matplotlib/seaborn
- joblib
- kagglehub

## Metodologia resumida

O trabalho usa uma abordagem experimental em três etapas:

1. exploração e diagnóstico do dataset;
2. pré-processamento e split temporal;
3. comparação de modelos e seleção operacional.

A avaliação prioriza métricas da classe positiva, especialmente recall, precision, F1 e AUPRC. O custo operacional também é levado em conta por meio de uma matriz de custo configurável, com foco em reduzir falsos negativos e manter tempo de inferência viável.

## Limitações

- o dataset é sintético;
- os resultados não equivalem a validação com dados reais do Pix ou de uma instituição financeira;
- o projeto é acadêmico e experimental;
- o cenário mais “realista” foi aproximado por split temporal, mas ainda sem contexto operacional completo de produção.

## Documentação detalhada

Para documentação mais aprofundada, consulte:

- [docs/README.md](docs/README.md)
- [docs/01-visao-geral.md](docs/01-visao-geral.md)
- [docs/02-pipeline-execucao.md](docs/02-pipeline-execucao.md)
- [docs/03-resultados-e-artefatos.md](docs/03-resultados-e-artefatos.md)
- [especificacao-formal-projeto.md](especificacao-formal-projeto.md)

## Status do projeto

| Entregável | Status |
|---|---|
| EDA | Concluído |
| Pré-processamento | Concluído |
| Modelagem e avaliação | Concluído |
| Calibração de probabilidade | Concluído |
| Dashboard/visualização final | Planejado |
| Relatório consolidado | Em consolidação |

