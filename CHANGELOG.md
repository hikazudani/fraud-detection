# Changelog

Todas as mudanças relevantes deste projeto. O formato segue
[Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/) e o versionamento
segue [Semantic Versioning](https://semver.org/lang/pt-BR/).

## [1.0.0] - 2026-07-25

Primeira versão completa: pipeline experimental executado de ponta a ponta e
dashboard publicada com os resultados reais.

### Adicionado

- **EDA** (`notebooks/eda_fraud_detection_refatorado.ipynb`): download do PaySim
  via `kagglehub`, distribuição das classes, padrões de fraude e diagnóstico de
  inconsistências de saldo.
- **Pré-processamento** (`notebooks/preprocessamento_fraud_detection.ipynb`):
  remoção de colunas com risco de leakage e alta cardinalidade, features
  derivadas (`log_amount`, indicadores de erro de saldo), split temporal por
  `step` e escrita dos conjuntos em `data/processado`.
- **Modelagem** (`notebooks/modelagem_fraud_detection_refatorado_limpo.ipynb`):
  baselines, Regressão Logística, Random Forest, XGBoost e LightGBM, com e sem
  SMOTE, ajuste de threshold operacional, robustez por imbalance ratio e
  medição de tempo de inferência.
- **Análise de calibração** de probabilidade para os modelos de gradient
  boosting.
- **Dashboard Streamlit** (`app/`) com cinco páginas: visão geral, modelo final,
  simulador de custo, robustez por imbalance ratio e tempo de inferência. É uma
  aplicação de leitura — não treina nem executa modelos.
- **Resolução de artefatos em cascata**: cada arquivo é procurado
  individualmente em `$FRAUD_DASHBOARD_DATA_DIR`, `data/modelagem`,
  `notebooks/data/modelagem`, `app/data/dashboard` e `app/data/exemplo`, o que
  permite misturar fontes e abrir o app sem nenhum artefato em disco.
- **Seed de demonstração** versionado em `app/data/exemplo`, transcrito da
  especificação, com aviso explícito na barra lateral quando está em uso.
- **Script de exportação** (`scripts/exportar_artefatos_dashboard.py`): deriva a
  curva Precision-Recall e o sweep de threshold das predições do conjunto de
  teste, substituindo ~89 mil linhas por algumas dezenas de KB. A flag
  `--incluir-resultados` copia também os arquivos de resultados, viabilizando o
  deploy com os números reais.
- **Artefatos publicados** em `app/data/dashboard` (47 KB), gerados por uma
  execução completa dos notebooks.
- **Requirements enxuto de deploy** (`app/requirements.txt`), sem as bibliotecas
  de treino, para acelerar o build no Streamlit Community Cloud.
- **Documentação** em `docs/`: visão geral, pipeline de execução, resultados e
  artefatos, e arquitetura da dashboard, além da especificação formal do
  projeto.
- **Dashboard publicada** em
  [fraud-detection-cteia.streamlit.app](https://fraud-detection-cteia.streamlit.app/).

### Resultados

Cenário operacional `sem_pos_transacao`, com matriz de custo FP = 10 e
FN = 500 sobre o conjunto de teste do split temporal (89.466 transações,
1.252 fraudes, prevalência de 1,40%):

| Modelo | Threshold | Precision | Recall | F1 | AUPRC | Custo total |
|---|---|---|---|---|---|---|
| LightGBM + SMOTE (selecionado) | 0,99 | 0,827 | 0,911 | 0,867 | 0,959 | 58.380 |
| LightGBM scale_pos_weight | 0,99 | 0,896 | 0,867 | 0,881 | 0,954 | 84.760 |
| XGBoost scale_pos_weight | 0,99 | 0,918 | 0,844 | 0,880 | 0,957 | 98.440 |
| Baseline regra type + amount | — | 0,310 | 0,288 | 0,299 | 0,099 | 454.000 |

O modelo final foi escolhido pelo menor custo total entre os elegíveis, e não
pelo maior F1: LightGBM + SMOTE troca precision por recall, o que reduz falsos
negativos — o erro caro nessa matriz. Tempo de inferência de 0,0027 ms por
transação, bem abaixo da meta de 50 ms.

### Limitações conhecidas

- O dataset é sintético; os resultados não equivalem a validação com dados reais
  de uma instituição financeira.
- A modelagem roda com `MODO_RAPIDO = True`, que limita as transações legítimas
  do **treino** a 200 mil. O conjunto de teste não é subamostrado.
- O split temporal aproxima o cenário de produção, mas sem contexto operacional
  completo.

[1.0.0]: https://github.com/hikazudani/fraud-detection/releases/tag/v1.0.0
