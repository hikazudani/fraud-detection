# Detecção de Fraude em Transações Financeiras

Projeto acadêmico da disciplina **Projetos de IA**, que avalia modelos de machine learning para detecção de fraude em transações financeiras sintéticas, considerando desempenho na classe minoritária, robustez ao desbalanceamento extremo, custo do erro e tempo de inferência.

Dataset: [PaySim Synthetic Financial Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) (Kaggle, CC BY-SA 4.0) — ~6,3 milhões de transações, ~0,13% de fraude.

Proposta completa e resultados detalhados: [`proposta-projeto.md`](./proposta-projeto.md)

---

## Resultado principal

O modelo final selecionado foi o **XGBoost com `scale_pos_weight` e threshold otimizado**, no cenário sem variáveis pós-transação:

| Métrica | Resultado |
|---|---:|
| F1-score (fraude) | 0,8797 |
| Recall (fraude) | 0,8442 |
| Precision (fraude) | 0,9183 |
| AUPRC | 0,9570 |
| Tempo médio de inferência | ~0,0044 ms/transação |

Todos os critérios de sucesso definidos na proposta foram atendidos. Detalhes da comparação entre modelos, robustez por imbalance ratio e análise de custo estão na seção 6 da proposta.

---

## Estrutura do projeto

```
fraud-detection/
├── proposta-projeto.md                          # Proposta formal + resultados consolidados
├── requirements.txt
├── notebooks/
│   ├── eda_fraud_detection_refatorado.ipynb # Análise exploratória
│   ├── preprocessamento_fraud_detection.ipynb # Limpeza, features, split temporal
│   └── modelagem_fraud_detection_refatorado_limpo.ipynb # Baselines, modelos, avaliação
├── data/
│   ├── processado/                              # Saída do preprocessamento (gerado)
│   └── modelagem/                                # Resultados e modelo final (gerado)
└── README.md
```

> As pastas `data/processado` e `data/modelagem` são geradas ao executar os notebooks e **não devem ser versionadas** (ver `.gitignore`).

## Pipeline

Os notebooks devem ser executados nesta ordem:

1. **`eda_fraud_detection_refatorado.ipynb`** — baixa o dataset via `kagglehub`, faz a análise exploratória (distribuição de classes, tipos de transação, valores, inconsistências de saldo).
2. **`preprocessamento_fraud_detection.ipynb`** — remove colunas de alto risco de leakage/cardinalidade (`isFlaggedFraud`, `nameOrig`, `nameDest`), cria features (`log_amount`, variáveis de erro de saldo, features temporais), faz o split temporal por `step` e salva os conjuntos em `data/processado/`.
3. **`modelagem_fraud_detection_refatorado_limpo.ipynb`** — carrega os dados processados, treina baselines e modelos (Regressão Logística, Random Forest, XGBoost, com e sem SMOTE), avalia métricas/custo/tempo de inferência, testa robustez por imbalance ratio e salva os resultados e o modelo final (`melhor_modelo.joblib`) em `data/modelagem/`.

## Como rodar

```bash
git clone https://github.com/hikazudani/fraud-detection.git
cd fraud-detection

python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/Mac

pip install -r requirements.txt
```

Configure suas credenciais do Kaggle (necessário para o `kagglehub` baixar o dataset — veja [docs do kagglehub](https://github.com/Kagglehub/kagglehub) sobre `kaggle.json`).

Depois, execute os notebooks na ordem listada acima (Jupyter, VS Code ou Colab).

## Stack

Python · pandas · numpy · scikit-learn · XGBoost · imbalanced-learn (SMOTE) · matplotlib/seaborn · Streamlit (dashboard, planejado)

## Status

| Entregável | Status |
|---|---|
| Proposta formal | Concluído |
| EDA e pré-processamento | Concluído |
| Modelagem e avaliação | Concluído |
| Dashboard Streamlit | Planejado |
| Relatório final | Em consolidação |

## Limitações

Dataset sintético, os resultados não equivalem a validação em dados reais de Pix ou de instituições financeiras. Mais detalhes em "Riscos, limitações e próximos passos" na proposta.
