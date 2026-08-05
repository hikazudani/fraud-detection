# Seed de demonstração

Estes arquivos existem **apenas** para que a dashboard rode em um clone limpo,
antes de o pipeline ser executado. Eles reproduzem o formato dos artefatos que o
notebook `modelagem_fraud_detection_refatorado_limpo.ipynb` grava em
`data/modelagem`.

Quando a dashboard usa este seed, ela exibe um aviso de **modo demonstração**.
Assim que os artefatos reais existirem, eles têm precedência automaticamente e o
aviso desaparece — sem alteração de código.

## Procedência dos números

Todos os valores foram transcritos de `docs/00-project-specification.md`:

| Arquivo | Origem |
|---|---|
| `resultados_teste.csv` | seção 6.2 (comparação de modelos) + 6.3 (matriz de confusão) + 6.8 (tempos) |
| `resultado_modelo_final.csv` | seções 6.3 e 6.8 |
| `resultados_robustez_imbalance_ratio.csv` | seção 6.7 |
| `metadata_modelagem.json` | seções 4.6 (custos), 5 (metas) e 6.3 (modelo/cenário) |

As colunas `tp` e `tn` **não** aparecem nas tabelas do documento: foram derivadas
de `fp`/`fn` e dos totais do conjunto de teste (1.252 fraudes e 88.214 legítimas,
obtidos da matriz de confusão da seção 6.3). A derivação foi validada
reproduzindo a `precision`, o `recall`, o `F1` e o `custo_total` publicados —
todos conferem com erro inferior a 5e-5, e os custos batem exatamente.

O mesmo vale para a tabela de imbalance ratio: `fp`/`fn` foram derivados da
`precision`/`recall` publicadas e do tamanho de amostra que o notebook constrói
(`n_fraudes = min(fraudes, legitimas // ir, 300)`), com validação equivalente.

## O que não está aqui

Campos que os documentos não publicam ficam **vazios**, nunca preenchidos com
valores inventados:

- o valor numérico do threshold operacional;
- `roc_auc`;
- tempo de inferência dos modelos que não são o modelo final;
- a curva Precision-Recall completa e o sweep de threshold.

Por isso, em modo demonstração a página do modelo final mostra apenas o ponto de
operação em vez da curva PR completa, e o slider de threshold do simulador de
custo fica desabilitado. Ambos passam a funcionar quando os artefatos reais (ou
os derivados de `app/data/dashboard`) estiverem disponíveis.
