# Dashboard Streamlit

Aplicação de leitura que apresenta os resultados já salvos pelo pipeline de
modelagem. Ela **não treina nem executa modelos**: todos os números vêm de
arquivos gerados pelos notebooks.

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

O app abre mesmo sem nenhum artefato em disco, usando o seed de demonstração
versionado em `app/data/exemplo`. Nesse caso, a barra lateral exibe um aviso de
modo demonstração.

Para usar os resultados reais, execute os notebooks na ordem documentada em
[02-pipeline-execucao.md](02-pipeline-execucao.md). O app passa a usá-los
automaticamente, sem alteração de código.

Opcionalmente, gere os artefatos leves de curva e sweep:

```bash
python scripts/exportar_artefatos_dashboard.py
```

## Publicar na web (Streamlit Community Cloud)

O repositório é público, o que atende ao plano gratuito. Passos:

1. acesse [share.streamlit.io](https://share.streamlit.io) e conecte a conta do GitHub;
2. escolha o repositório, a branch e informe `app/streamlit_app.py` como arquivo principal;
3. publique.

As dependências do deploy ficam em `app/requirements.txt`, propositalmente
enxuto: apenas `streamlit`, `altair`, `pandas`, `numpy` e `scikit-learn`. Como a
dashboard não treina modelos, `xgboost`, `lightgbm`, `imbalanced-learn`,
`kagglehub`, `matplotlib` e `seaborn` ficam fora do build. O `requirements.txt`
da raiz continua sendo o ambiente completo, usado pelos notebooks.

### Publicar com os números reais

O ambiente de deploy tem apenas o que está versionado, e as pastas `data/` não
são versionadas. Sem nenhuma providência, a versão publicada abre em modo
demonstração.

Para publicar com os resultados reais, gere os artefatos leves incluindo os
arquivos de resultados e versione a pasta:

```bash
python scripts/exportar_artefatos_dashboard.py --incluir-resultados
git add app/data/dashboard
git commit -m "chore(dashboard): update published results"
```

São cerca de 45 KB no total. Com eles a versão publicada mostra a comparação de
modelos, a matriz de confusão, a curva Precision-Recall completa, o simulador
com slider de threshold ativo, a robustez por imbalance ratio e os tempos de
inferência — tudo sem as ~89 mil linhas de predições nem o dataset de 470 MB.

O que **não** deve ser versionado: `predicoes_teste_melhor_modelo.csv` (~3 MB),
`melhor_modelo.joblib` e qualquer conteúdo de `data/processado`.

## Páginas

| Página | Conteúdo |
|---|---|
| Visão geral (`app/streamlit_app.py`) | cartões do modelo final e comparação de modelos por precision, recall, F1, AUPRC e custo total |
| Modelo final | matriz de confusão e curva Precision-Recall |
| Simulador de custo | sliders de `custo_FP`, `custo_FN` e threshold, com recálculo em tempo real |
| Robustez por imbalance ratio | métricas por cenário de 1:100 a 1:2000 |
| Tempo de inferência | tempo médio e p95 por modelo, comparados às metas |

## Onde o app busca os dados

Cada artefato é procurado **individualmente** nesta ordem, e o primeiro
encontrado vale. Isso permite misturar fontes: predições reais podem coexistir
com o seed para o que ainda não foi gerado.

1. `$FRAUD_DASHBOARD_DATA_DIR` (se a variável de ambiente estiver definida)
2. `data/modelagem/`
3. `notebooks/data/modelagem/`
4. `app/data/dashboard/` (derivados leves do script de exportação)
5. `app/data/exemplo/` (seed de demonstração)

A barra lateral mostra a origem efetiva de cada arquivo carregado e lista os
ausentes.

## Artefatos consumidos

Gerados pelo notebook de modelagem:

| Arquivo | Colunas usadas | Onde é usado |
|---|---|---|
| `resultados_teste.csv` | `conjunto, cenario_features, modelo, threshold, precision_fraude, recall_fraude, f1_fraude, auprc, tn, fp, fn, tp, custo_total, tempo_medio_ms_por_transacao, tempo_p95_ms_por_transacao` | visão geral, tempo de inferência |
| `resultado_modelo_final.csv` | mesmo schema, uma linha | modelo final, simulador |
| `resultados_robustez_imbalance_ratio.csv` | acima + `imbalance_ratio_alvo, total_amostra, fraudes_amostra, legitimas_amostra` | robustez |
| `metadata_modelagem.json` | `melhor_modelo_base, melhor_cenario_features, melhor_threshold, criterio_selecao, custos, metas, data_execucao` | barra lateral, defaults dos sliders, linhas de meta |
| `predicoes_teste_melhor_modelo.csv` | `y_true, probabilidade_fraude` | curva PR e sweep de threshold |
| `tabela_principal_modelos.csv` | formato de apresentação | fallback da visão geral (não traz p95) |

Derivados por `scripts/exportar_artefatos_dashboard.py` em `app/data/dashboard/`:

| Arquivo | Colunas |
|---|---|
| `curva_precision_recall.csv` | `recall, precision, threshold` |
| `curva_threshold.csv` | `threshold, tp, fp, fn, tn, precision, recall, f1` |
| `manifest.json` | metadados da exportação, incluindo AUPRC e prevalência |

Esses dois CSVs somam algumas dezenas de kilobytes e substituem as ~89 mil
linhas de predições, permitindo que a dashboard funcione em qualquer clone.

## Por que o sweep de threshold

O custo é linear nas contagens de erro:

```text
custo_total = FP * custo_FP + FN * custo_FN
```

Com `tp/fp/fn/tn` tabelados por threshold, o custo para qualquer par de custos é
recalculado por aritmética vetorizada sobre poucas centenas de linhas. Por isso
os sliders respondem em tempo real sem reprocessar predições e sem cache.

## Cache

| O quê | Estratégia |
|---|---|
| leitura de CSV e JSON | `st.cache_data`, com o `mtime` do arquivo na assinatura — regravar um artefato invalida o cache |
| sweep e curva PR a partir das predições | `st.cache_data`, com `mtime` e parâmetros na chave |
| recálculo de custo pelos sliders | sem cache; é aritmética vetorizada |

`st.cache_resource` não é usado, porque o app não carrega `melhor_modelo.joblib`
nem faz inferência.

## Degradação quando falta dado

O app nunca preenche lacuna com valor estimado. Quando um dado não existe na
fonte carregada, ele diz isso:

- sem `predicoes_teste_melhor_modelo.csv` nem `curva_threshold.csv`, o slider de
  threshold é omitido e o simulador opera sobre o FP/FN fixo do modelo final;
- sem a curva PR, a página do modelo final mostra apenas o ponto de operação
  publicado e a linha de prevalência;
- modelos sem tempo registrado aparecem como "sem dado" e ficam fora do gráfico,
  em vez de serem plotados como zero;
- o threshold operacional aparece como "não publicado" quando o metadata não o
  traz.

## Decisões de visualização

- nenhum gráfico usa dois eixos y;
- cores categóricas em ordem fixa por identidade da série, então filtrar séries
  não repinta as que sobraram;
- passos de cor escolhidos separadamente para tema claro e escuro e validados
  contra as superfícies reais do Streamlit (banda de luminosidade, piso de
  croma, separação para daltonismo protan/deutan e piso de visão normal);
- a matriz de confusão usa rampa de um matiz com as contagens sempre visíveis
  como rótulo, porque com fraude a 0,13% acertos e erros diferem em três ordens
  de grandeza;
- o gráfico por imbalance ratio traz rótulos diretos e tabela equivalente,
  exigidos porque duas de suas cores ficam abaixo de 3:1 no tema claro;
- o eixo de threshold é logarítmico, já que os thresholds úteis em classe rara
  se concentram nos valores baixos.
