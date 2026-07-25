# Entrega 1 - Especificação Formal do Projeto (Grupo 3)

**Título do projeto:** Detecção de Fraude em Transações Financeiras: avaliação de desempenho, desbalanceamento extremo, custo do erro e tempo de inferência

**Entrega:** Especificação formal do projeto + resultados preliminares de modelagem  
**Data:** 30 de junho de 2026  
**Disciplina:** Projetos de IA  
**Professor:** Nicksson Freitas  
**Equipe:** George Lucas Lopes da Silva; Lucas Gabriel Carvalho dos Ramos; Alexsandro Barreto de Abreu; Lucas Jundi Hikazudani; Martony Demes da Silva; Stefferson Bruno Costa Ferreira; João Pedro Piccino Marafiotti  

---

## Resumo executivo

Este documento consolida a proposta formal do projeto e incorpora resultados preliminares do primeiro ciclo de modelagem. O objetivo foi avaliar modelos de machine learning para detecção de fraude em transações financeiras sintéticas, considerando desempenho na classe minoritária, robustez a diferentes níveis de desbalanceamento, custo dos erros e tempo de inferência.

O modelo selecionado no ciclo preliminar foi o **XGBoost com `scale_pos_weight` e threshold otimizado**, no cenário **sem variáveis pós-transação**, por apresentar o melhor compromisso entre detecção, custo e viabilidade operacional.

---

## 1. Definição do problema

### 1.1 Contexto e justificativa

Sistemas de pagamento digitais processam grandes volumes de transações em pouco tempo. Nesse contexto, a detecção de fraude precisa equilibrar dois objetivos que podem entrar em conflito: identificar corretamente transações fraudulentas e manter uma resposta suficientemente rápida para não prejudicar a experiência do usuário.

O tema do projeto é a classificação de transações financeiras sintéticas como legítimas ou fraudulentas, utilizando o dataset **PaySim Synthetic Financial Dataset**. O problema é relevante porque fraudes representam uma parcela muito pequena das transações, mas podem gerar impacto financeiro significativo. A análise exploratória inicial identificou **6.362.620 registros**, sendo apenas **8.213 transações fraudulentas**. Isso corresponde a aproximadamente **0,129% de fraudes**, ou cerca de **1 fraude para cada 774 transações legítimas**.

Esse desbalanceamento torna a acurácia uma métrica inadequada como critério principal. Um modelo que classificasse todas as transações como legítimas teria acurácia muito alta, mas falharia no objetivo real do projeto: detectar fraudes. Portanto, a avaliação foi orientada por métricas da classe minoritária, como **recall**, **precision**, **F1-score** e **área sob a curva Precision-Recall**.

Além disso, seguindo o feedback recebido na pré-especificação, o projeto não buscou apenas o melhor resultado preditivo possível. O foco foi avaliar o compromisso entre desempenho e viabilidade operacional. Em uma transação financeira, um modelo muito preciso, mas lento, pode ser inadequado. Por isso, o tempo de inferência foi tratado como critério de sucesso junto com as métricas de detecção.

### 1.2 Formulação do problema

O problema foi tratado como uma tarefa de classificação binária supervisionada:

- **Entrada:** atributos de uma transação financeira sintética, como tipo da transação, valor, saldos antes/depois e tempo da simulação.
- **Saída:** classe prevista da transação: legítima ou fraudulenta.
- **Classe positiva:** fraude (`isFraud = 1`).
- **Classe negativa:** transação legítima (`isFraud = 0`).

**Pergunta central:** qual abordagem de classificação supervisionada apresenta o melhor equilíbrio entre detecção de fraudes, robustez ao desbalanceamento extremo e tempo de inferência em transações financeiras sintéticas?

**Pergunta complementar:** até que nível de desbalanceamento um modelo consegue manter recall e F1-score aceitáveis sem ultrapassar um tempo de resposta operacionalmente viável?

### 1.3 Hipótese

A hipótese do projeto é que modelos baseados em árvores, especialmente o XGBoost com ponderação da classe minoritária e ajuste de threshold, apresentam desempenho superior às baselines simples e à Regressão Logística na detecção de fraude. Espera-se que essa abordagem alcance F1-score de pelo menos 0,85 para a classe fraude, mantendo tempo de inferência compatível com o limite operacional acadêmico definido.

### 1.4 Objetivo geral

Avaliar modelos de machine learning para detecção de fraude em transações financeiras sintéticas, considerando simultaneamente desempenho na classe minoritária, robustez a diferentes níveis de desbalanceamento, custo dos erros e tempo de inferência compatível com um cenário de decisão rápida.

### 1.5 Objetivos específicos

1. Investigar abordagens utilizadas na literatura recente para detecção de fraude financeira em dados desbalanceados.
2. Construir baselines simples e interpretáveis, incluindo regra por tipo/valor e uso da coluna `isFlaggedFraud` apenas como referência comparativa.
3. Treinar modelos supervisionados clássicos e eficientes para dados tabulares, como Regressão Logística, Random Forest e XGBoost.
4. Comparar estratégias para lidar com desbalanceamento, incluindo pesos de classe, SMOTE e ajuste de threshold de decisão.
5. Avaliar os modelos com foco na classe fraude, usando recall, precision, F1-score, AUPRC e matriz de confusão.
6. Medir o tempo médio de inferência por transação e comparar o custo operacional dos modelos.
7. Testar a robustez dos modelos sob diferentes imbalance ratios.
8. Simular o impacto de falsos positivos e falsos negativos por meio de matriz de custo configurável.

### 1.6 Escopo e limitações

O projeto não pretende criar um sistema real de prevenção a fraude para produção. O escopo é acadêmico e experimental. O dataset é sintético e não representa diretamente dados reais do Pix ou de uma instituição financeira brasileira. Ainda assim, ele é adequado para estudar alto volume, classe minoritária rara, efeitos de desbalanceamento e comparação entre estratégias de modelagem.

Não foram priorizadas, nesta primeira versão, abordagens de deep learning, graph neural networks ou modelos multimodais, pois essas técnicas exigem mais dados contextuais, maior custo computacional e maior complexidade de implementação. Elas foram discutidas no estado da arte como possibilidades futuras, mas o projeto deu foco a modelos clássicos mais viáveis para dados tabulares e inferência rápida.

---

## 2. Estado da arte e decisões de projeto

### 2.1 O que a literatura recente indica

A literatura recente mostra que a detecção de fraude financeira continua sendo um problema desafiador devido a desbalanceamento extremo, mudança no padrão das fraudes ao longo do tempo, alto custo de falsos negativos, excesso de falsos positivos e necessidade de resposta em tempo real.

Uma revisão recente de 2026 sobre detecção de fraude financeira aponta que sistemas modernos precisam lidar com múltiplas modalidades de dados, custo operacional, interpretabilidade, privacidade, concept drift e restrições de latência. Ao mesmo tempo, métodos clássicos supervisionados continuam relevantes em ambientes operacionais por serem mais simples, eficientes e fáceis de implantar em dados tabulares, enquanto técnicas como GNNs, transformers e aprendizado federado aparecem como tendências para cenários mais complexos.

Neste projeto, a literatura foi usada para evitar escolhas ingênuas. O objetivo não foi implementar a técnica mais avançada possível, mas selecionar métodos adequados ao escopo, medir seus limites e justificar por que algumas alternativas foram testadas ou deixadas fora da primeira versão.

| Tema observado na literatura | O que já foi tentado | Decisão para este projeto |
|---|---|---|
| Dados sintéticos de fraude | PaySim foi proposto para contornar a dificuldade de acesso a dados financeiros reais. | Usar PaySim como base experimental, deixando explícito que os resultados não equivalem a validação em Pix real. |
| Modelos tabulares | Regressão Logística, Random Forest, XGBoost e variações de gradient boosting são recorrentes. | Usar Regressão Logística como baseline supervisionado e Random Forest/XGBoost como modelos principais. |
| Desbalanceamento | SMOTE, undersampling, pesos de classe, threshold moving e métodos híbridos são frequentes. | Comparar SMOTE, pesos de classe e ajuste de threshold, sem assumir previamente que SMOTE será melhor. |
| Vazamento de dados | Estudos recentes mostram que aplicar sampling antes do split pode inflar artificialmente os resultados. | Aplicar SMOTE apenas no treino, depois da separação treino/validação/teste. |
| Métricas | Acurácia é inadequada em fraude; precision, recall, F1 e AUPRC são mais informativas para a classe minoritária. | Não usar acurácia como critério principal. Usar recall, precision, F1 e AUPRC. |
| Tempo de resposta | Sistemas reais exigem restrições de latência e modelos operacionalmente viáveis. | Medir tempo médio e p95 de inferência; selecionar o melhor modelo por equilíbrio entre detecção e tempo. |
| Custo do erro | Falso negativo e falso positivo possuem impactos diferentes. | Implementar análise de matriz de custo e selecionar modelo por custo total simulado. |

### 2.2 Referências recentes que orientaram a implementação

A escolha das referências priorizou trabalhos de 2022 a 2026. Referências anteriores a 2020 foram mantidas apenas quando são fundacionais, como o artigo original do PaySim.

- **Lopez-Rojas, Elmir e Axelsson (2016):** apresentam o PaySim como simulador financeiro para geração de dados sintéticos de mobile money, motivado pela dificuldade de acesso a dados reais de transações financeiras. Esta referência justifica a origem do dataset.
- **Isangediok e Gajamannage (2022):** comparam técnicas otimizadas de machine learning em bases de fraude desbalanceadas e avaliam AUC ROC e AUC PR, reforçando o uso de XGBoost e métricas adequadas para classe minoritária.
- **Velarde et al. (2023):** avaliam XGBoost sob diferentes distribuições de classe e mostram que o desempenho tende a piorar conforme o desbalanceamento aumenta. Isso justifica o teste por imbalance ratio.
- **Kabane (2024):** analisa XGBoost em fraude com diferentes formas de aplicação de sampling e evidencia o risco de data leakage quando o balanceamento é feito antes do split. Isso orientou o protocolo experimental.
- **Hoffmann et al. (2025):** avaliam Random Forest e XGBoost com e sem SMOTE em detecção de fraude, mostrando o trade-off entre recall e precision. Esse estudo reforça que o melhor modelo depende da prioridade institucional.
- **Thivaios et al. (2026):** revisão que discute desafios de implantação, como desbalanceamento, concept drift, privacidade, interpretabilidade, custo operacional e latência em tempo real. Essa revisão orientou o recorte desempenho + tempo de inferência.
- **Batsyas e Yaduwanshi (2026):** trabalho recente que utiliza PaySim, CRISP-DM, EDA, Regressão Logística, Decision Tree, Random Forest, XGBoost, SMOTE e GridSearchCV. Por ser próximo do tema, foi usado como referência para não repetir cegamente uma solução sem avaliar limitações.
- **Fraud Detection Handbook:** discute métricas para fraude e mostra que acurácia pode ser enganosa em datasets com baixa proporção de fraude; recomenda métricas como recall, precision, F1 e Average Precision/Precision-Recall.
- **Banco Central do Brasil - Fórum Pix / MED 2.0 (2025/2026):** usado apenas como contexto de relevância, não como afirmação de que o dataset PaySim representa Pix real.

### 2.3 O que foi evitado

1. Usar acurácia como métrica principal, pois ela mascara o problema da classe minoritária.
2. Aplicar SMOTE antes da separação dos dados, pois isso pode causar data leakage e inflar os resultados.
3. Usar `isFlaggedFraud` como variável de treino principal, pois ela representa uma regra do próprio simulador e pode contaminar a avaliação.
4. Usar `nameOrig` e `nameDest` diretamente, devido à alta cardinalidade e risco de memorização.
5. Escolher o modelo apenas pelo maior F1, ignorando tempo de inferência e custo dos erros.
6. Prometer um sistema real para Pix, já que o dataset é sintético e tem limitações de representatividade.

---

## 3. Dados e estratégia de pré-processamento

### 3.1 Origem dos dados

O dataset escolhido é o **PaySim Synthetic Financial Dataset**, disponível no Kaggle. Ele contém transações financeiras sintéticas geradas pelo simulador PaySim, inspirado em transações de mobile money. O dataset possui aproximadamente **6,3 milhões de transações**, cerca de **470 MB** e **11 variáveis**.

**Licença:** Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0), conforme documentação do dataset.  
**Fonte:** https://www.kaggle.com/datasets/ealaxi/paysim1  
**Data de acesso:** 29 de maio de 2026.  
**Versão/snapshot:** Version 2 disponibilizada no Kaggle.  
**Autenticação:** o download requer conta e credenciais configuradas no Kaggle, utilizadas no projeto por meio da biblioteca `kagglehub`.

### 3.2 Estrutura dos dados

| Variável | Descrição | Uso planejado |
|---|---|---|
| `step` | Tempo da simulação, em horas | Usar para separação temporal e features de tempo |
| `type` | Tipo da transação | Usar com one-hot encoding |
| `amount` | Valor da transação | Usar com transformação `log1p` |
| `nameOrig` | Identificador do cliente de origem | Não usar diretamente; risco de alta cardinalidade |
| `oldbalanceOrg` | Saldo inicial da origem | Usar com cautela; avaliar risco de leakage |
| `newbalanceOrig` | Saldo final da origem | Usar com cautela; avaliar risco de leakage |
| `nameDest` | Identificador do destino | Não usar diretamente; risco de alta cardinalidade |
| `oldbalanceDest` | Saldo inicial do destino | Usar com cautela; avaliar risco de leakage |
| `newbalanceDest` | Saldo final do destino | Usar com cautela; avaliar risco de leakage |
| `isFraud` | Variável alvo | Classe positiva do modelo |
| `isFlaggedFraud` | Flag do simulador | Usar apenas como baseline comparativo, não no treino principal |

### 3.3 Achados da EDA

- O dataset não apresentou valores nulos nem registros duplicados.
- As fraudes são extremamente raras: 8.213 fraudes em 6.362.620 registros.
- As fraudes estão concentradas principalmente em transações dos tipos `TRANSFER` e `CASH_OUT`.
- A variável `amount` possui forte assimetria e presença de outliers, indicando a necessidade de testar `log1p(amount)`.
- A coluna `isFlaggedFraud` sinaliza apenas 16 transações, embora existam 8.213 fraudes marcadas em `isFraud`, indicando recall muito baixo.
- Foram observadas inconsistências frequentes nas relações de saldo, como `oldbalanceOrg - amount != newbalanceOrig`, o que exigiu investigação antes de usar essas variáveis.

### 3.4 Riscos dos dados

| Risco | Impacto | Mitigação |
|---|---|---|
| Dataset sintético | Resultados podem não generalizar para transações reais | Declarar limitação e avaliar apenas como simulação |
| Desbalanceamento extremo | Modelo pode ignorar a classe fraude | Usar métricas da classe minoritária e estratégias de balanceamento |
| Data leakage | Resultados artificialmente altos | Separar dados antes de sampling; auditar variáveis de saldo e flags |
| Alta cardinalidade em IDs | Memorização de clientes em vez de aprendizado geral | Não usar IDs diretamente |
| Split aleatório otimista | Avaliação pode ficar distante de cenário real | Priorizar split temporal por `step` |
| Custo computacional | Modelos lentos podem ser inviáveis | Medir tempo de treino e inferência |

### 3.5 Ética e privacidade

O dataset PaySim é sintético e não contém informações pessoais reais de clientes. Assim, este estudo não utiliza dados pessoais identificáveis nem dados financeiros reais. Ainda assim, sistemas de detecção de fraude aplicados em ambientes reais podem afetar usuários legítimos por meio de falsos positivos, como bloqueios indevidos ou atrito na experiência de pagamento.

Por esse motivo, o projeto avalia não apenas recall, mas também precision, custo dos erros e limitações de generalização. A licença CC BY-SA 4.0 permite o uso acadêmico do dataset, desde que a atribuição à fonte seja preservada.

### 3.6 Pré-processamento executado

1. Remoção de `isFlaggedFraud` das features principais, mantendo-a apenas como referência comparativa.
2. Remoção de `nameOrig` e `nameDest` como variáveis diretas.
3. Codificação da variável `type` por one-hot encoding.
4. Criação de `log_amount = log1p(amount)`.
5. Criação de variáveis de consistência de saldo, como `erro_saldo_origem` e `erro_saldo_destino`.
6. Criação de features temporais a partir de `step`, como `dia_simulado` e `hora_simulada`.
7. Separação temporal em treino, validação e teste, simulando aprendizado em transações passadas e avaliação em transações futuras.
8. Aplicação de SMOTE somente no conjunto de treino, nunca antes do split.
9. Padronização quando necessário, especialmente para Regressão Logística.

A separação temporal gerou conjuntos com proporções diferentes de fraude, pois a distribuição das fraudes varia ao longo do tempo. Por isso, além do teste temporal, foi realizado teste complementar por imbalance ratio.

---

## 4. Metodologia experimental

### 4.1 Stack técnica

O projeto foi desenvolvido em Python, utilizando pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, matplotlib, seaborn, joblib e kagglehub. O ambiente é isolado com `venv`, as dependências estão registradas em `requirements.txt` e os principais experimentos utilizam sementes fixas para favorecer a reprodutibilidade.

### 4.2 Protocolo de validação

A avaliação foi feita com separação temporal usando a variável `step`. Essa escolha simula um cenário mais realista: o modelo aprende com transações anteriores e é avaliado em transações futuras. Todas as técnicas de balanceamento foram aplicadas apenas no conjunto de treino, evitando data leakage.

Para viabilizar os experimentos no ambiente acadêmico, os modelos foram treinados com todas as transações fraudulentas disponíveis no conjunto de treino e uma amostra aleatória reprodutível de até 200.000 transações legítimas. Nos experimentos com SMOTE, o limite de transações legítimas foi reduzido para 50.000 devido ao custo computacional do oversampling. Os conjuntos temporais de validação e teste foram mantidos íntegros para a avaliação dos resultados.

### 4.3 Baselines

| Baseline | Descrição | Objetivo |
|---|---|---|
| Baseline 0 | Classificar tudo como legítimo | Mostrar por que acurácia é enganosa |
| Referência diagnóstica | Avaliar `isFlaggedFraud` como detector nativo do simulador | Evidenciar sua cobertura muito baixa e justificar que ela não entra no treino principal |
| Baseline 2 | Regra `type in {TRANSFER, CASH_OUT}` + threshold de `amount` | Criar referência simples e interpretável |

### 4.4 Modelos e experimentos

| Experimento | Modelo | Tratamento do desbalanceamento | Objetivo |
|---|---|---|---|
| E1 | Regressão Logística | `class_weight='balanced'` | Baseline supervisionado simples e rápido |
| E2 | Random Forest | `class_weight='balanced'` | Modelo não linear robusto para dados tabulares |
| E3 | XGBoost | `scale_pos_weight` | Modelo principal por bom desempenho e eficiência em dados tabulares |
| E4 | Random Forest + SMOTE | SMOTE apenas no treino | Avaliar ganho/perda do oversampling |
| E5 | XGBoost + SMOTE | SMOTE apenas no treino | Comparar oversampling com pesos de classe |
| E6 | Melhor modelo + ajuste de threshold | Threshold otimizado na validação | Ajustar trade-off entre recall, precision, F1 e custo |

### 4.5 Teste por imbalance ratio

| Cenário | Proporção aproximada | Objetivo |
|---|---|---|
| IR 1:100 | 1 fraude para 100 legítimas | Cenário menos extremo |
| IR 1:500 | 1 fraude para 500 legítimas | Próximo do problema original |
| IR 1:1000 | 1 fraude para 1000 legítimas | Cenário extremo e alvo principal |
| IR 1:2000 | 1 fraude para 2000 legítimas | Teste de estresse adicional |

Esse teste responde diretamente ao feedback recebido, pois permite estimar até qual nível de desbalanceamento a solução continua aceitável.

### 4.6 Tempo de inferência e matriz de custo

Cada modelo foi avaliado também pelo tempo médio de inferência por transação, p95 de inferência por transação e tempo total de classificação do conjunto de teste. A meta operacional acadêmica inicial foi:

- tempo médio <= 50 ms por transação;
- p95 <= 100 ms por transação.

A matriz de custo foi definida como:

```text
custo_total = FP * custo_FP + FN * custo_FN
```

Onde:

- **FP:** transação legítima marcada como fraude, gerando atrito para o cliente.
- **FN:** fraude classificada como legítima, gerando perda financeira.

---

## 5. Métricas de sucesso

| Dimensão | Métrica | Valor mínimo aceitável | Justificativa |
|---|---|---|---|
| Detecção | F1-score da classe fraude | >= 0,85 | Equilibra precision e recall |
| Detecção | Recall da classe fraude | >= 0,80 | Reduz fraudes que passam despercebidas |
| Detecção | Precision da classe fraude | >= 0,60 | Evita excesso de falsos alertas |
| Ranking | AUPRC | Maior que todos os baselines | Mais informativa que acurácia em classe rara |
| Robustez | Recall >= 0,80 até IR 1:1000 | Desejável | Mede resistência ao desbalanceamento extremo |
| Operação | Tempo médio de inferência | <= 50 ms/transação | Critério de viabilidade em resposta rápida |
| Operação | p95 de inferência | <= 100 ms/transação | Evita atrasos em casos mais lentos |
| Negócio | Custo total simulado | Menor que baseline de regra | Conecta resultado estatístico ao impacto prático |

A seleção do modelo final seguiu uma lógica de compromisso. O melhor modelo não foi necessariamente aquele com maior F1 isolado, mas aquele que apresentou melhor equilíbrio entre detecção, tempo de inferência, robustez ao imbalance ratio e custo total esperado.

---

## 6. Resultados da modelagem

### 6.1 Baselines

A baseline que classifica todas as transações como legítimas teve F1-score igual a zero para a classe fraude. Esse resultado confirma que a acurácia, isoladamente, não é uma métrica adequada para este problema. Como a maioria das transações é legítima, um modelo pode apresentar uma acurácia aparentemente alta simplesmente por ignorar a classe minoritária. Como referência diagnóstica adicional, a coluna `isFlaggedFraud` sinalizou apenas 16 transações diante das 8.213 fraudes presentes no dataset. Portanto, embora represente uma regra nativa do simulador, ela possui cobertura muito baixa e foi mantida apenas como referência diagnóstica, não como feature do treino principal.

A baseline baseada em uma regra simples usando tipo da transação e valor apresentou desempenho melhor do que a baseline ingênua, mas ainda muito inferior aos modelos supervisionados. No conjunto de teste, essa regra obteve precision de **0,3103**, recall de **0,2875** e F1-score de **0,2985** para a classe fraude. Apesar de ser interpretável, ela não foi suficiente para capturar a complexidade do problema.

### 6.2 Comparação dos modelos no cenário operacional

A tabela abaixo resume os principais resultados no cenário `sem_pos_transacao`, escolhido como cenário operacional mais conservador.

| Modelo | Precision | Recall | F1 | AUPRC | FP | FN | Custo total |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost `scale_pos_weight` | 0,9183 | 0,8442 | 0,8797 | 0,9570 | 94 | 195 | 98.440 |
| XGBoost + SMOTE | 0,9532 | 0,7971 | 0,8682 | 0,9498 | 49 | 254 | 127.490 |
| Random Forest `class_weight` | 0,9059 | 0,7995 | 0,8494 | 0,9096 | 104 | 251 | 126.540 |
| Random Forest + SMOTE | 0,9128 | 0,7524 | 0,8249 | 0,8716 | 90 | 310 | 155.900 |
| Regressão Logística `class_weight` | 0,7127 | 0,5527 | 0,6226 | 0,6593 | 279 | 560 | 282.790 |
| Regra `type + amount` | 0,3103 | 0,2875 | 0,2985 | 0,0992 | 800 | 892 | 454.000 |
| Tudo legítimo | 0,0000 | 0,0000 | 0,0000 | 0,0140 | 0 | 1252 | 626.000 |

### 6.3 Modelo final escolhido

O melhor modelo operacional foi o **XGBoost com `scale_pos_weight` e threshold otimizado**, utilizando o cenário `sem_pos_transacao`. Esse modelo obteve precision de **0,9183**, recall de **0,8442** e F1-score de **0,8797** para a classe fraude no conjunto de teste. Além disso, apresentou AUPRC de **0,9570**, indicando boa capacidade de ranquear transações fraudulentas acima das legítimas.

|  | Prevista legítima | Prevista fraude |
|---|---:|---:|
| Real legítima | 88.120 | 94 |
| Real fraude | 195 | 1.057 |

Esse resultado mostra que o modelo conseguiu detectar **1.057 das 1.252 fraudes** presentes no conjunto de teste, deixando passar **195 fraudes**. Ao mesmo tempo, classificou incorretamente **94 transações legítimas** como fraude.

### 6.4 Atendimento às metas

| Métrica | Meta | Resultado | Status |
|---|---:|---:|---|
| F1-score da classe fraude | >= 0,85 | 0,8797 | Atendida |
| Recall da classe fraude | >= 0,80 | 0,8442 | Atendida |
| Precision da classe fraude | >= 0,60 | 0,9183 | Atendida |
| Tempo médio por transação | <= 50 ms | 0,004371 ms | Atendida |
| p95 por transação | <= 100 ms | 0,004517 ms | Atendida |

Portanto, o modelo final atende tanto aos critérios de desempenho quanto aos critérios operacionais definidos inicialmente.

### 6.5 Diagnóstico do cenário completo

Também foi avaliado um cenário chamado `completo`, que utilizava todas as variáveis disponíveis após o pré-processamento, incluindo variáveis de saldo pós-transação e variáveis derivadas de inconsistências de saldo. Nesse cenário, alguns modelos apresentaram desempenho quase perfeito, com F1-score próximo de 1,0.

Apesar de parecer um resultado excelente, esse desempenho deve ser interpretado com cautela. Variáveis como `newbalanceOrig`, `newbalanceDest`, `erro_saldo_origem` e `erro_saldo_destino` podem carregar informações muito próximas do resultado final da transação ou capturar artefatos específicos do simulador PaySim. Isso pode gerar avaliação otimista demais e reduzir a capacidade de generalização do modelo para cenários reais. Por esse motivo, o cenário completo foi mantido apenas como diagnóstico.

### 6.6 Análise do SMOTE

O SMOTE foi aplicado apenas ao conjunto de treino, após a separação temporal dos dados, evitando vazamento de informação. Essa decisão segue a metodologia definida na proposta, pois aplicar oversampling antes da divisão entre treino, validação e teste poderia gerar resultados artificialmente altos.

Nos experimentos realizados, o uso de SMOTE melhorou a precision em alguns casos, mas não superou o XGBoost com `scale_pos_weight` no equilíbrio geral. O XGBoost + SMOTE obteve precision de **0,9532**, recall de **0,7971** e F1-score de **0,8682**. Apesar da alta precision, o recall ficou abaixo da meta de 0,80 e o custo total foi maior que o do XGBoost com `scale_pos_weight`. Esse resultado reforça que SMOTE não deve ser assumido como melhor solução automaticamente.

### 6.7 Robustez por imbalance ratio

| Cenário | Precision | Recall | F1-score | AUPRC |
|---|---:|---:|---:|---:|
| IR 1:100 | 0,9053 | 0,8600 | 0,8821 | 0,9533 |
| IR 1:500 | 0,6225 | 0,8807 | 0,7294 | 0,8806 |
| IR 1:1000 | 0,4371 | 0,8295 | 0,5725 | 0,7762 |
| IR 1:2000 | 0,2791 | 0,8182 | 0,4162 | 0,7012 |

O modelo manteve recall acima de **0,80** até o cenário **IR 1:2000**, o que indica boa capacidade de capturar fraudes mesmo quando elas se tornam extremamente raras. Porém, conforme o desbalanceamento aumenta, a precision e o F1-score caem de forma relevante. Isso significa que o modelo continua encontrando a maior parte das fraudes, mas passa a gerar proporcionalmente mais falsos positivos.

### 6.8 Tempo de inferência

O tempo de inferência foi um dos critérios centrais do projeto. O modelo final apresentou tempo médio de aproximadamente **0,004371 ms por transação** e p95 de aproximadamente **0,004517 ms por transação** no ambiente experimental. Esses valores ficaram muito abaixo das metas acadêmicas definidas, que eram de até 50 ms em média e até 100 ms no p95.

Esses tempos não representam garantia de desempenho em produção real, pois dependeriam da infraestrutura, do volume de requisições simultâneas e da integração com sistemas externos. Ainda assim, a comparação entre modelos mostra que o XGBoost apresentou uma relação favorável entre desempenho preditivo e velocidade.

---

## 7. Interpretação final

A modelagem confirmou que a detecção de fraude em dados extremamente desbalanceados exige métricas específicas para a classe minoritária. A baseline que classifica tudo como legítimo demonstrou que a acurácia pode ser enganosa, enquanto a regra simples baseada em tipo e valor mostrou que abordagens interpretáveis são úteis como referência, mas insuficientes como solução principal.

Entre os modelos avaliados, o **XGBoost com `scale_pos_weight` e threshold otimizado** foi selecionado como melhor modelo operacional. Ele atingiu F1-score de **0,8797**, recall de **0,8442**, precision de **0,9183** e AUPRC de **0,9570** no conjunto de teste, além de apresentar baixo custo total simulado e tempo de inferência muito inferior ao limite definido.

O cenário completo apresentou desempenho quase perfeito, mas foi tratado apenas como diagnóstico devido ao possível risco de vazamento ou uso de artefatos do simulador. A escolha final pelo cenário `sem_pos_transacao` torna a avaliação mais conservadora e mais adequada para discussão acadêmica.

Por fim, o teste por imbalance ratio mostrou que o modelo mantém recall acima de **0,80** até **IR 1:2000**, mas com queda progressiva de precision e F1-score. Assim, a conclusão técnica é que o modelo final é adequado para detectar fraudes em cenários altamente desbalanceados, principalmente quando o objetivo é reduzir fraudes não detectadas, mas seu uso exigiria ajuste de threshold e análise de custo conforme a tolerância da instituição a falsos positivos.

---

## 8. Entregáveis e status do projeto

| Entregável | Status | Observação |
|---|---|---|
| `proposta-projeto.md` / proposta formal | Concluído | Documento consolidado com problema, dados, metodologia, métricas e literatura. |
| Notebook de EDA e pré-processamento | Concluído | EDA, engenharia de atributos, split temporal e geração dos dados processados. |
| Notebook de modelagem | Concluído | Baselines, Random Forest, XGBoost, SMOTE, threshold, custo e tempo. |
| Tabela comparativa | Concluído | Tabela com métricas de detecção, custo total e tempo de inferência. |
| Dashboard Streamlit | Planejado | Usar resultados salvos para análise de threshold e matriz de custo. |
| Relatório final | Em consolidação | Discussão dos resultados, limitações e próximos passos. |

---

## 9. Riscos, limitações e próximos passos

| Risco/Limitação | Consequência | Tratamento/Próximo passo |
|---|---|---|
| Dataset sintético | Os resultados não podem ser interpretados como validação em Pix real. | Usar como estudo experimental e discutir necessidade de dados reais anonimizados. |
| Vazamento por variáveis de saldo | Cenário completo pode inflar resultados. | Usar cenário `sem_pos_transacao` como conclusão principal. |
| Queda de precision em IR extremo | Mais falsos positivos em cenários muito raros. | Ajustar threshold conforme custo institucional. |
| SMOTE não ser sempre superior | Oversampling pode aumentar precision, mas reduzir recall/custo total. | Comparar com pesos de classe e threshold, como feito. |
| Ambiente experimental | Tempo de inferência pode mudar em produção. | Testar em infraestrutura real e com requisições simultâneas. |

Como próximos passos, recomenda-se construir o dashboard Streamlit usando a tabela comparativa salva, permitindo simular custos de falso positivo e falso negativo, ajustar thresholds e visualizar a curva Precision-Recall do modelo final. Também seria interessante avaliar modelos adicionais de gradient boosting, como LightGBM, e testar estratégias de calibração de probabilidade.

---

## 10. Referências

- BANCO CENTRAL DO BRASIL. Fórum Pix - 27ª Reunião Plenária. 2025. Disponível em: https://www.bcb.gov.br/content/estabilidadefinanceira/pix/Forum_Pix_Plenaria/20251204-Forum_Pix.pdf
- BATSYAS, Ranya; YADUWANSHI, Ritesh. Fraud Detection System for Banking Transactions. arXiv:2604.07952, 2026. Disponível em: https://arxiv.org/abs/2604.07952
- FRAUD DETECTION HANDBOOK. Reproducible Machine Learning for Credit Card Fraud Detection - Performance Metrics. Disponível em: https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_4_PerformanceMetrics/Introduction.html
- HOFFMANN, Matheus H. R. P. et al. A Machine Learning-Based Framework for Detecting Financial Fraudulent Transactions. Iberoamerican Journal of Applied Computing, v. 13, n. 1, 2025. Disponível em: https://revistas.uepg.br/index.php/ijac/article/view/25781
- ISANGEDIOK, Mary; GAJAMANNAGE, Kelum. Fraud Detection Using Optimized Machine Learning Tools Under Imbalance Classes. arXiv:2209.01642, 2022. Disponível em: https://arxiv.org/abs/2209.01642
- KABANE, Siyaxolisa. Impact of Sampling Techniques and Data Leakage on XGBoost Performance in Credit Card Fraud Detection. arXiv:2412.07437, 2024. Disponível em: https://arxiv.org/abs/2412.07437
- KAGGLE. Synthetic Financial Datasets For Fraud Detection - PaySim. Disponível em: https://www.kaggle.com/datasets/ealaxi/paysim1
- LOPEZ-ROJAS, Edgar; ELMIR, Ahmad; AXELSSON, Stefan. PaySim: A Financial Mobile Money Simulator for Fraud Detection. 28th European Modeling and Simulation Symposium, 2016. Disponível em: https://www.msc-les.org/proceedings/emss/2016/EMSS2016_249.pdf
- MCDERMOTT, Matthew B. A. et al. A Closer Look at AUROC and AUPRC under Class Imbalance. arXiv:2401.06091, 2024. Disponível em: https://arxiv.org/abs/2401.06091
- THIVAIOS, S. et al. A Survey of Machine Learning and Deep Learning for Financial Fraud Detection: Architectures, Data Modalities, and Real-World Deployment Challenges. Algorithms, v. 19, n. 5, 354, 2026. Disponível em: https://www.mdpi.com/1999-4893/19/5/354
- VELARDE, Gissel et al. Evaluating XGBoost for Balanced and Imbalanced Data: Application to Fraud Detection. arXiv:2303.15218, 2023. Disponível em: https://arxiv.org/abs/2303.15218
