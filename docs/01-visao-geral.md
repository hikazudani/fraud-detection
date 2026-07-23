# Visão geral

## Tema

Detecção de fraude em transações financeiras sintéticas usando aprendizado supervisionado.

## Problema

A fraude representa uma fração muito pequena do total de transações. Isso torna a acurácia pouco útil como métrica principal, porque um modelo que classifica tudo como legítimo pode ter acurácia alta e ainda falhar no objetivo real.

## Objetivo principal

Encontrar um modelo que equilibre:

- recall para fraudes;
- precision aceitável;
- F1-score e AUPRC;
- custo de falsos positivos e falsos negativos;
- velocidade de inferência.

## Dataset

O repositório usa o PaySim Synthetic Financial Dataset, extraído do Kaggle e processado localmente via notebooks.

## Principais abordagens

- baselines interpretáveis;
- Regressão Logística;
- Random Forest;
- XGBoost;
- LightGBM;
- estratégias com e sem SMOTE;
- calibração de probabilidades com sigmoid e isotonic.
