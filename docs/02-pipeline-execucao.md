# Pipeline de execução

## Ordem recomendada

1. Abrir o notebook de EDA.
2. Executar o notebook de pré-processamento.
3. Executar o notebook de modelagem.

## Dependências

As dependências do projeto estão listadas em `requirements.txt` e devem ser atendidas pelo ambiente Python já disponível para execução local.

## Arquivos de entrada

O notebook de modelagem depende dos conjuntos processados em `data/processado`.

## Arquivos de saída

Os resultados são gerados em `data/modelagem` ou, dependendo do ambiente de execução do notebook, em `notebooks/data/modelagem`.

## Observação importante

As pastas de artefatos gerados devem ser tratadas como saídas temporárias e não como parte do histórico de código.
