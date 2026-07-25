"""Utilitários da dashboard de detecção de fraude.

Os módulos deste pacote separam três responsabilidades:

- `paths`: descobre onde estão os artefatos salvos pelo pipeline;
- `loaders`: lê e normaliza esses artefatos com cache do Streamlit;
- `metrics`: cálculos puros (sweep de threshold, matriz de custo, curva PR);
- `charts`: montagem dos gráficos Altair.

Apenas `loaders` e `charts` dependem de Streamlit. `metrics` é puro
pandas/numpy para poder ser reaproveitado pelo script de exportação.
"""
