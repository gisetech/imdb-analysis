# Movie IMDb Project

Este projeto realiza **análise exploratória de dados (EDA)** e **modelagem preditiva** usando a base reduzida do 
[IMDb Top 1000 (Kaggle)](https://www.kaggle.com/datasets):

- Quais fatores estão relacionados com alta nota e faturamento?
- É possível prever o gênero de um filme a partir da sinopse (*Overview*)?
- Como estimar a nota IMDb de um novo filme?

## Tecnologias utilizadas
- Python 3.10+
- Pandas, Numpy
- Scikit-learn (Pipeline, TfidfVectorizer, LogisticRegression)
- Jupyter Notebook

Modelos Implementados
1. Classificação de Gênero
Técnica: TF-IDF + LogisticRegression (OneVsRest)

Acurácia: 85%

Input: Texto da sinopse (Overview)

Output: Probabilidades para múltiplos gêneros

2. Previsão de Nota IMDB
Técnica: Pipeline com pré-processamento + RandomForestRegressor

Performance: RMSE 0.45, R² 0.72

Features: Runtime, votos, bilheteria, metascore, certificado, gênero, diretor, sinopse, ano

Output: Nota IMDB prevista (0-10)

## Como executar no Codespaces
1. Clone este repositório e abra no GitHub Codespaces.
2. Crie e ative um ambiente virtual:
   ```
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows
   ```
