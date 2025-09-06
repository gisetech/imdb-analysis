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

## Estrutura do projeto
- notebooks/01_EDA.ipynb → análise exploratória, gráficos e hipóteses
- notebooks/02_Modeling.ipynb → pipeline TF-IDF + LogisticRegression, avaliação de acurácia
- src/modeling.py → script para treinar e salvar o modelo
- models/movie_genre_model.pkl → modelo salvo
- requirements.txt → pacotes necessários

## Como executar no Codespaces
1. Clone este repositório e abra no GitHub Codespaces.
2. Crie e ative um ambiente virtual:
   ```
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows
   ```
