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

# Análise de Filmes - Respostas às Questões

## 1. Qual filme você recomendaria para uma pessoa que você não conhece?

Com base no dataset utilizado, recomendaria **um filme com as seguintes características**:
- **IMDB Rating**: acima de 7.5
- **Meta_score**: acima de 70
- **No_of_Votes**: alto número (indicando popularidade)
- **Gênero**: Drama ou Drama/Thriller (gêneros mais universais)

**Justificativa**: Filmes com nota alta no IMDB e Metacritic tendem a ter qualidade técnica e narrativa reconhecidas. Um alto número de votos indica que o filme foi amplamente assistido e aprovado.

## 2. Principais fatores relacionados com alta expectativa de faturamento

Com base no modelo desenvolvido, os principais fatores são:

### Fatores Identificados:
- **Atores e Diretores Famosos**: Confirmado pela inclusão da variável `Director` no modelo
- **Meta_score**: Filmes com melhor recepção crítica tendem a ter melhor desempenho
- **Gênero**: Alguns gêneros historicamente performam melhor (Action, Adventure, Sci-Fi)
- **Runtime_min**: Duração adequada (nem muito curto, nem excessivamente longo)
- **Ano de Lançamento**: Filmes mais recentes podem ter melhor bilheteria

### Justificativa:
O modelo utiliza `Gross_clean` como uma das features, sugerindo que faturamento está correlacionado com qualidade percebida (ratings).

## 3. Insights da coluna Overview

### O que é a coluna Overview:
Com base no exemplo do IMDb, a coluna **Overview** contém o **resumo/sinopse do filme**. No exemplo de "12 Homens e uma Sentença": *"O julgamento de um assassinato em Nova Iorque é frustrado por um único membro, cujo ceticismo força o júri a considerar cuidadosamente as evidências antes de dar o veredito."*

### Possibilidade de inferir gênero:
**SIM**, é possível inferir o gênero através da Overview usando:

#### Técnicas utilizadas no código:
- **TF-IDF Vectorizer**: Extrai palavras-chave características
- **N-gramas (1,2)**: Captura combinações de palavras
- **Max_features=500**: Foca nas palavras mais relevantes

#### Como funciona:
- **Contexto narrativo**: "julgamento", "evidências" → Drama judicial
- **Combinações de palavras**: "nave espacial" → Sci-Fi, "romance proibido" → Romance

## 4. Previsão da Nota IMDB - Variáveis e Transformações

### Variáveis Utilizadas:
```python
features = ['Runtime_min', 'No_of_Votes', 'Gross_clean', 'Meta_score', 
            'Certificate', 'Genre', 'Director', 'Overview', 'Year_num']
```

### Transformações Aplicadas:

#### Variáveis Numéricas:
- **SimpleImputer(median)**: Tratamento de valores ausentes
- **StandardScaler**: Normalização das escalas
- **Variáveis**: Runtime_min, No_of_Votes, Gross_clean, Meta_score, Year_num

#### Variáveis Categóricas:
- **OneHotEncoder**: Transformação em variáveis dummy
- **Variáveis**: Certificate, Director, Genre

#### Texto (Overview):
- **TfidfVectorizer**: Conversão de texto em features numéricas
- **Stop_words**: Remoção de palavras irrelevantes
- **N-gramas (1,2)**: Captura contexto das palavras

### Justificativa das Escolhas:
- **Meta_score**: Forte correlação com qualidade do filme
- **No_of_Votes

## Como executar no Codespaces
1. Clone este repositório e abra no GitHub Codespaces.
2. Crie e ative um ambiente virtual:
   ```
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows
   ```
