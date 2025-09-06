# src/modeling.py
import numpy as np
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

# -------------------------
# Funções auxiliares
# -------------------------
def extract_runtime(X):
    return X["Runtime"].str.replace(" min", "").astype(float).to_frame()

def log_votes(X):
    return np.log1p(X[["No_of_Votes"]])

def log_gross(X):
    gross = X["Gross"].str.replace(",", "").astype(float)
    return np.log1p(gross).to_frame()

def primary_genre(X):
    return X["Genre"].str.split(",").str[0].to_frame()

# -------------------------
# Criar pipeline
# -------------------------
def build_pipeline():
    # Features numéricas
    num_transformers = [
        ("runtime", FunctionTransformer(extract_runtime, validate=False)),
        ("votes_log", FunctionTransformer(log_votes, validate=False)),
        ("gross_log", FunctionTransformer(log_gross, validate=False))
    ]

    # Cada transformer gera uma coluna
    preprocessor = ColumnTransformer(
        transformers=[
            ("runtime", FunctionTransformer(extract_runtime, validate=False), ["Runtime"]),
            ("votes_log", FunctionTransformer(log_votes, validate=False), ["No_of_Votes"]),
            ("gross_log", FunctionTransformer(log_gross, validate=False), ["Gross"]),
            ("genre", Pipeline([
                ("extract", FunctionTransformer(primary_genre, validate=False)),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), ["Genre"]),
            ("overview", TfidfVectorizer(stop_words="english", max_features=5000), "Overview")
        ],
        remainder="drop"
    )

    # Modelo final (regressão linear simples)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    return pipeline

# -------------------------
# Treinar modelo
# -------------------------
def train_model(df):
    X = df[["Runtime", "No_of_Votes", "Gross", "Genre", "Overview"]]
    y = df["IMDB_Rating"]

    pipeline = build_pipeline()
    pipeline.fit(X, y)
    return pipeline

# -------------------------
# Salvar modelo
# -------------------------
def save_model(model, path="models/imdb_model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)

# -------------------------
# Carregar modelo
# -------------------------
def load_model(path="models/imdb_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)