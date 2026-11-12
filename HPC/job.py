# -*- coding: utf-8 -*-
import os

import matplotlib
matplotlib.use('Agg')

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

STORAGE_URL = "sqlite:///tiamat_optuna.db"

import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.io import arff
from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import optuna

# Configuracoes Gerais
SEMENTE_ALEATORIA = 67
N_TENTATIVAS = 100
N_DOBRAS = 5
PROPORCAO_TESTE = 0.20


# 1. Coleta e Pre-processamento
dados_brutos, metadados = arff.loadarff("BioResponse.arff")
df = pd.DataFrame(dados_brutos)

colunas_texto = df.select_dtypes(include=[object]).columns
for col in colunas_texto:
    df[col] = df[col].str.decode("utf-8")

X = df.drop(columns=['target'])
y = df['target'].astype(int)

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=PROPORCAO_TESTE, random_state=SEMENTE_ALEATORIA, stratify=y
)

cv_stratified = StratifiedKFold(n_splits=N_DOBRAS, shuffle=True, random_state=SEMENTE_ALEATORIA)

print(f"Treino: {X_treino.shape[0]} moleculas | Teste: {X_teste.shape[0]} moleculas | Atributos: {X_treino.shape[1]}")

# 2. Avaliacao do Baseline
baseline = DummyClassifier(strategy="stratified", random_state=SEMENTE_ALEATORIA)
baseline.fit(X_treino, y_treino)
y_prob_base = baseline.predict_proba(X_teste)[:, 1]
auc_base = roc_auc_score(y_teste, y_prob_base)
print(f"Baseline ROC-AUC no Teste: {auc_base:.4f}")

# 3. Otimizacao de Hiperparametros com Optuna para os Modelos
optuna.logging.set_verbosity(optuna.logging.WARNING)

estudos = {}
melhores_modelos = {}

# Arvore de Decisao 
def objetivo_ad(trial):
    params = {
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "max_depth": trial.suggest_int("max_depth", 2, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
        "max_features": trial.suggest_float("max_features", 0.02, 1.0),
        "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-6, 1e-2, log=True),
        "random_state": SEMENTE_ALEATORIA
    }
    model = DecisionTreeClassifier(**params)
    scores = cross_val_score(model, X_treino, y_treino, cv=cv_stratified, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

print("Otimizando Arvore de Decisao...")
study_ad = optuna.create_study(study_name="arvore_decisao", storage=STORAGE_URL, load_if_exists=True, direction="maximize")
study_ad.optimize(objetivo_ad, n_trials=N_TENTATIVAS)
estudos['Arvore de Decisao'] = study_ad
melhores_modelos['Arvore de Decisao'] = DecisionTreeClassifier(**study_ad.best_params, random_state=SEMENTE_ALEATORIA)


# Random Forest 
def objetivo_fa(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_float("max_features", 0.05, 1.0),
        "random_state": SEMENTE_ALEATORIA,
        "n_jobs": -1
    }
    model = RandomForestClassifier(**params)
    scores = cross_val_score(model, X_treino, y_treino, cv=cv_stratified, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

print("Otimizando Random Forest...")
study_fa = optuna.create_study(study_name="random_forest", storage=STORAGE_URL, load_if_exists=True, direction="maximize")
study_fa.optimize(objetivo_fa, n_trials=N_TENTATIVAS)
estudos['Random Forest'] = study_fa
melhores_modelos['Random Forest'] = RandomForestClassifier(**study_fa.best_params, random_state=SEMENTE_ALEATORIA, n_jobs=-1)


# XGBoost 
def objetivo_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "eval_metric": "logloss",
        "random_state": SEMENTE_ALEATORIA
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_treino, y_treino, cv=cv_stratified, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

print("Otimizando XGBoost...")
study_xgb = optuna.create_study(study_name="xgboost", storage=STORAGE_URL, load_if_exists=True, direction="maximize")
study_xgb.optimize(objetivo_xgb, n_trials=N_TENTATIVAS)
estudos['XGBoost'] = study_xgb
melhores_modelos['XGBoost'] = XGBClassifier(**study_xgb.best_params, random_state=SEMENTE_ALEATORIA, n_jobs=-1)


# LightGBM 
def objetivo_lgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": SEMENTE_ALEATORIA,
        "n_jobs": -1,
        "verbose": -1
    }
    model = LGBMClassifier(**params)
    scores = cross_val_score(model, X_treino, y_treino, cv=cv_stratified, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

print("Otimizando LightGBM...")
study_lgb = optuna.create_study(study_name="lightgbm", storage=STORAGE_URL, load_if_exists=True, direction="maximize")
study_lgb.optimize(objetivo_lgb, n_trials=N_TENTATIVAS)
estudos['LightGBM'] = study_lgb
melhores_modelos['LightGBM'] = LGBMClassifier(**study_lgb.best_params, random_state=SEMENTE_ALEATORIA, n_jobs=-1, verbose=-1)


# KNN com Pipeline (Scaler)
def objetivo_knn(trial):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", KNeighborsClassifier(
            n_neighbors=trial.suggest_int("n_neighbors", 1, 60),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
            metric=trial.suggest_categorical("metric", ["euclidean", "manhattan", "cosine"]),
            n_jobs=-1
        ))
    ])
    scores = cross_val_score(pipeline, X_treino, y_treino, cv=cv_stratified, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

print("Otimizando KNN...")
study_knn = optuna.create_study(study_name="knn", storage=STORAGE_URL, load_if_exists=True, direction="maximize")
study_knn.optimize(objetivo_knn, n_trials=N_TENTATIVAS)
estudos['KNN'] = study_knn

# KNN final pipeline
best_knn_params = study_knn.best_params
melhores_modelos['KNN'] = Pipeline([
    ("scaler", StandardScaler()),
    ("modelo", KNeighborsClassifier(
        n_neighbors=best_knn_params["n_neighbors"],
        weights=best_knn_params["weights"],
        metric=best_knn_params["metric"],
        n_jobs=-1
    ))
])

# 4. Avaliacao Final no Conjunto de Teste
print("\n=== RESULTADOS FINAIS NO CONJUNTO DE TESTE ===")
resultados = [{'Modelo': 'Baseline', 'ROC-AUC': auc_base}]

for nome, modelo in melhores_modelos.items():
    modelo.fit(X_treino, y_treino)
    y_prob = modelo.predict_proba(X_teste)[:, 1]
    auc = roc_auc_score(y_teste, y_prob)
    resultados.append({'Modelo': nome, 'ROC-AUC': auc})
    print(f"[{nome}] ROC-AUC: {auc:.4f}")

df_res = pd.DataFrame(resultados).sort_values(by='ROC-AUC', ascending=False)
print("\nTabela Geral:")
print(df_res.to_string(index=False))

# Salvar modelo treinado opcionalmente
os.makedirs("models", exist_ok=True)
with open("models/melhores_modelos.pkl", "wb") as f:
    pickle.dump(melhores_modelos, f)

print("Processo concluido com sucesso!")