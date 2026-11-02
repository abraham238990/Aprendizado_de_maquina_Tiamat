# -*- coding: utf-8 -*-
import os
import matplotlib
matplotlib.use('Agg')

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from scipy.io import arff

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEMENTE_ALEATORIA = 67
STORAGE_URL = "sqlite:///estudos_optuna.db"

def carregar_dados():
    try:
        dados_brutos, _ = arff.loadarff("BioResponse.arff")
        df = pd.DataFrame(dados_brutos)
        for col in df.select_dtypes(include=[object]).columns:
            df[col] = df[col].str.decode("utf-8")
        X = df.drop(columns=['target'])
        y = df['target'].astype(int)
    except Exception:
        print("Arquivo ARFF nao encontrado localmente. Baixando via OpenML...")
        bioresponse = fetch_openml(data_id=4134, as_frame=True, parser='auto')
        X = bioresponse.data
        y = bioresponse.target.astype(int)
    return X, y

print("Carregando os dados...")
X, y = carregar_dados()

print("Informacoes do Dataset:")
print(f"Total de exemplos (amostras): {X.shape[0]}")
print(f"Total de atributos (features): {X.shape[1]}")
print(f"Valores ausentes: {X.isnull().sum().sum()}")

=======
import os
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from scipy.io import arff
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Configura o Matplotlib para rodar em modo headless (sem interface grÃ¡fica)
plt.switch_backend('Agg')

# Desativa logs verbozos do Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

#  Carregar o arquivo ARFF
dados_brutos, metadados = arff.loadarff("BioResponse.arff")
df = pd.DataFrame(dados_brutos)

# Decodificar strings (bytes -> utf-8)
colunas_texto = df.select_dtypes(include=[object]).columns
for col in colunas_texto:
    df[col] = df[col].str.decode("utf-8")

# Separar atributos (X) e alvo (y)
X = df.drop(columns=['target'])
y = df['target'].astype(int)

SEMENTE_ALEATORIA = 67

#  DivisÃ£o inicial (80% treino, 20% teste)
>>>>>>> 08970b513e49b5c478f87028f8ad75cb1ed485eb
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=SEMENTE_ALEATORIA, stratify=y
)

<<<<<<< HEAD
cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE_ALEATORIA)

plt.figure(figsize=(6, 4))
sns.countplot(x=y, palette='Paired', hue=y, legend=False)
plt.title('Distribuicao das classes')
plt.xlabel('Classe')
plt.ylabel('Contagem')
plt.savefig('distribuicao_classes.png', dpi=300, bbox_inches='tight')
plt.close()

scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
pca = PCA()
pca.fit(X_treino_scaled)
variancia_acumulada = np.cumsum(pca.explained_variance_ratio_)

limiar_variancia = 0.90
n_componentes_90 = np.argmax(variancia_acumulada >= limiar_variancia) + 1

plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, len(variancia_acumulada) + 1), variancia_acumulada, marker='o', linestyle='-', color='aquamarine')
plt.axhline(limiar_variancia, linestyle='--', color='red', label=f'Limiar de {int(limiar_variancia*100)}% ({n_componentes_90} PCs)')
plt.title('Variancia Acumulada das Componentes Principais (PCA)')
plt.xlabel('Numero de Componentes Principais')
plt.ylabel('Variancia Explicada Acumulada')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('pca_variancia.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nNumero de componentes principais necessarias para reter 90% da variancia: {n_componentes_90}")

amostra_features = X.iloc[:, :15]
matriz_corr = amostra_features.corr()
plt.figure(figsize=(10, 8))
mascara = np.triu(np.ones_like(matriz_corr, dtype=bool))
sns.heatmap(
    matriz_corr,
    mask=mascara,
    cmap="PuOr",
    center=0,
    square=True,
    linewidths=0.5,
    annot=True,
    fmt=".2f"
)
plt.title("Mapa de Calor da Matriz de Correlacao (Pearson - Amostra D1-D15)")
plt.savefig('matriz_correlacao.png', dpi=300, bbox_inches='tight')
plt.close()

# PCA + Random Forest (Com persistência)
def objetivo_pca(trial):
    n_components = trial.suggest_int("n_components", 1, min(100, X_treino.shape[1]))
    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    max_depth = trial.suggest_int("max_depth", 2, 15)
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components)),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=SEMENTE_ALEATORIA,
            n_jobs=-1
        ))
    ])
    scores = cross_val_score(pipeline, X_treino, y_treino, scoring="accuracy", cv=cv_stratified, n_jobs=-1)
    return scores.mean()

print("\nGerenciando estudo Optuna: PCA + Random Forest...")
try:
    study_pca = optuna.load_study(study_name="pca_random_forest", storage=STORAGE_URL)
    print("Estudo PCA + RF carregado do disco.")
except Exception:
    study_pca = optuna.create_study(study_name="pca_random_forest", storage=STORAGE_URL, direction="maximize")
    study_pca.optimize(objetivo_pca, n_trials=20)
print("Melhor acuracia (PCA + RF) na validacao:", study_pca.best_value)

baseline = DummyClassifier(strategy="stratified", random_state=SEMENTE_ALEATORIA)
baseline.fit(X_treino, y_treino)
y_pred_base = baseline.predict(X_teste)
y_prob_base = baseline.predict_proba(X_teste)[:, 1]
acc_base = accuracy_score(y_teste, y_pred_base)
auc_base = roc_auc_score(y_teste, y_prob_base)

print(f"Acuracia do Baseline: {acc_base:.4f}")
print(f"ROC-AUC do Baseline: {auc_base:.4f}")

estudos = {}
melhores_modelos = {}
curvas_roc = {}

fpr_base, tpr_base, _ = roc_curve(y_teste, y_prob_base)
curvas_roc['Baseline'] = (fpr_base, tpr_base)

# Arvore de Decisao
def objetivo_ad(trial):
=======
# Escalonamento especÃ­fico para algoritmos sensÃ­veis Ã  distÃ¢ncia (KNN)
scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled = scaler.transform(X_teste)

# ValidaÃ§Ã£o cruzada padrÃ£o para os modelos
cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE_ALEATORIA)

#  Baseline
baseline = DummyClassifier(strategy="stratified", random_state=SEMENTE_ALEATORIA)
baseline.fit(X_treino, y_treino)

y_pred_base = baseline.predict(X_teste)
y_prob_base = baseline.predict_proba(X_teste)[:, 1]

acc_base = accuracy_score(y_teste, y_pred_base)
auc_base = roc_auc_score(y_teste, y_prob_base)

print(f"AcurÃ¡cia do Baseline: {acc_base:.4f}")
print(f"ROC-AUC do Baseline:  {auc_base:.4f}")

estudos = {}
melhores_modelos = {}

#  Ãrvore de DecisÃ£o 
def objectivo_ad(trial):
>>>>>>> 08970b513e49b5c478f87028f8ad75cb1ed485eb
    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "random_state": SEMENTE_ALEATORIA
    }
    model = DecisionTreeClassifier(**params)
<<<<<<< HEAD
    scores = cross_val_score(model, X_treino, y_treino, cv=cv_stratified, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

print("\nGerenciando estudo Optuna: Arvore de Decisao...")
try:
    study_ad = optuna.load_study(study_name="arvore_decisao", storage=STORAGE_URL)
    print("Estudo de Arvore de Decisao carregado do disco.")
except Exception:
    study_ad = optuna.create_study(study_name="arvore_decisao", storage=STORAGE_URL, direction="maximize")
    study_ad.optimize(objetivo_ad, n_trials=100)

estudos['Arvore de Decisao'] = study_ad
melhores_modelos['Arvore de Decisao'] = DecisionTreeClassifier(**study_ad.best_params, random_state=SEMENTE_ALEATORIA)

# Random Forest
def objetivo_fa(trial):
=======
    scores = cross_val_score(model, X_treino, y_treino, cv=cv_stratified, scoring="roc_auc")
    return scores.mean()

study_ad = optuna.create_study(direction="maximize")
study_ad.optimize(objectivo_ad, n_trials=100)
estudos['Ãrvore de DecisÃ£o'] = study_ad
melhores_modelos['Ãrvore de DecisÃ£o'] = DecisionTreeClassifier(**study_ad.best_params, random_state=SEMENTE_ALEATORIA)
print(f"[Ãrvore de DecisÃ£o] Melhor ROC-AUC (CV): {study_ad.best_value:.4f}")

#  Random Forest 
def objectivo_fa(trial):
>>>>>>> 08970b513e49b5c478f87028f8ad75cb1ed485eb
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 250, log=True),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
        "random_state": SEMENTE_ALEATORIA,
        "n_jobs": -1
    }
    model = RandomForestClassifier(**params)
<<<<<<< HEAD
    scores = cross_val_score(model, X_treino, y_treino, cv=cv_stratified, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

print("Gerenciando estudo Optuna: Random Forest...")
try:
    study_fa = optuna.load_study(study_name="random_forest", storage=STORAGE_URL)
    print("Estudo de Random Forest carregado do disco.")
except Exception:
    study_fa = optuna.create_study(study_name="random_forest", storage=STORAGE_URL, direction="maximize")
    study_fa.optimize(objetivo_fa, n_trials=100)

estudos['Random Forest'] = study_fa
melhores_modelos['Random Forest'] = RandomForestClassifier(**study_fa.best_params, random_state=SEMENTE_ALEATORIA, n_jobs=-1)

# XGBoost
def objetivo_xgb(trial):
=======
    scores = cross_val_score(model, X_treino, y_treino, cv=cv_stratified, scoring="roc_auc")
    return scores.mean()

study_fa = optuna.create_study(direction="maximize")
study_fa.optimize(objectivo_fa, n_trials=100)
estudos['Random Forest'] = study_fa
melhores_modelos['Random Forest'] = RandomForestClassifier(**study_fa.best_params, random_state=SEMENTE_ALEATORIA, n_jobs=-1)
print(f"[Random Forest] Melhor ROC-AUC (CV): {study_fa.best_value:.4f}")

#  XGBoost 
def objectivo_xgb(trial):
>>>>>>> 08970b513e49b5c478f87028f8ad75cb1ed485eb
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 250, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": SEMENTE_ALEATORIA,
        "n_jobs": -1,
        "eval_metric": "logloss"
    }
    model = XGBClassifier(**params)
<<<<<<< HEAD
    scores = cross_val_score(model, X_treino, y_treino, cv=cv_stratified, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

print("Gerenciando estudo Optuna: XGBoost...")
try:
    study_xgb = optuna.load_study(study_name="xgboost", storage=STORAGE_URL)
    print("Estudo de XGBoost carregado do disco.")
except Exception:
    study_xgb = optuna.create_study(study_name="xgboost", storage=STORAGE_URL, direction="maximize")
    study_xgb.optimize(objetivo_xgb, n_trials=100)

estudos['XGBoost'] = study_xgb
melhores_modelos['XGBoost'] = XGBClassifier(**study_xgb.best_params, random_state=SEMENTE_ALEATORIA, n_jobs=-1, eval_metric="logloss")

# KNN
def objetivo_knn(trial):
=======
    scores = cross_val_score(model, X_treino, y_treino, cv=cv_stratified, scoring="roc_auc")
    return scores.mean()

study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objectivo_xgb, n_trials=100)
estudos['XGBoost'] = study_xgb
melhores_modelos['XGBoost'] = XGBClassifier(**study_xgb.best_params, random_state=SEMENTE_ALEATORIA, n_jobs=-1, eval_metric="logloss")
print(f"[XGBoost] Melhor ROC-AUC (CV): {study_xgb.best_value:.4f}")

#  KNN 
def objectivo_knn(trial):
>>>>>>> 08970b513e49b5c478f87028f8ad75cb1ed485eb
    params = {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 50),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"])
    }
<<<<<<< HEAD
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(**params))
    ])
    scores = cross_val_score(pipeline, X_treino, y_treino, cv=cv_stratified, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

print("Gerenciando estudo Optuna: KNN...")
try:
    study_knn = optuna.load_study(study_name="knn", storage=STORAGE_URL)
    print("Estudo de KNN carregado do disco.")
except Exception:
    study_knn = optuna.create_study(study_name="knn", storage=STORAGE_URL, direction="maximize")
    study_knn.optimize(objetivo_knn, n_trials=100)

estudos['KNN'] = study_knn
melhores_modelos['KNN'] = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(**study_knn.best_params))
])

# LightGBM
def objetivo_lgb(trial):
=======
    model = KNeighborsClassifier(**params)
    scores = cross_val_score(model, X_treino_scaled, y_treino, cv=cv_stratified, scoring="roc_auc")
    return scores.mean()

study_knn = optuna.create_study(direction="maximize")
study_knn.optimize(objectivo_knn, n_trials=100)
estudos['KNN'] = study_knn
melhores_modelos['KNN'] = KNeighborsClassifier(**study_knn.best_params)
print(f"[KNN] Melhor ROC-AUC (CV): {study_knn.best_value:.4f}")

#  LightGBM 
def objectivo_lgb(trial):
>>>>>>> 08970b513e49b5c478f87028f8ad75cb1ed485eb
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 250, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "random_state": SEMENTE_ALEATORIA,
        "n_jobs": -1,
        "verbose": -1
    }
    model = LGBMClassifier(**params)
<<<<<<< HEAD
    scores = cross_val_score(model, X_treino, y_treino, cv=cv_stratified, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

print("Gerenciando estudo Optuna: LightGBM...")
try:
    study_lgb = optuna.load_study(study_name="lightgbm", storage=STORAGE_URL)
    print("Estudo de LightGBM carregado do disco.")
except Exception:
    study_lgb = optuna.create_study(study_name="lightgbm", storage=STORAGE_URL, direction="maximize")
    study_lgb.optimize(objetivo_lgb, n_trials=100)

estudos['LightGBM'] = study_lgb
melhores_modelos['LightGBM'] = LGBMClassifier(**study_lgb.best_params, random_state=SEMENTE_ALEATORIA, n_jobs=-1, verbose=-1)

# Avaliação Final no Conjunto de Teste
resultados_desempenho = []

resultados_desempenho.append({
    'Modelo': 'Baseline',
    'Acuracia': acc_base,
    'ROC-AUC': auc_base
})

for nome, modelo in melhores_modelos.items():
    modelo.fit(X_treino, y_treino)
    y_pred = modelo.predict(X_teste)
    y_prob = modelo.predict_proba(X_teste)[:, 1]
=======
    scores = cross_val_score(model, X_treino, y_treino, cv=cv_stratified, scoring="roc_auc")
    return scores.mean()

study_lgb = optuna.create_study(direction="maximize")
study_lgb.optimize(objectivo_lgb, n_trials=100)
estudos['LightGBM'] = study_lgb
melhores_modelos['LightGBM'] = LGBMClassifier(**study_lgb.best_params, random_state=SEMENTE_ALEATORIA, n_jobs=-1, verbose=-1)
print(f"[LightGBM] Melhor ROC-AUC (CV): {study_lgb.best_value:.4f}")

#  AvaliaÃ§Ã£o Final no Conjunto de Teste
resultados_desempenho = []
curvas_roc = {}

# AvaliaÃ§Ã£o do Baseline
resultados_desempenho.append({
    'Modelo': 'Baseline',
    'AcurÃ¡cia': acc_base,
    'ROC-AUC': auc_base
})
curvas_roc['Baseline'] = (roc_curve(y_teste, y_prob_base), auc_base)

# AvaliaÃ§Ã£o dos demais modelos
for nome, modelo in melhores_modelos.items():
    X_tr = X_treino_scaled if nome == 'KNN' else X_treino
    X_te = X_teste_scaled if nome == 'KNN' else X_teste
    
    modelo.fit(X_tr, y_treino)
    y_pred = modelo.predict(X_te)
    y_prob = modelo.predict_proba(X_te)[:, 1]
>>>>>>> 08970b513e49b5c478f87028f8ad75cb1ed485eb
    
    acc = accuracy_score(y_teste, y_pred)
    auc = roc_auc_score(y_teste, y_prob)
    
    resultados_desempenho.append({
        'Modelo': nome,
<<<<<<< HEAD
        'Acuracia': acc,
        'ROC-AUC': auc
    })
    
    fpr, tpr, _ = roc_curve(y_teste, y_prob)
    curvas_roc[nome] = (fpr, tpr)
=======
        'AcurÃ¡cia': acc,
        'ROC-AUC': auc
    })
    curvas_roc[nome] = (roc_curve(y_teste, y_prob), auc)
>>>>>>> 08970b513e49b5c478f87028f8ad75cb1ed485eb

df_resultados = pd.DataFrame(resultados_desempenho).sort_values(by='ROC-AUC', ascending=False)
print("\nTabela de Desempenho no Conjunto de Teste:")
print(df_resultados.to_string(index=False))

<<<<<<< HEAD
df_resultados.to_csv('resultados_desempenho.csv', index=False)

plt.figure(figsize=(8, 6))
for nome, (fpr, tpr) in curvas_roc.items():
    if nome == 'Baseline':
        auc_val = auc_base
    else:
        auc_val = df_resultados.loc[df_resultados['Modelo'] == nome, 'ROC-AUC'].values[0]
    plt.plot(fpr, tpr, lw=2, label=f'{nome} (AUC = {auc_val:.4f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Aleatorio (AUC = 0.5000)')
=======
# Plotagem e Salvamento da Curva ROC Comparativa
plt.figure(figsize=(8, 6))

for nome, ((fpr, tpr, _), auc_val) in curvas_roc.items():
    plt.plot(fpr, tpr, lw=2, label=f'{nome} (AUC = {auc_val:.4f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='AleatÃ³rio (AUC = 0.5000)')
>>>>>>> 08970b513e49b5c478f87028f8ad75cb1ed485eb
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=11)
plt.ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)', fontsize=11)
plt.title('Curvas ROC Comparativas no Conjunto de Teste', fontsize=12, fontweight='bold')
plt.legend(loc="lower right", fontsize=9)
plt.grid(True)
<<<<<<< HEAD
plt.savefig('curvas_roc_comparativas.png', dpi=300, bbox_inches='tight')
plt.close()

melhor_nome_estudo = max(estudos, key=lambda k: estudos[k].best_value)
importances = optuna.importance.get_param_importances(estudos[melhor_nome_estudo])
nomes = list(importances.keys())
valores = list(importances.values())

plt.figure(figsize=(8, 5))
plt.barh(nomes[::-1], valores[::-1])
plt.xlabel("Importancia relativa")
plt.ylabel("Hiperparametros")
plt.title(f"Importancia dos hiperparametros - {melhor_nome_estudo} (Optuna)")
plt.savefig('importancia_hiperparametros.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nExecucao finalizada com sucesso! Graficos e 'resultados_desempenho.csv' salvos.")
=======
plt.tight_layout()
plt.savefig('curvas_roc_comparativas.png', dpi=300)
plt.close()

# Plotagem e Salvamento do HistÃ³rico do Optuna
melhor_nome_estudo = max(estudos, key=lambda k: estudos[k].best_value)
fig_hist = optuna.visualization.matplotlib.plot_optimization_history(estudos[melhor_nome_estudo])
plt.title(f'HistÃ³rico de ConvergÃªncia do Optuna ({melhor_nome_estudo})', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('optuna_historico_convergencia.png', dpi=300)
plt.close()

print("\nProcessamento concluÃ­do")
>>>>>>> 08970b513e49b5c478f87028f8ad75cb1ed485eb
