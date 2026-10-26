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

# Configura o Matplotlib para rodar em modo headless (sem interface gráfica)
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

#  Divisão inicial (80% treino, 20% teste)
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=SEMENTE_ALEATORIA, stratify=y
)

# Escalonamento específico para algoritmos sensíveis à distância (KNN)
scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled = scaler.transform(X_teste)

# Validação cruzada padrão para os modelos
cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE_ALEATORIA)

#  Baseline
baseline = DummyClassifier(strategy="stratified", random_state=SEMENTE_ALEATORIA)
baseline.fit(X_treino, y_treino)

y_pred_base = baseline.predict(X_teste)
y_prob_base = baseline.predict_proba(X_teste)[:, 1]

acc_base = accuracy_score(y_teste, y_pred_base)
auc_base = roc_auc_score(y_teste, y_prob_base)

print(f"Acurácia do Baseline: {acc_base:.4f}")
print(f"ROC-AUC do Baseline:  {auc_base:.4f}")

estudos = {}
melhores_modelos = {}

#  Árvore de Decisão 
def objectivo_ad(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "random_state": SEMENTE_ALEATORIA
    }
    model = DecisionTreeClassifier(**params)
    scores = cross_val_score(model, X_treino, y_treino, cv=cv_stratified, scoring="roc_auc")
    return scores.mean()

study_ad = optuna.create_study(direction="maximize")
study_ad.optimize(objectivo_ad, n_trials=100)
estudos['Árvore de Decisão'] = study_ad
melhores_modelos['Árvore de Decisão'] = DecisionTreeClassifier(**study_ad.best_params, random_state=SEMENTE_ALEATORIA)
print(f"[Árvore de Decisão] Melhor ROC-AUC (CV): {study_ad.best_value:.4f}")

#  Random Forest 
def objectivo_fa(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 250, log=True),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
        "random_state": SEMENTE_ALEATORIA,
        "n_jobs": -1
    }
    model = RandomForestClassifier(**params)
    scores = cross_val_score(model, X_treino, y_treino, cv=cv_stratified, scoring="roc_auc")
    return scores.mean()

study_fa = optuna.create_study(direction="maximize")
study_fa.optimize(objectivo_fa, n_trials=100)
estudos['Random Forest'] = study_fa
melhores_modelos['Random Forest'] = RandomForestClassifier(**study_fa.best_params, random_state=SEMENTE_ALEATORIA, n_jobs=-1)
print(f"[Random Forest] Melhor ROC-AUC (CV): {study_fa.best_value:.4f}")

#  XGBoost 
def objectivo_xgb(trial):
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
    scores = cross_val_score(model, X_treino, y_treino, cv=cv_stratified, scoring="roc_auc")
    return scores.mean()

study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objectivo_xgb, n_trials=100)
estudos['XGBoost'] = study_xgb
melhores_modelos['XGBoost'] = XGBClassifier(**study_xgb.best_params, random_state=SEMENTE_ALEATORIA, n_jobs=-1, eval_metric="logloss")
print(f"[XGBoost] Melhor ROC-AUC (CV): {study_xgb.best_value:.4f}")

#  KNN 
def objectivo_knn(trial):
    params = {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 50),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"])
    }
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
    scores = cross_val_score(model, X_treino, y_treino, cv=cv_stratified, scoring="roc_auc")
    return scores.mean()

study_lgb = optuna.create_study(direction="maximize")
study_lgb.optimize(objectivo_lgb, n_trials=100)
estudos['LightGBM'] = study_lgb
melhores_modelos['LightGBM'] = LGBMClassifier(**study_lgb.best_params, random_state=SEMENTE_ALEATORIA, n_jobs=-1, verbose=-1)
print(f"[LightGBM] Melhor ROC-AUC (CV): {study_lgb.best_value:.4f}")

#  Avaliação Final no Conjunto de Teste
resultados_desempenho = []
curvas_roc = {}

# Avaliação do Baseline
resultados_desempenho.append({
    'Modelo': 'Baseline',
    'Acurácia': acc_base,
    'ROC-AUC': auc_base
})
curvas_roc['Baseline'] = (roc_curve(y_teste, y_prob_base), auc_base)

# Avaliação dos demais modelos
for nome, modelo in melhores_modelos.items():
    X_tr = X_treino_scaled if nome == 'KNN' else X_treino
    X_te = X_teste_scaled if nome == 'KNN' else X_teste
    
    modelo.fit(X_tr, y_treino)
    y_pred = modelo.predict(X_te)
    y_prob = modelo.predict_proba(X_te)[:, 1]
    
    acc = accuracy_score(y_teste, y_pred)
    auc = roc_auc_score(y_teste, y_prob)
    
    resultados_desempenho.append({
        'Modelo': nome,
        'Acurácia': acc,
        'ROC-AUC': auc
    })
    curvas_roc[nome] = (roc_curve(y_teste, y_prob), auc)

df_resultados = pd.DataFrame(resultados_desempenho).sort_values(by='ROC-AUC', ascending=False)
print("\nTabela de Desempenho no Conjunto de Teste:")
print(df_resultados.to_string(index=False))

# Plotagem e Salvamento da Curva ROC Comparativa
plt.figure(figsize=(8, 6))

for nome, ((fpr, tpr, _), auc_val) in curvas_roc.items():
    plt.plot(fpr, tpr, lw=2, label=f'{nome} (AUC = {auc_val:.4f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Aleatório (AUC = 0.5000)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=11)
plt.ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)', fontsize=11)
plt.title('Curvas ROC Comparativas no Conjunto de Teste', fontsize=12, fontweight='bold')
plt.legend(loc="lower right", fontsize=9)
plt.grid(True)
plt.tight_layout()
plt.savefig('curvas_roc_comparativas.png', dpi=300)
plt.close()

# Plotagem e Salvamento do Histórico do Optuna
melhor_nome_estudo = max(estudos, key=lambda k: estudos[k].best_value)
fig_hist = optuna.visualization.matplotlib.plot_optimization_history(estudos[melhor_nome_estudo])
plt.title(f'Histórico de Convergência do Optuna ({melhor_nome_estudo})', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('optuna_historico_convergencia.png', dpi=300)
plt.close()

print("\nProcessamento concluído")