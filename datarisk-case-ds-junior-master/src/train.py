#%%
import pandas as pd

import matplotlib.pyplot as plt
from feature_engineering import criar_tabela_analitica, calcular_medianas_imputacao, aplicar_imputacao
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
#%%

df_info = pd.read_csv('../data/base_info.csv', sep=';')
df_cad = pd.read_csv('../data/base_cadastral.csv', sep=';')
df_dev = pd.read_csv('../data/base_pagamentos_desenvolvimento.csv', sep=';')

df_dev = df_dev.dropna(subset=['DATA_PAGAMENTO']).copy()
df_dev['DATA_VENCIMENTO'] = pd.to_datetime(df_dev['DATA_VENCIMENTO'])
df_dev['DATA_PAGAMENTO'] = pd.to_datetime(df_dev['DATA_PAGAMENTO'])
df_dev['DIAS_ATRASO'] = (df_dev['DATA_PAGAMENTO'] - df_dev['DATA_VENCIMENTO']).dt.days
df_dev['INADIMPLENTE'] = (df_dev['DIAS_ATRASO'] >= 5).astype(int)
df_dev['BASE_ORIGEM'] = 'TREINO'

df_modelagem = criar_tabela_analitica(df_dev, df_info, df_cad)

data_corte_teste = '2020-07-01'
data_corte_oot = '2021-01-01'

df_train = df_modelagem[df_modelagem['SAFRA_REF'] < data_corte_teste].copy()
df_test = df_modelagem[(df_modelagem['SAFRA_REF'] >= data_corte_teste) & (df_modelagem['SAFRA_REF'] < data_corte_oot)].copy()
df_oot = df_modelagem[df_modelagem['SAFRA_REF'] >= data_corte_oot].copy()
medianas_imputacao = calcular_medianas_imputacao(df_train)

df_train = aplicar_imputacao(df_train, medianas_imputacao)
df_test = aplicar_imputacao(df_test, medianas_imputacao)
df_oot = aplicar_imputacao(df_oot, medianas_imputacao)

features_categoricas = ['PORTE', 'SEGMENTO_INDUSTRIAL', 'REGIAO']

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(df_train[features_categoricas])
#%%

def preparar_x_y(df):
    encoded_data = encoder.transform(df[features_categoricas])
    encoded_cols = encoder.get_feature_names_out(features_categoricas)
    df_encoded = pd.DataFrame(encoded_data, columns=encoded_cols, index=df.index)
    
    X = df.drop(columns=['ID_CLIENTE', 'SAFRA_REF', 'DATA_EMISSAO_DOCUMENTO', 'BASE_ORIGEM', 'INADIMPLENTE'] + features_categoricas)
    X = pd.concat([X, df_encoded], axis=1)
    y = df['INADIMPLENTE']
    return X, y

X_train, y_train = preparar_x_y(df_train)
X_test, y_test = preparar_x_y(df_test)
X_oot, y_oot = preparar_x_y(df_oot)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_oot_scaled = scaler.transform(X_oot)

#%%

# Treinamento dos modelos

rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

log_reg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

#%%

y_prob_rf = rf_model.predict_proba(X_oot)[:, 1]
y_prob_log = log_reg.predict_proba(X_oot_scaled)[:, 1]
y_prob_nb = nb_model.predict_proba(X_oot_scaled)[:, 1]

fpr_rf, tpr_rf, _ = roc_curve(y_oot, y_prob_rf)
fpr_log, tpr_log, _ = roc_curve(y_oot, y_prob_log)
fpr_nb, tpr_nb, _ = roc_curve(y_oot, y_prob_nb)

auc_rf = roc_auc_score(y_oot, y_prob_rf)
auc_log = roc_auc_score(y_oot, y_prob_log)
auc_nb = roc_auc_score(y_oot, y_prob_nb)
#%%
# AUC no TESTE 
y_prob_rf_train = rf_model.predict_proba(X_train)[:, 1]
y_prob_rf_test = rf_model.predict_proba(X_test)[:, 1]
auc_rf_train = roc_auc_score(y_train, y_prob_rf_train)
auc_rf_test = roc_auc_score(y_test, y_prob_rf_test)
print(f"AUC Random Forest - Treino: {auc_rf_train:.4f}")
print(f"AUC Random Forest - Teste:  {auc_rf_test:.4f}")
#%%
auc_rf = roc_auc_score(y_oot, y_prob_rf)
auc_log = roc_auc_score(y_oot, y_prob_log)
auc_nb = roc_auc_score(y_oot, y_prob_nb)
#%%
import seaborn as sns

plt.figure(figsize=(10, 8))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.4f})', color='blue', linewidth=2)
plt.plot(fpr_log, tpr_log, label=f'Regressão Logística (AUC = {auc_log:.4f})', color='green', linewidth=2)
plt.plot(fpr_nb, tpr_nb, label=f'Naive Bayes (AUC = {auc_nb:.4f})', color='red', linewidth=2)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva ROC - Validação Out-of-Time (OOT)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()
#%%

# Extrair as variáveis que o modelo considerou mais importantes
importancias = pd.DataFrame({
    'Feature': X_train.columns,
    'Importância': rf_model.feature_importances_
}).sort_values(by='Importância', ascending=False)

print("TOP 10 VARIÁVEIS MAIS IMPORTANTES")
print(importancias.head(10))

# Plotar o gráfico de importância
plt.figure(figsize=(10, 6))
plt.barh(importancias['Feature'][:10][::-1], importancias['Importância'][:10][::-1], color='steelblue')
plt.title('Quais variáveis mais ditam o Risco de Inadimplência?')
plt.xlabel('Importância (0 a 1)')
plt.tight_layout()
plt.show()
#%%

df_oot['PROBABILIDADE_INADIMPLENCIA'] = y_prob_rf

sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 4))
sns.histplot(df_oot['PROBABILIDADE_INADIMPLENCIA'], bins=50, kde=True, color='royalblue')
plt.title('Distribuição da Probabilidade de Inadimplência', fontsize=12)
plt.show()
#%%

model_artifact = pd.Series({
    'model': rf_model,
    'encoder': encoder,
    'features': list(X_train.columns),
    'medianas_imputacao': medianas_imputacao
})

model_artifact.to_pickle('model.pkl')
# %%
mask_com_historico_test = df_test['QTD_COBRANCAS_ANTERIORES'] > 0
mask_sem_historico_test = df_test['QTD_COBRANCAS_ANTERIORES'] == 0

auc_test_com_historico = roc_auc_score(
    y_test[mask_com_historico_test], y_prob_rf_test[mask_com_historico_test.values]
)
auc_test_sem_historico = roc_auc_score(
    y_test[mask_sem_historico_test], y_prob_rf_test[mask_sem_historico_test.values]
)

print(f"\nNo TESTE:")
print(f"Qtd linhas COM histórico: {mask_com_historico_test.sum()} | AUC: {auc_test_com_historico:.4f}")
print(f"Qtd linhas SEM histórico: {mask_sem_historico_test.sum()} | AUC: {auc_test_sem_historico:.4f}")
# %%
