#%%
import pandas as pd
from feature_engineering import criar_tabela_analitica, aplicar_imputacao
#%%

df_teste_novo = pd.read_csv('../data/base_pagamentos_teste.csv', sep=';')
df_info = pd.read_csv('../data/base_info.csv', sep=';')
df_cad = pd.read_csv('../data/base_cadastral.csv', sep=';')

df_dev_passado = pd.read_csv('../data/base_pagamentos_desenvolvimento.csv', sep=';')
df_dev_passado = df_dev_passado.dropna(subset=['DATA_PAGAMENTO']).copy()
df_dev_passado['DATA_VENCIMENTO'] = pd.to_datetime(df_dev_passado['DATA_VENCIMENTO'])
df_dev_passado['DATA_PAGAMENTO'] = pd.to_datetime(df_dev_passado['DATA_PAGAMENTO'])
df_dev_passado['DIAS_ATRASO'] = (df_dev_passado['DATA_PAGAMENTO'] - df_dev_passado['DATA_VENCIMENTO']).dt.days
df_dev_passado['INADIMPLENTE'] = (df_dev_passado['DIAS_ATRASO'] >= 5).astype(int)


df_dev_passado['BASE_ORIGEM'] = 'PASSADO'
df_teste_novo['BASE_ORIGEM'] = 'PRESENTE'

df_empilhado = pd.concat([df_dev_passado, df_teste_novo], ignore_index=True)
df_processado = criar_tabela_analitica(df_empilhado, df_info, df_cad)


df_teste_final = df_processado[df_processado['BASE_ORIGEM'] == 'PRESENTE'].copy()


model_artifact = pd.read_pickle('model.pkl')

#%%
rf_model = model_artifact['model']
encoder = model_artifact['encoder']
colunas_treino = model_artifact['features']
medianas_imputacao = model_artifact['medianas_imputacao']

df_teste_final = aplicar_imputacao(df_teste_final, medianas_imputacao)

features_categoricas = ['PORTE', 'SEGMENTO_INDUSTRIAL', 'REGIAO']
encoded_data = encoder.transform(df_teste_final[features_categoricas]) 
encoded_cols = encoder.get_feature_names_out(features_categoricas)
df_encoded = pd.DataFrame(encoded_data, columns=encoded_cols, index=df_teste_final.index)

#%%
X_predict = df_teste_final.drop(columns=['ID_CLIENTE', 'SAFRA_REF', 'DATA_EMISSAO_DOCUMENTO', 'BASE_ORIGEM', 'INADIMPLENTE'] + features_categoricas, errors='ignore')
X_predict = pd.concat([X_predict, df_encoded], axis=1)

X_predict = X_predict.reindex(columns=colunas_treino, fill_value=0)

df_teste_final['PROBABILIDADE_INADIMPLENCIA'] = rf_model.predict_proba(X_predict)[:, 1]

#%%
submissao = df_teste_final[['ID_CLIENTE', 'SAFRA_REF', 'PROBABILIDADE_INADIMPLENCIA']].copy()
submissao['SAFRA_REF'] = submissao['SAFRA_REF'].dt.strftime('%Y-%m')
submissao['PROBABILIDADE_INADIMPLENCIA'] = submissao['PROBABILIDADE_INADIMPLENCIA'].round(4)

submissao.to_csv('../data/submissao_case.csv', index=False, sep=';')
print("Sucesso! O arquivo final foi salvo em '../data/submissao_case.csv'.")
# %%
import matplotlib.pyplot as plt
import seaborn as sns


# Ajuste os cortes conforme o apetite de risco da empresa
bins = [0, 0.3, 0.7, 1.0]
labels = ['Baixo Risco (<30%)', 'Médio Risco (30-70%)', 'Alto Risco (>70%)']
df_teste_final['FAIXA_RISCO'] = pd.cut(df_teste_final['PROBABILIDADE_INADIMPLENCIA'], bins=bins, labels=labels)

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Distribuição Geral das Probabilidades
sns.histplot(df_teste_final['PROBABILIDADE_INADIMPLENCIA'], bins=50, kde=True, ax=axes[0], color='royalblue')
axes[0].set_title('Distribuição da Probabilidade de Inadimplência', fontsize=12)
axes[0].set_xlabel('Probabilidade')
axes[0].set_ylabel('Quantidade de Documentos')


risk_counts = df_teste_final['FAIXA_RISCO'].value_counts(sort=False)
sns.barplot(x=risk_counts.index, y=risk_counts.values, ax=axes[1], palette=['#2ecc71', '#f1c40f', '#e74c3c'])
axes[1].set_title('Volume de Documentos por Faixa de Risco', fontsize=12)
axes[1].set_ylabel('Quantidade')
axes[1].tick_params(axis='x', rotation=15)

# Usando a base antes do drop das features originais para pegar a REGIAO legível
risco_regiao = df_teste_final.groupby('REGIAO')['PROBABILIDADE_INADIMPLENCIA'].mean().sort_values()
sns.barplot(x=risco_regiao.values, y=risco_regiao.index, ax=axes[2], palette='viridis')
axes[2].set_title('Probabilidade Média de Atraso por Região', fontsize=12)
axes[2].set_xlabel('Probabilidade Média')
axes[2].set_ylabel('')

plt.tight_layout()
plt.savefig('../data/analise_predicoes.png', dpi=300) 
plt.show()

print("\nResumo da Ação Sugerida:")
for faixa, qtd in risk_counts.items():
    perc = (qtd / len(df_teste_final)) * 100
    print(f"- {faixa}: {qtd} boletos ({perc:.1f}%)")
# %%
