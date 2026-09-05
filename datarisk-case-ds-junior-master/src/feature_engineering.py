
# %%
import pandas as pd

imputar_mediana_cols = ['VALOR_A_PAGAR', 'RENDA_MES_ANTERIOR'] 

def limpar_dados_cadastrais(df_cadastral):
    df = df_cadastral.copy()
    
    # Tratamento de flags e nulos
    df['FLAG_PF'] = df['FLAG_PF'].map({'X': 1}).fillna(0).astype(int)
    df['DOMINIO_EMAIL'] = df['DOMINIO_EMAIL'].fillna('Outros ')
    df[['PORTE', 'SEGMENTO_INDUSTRIAL']] = df[['PORTE', 'SEGMENTO_INDUSTRIAL']].fillna('Outros / Nao aplica ')

    # Mapeamento de DDD para Região
    regioes_ddd = {
        "Norte": ["63", "68", "69", "91", "92", "93", "94", "95", "96", "97"],
        "Nordeste": ["71", "73", "74", "75", "77", "79", "81", "82", "83", "84", "85", "86", "87", "88", "89", "98", "99"],
        "Centro-Oeste": ["61", "62", "64", "65", "66", "67"],
        "Sudeste": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "21", "22", "24", "27", "28", "31", "32", "33", "34", "35", "37", "38"],
        "Sul": ["41", "42", "43", "44", "45", "46", "47", "48", "49", "51", "53", "54", "55"]
    }
    ddd_to_regiao = {ddd: regiao for regiao, ddds in regioes_ddd.items() for ddd in ddds}
    df['REGIAO'] = df['DDD'].map(ddd_to_regiao).fillna('Desconhecido')
    df['DATA_CADASTRO'] = pd.to_datetime(df['DATA_CADASTRO'], format='%Y-%m-%d', errors='coerce')
    
    return df


def criar_tabela_analitica(df_pagamentos, df_info, df_cadastral):

    # Padronização de datas 
    df_info['SAFRA_REF'] = pd.to_datetime(df_info['SAFRA_REF'], format='%Y-%m')
    df_pagamentos['SAFRA_REF'] = pd.to_datetime(df_pagamentos['SAFRA_REF'], format='%Y-%m')
    df_pagamentos['DATA_EMISSAO_DOCUMENTO'] = pd.to_datetime(df_pagamentos['DATA_EMISSAO_DOCUMENTO'])
        
    # Merges
    df_abt = df_pagamentos.merge(df_info, on=['ID_CLIENTE', 'SAFRA_REF'], how='left')
    df_cad_limpo = limpar_dados_cadastrais(df_cadastral)
    df_abt = df_abt.merge(df_cad_limpo, on='ID_CLIENTE', how='left')
    
    df_abt['DIAS_DE_CADASTRO'] = (df_abt['DATA_EMISSAO_DOCUMENTO'] - df_abt['DATA_CADASTRO']).dt.days
    
    # Features de Histórico (ordenar no tempo para evitar data leakage)
    df_abt = df_abt.sort_values(by=['ID_CLIENTE', 'DATA_EMISSAO_DOCUMENTO'])
    df_abt['QTD_COBRANCAS_ANTERIORES'] = df_abt.groupby('ID_CLIENTE').cumcount()
    

    if 'INADIMPLENTE' in df_abt.columns:
        df_abt['FLAG_INADIMPLENTE_ANTERIOR'] = df_abt.groupby('ID_CLIENTE')['INADIMPLENTE'].shift(1).fillna(0)
        df_abt['QTD_ATRASOS_ANTERIORES'] = df_abt.groupby('ID_CLIENTE')['FLAG_INADIMPLENTE_ANTERIOR'].cumsum()
        df_abt['TAXA_ATRASO_HISTORICA'] = (df_abt['QTD_ATRASOS_ANTERIORES'] / df_abt['QTD_COBRANCAS_ANTERIORES']).fillna(0)
        df_abt = df_abt.drop(columns=['FLAG_INADIMPLENTE_ANTERIOR'])
    
    
    for col in imputar_mediana_cols:
        if col in df_abt.columns:
            media_historica_cliente = df_abt.groupby('ID_CLIENTE')[col].transform(
                lambda s: s.shift(1).expanding().mean()
            )
            df_abt[col] = df_abt[col].fillna(media_historica_cliente)
    
    # separação de colunas finais
    features_numericas = [
        'VALOR_A_PAGAR', 'TAXA', 'RENDA_MES_ANTERIOR', 'DIAS_DE_CADASTRO', 
        'QTD_COBRANCAS_ANTERIORES', 'QTD_ATRASOS_ANTERIORES', 'TAXA_ATRASO_HISTORICA', 'FLAG_PF'
    ]
    features_categoricas = ['PORTE', 'SEGMENTO_INDUSTRIAL', 'REGIAO']
    
    cols_manter = ['ID_CLIENTE', 'SAFRA_REF', 'DATA_EMISSAO_DOCUMENTO', 'BASE_ORIGEM'] + features_numericas + features_categoricas
    
    if 'INADIMPLENTE' in df_abt.columns:
        cols_manter.append('INADIMPLENTE')
        
    df_final = df_abt[[col for col in cols_manter if col in df_abt.columns]].copy()
            
    return df_final
# %%
def calcular_medianas_imputacao(df_train):
    return {col: df_train[col].median() for col in imputar_mediana_cols if col in df_train.columns}


def aplicar_imputacao(df, medianas):
    df = df.copy()
    for col, mediana in medianas.items():
        if col in df.columns:
            df[col] = df[col].fillna(mediana)
    return df
