# Credit Risk Prediction (PD Model)
O objetivo do projeto é desenvolver um modelo que ajude  a prever situações de inadimplência por atraso no pagamento, com base  no histórico de comportamento e nas características dos clientes.

## Sumário

- [Sumário Executivo](#sumário-executivo)
- [Fluxo do trabalho](#fluxo-do-trabalho)
- [Explore: Análise Exploratória dos Dados (EDA)](#explore-análise-exploratória-dos-dados-eda)
  - [Distribuição da variável-alvo](#distribuição-da-variável-alvo)
  - [Achados por dimensão](#achados-por-dimensão)
  - [Conexão com a etapa de Modelagem](#conexão-com-a-etapa-de-modelagem)
- [Matriz de Impacto: Das Variáveis Selecionadas à Ação de Negócio](#matriz-de-impacto-das-variáveis-selecionadas-à-ação-de-negócio)
- [Metodologia & Engenharia do Pipeline](#metodologia--engenharia-do-pipeline)
- [Resultados do Modelo e Métricas Executivas](#resultados-do-modelo-e-métricas-executivas)
- [Arquitetura e Organização do Repositório](#arquitetura-e-organização-do-repositório)

---

## Resumo Executivo

Este projeto implementa um **pipeline preditivo de Machine Learning para Risco de Crédito**. O modelo calcula a Probabilidade de Inadimplência (Probability of Default - PD) no grão de documento/fatura, identificando antecipadamente se uma cobrança sofrerá um atraso igual ou superior a 5 dias.

A partir de uma arquitetura modular de engenharia de features (com foco em histórico comportamental e dados cadastrais), o sistema entrega uma probabilidade de risco acionável para balizar estratégias de cobrança preventiva e adequação de limites de crédito.

---

## Fluxo do trabalho

* **EDA (Análise Exploratória):** Identificação de distribuições financeiras, tratamento de valores nulos (Pessoas Físicas vs. PJ) e mapeamento de risco por região e segmento.
* **Feature Engineering:** Construção robusta de variáveis de histórico (atrasos passados, taxa de inadimplência histórica) garantindo isolamento temporal estrito.
* **Modelagem e OOT Validation:** Treinamento de baselines (Logística, Naive Bayes) e modelos avançados (Random Forest), validados em uma janela *Out-of-Time* (safras futuras).
* **Pipeline de Produção:** Empacotamento de transformadores (`OneHotEncoder`) e algoritmos preditivos para inferência cega e geração de matrizes de submissão.

---

## Explore: Análise Exploratória dos Dados (EDA)

A etapa de análise exploratória buscou responder às seguintes perguntas de negócio:

1. Qual é a correlação entre atrasos passados e a probabilidade de um novo calote?
2. Como características cadastrais (Região, Porte, Segmento) afetam o risco financeiro?
3. O modelo de negócios atende perfis mistos (Pessoas Físicas e Jurídicas) e como isso impacta a qualidade do dado?

As bases analisadas (Pagamentos, Info e Cadastral) foram integradas e avaliadas sob o contexto temporal das safras.

### Distribuição da variável-alvo
A variável-alvo (`INADIMPLENTE`) foi definida como um atraso de 5 dias ou mais no pagamento do boleto. Tratou-se de uma base naturalmente desbalanceada, reflexo direto da proporção entre bons pagadores e inadimplentes na carteira da empresa.

### Achados por dimensão

**Histórico de Comportamento.** O preditor mais forte de risco é o comportamento passado. Clientes com atrasos anteriores possuem uma taxa de reincidência significativamente maior.

**Tempo de Relacionamento (`DIAS_DE_CADASTRO`).** Clientes mais antigos (maior antiguidade no momento da emissão da cobrança) apresentam maior estabilidade e menor risco de default comparado aos novos entrantes (Cold Start).

**Perfil Demográfico e Nulos.** Identificou-se que a ausência de informações de "Porte" e "Segmento" (`NaNs`) tem correlação direta com Pessoas Físicas (PF) e empresas com cadastros incompletos, tornando a preservação dessa categoria ("Outros") essencial para o mapa de risco.

**Risco Regional.** Regiões distintas do Brasil apresentaram níveis de risco variáveis, indicando que o fator macroeconômico e geográfico influencia a probabilidade de pagamento em dia.

### Conexão com a etapa de Modelagem
Os fatores identificados na EDA convergiram perfeitamente com a *Feature Importance* extraída do modelo Random Forest. Variáveis construídas como `TAXA_ATRASO_HISTORICA`, `QTD_ATRASOS_ANTERIORES` e `DIAS_DE_CADASTRO` dominaram o poder preditivo, confirmando que a engenharia de features traduziu com sucesso os padrões de negócio para o algoritmo.

---

## Matriz de Impacto: Das Variáveis Selecionadas à Ação de Negócio

Em modelos de crédito, a interpretação das features deve guiar as políticas de cobrança e concessão:

| Variável Preditiva | Interpretação de Negócio ("So What?") | Plano de Ação Recomendado (Risco / Cobrança) |
| --- | --- | --- |
| **`TAXA_ATRASO_HISTORICA`** e **`QTD_ATRASOS_ANTERIORES`** | O comportamento pregresso é o maior indicativo de risco futuro. Clientes habituados a atrasar têm alta probabilidade de repetição. | **Régua de Cobrança Preventiva:** Disparar SMS/WhatsApp ou ligações preventivas 3 a 5 dias *antes* do vencimento para clientes com histórico de atraso. |
| **`DIAS_DE_CADASTRO`** | Clientes novos representam risco desconhecido, enquanto clientes maduros costumam estabilizar seu comportamento de pagamento. | **Adequação de Limites:** Restringir prazos ou valores de faturamento para clientes novos (Cold Start) até que construam um histórico confiável. |
| **`VALOR_A_PAGAR`** e **`RENDA_MES_ANTERIOR`** | O grau de alavancagem do cliente (Valor da Fatura vs. Renda Reportada) afeta a capacidade de honrar o compromisso na data estipulada. | **Renegociação Antecipada:** Se o valor da fatura destoar da renda histórica, oferecer opções de parcelamento antes do atraso consolidar. |
| **`REGIAO`**, **`PORTE`**, **`SEGMENTO_INDUSTRIAL`** | Fatores geográficos e corporativos ditam fluxos de caixa distintos e exposições macroeconômicas variadas. | **Políticas Regionais:** Ajustar o apetite de risco e as taxas de juros (`TAXA`) de acordo com o segmento de atuação ou a região geográfica do cliente. |

---

## Metodologia & Engenharia do Pipeline

1. **Prevenção Máxima de Data Leakage:**
   * Utilização de ordenação cronológica estrita no `feature_engineering.py` para calcular atrasos anteriores (`shift`, `cumsum`), garantindo que o status de inadimplência da fatura *atual* nunca contamine o próprio histórico.
2. **Validação Out-of-Time (OOT):**
   * Em vez de um split aleatório (que ignora a inflação e sazonalidades), a base foi cortada no tempo:
     * **Treino:** Até Jun/2020.
     * **Teste:** Jul/2020 a Dez/2020.
     * **OOT (Validação):** Jan/2021 em diante. Isso garante que o modelo aprendeu padrões universais de crédito e não memorizou uma safra específica.
3. **Robustez a Novos Dados (`OneHotEncoder`):**
   * Implementação do `OneHotEncoder` com `handle_unknown='ignore'`. Diferente do `get_dummies` estático, o encoder foi salvo para lidar automaticamente com categorias ausentes ou inéditas no momento da inferência, prevenindo quebras em produção.
4. **Tratamento de Dados Categóricos e Nulos:**
   * Imputação de nulos orientada a negócio (Preservação de PFs como "Outros" em variáveis empresariais) e uso de medianas aprendidas na base de treino para variáveis contínuas (ex: `RENDA_MES_ANTERIOR`).
5. **Portabilidade do Modelo:**
   * Exportação centralizada via `pd.Series` e `Pickle` (`model.pkl`), agrupando no mesmo arquivo: a lista exata de features, o transformador de categorias e o algoritmo preditivo, garantindo sincronia total em produção.

---

## Resultados do Modelo e Métricas Executivas

O case comparou algoritmos lineares, probabilísticos e baseados em árvores: **Regressão Logística**, **Naive Bayes** e **Random Forest**. O balanceamento de classes (`class_weight='balanced'`) foi aplicado para maximizar a captura de calotes.

Devido à sua robustez contra outliers financeiros e capacidade de capturar relações não-lineares, o **Random Forest** foi o modelo vencedor. Avaliamos a performance primariamente através da curva **ROC-AUC**.

| Partição | Random Forest (AUC) | Regressão Logística (AUC) | Naive Bayes (AUC) |
| :--- | :---: | :---: | :---: |
| **Teste (Jul-Dez 2020)** | **~0.909** | *Baseline Test* | *Baseline Test* |
| **OOT (2021 em diante)** | **~0.918** | *Baseline OOT* | *Baseline OOT* |

*Nota Analítica:* Um modelo de crédito sustentando AUC superior a 0.90 em uma janela temporal inédita (Out-of-Time) demonstra altíssima estabilidade, provando que as features criadas são resilientes a variações econômicas ao longo dos meses.

---

## Arquitetura e Organização do Repositório

```text
├── data/
│   ├── base_info.csv                       # Informações de Renda
│   ├── base_cadastral.csv                  # Dados Demográficos e de Perfil
│   ├── base_pagamentos_desenvolvimento.csv # Histórico de Faturas para Treino
│   ├── base_pagamentos_teste.csv           # Dados cegos para submissão final
│   └── submissao_case.csv                  # Saída com as probabilidades de calote
├── models/
│   └── model.pkl                           # Pacote serializado (RF Model + Encoder + Features)
├── src/
│   ├── feature_engineering.py              # Motor de joins, limpeza e variáveis de lag histórico
│   ├── train.py                            # Rotina de Split Temporal, Pipeline de Treino e Pickle
│   └── predict.py                          # Inferência OOT, simulação de Produção e visualizações
