# News Headline Classifier

> Classificação automática de manchetes de notícias com Machine Learning

---

## Integrantes e RAs

| Nome | RA |
|------|----|
| *Miguel Guarnetti de Moraes* | *1999154* |
| *Gabriel Fante Javarotti* | *1990554* |
| *Leonardo Lopes Martins Silva* | *2010503* |

---

## Descrição do Problema

O volume de notícias publicadas diariamente na internet torna inviável a categorização manual de conteúdo em larga escala. Portanto, automatizar a classificação de manchetes é essencial para sistemas de recomendação, agregadores de notícias e organização editorial. Este projeto aborda esse problema desenvolvendo um classificador capaz de identificar a categoria de uma notícia a partir apenas do texto de sua manchete.

---

## Objetivo do Projeto

Desenvolver um modelo de Machine Learning capaz de classificar manchetes de notícias em inglês em 5 categorias temáticas, e disponibilizá-lo como uma aplicação web interativa via Streamlit.

---

## Dataset Utilizado

- **Nome:** News Category Dataset
- **Fonte:** [Kaggle — rmisra/news-category-dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
- **Origem dos dados:** HuffPost
- **Total original:** 209.527 manchetes com 42 categorias
- **Após filtragem (Top 5 categorias):** ~42.000 registros
- **Variável de entrada:** `headline` (texto da manchete)
- **Variável-alvo:** `category` (categoria da notícia)

As 5 categorias selecionadas foram as mais frequentes do dataset:

| # | Categoria |
|---|-----------|
| 1 | POLITICS |
| 2 | WELLNESS |
| 3 | ENTERTAINMENT |
| 4 | TRAVEL |
| 5 | STYLE & BEAUTY |

---

## Tipo de Problema de Machine Learning

**Classificação multiclasse supervisionada** — o modelo aprende a partir de exemplos rotulados e prevê a qual das 5 categorias uma nova manchete pertence.

---

## Metodologia

1. **Carregamento dos dados** — leitura do dataset em formato JSON Lines
2. **Análise exploratória (EDA)** — distribuição de categorias, tamanho das manchetes, palavras mais frequentes por categoria e nuvens de palavras
3. **Pré-processamento** — conversão para minúsculas, remoção de pontuação e caracteres especiais via `limpar_texto()`; remoção de duplicatas e manchetes vazias
4. **Vetorização** — `TfidfVectorizer` com unigramas e bigramas, `max_features=50.000`, `stop_words='english'`, `min_df=2`, `max_df=0.95`
5. **Divisão dos dados** — estratificada 60% treino / 20% validação / 20% teste
6. **Treinamento e validação cruzada** — `StratifiedKFold (k=3)` com métricas macro sobre o conjunto de treino
7. **Avaliação final** — métricas no conjunto de validação (para seleção) e no conjunto de teste (para reporte)
8. **Análise de erros** — identificação de confusões entre categorias, confiança do modelo e exemplos mal classificados
9. **Salvamento do modelo** — pipeline final exportado com `joblib`

---

## Modelos Treinados

| Modelo | Acurácia (Teste) | F1-Score Macro (Teste) |
|--------|-----------------|------------------------|
| MultinomialNB | ~85% | ~0.848 |
| LogisticRegression | ~87% | ~0.865 |
| **LinearSVC** | **~88%** | **~0.869** |

Todos os modelos foram avaliados com **Stratified K-Fold (k=3)** no treino, e depois avaliados separadamente em validação e teste.

---

## Modelo Final Escolhido

**LinearSVC** com `C=1.0` e `max_iter=2000`.

**Justificativa:** O LinearSVC apresentou o melhor F1-Score Macro no conjunto de validação (~0.869), resultado que se confirmou no teste. SVMs lineares são especialmente eficientes para dados de texto vetorizados com TF-IDF, pois a representação é esparsa e de alta dimensão (~42.000 features) — cenário em que uma fronteira de decisão de margem máxima separa bem as classes. A diferença em relação à LogisticRegression é marginal (~0.004 em F1), mas o LinearSVC foi consistentemente superior em todas as etapas de avaliação.

---

## Métricas de Avaliação

| Métrica | Descrição |
|---------|-----------|
| **Acurácia** | Proporção de predições corretas sobre o total |
| **Precisão Macro** | Média não ponderada da precisão por classe |
| **Recall Macro** | Média não ponderada do recall por classe |
| **F1-Score Macro** | Média harmônica entre precisão e recall (macro) — principal métrica, pois penaliza igualmente erros em classes menores |

A métrica principal utilizada para seleção do modelo foi o **F1-Score Macro**, por ser mais adequada a dados com leve desbalanceamento entre categorias.

---

## Principais Resultados

- **Acurácia no teste:** ~88%
- **F1-Score Macro no teste:** ~0.869
- **Categoria com melhor desempenho:** POLITICS (F1 ≈ 0.93) — maior volume de amostras e vocabulário muito característico
- **Categoria mais desafiadora:** TRAVEL — confundida com WELLNESS por sobreposição temática (estilo de vida)
- **Generalização:** métricas consistentes entre CV, validação e teste (variação < 0.01), indicando ausência de overfitting
- **Erros analisados:** concentrados em manchetes curtas e em categorias com vocabulário similar

---

## Estrutura dos Arquivos

```
news-headline-classifier/
│
├── app.py                          # Aplicação Streamlit
├── requirements.txt                # Dependências do projeto
├── README.md                       # Documentação do projeto
│
├── notebooks/
│   └── News_Grupo_L_Revisado_1.ipynb   # Notebook revisado (P2)
│
├── model/
│   ├── modelo_final.joblib         # Modelo LinearSVC treinado
|   └── tfidf_vectorizer.joblib     #  # Vetorizador TF-IDF ajustado no treino
│
├── reports/
│   └── Relatorio_Final_Desafio13_Revisado.pdf    # Relatório final em PDF
│
└── data/
    └── # Instruções para obter o dataset na seção abaixo
```

> **Nota:** o dataset não está versionado por ser grande. Veja a seção de instruções abaixo para obtê-lo.

---

## Tecnologias Utilizadas

| Tecnologia | Uso |
|------------|-----|
| Python 3.10+ | Linguagem principal |
| pandas | Manipulação de dados |
| numpy | Operações numéricas |
| scikit-learn | Modelos, vetorização e métricas |
| matplotlib / seaborn | Visualizações e gráficos |
| wordcloud | Nuvens de palavras |
| joblib | Serialização do modelo |
| Streamlit | Interface web da aplicação |

---

## Instruções para Executar o Notebook

**1. Obtenha o dataset:**
- Acesse [kaggle.com/datasets/rmisra/news-category-dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
- Faça download do arquivo `News_Category_Dataset_v3.json`
- No Google Colab: faça upload do arquivo clicando no ícone de pasta → primeiro ícone da seta para cima → escolha o arquivo `News_Category_Dataset_v3.json`

**2. Instale as dependências:**
```bash
pip install -r requirements.txt
```

**3. Abra e execute o notebook:**
```bash
jupyter notebook notebooks/notebook_atualizado.ipynb
```
> Ou abra diretamente no Google Colab via *File → Upload notebook*

Execute as células em ordem. O notebook irá carregar os dados, treinar os modelos, exibir os gráficos e salvar o modelo final na pasta `model/`.

---

## Instruções para Executar o App Streamlit

**Localmente:**

```bash
# 1. Clone o repositório
git clone https://github.com/SEU_USUARIO/news-headline-classifier.git
cd news-headline-classifier

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Certifique-se de que o modelo está salvo em model/modelo_final.joblib
#    (execute o notebook primeiro, se ainda não tiver feito)

# 4. Execute o app
streamlit run app.py
```

O app abrirá automaticamente no navegador em `http://localhost:8501`.

---

## Link do App Publicado

 **[Acesse a aplicação aqui]([https://INSIRA-O-LINK-AQUI.streamlit.app](https://news-headline-classifier-adapsbgurzudtwor6he897.streamlit.app/))**

---

## Limitações

- O modelo foi treinado apenas com manchetes em **inglês** — manchetes em outros idiomas não serão classificadas corretamente
- O conjunto de categorias é fixo em 5 — manchetes de outras áreas (ex.: ciência, tecnologia, esporte) serão forçadas a uma das 5 categorias disponíveis
- Manchetes muito curtas (1–3 palavras) tendem a ter menor precisão, pois fornecem menos contexto para o TF-IDF
- O modelo não captura contexto semântico profundo — abordagens como BERT ou outros modelos de linguagem poderiam melhorar a performance em categorias com vocabulário similar (ex.: TRAVEL × WELLNESS)
- O dataset é de 2012–2022 (HuffPost), o que pode gerar viés temporal em manchetes sobre eventos muito recentes

---

## Conclusão

O projeto demonstrou que é possível classificar manchetes de notícias com alta acurácia utilizando técnicas clássicas de NLP. O pipeline TF-IDF + LinearSVC atingiu ~88% de acurácia e ~0.87 de F1-Score Macro no conjunto de teste, com boa generalização entre as etapas de avaliação.

A análise de erros revelou que as confusões entre categorias são coerentes com a sobreposição temática real (ex.: TRAVEL × WELLNESS), e não representam falhas do classificador. O modelo foi disponibilizado como uma aplicação web interativa via Streamlit, permitindo que qualquer usuário classifique manchetes em tempo real.

Como trabalhos futuros, a substituição do TF-IDF por embeddings contextuais (BERT, DistilBERT) e a expansão para mais categorias seriam os próximos passos naturais para melhorar a cobertura e a precisão do sistema.
