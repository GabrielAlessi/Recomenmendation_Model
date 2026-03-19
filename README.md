[README_RECOMMENDATION.md](https://github.com/user-attachments/files/26119598/README_RECOMMENDATION.md)
# 🎯 Sistema de Recomendação de Produtos Financeiros

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-34d399?style=for-the-badge)](LICENSE)

> Pipeline completo de recomendação personalizada para produtos financeiros usando **Neural Collaborative Filtering (NeuMF)** — combinando GMF linear com MLP não-linear para capturar padrões complexos de preferência de investidores.

---

## 🖥️ Aplicação em Produção

🔗 **[Acessar o Dashboard](https://recommendation-dashboard.streamlit.app)**

---

## 🧠 Arquitetura do Modelo

O **NeuMF (Neural Matrix Factorization)** combina duas abordagens complementares:

```
Usuário ──► Embedding GMF ──► Produto Elemento a Elemento ──┐
                                                              ├──► Linear ──► Sigmoid ──► Score
Usuário ──► Embedding MLP ──► Concatenar ──► MLP Tower ──────┘
```

- **GMF (Generalized Matrix Factorization)** — captura interações lineares entre usuários e produtos
- **MLP Tower** — captura padrões não-lineares com camadas `[64 → 32 → 16]` + BatchNorm + Dropout
- **Negative Sampling** — 4 negativos por positivo para melhorar qualidade do ranking

---

## 📊 Resultados

| Modelo | Hit Rate@10 | NDCG@10 | Coverage@10 |
|:-------|:-----------:|:-------:|:-----------:|
| **NeuMF (NCF)** | **Melhor** | **Melhor** | Boa |
| SVD | Médio | Médio | Boa |
| Popularidade | Baixo | Baixo | Baixa |

---

## 🗂️ Estrutura do Projeto

```
recommendation-system/
│
├── 📓 recommendation_system.ipynb   # Notebook principal (23 células)
├── 🖥️  recommendation_dashboard.py  # Aplicação Streamlit em produção
├── requirements.txt
└── README.md
```

---

## 📋 Conteúdo do Notebook

| # | Seção | Conteúdo |
|:--|:------|:---------|
| 1 | Configuração | Imports, DEVICE, tema dark |
| 2 | Dataset | 3.000 usuários · 15 produtos · 5 perfis de investidor |
| 3 | EDA | Distribuição de ratings, long tail, heatmap perfil × categoria |
| 4 | Pré-processamento | Label Encoding, leave-one-out split, matriz esparsa |
| 5 | Baselines | Popularidade + SVD (Matrix Factorization) |
| 6 | NeuMF | Arquitetura GMF + MLP, negative sampling, treinamento |
| 7 | Avaliação | Hit Rate@10, NDCG@10, Coverage@10 |
| 8 | Explicabilidade | Similaridade de embeddings, razão por recomendação |
| 9 | Produção | Link para o dashboard Streamlit |
| 10 | Conclusões | Comparativo final e próximos passos |

---

## 🏦 Produtos Financeiros no Catálogo

| Categoria | Produtos | Risco |
|:----------|:---------|:-----:|
| Renda Fixa | CDB 90 dias, Tesouro Selic, LCI 12 meses | 🟢 Baixo |
| Multimercado | Fundo Multimercado | 🟡 Médio |
| FII | FII Tijolo, FII Papel | 🟡 Médio |
| Renda Variável | Ações Blue Chips, ETF S&P 500, BDR Tech | 🔴 Alto |
| Cripto | Criptomoedas | 🔴 Alto |
| Previdência | PGBL | 🟡 Médio |
| Seguro | Seguro de Vida | 🟢 Baixo |
| Crédito | Cartão Premium, Empréstimo, Financiamento | 🟢/🟡 |

---

## 👤 Perfis de Investidor

| Perfil | Distribuição | Preferência Principal |
|:-------|:------------:|:----------------------|
| Conservador | 30% | Renda Fixa (70%) |
| Moderado | 30% | Renda Fixa + Multimercado |
| Arrojado | 20% | Renda Variável + Cripto |
| Jovem Digital | 15% | Cripto + Renda Variável |
| Aposentador | 5% | Renda Fixa + Previdência |

---

## 🚀 Como Executar

### 1. Clone o repositório
```bash
git clone https://github.com/GabrielAlessi/recommendation-system.git
cd recommendation-system
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Execute o notebook
```bash
jupyter notebook recommendation_system.ipynb
```

### 4. Rode o dashboard localmente
```bash
streamlit run recommendation_dashboard.py
```

---

## 🔬 Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

---

## 🚀 Próximos Passos

- [ ] **Session-based recommendations** — incorporar sequência temporal das interações
- [ ] **Features contextuais** — incluir idade, renda e perfil de risco diretamente no modelo
- [ ] **Two-Tower Model** — arquitetura escalável para retrieval em tempo real
- [ ] **A/B Test** — validar uplift em CTR e conversão em ambiente real
- [ ] **Cold Start** — estratégia para novos usuários via content-based híbrido

---

*Desenvolvido por **Gabriel Alessi Naumann** | [LinkedIn](https://www.linkedin.com/in/gabriel-alessi-naumann/) | [Kaggle](https://www.kaggle.com/gabrielalessinaumann)*
