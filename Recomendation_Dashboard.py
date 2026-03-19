"""
Sistema de Recomendação de Produtos Financeiros
Neural Collaborative Filtering — Dashboard Streamlit
Gabriel Alessi Naumann
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FinRec · Recomendação Financeira",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Background ── */
.stApp { background-color: #06080f; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0d14 0%, #0d1120 100%);
    border-right: 1px solid #1e2d3d;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1829 0%, #0a1020 100%);
    border: 1px solid #1e2d3d;
    border-radius: 12px;
    padding: 1rem 1.25rem;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 1.6rem !important;
    color: #14b8a6 !important;
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.8rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ── Section headers ── */
h1 { color: #f1f5f9 !important; font-weight: 700 !important; letter-spacing: -0.8px; }
h2 { color: #e2e8f0 !important; font-size: 1.3rem !important; font-weight: 600 !important; border-left: 3px solid #14b8a6; padding-left: 0.75rem; margin-bottom: 1rem !important; }
h3 { color: #cbd5e1 !important; font-size: 1rem !important; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif !important;
    color: #64748b !important;
    font-size: 0.9rem !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #14b8a6 !important;
    border-bottom-color: #14b8a6 !important;
}

/* ── Selectbox / inputs ── */
[data-baseweb="select"] { background: #0d1829 !important; border-color: #1e2d3d !important; }
[data-baseweb="input"]  { background: #0d1829 !important; border-color: #1e2d3d !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%);
    color: #06080f !important;
    border: none; border-radius: 8px;
    font-weight: 600; font-size: 0.9rem;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s ease;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #1e2d3d; border-radius: 10px; }

/* ── Divider ── */
hr { border-color: #1e2d3d !important; }

/* ── Slider ── */
[data-testid="stSlider"] > div > div { background: #1e2d3d !important; }

/* ── Info / warning boxes ── */
.stInfo    { background: rgba(20,184,166,0.08)  !important; border-left: 3px solid #14b8a6 !important; }
.stWarning { background: rgba(245,158,11,0.08)  !important; border-left: 3px solid #f59e0b !important; }
.stSuccess { background: rgba(52,211,153,0.08)  !important; border-left: 3px solid #34d399 !important; }

/* ── Product cards ── */
.rec-card {
    background: linear-gradient(135deg, #0d1829 0%, #0a1020 100%);
    border: 1px solid #1e2d3d;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s ease;
}
.rec-card:hover { border-color: #14b8a6; }
.rec-rank {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem; color: #14b8a6; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
}
.rec-name {
    font-size: 1.05rem; font-weight: 600; color: #f1f5f9;
    margin: 0.25rem 0 0.5rem;
}
.rec-meta { font-size: 0.8rem; color: #64748b; }
.rec-score {
    font-family: 'DM Mono', monospace;
    font-size: 0.9rem; color: #14b8a6; font-weight: 500;
}
.rec-reason {
    font-size: 0.82rem; color: #94a3b8;
    background: rgba(14,165,233,0.07);
    border-radius: 6px; padding: 0.35rem 0.6rem;
    margin-top: 0.4rem; display: inline-block;
}
.risk-baixo  { color: #34d399; font-weight: 600; }
.risk-médio  { color: #f59e0b; font-weight: 600; }
.risk-alto   { color: #f87171; font-weight: 600; }

/* ── Stat pill ── */
.stat-pill {
    display: inline-block;
    background: rgba(20,184,166,0.1);
    border: 1px solid rgba(20,184,166,0.25);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.8rem; color: #14b8a6; font-weight: 600;
    margin-right: 0.4rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA & MODEL (cached)
# ─────────────────────────────────────────────
PRODUCTS = {
    'P001': {'name': 'CDB 90 dias',         'category': 'Renda Fixa',     'risk': 'baixo',  'min_value': 500,  'yield': '14.2% a.a.'},
    'P002': {'name': 'Tesouro Selic',        'category': 'Renda Fixa',     'risk': 'baixo',  'min_value': 100,  'yield': '13.65% a.a.'},
    'P003': {'name': 'LCI 12 meses',         'category': 'Renda Fixa',     'risk': 'baixo',  'min_value': 1000, 'yield': '13.0% a.a.'},
    'P004': {'name': 'Fundo Multimercado',   'category': 'Multimercado',   'risk': 'médio',  'min_value': 1000, 'yield': 'CDI +2%'},
    'P005': {'name': 'FII — Tijolo',         'category': 'FII',            'risk': 'médio',  'min_value': 100,  'yield': 'DY 9.2%'},
    'P006': {'name': 'FII — Papel',          'category': 'FII',            'risk': 'médio',  'min_value': 100,  'yield': 'DY 11.5%'},
    'P007': {'name': 'Ações BR Blue Chips',  'category': 'Renda Variável', 'risk': 'alto',   'min_value': 100,  'yield': 'Variável'},
    'P008': {'name': 'ETF S&P 500',          'category': 'Renda Variável', 'risk': 'alto',   'min_value': 100,  'yield': 'Variável'},
    'P009': {'name': 'BDR Tech',             'category': 'Renda Variável', 'risk': 'alto',   'min_value': 100,  'yield': 'Variável'},
    'P010': {'name': 'Criptomoedas',         'category': 'Cripto',         'risk': 'alto',   'min_value': 50,   'yield': 'Variável'},
    'P011': {'name': 'Previdência PGBL',     'category': 'Previdência',    'risk': 'médio',  'min_value': 200,  'yield': 'IPCA +4%'},
    'P012': {'name': 'Seguro de Vida',       'category': 'Seguro',         'risk': 'baixo',  'min_value': 50,   'yield': 'Proteção'},
    'P013': {'name': 'Cartão Premium',       'category': 'Crédito',        'risk': 'baixo',  'min_value': 0,    'yield': 'Benefícios'},
    'P014': {'name': 'Empréstimo Pessoal',   'category': 'Crédito',        'risk': 'médio',  'min_value': 0,    'yield': 'a partir 1.5% a.m.'},
    'P015': {'name': 'Financiamento Imóvel', 'category': 'Crédito',        'risk': 'médio',  'min_value': 0,    'yield': 'IPCA +7%'},
}
PRODUCT_IDS  = list(PRODUCTS.keys())
PALETTE      = ['#14b8a6','#818cf8','#f59e0b','#f87171','#34d399','#60a5fa','#a78bfa','#fb7185']
RISK_COLOR   = {'baixo': '#34d399', 'médio': '#f59e0b', 'alto': '#f87171'}
CAT_ICON     = {
    'Renda Fixa': '🏦', 'Multimercado': '⚖️', 'FII': '🏢',
    'Renda Variável': '📈', 'Cripto': '₿', 'Previdência': '🛡️',
    'Seguro': '🔒', 'Crédito': '💳',
}
PROPENSITY = {
    'Conservador':   {'Renda Fixa': 0.70, 'Previdência': 0.15, 'FII': 0.05, 'Renda Variável': 0.04, 'Cripto': 0.01, 'Seguro': 0.03, 'Crédito': 0.02},
    'Moderado':      {'Renda Fixa': 0.35, 'Multimercado': 0.25, 'FII': 0.15, 'Renda Variável': 0.10, 'Previdência': 0.08, 'Seguro': 0.04, 'Cripto': 0.02, 'Crédito': 0.01},
    'Arrojado':      {'Renda Variável': 0.40, 'Cripto': 0.20, 'FII': 0.15, 'Multimercado': 0.10, 'Renda Fixa': 0.08, 'Crédito': 0.05, 'Previdência': 0.02},
    'Jovem Digital': {'Cripto': 0.30, 'Renda Variável': 0.25, 'FII': 0.15, 'Crédito': 0.15, 'Multimercado': 0.10, 'Renda Fixa': 0.05},
    'Aposentador':   {'Renda Fixa': 0.50, 'Previdência': 0.25, 'Seguro': 0.10, 'FII': 0.08, 'Multimercado': 0.05, 'Renda Variável': 0.02},
}

@st.cache_resource
def build_system(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    N_USERS = 3_000
    profiles_list = np.random.choice(
        ['Conservador','Moderado','Arrojado','Jovem Digital','Aposentador'],
        size=N_USERS, p=[0.30, 0.30, 0.20, 0.15, 0.05]
    )
    records = []
    for uid in range(N_USERS):
        profile = profiles_list[uid]
        age     = np.random.randint(22, 70)
        income  = np.random.lognormal(9.0, 0.7)
        prop    = PROPENSITY[profile]
        n_int   = np.random.randint(2, 11)
        weights = []
        for pid, pinfo in PRODUCTS.items():
            w = prop.get(pinfo['category'], 0.01)
            if pinfo['min_value'] > income * 0.1: w *= 0.3
            weights.append(w)
        weights = np.array(weights); weights /= weights.sum()
        chosen = np.random.choice(PRODUCT_IDS, size=min(n_int, len(PRODUCT_IDS)), replace=False, p=weights)
        for pid in chosen:
            base = np.random.choice([3,4,5], p=[0.2,0.4,0.4])
            if prop.get(PRODUCTS[pid]['category'], 0) > 0.2: base = min(5, base+1)
            records.append({'user_idx': uid, 'product_id': pid,
                            'rating': base, 'profile': profile,
                            'age': age, 'income': round(income, 2)})
    df = pd.DataFrame(records)
    prod_enc = LabelEncoder().fit(PRODUCT_IDS)
    df['product_idx'] = prod_enc.transform(df['product_id'])
    N_P = len(PRODUCT_IDS)

    # SVD baseline
    mat = csr_matrix((df['rating'], (df['user_idx'], df['product_idx'])),
                     shape=(N_USERS, N_P))
    k   = min(20, min(mat.shape)-1)
    U, sigma, Vt = svds(mat.astype(float), k=k)
    predicted_svd = np.dot(np.dot(U, np.diag(sigma)), Vt)

    # NeuMF
    class NeuMF(nn.Module):
        def __init__(self, n_u, n_p, eg=32, em=32, layers=[64,32,16], dropout=0.2):
            super().__init__()
            self.ug = nn.Embedding(n_u, eg); self.pg = nn.Embedding(n_p, eg)
            self.um = nn.Embedding(n_u, em); self.pm = nn.Embedding(n_p, em)
            mlp_in, seq = em*2, []
            for out in layers:
                seq += [nn.Linear(mlp_in,out), nn.ReLU(), nn.Dropout(dropout), nn.BatchNorm1d(out)]
                mlp_in = out
            self.mlp = nn.Sequential(*seq)
            self.out = nn.Linear(eg + layers[-1], 1)
            self.sig = nn.Sigmoid()
            for m in self.modules():
                if isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.01)
                elif isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        def forward(self, u, p):
            g = self.ug(u) * self.pg(p)
            m = self.mlp(torch.cat([self.um(u), self.pm(p)], dim=-1))
            return self.sig(self.out(torch.cat([g,m], dim=-1))).squeeze()

    model = NeuMF(N_USERS, N_P)
    # Mini-training (8 epochs for speed)
    pos_users = torch.LongTensor(df['user_idx'].values)
    pos_prods = torch.LongTensor(df['product_idx'].values)
    labels    = torch.FloatTensor(df['rating'].values / 5.0)
    user_pos  = defaultdict(set)
    for u, p in zip(df['user_idx'].values, df['product_idx'].values): user_pos[u].add(p)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = nn.BCELoss()
    model.train()
    for epoch in range(8):
        perm = torch.randperm(len(pos_users))
        for i in range(0, len(pos_users), 512):
            idx = perm[i:i+512]
            u, p, l = pos_users[idx], pos_prods[idx], labels[idx]
            # add negatives
            neg_p = []
            for uu in u.numpy():
                while True:
                    np_ = np.random.randint(N_P)
                    if np_ not in user_pos[uu]: break
                neg_p.append(np_)
            neg_p = torch.LongTensor(neg_p)
            u_all = torch.cat([u, u])
            p_all = torch.cat([p, neg_p])
            l_all = torch.cat([l, torch.zeros(len(neg_p))])
            opt.zero_grad()
            pred = model(u_all, p_all)
            loss = crit(pred, (l_all > 0).float())
            loss.backward(); opt.step()

    model.eval()
    return df, prod_enc, model, predicted_svd, user_pos, profiles_list

df, prod_enc, model, predicted_svd, user_pos, profiles_list = build_system()
N_USERS = df['user_idx'].nunique()
N_P     = len(PRODUCT_IDS)

def get_scores_ncf(user_idx):
    with torch.no_grad():
        u = torch.full((N_P,), user_idx, dtype=torch.long)
        p = torch.arange(N_P, dtype=torch.long)
        scores = model(u, p).numpy()
    return scores

def recommend(user_idx, n=5, method='ncf', exclude_seen=True):
    if method == 'ncf':
        scores = get_scores_ncf(user_idx)
    elif method == 'svd':
        scores = predicted_svd[user_idx].copy()
    else:  # popularity
        pop = df.groupby('product_idx').size()
        scores = np.array([pop.get(i, 0) for i in range(N_P)], dtype=float)
        scores = scores / scores.max()

    if exclude_seen:
        for s in user_pos.get(user_idx, set()): scores[s] = -np.inf
    top = np.argsort(scores)[::-1][:n]
    return [(prod_enc.classes_[i], scores[i]) for i in top]

def explain_reason(pid, profile, fav_cats):
    cat  = PRODUCTS[pid]['category']
    risk = PRODUCTS[pid]['risk']
    if cat in fav_cats[:2]:    return f"alinhado com suas categorias favoritas"
    if risk == 'baixo' and profile in ['Conservador','Aposentador']: return f"baixo risco — compatível com perfil {profile}"
    if risk == 'alto'  and profile in ['Arrojado','Jovem Digital']:  return f"alto potencial — compatível com perfil {profile}"
    return "alta adesão entre clientes similares"

def dark_fig(figsize=(10,4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor='#06080f')
    ax.set_facecolor('#0a0d14')
    ax.tick_params(colors='#475569', labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor('#1e2d3d')
    return fig, ax

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 FinRec")
    st.markdown("<p style='font-size:0.8rem;color:#475569;margin-top:-0.5rem;'>Neural Collaborative Filtering</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Configurar Usuário**")
    profile_sel = st.selectbox("Perfil", list(PROPENSITY.keys()), index=1)
    age_sel     = st.slider("Idade", 22, 70, 35)
    income_sel  = st.slider("Renda Mensal (R$)", 2_000, 50_000, 8_000, step=500,
                            format="R$ %d")
    patrimonio  = st.slider("Patrimônio Investido (R$)", 0, 500_000, 30_000, step=5_000,
                            format="R$ %d")
    st.markdown("---")
    st.markdown("**Recomendação**")
    method_sel  = st.selectbox("Modelo", ["NeuMF (NCF)", "SVD", "Popularidade"])
    n_recs      = st.slider("Nº de Recomendações", 3, 10, 5)
    exclude_seen= st.checkbox("Excluir produtos já contratados", value=True)
    st.markdown("---")

    # Encontrar usuário real com perfil similar
    profile_key  = profile_sel.lower().replace(' ', '_').replace('jovem_', 'Jovem ')
    profile_map  = {'conservador': 'Conservador', 'moderado': 'Moderado',
                    'arrojado': 'Arrojado', 'jovem_digital': 'Jovem Digital',
                    'aposentador': 'Aposentador'}
    users_profile = df[df['profile'] == profile_sel]['user_idx'].unique()
    sim_user_idx  = int(users_profile[0]) if len(users_profile) > 0 else 0

    st.markdown(f"<p style='font-size:0.78rem;color:#475569;'>Simulando usuário #{sim_user_idx:05d}</p>", unsafe_allow_html=True)
    btn = st.button("⚡ Gerar Recomendações", use_container_width=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 🎯 Sistema de Recomendação Financeira")
st.markdown("<p style='color:#64748b;margin-top:-0.5rem;font-size:1rem;'>Neural Collaborative Filtering · Produtos Financeiros Personalizados</p>", unsafe_allow_html=True)
st.markdown("---")

# ── KPIs ─────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
total_int = len(df)
total_u   = df['user_idx'].nunique()
avg_r     = df['rating'].mean()
user_hist = df[df['user_idx'] == sim_user_idx]
n_hist    = len(user_hist)
with c1: st.metric("Usuários Ativos",    f"{total_u:,}")
with c2: st.metric("Interações Totais",  f"{total_int:,}")
with c3: st.metric("Produtos no Catálogo", f"{N_P}")
with c4: st.metric("Rating Médio",       f"{avg_r:.2f} ⭐")
with c5: st.metric("Histórico do Usuário", f"{n_hist} produtos")

st.markdown("---")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Recomendações",
    "📊 Análise do Perfil",
    "🔬 Comparação de Modelos",
    "📚 Catálogo de Produtos",
])

# ─────────────────────────────────────────────
# TAB 1 — RECOMENDAÇÕES
# ─────────────────────────────────────────────
with tab1:
    method_key = {'NeuMF (NCF)': 'ncf', 'SVD': 'svd', 'Popularidade': 'popularity'}[method_sel]
    recs = recommend(sim_user_idx, n=n_recs, method=method_key, exclude_seen=exclude_seen)

    # Categorias favoritas do usuário
    if len(user_hist) > 0:
        fav_df  = user_hist.groupby(user_hist['product_id'].map(lambda x: PRODUCTS[x]['category']))['rating'].mean()
        fav_cats = fav_df.sort_values(ascending=False).index.tolist()
    else:
        fav_cats = []

    col_recs, col_hist = st.columns([3, 2])

    with col_recs:
        st.markdown("## Recomendações Personalizadas")
        st.markdown(f"<p style='color:#64748b;font-size:0.9rem;'>Modelo: <strong style='color:#14b8a6'>{method_sel}</strong> · Perfil: <strong style='color:#14b8a6'>{profile_sel}</strong></p>", unsafe_allow_html=True)

        for rank, (pid, score) in enumerate(recs, 1):
            pinfo  = PRODUCTS[pid]
            reason = explain_reason(pid, profile_sel, fav_cats)
            risk_cls = f"risk-{pinfo['risk']}"
            cat_icon = CAT_ICON.get(pinfo['category'], '📦')
            score_pct = max(0, min(1, float(score))) * 100

            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-rank">#{rank} &nbsp;·&nbsp; {cat_icon} {pinfo['category']}</div>
                <div class="rec-name">{pinfo['name']}</div>
                <div class="rec-meta">
                    <span class="{risk_cls}">● {pinfo['risk'].upper()}</span>
                    &nbsp;·&nbsp; Mín: R$ {pinfo['min_value']:,}
                    &nbsp;·&nbsp; Retorno: {pinfo['yield']}
                </div>
                <div style="margin-top:0.5rem;background:#0d1829;border-radius:6px;height:5px;overflow:hidden;">
                    <div style="width:{score_pct:.0f}%;height:100%;background:linear-gradient(90deg,#14b8a6,#818cf8);border-radius:6px;"></div>
                </div>
                <div style="display:flex;justify-content:space-between;margin-top:0.3rem;">
                    <span class="rec-reason">💡 {reason}</span>
                    <span class="rec-score">{score:.4f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_hist:
        st.markdown("## Histórico do Usuário")
        if len(user_hist) > 0:
            for _, row in user_hist.iterrows():
                pid    = row['product_id']
                pinfo  = PRODUCTS[pid]
                stars  = "⭐" * int(row['rating'])
                cat_icon = CAT_ICON.get(pinfo['category'], '📦')
                st.markdown(f"""
                <div style='padding:0.6rem 0.8rem;background:#0a0d14;border:1px solid #1e2d3d;
                             border-radius:8px;margin-bottom:0.4rem;'>
                    <span style='font-size:0.85rem;color:#e2e8f0;font-weight:500;'>{cat_icon} {pinfo['name']}</span><br>
                    <span style='font-size:0.75rem;color:#475569;'>{pinfo['category']}</span>
                    <span style='float:right;font-size:0.78rem;'>{stars}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Nenhum histórico encontrado para este perfil.")

        # Mini donut — categorias recomendadas
        st.markdown("## Mix de Categorias")
        rec_cats = [PRODUCTS[pid]['category'] for pid, _ in recs]
        cat_ser  = pd.Series(rec_cats).value_counts()
        fig, ax  = plt.subplots(figsize=(4, 3.5), facecolor='#06080f')
        ax.set_facecolor('#06080f')
        colors_cat = [PALETTE[i % len(PALETTE)] for i in range(len(cat_ser))]
        wedges, texts, autotexts = ax.pie(
            cat_ser.values, labels=None,
            colors=colors_cat, autopct='%1.0f%%',
            startangle=90, pctdistance=0.75,
            wedgeprops={'edgecolor':'#06080f','linewidth':2}
        )
        for t in autotexts: t.set_color('#f1f5f9'); t.set_fontsize(9); t.set_fontweight('bold')
        ax.legend(cat_ser.index, loc='lower center', bbox_to_anchor=(0.5, -0.22),
                  ncol=2, facecolor='#0a0d14', labelcolor='#94a3b8', fontsize=7.5, framealpha=0)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ─────────────────────────────────────────────
# TAB 2 — ANÁLISE DO PERFIL
# ─────────────────────────────────────────────
with tab2:
    st.markdown("## Análise do Perfil de Investidor")

    col_a, col_b = st.columns(2)

    with col_a:
        # Propensão do perfil
        prop_data = PROPENSITY[profile_sel]
        prop_df   = pd.DataFrame(list(prop_data.items()), columns=['Categoria','Propensão']).sort_values('Propensão')
        fig, ax   = dark_fig((6, 4))
        colors_p  = [PALETTE[i % len(PALETTE)] for i in range(len(prop_df))]
        ax.barh(prop_df['Categoria'], prop_df['Propensão'], color=colors_p, alpha=0.9, height=0.6)
        ax.set_title(f'Propensão por Categoria — {profile_sel}', color='#cbd5e1', pad=10)
        ax.set_xlabel('Propensão', color='#475569')
        for i, (_, row) in enumerate(prop_df.iterrows()):
            ax.text(row['Propensão']+0.005, i, f"{row['Propensão']:.0%}", va='center', color='#cbd5e1', fontsize=8)
        ax.set_xlim(0, prop_df['Propensão'].max() * 1.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_b:
        # Distribuição de perfis no dataset
        profile_dist = df.drop_duplicates('user_idx')['profile'].value_counts()
        fig, ax = dark_fig((6, 4))
        colors_pr = [PALETTE[i % len(PALETTE)] for i in range(len(profile_dist))]
        bars = ax.bar(profile_dist.index, profile_dist.values, color=colors_pr, alpha=0.9)
        ax.set_title('Distribuição de Perfis na Base', color='#cbd5e1', pad=10)
        ax.set_ylabel('Usuários', color='#475569')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+10,
                    f'{int(bar.get_height()):,}', ha='center', color='#cbd5e1', fontsize=8)
        ax.set_xticklabels(profile_dist.index, rotation=15, ha='right')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Rating médio por perfil × categoria (heatmap)
    st.markdown("## Mapa de Calor: Perfil × Categoria")
    hm_data = (df.groupby(['profile','product_id'])['rating'].mean()
               .reset_index()
               .assign(category=lambda x: x['product_id'].map(lambda p: PRODUCTS[p]['category'])))
    hm_pivot = hm_data.groupby(['profile','category'])['rating'].mean().unstack(fill_value=0)
    fig, ax  = plt.subplots(figsize=(12, 4), facecolor='#06080f')
    ax.set_facecolor('#0a0d14')
    import matplotlib.colors as mc
    cmap = mc.LinearSegmentedColormap.from_list('teal', ['#0a0d14','#0d9488','#14b8a6','#5eead4'])
    im = ax.imshow(hm_pivot.values, cmap=cmap, aspect='auto')
    ax.set_xticks(range(len(hm_pivot.columns))); ax.set_xticklabels(hm_pivot.columns, rotation=30, ha='right', color='#94a3b8', fontsize=9)
    ax.set_yticks(range(len(hm_pivot.index)));   ax.set_yticklabels(hm_pivot.index, color='#94a3b8', fontsize=9)
    for i in range(len(hm_pivot.index)):
        for j in range(len(hm_pivot.columns)):
            v = hm_pivot.values[i,j]
            if v > 0: ax.text(j, i, f'{v:.1f}', ha='center', va='center', color='white', fontsize=8)
    ax.set_title('Rating Médio por Perfil × Categoria', color='#cbd5e1', pad=10, fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8).ax.tick_params(colors='#475569')
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ─────────────────────────────────────────────
# TAB 3 — COMPARAÇÃO DE MODELOS
# ─────────────────────────────────────────────
with tab3:
    st.markdown("## Comparação de Modelos de Recomendação")

    @st.cache_data
    def eval_models(sample=400):
        test_users = df['user_idx'].unique()[:sample]
        results = {}
        for mname, mkey in [('Popularidade','popularity'),('SVD','svd'),('NeuMF (NCF)','ncf')]:
            hits, ndcg_s = 0, []
            all_recs_set = set()
            for uid in test_users:
                recs_eval = recommend(uid, n=10, method=mkey, exclude_seen=False)
                rec_pids  = [prod_enc.transform([pid])[0] for pid, _ in recs_eval]
                # simular item de teste = último produto interagido
                user_prod = df[df['user_idx']==uid]['product_idx'].values
                if len(user_prod) == 0: continue
                true_item = user_prod[-1]
                all_recs_set.update(rec_pids)
                if true_item in rec_pids:
                    hits += 1
                    pos   = rec_pids.index(true_item)
                    ndcg_s.append(1.0 / np.log2(pos+2))
                else:
                    ndcg_s.append(0.0)
            results[mname] = {
                'Hit Rate@10': hits / len(test_users),
                'NDCG@10':     np.mean(ndcg_s),
                'Coverage@10': len(all_recs_set) / N_P,
            }
        return pd.DataFrame(results).T

    with st.spinner("Avaliando modelos..."):
        df_eval = eval_models()

    # Métricas em colunas
    m1, m2, m3 = st.columns(3)
    best_hr   = df_eval['Hit Rate@10'].idxmax()
    best_ndcg = df_eval['NDCG@10'].idxmax()
    best_cov  = df_eval['Coverage@10'].idxmax()
    with m1: st.metric("🏆 Melhor Hit Rate@10",   best_hr,   f"{df_eval.loc[best_hr,'Hit Rate@10']:.4f}")
    with m2: st.metric("🏆 Melhor NDCG@10",       best_ndcg, f"{df_eval.loc[best_ndcg,'NDCG@10']:.4f}")
    with m3: st.metric("🏆 Maior Coverage@10",    best_cov,  f"{df_eval.loc[best_cov,'Coverage@10']:.4f}")

    st.markdown("---")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#06080f')
    for i, (ax, metric) in enumerate(zip(axes, ['Hit Rate@10','NDCG@10','Coverage@10'])):
        ax.set_facecolor('#0a0d14')
        vals   = df_eval[metric].values
        colors = [PALETTE[4] if v == vals.max() else PALETTE[0] for v in vals]
        bars   = ax.bar(df_eval.index, vals, color=colors, alpha=0.9, width=0.55)
        ax.set_title(metric, color='#cbd5e1', pad=8, fontsize=11)
        ax.tick_params(colors='#475569', labelsize=8)
        ax.set_xticklabels(df_eval.index, rotation=10, ha='right', fontsize=8.5)
        for spine in ax.spines.values(): spine.set_edgecolor('#1e2d3d')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.002, f'{v:.4f}',
                    ha='center', color='#cbd5e1', fontsize=8.5)
    fig.suptitle('Avaliação Comparativa — Hit Rate, NDCG e Coverage', color='#e2e8f0', fontsize=12, y=1.02)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Tabela
    st.markdown("## Tabela de Resultados")
    st.dataframe(
        df_eval.style
            .highlight_max(color='rgba(20,184,166,0.25)', axis=0)
            .format("{:.4f}"),
        use_container_width=True
    )

# ─────────────────────────────────────────────
# TAB 4 — CATÁLOGO
# ─────────────────────────────────────────────
with tab4:
    st.markdown("## Catálogo de Produtos Financeiros")

    cat_filter = st.multiselect(
        "Filtrar por Categoria",
        options=sorted(set(p['category'] for p in PRODUCTS.values())),
        default=[]
    )
    risk_filter = st.multiselect(
        "Filtrar por Risco",
        options=['baixo','médio','alto'],
        default=[]
    )

    catalog_data = []
    for pid, pinfo in PRODUCTS.items():
        if cat_filter and pinfo['category'] not in cat_filter: continue
        if risk_filter and pinfo['risk'] not in risk_filter: continue
        pop  = len(df[df['product_id']==pid])
        avgr = df[df['product_id']==pid]['rating'].mean() if pop > 0 else 0
        catalog_data.append({
            'ID': pid,
            'Produto': f"{CAT_ICON.get(pinfo['category'],'📦')} {pinfo['name']}",
            'Categoria': pinfo['category'],
            'Risco': pinfo['risk'].upper(),
            'Mín (R$)': pinfo['min_value'],
            'Retorno': pinfo['yield'],
            'Interações': pop,
            'Rating Médio': round(avgr, 2),
        })

    df_cat = pd.DataFrame(catalog_data)
    if len(df_cat) > 0:
        st.dataframe(
            df_cat.style.apply(
                lambda x: ['color:#34d399' if v=='BAIXO' else 'color:#f59e0b' if v=='MÉDIO' else 'color:#f87171' if v=='ALTO' else '' for v in x],
                subset=['Risco']
            ).background_gradient(subset=['Interações'], cmap='Blues'),
            use_container_width=True, height=420
        )

        # Popularidade dos produtos
        st.markdown("## Popularidade dos Produtos")
        pop_by_prod = (df.groupby('product_id').size()
                       .reindex(PRODUCT_IDS, fill_value=0)
                       .sort_values(ascending=True))
        fig, ax = dark_fig((12, 5))
        colors_prod = [PALETTE[i % len(PALETTE)] for i in range(len(pop_by_prod))]
        ax.barh([PRODUCTS[p]['name'] for p in pop_by_prod.index],
                pop_by_prod.values, color=colors_prod, alpha=0.9, height=0.65)
        ax.set_xlabel('Interações', color='#475569')
        ax.set_title('Popularidade dos Produtos na Base', color='#cbd5e1', pad=10)
        for i, v in enumerate(pop_by_prod.values):
            if v > 0: ax.text(v+1, i, str(v), va='center', color='#cbd5e1', fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;padding:1rem 0;'>
    <p style='color:#334155;font-size:0.82rem;'>
        Desenvolvido por <strong style='color:#475569'>Gabriel Alessi Naumann</strong> · 
        Neural Collaborative Filtering · PyTorch · Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
