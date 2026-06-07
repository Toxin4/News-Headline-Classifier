import streamlit as st
import joblib
import os
import re
import numpy as np

# ── Configuração da página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Classificador de Notícias",
    page_icon="📰",
    layout="centered",
)

# ── Visual ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Source+Sans+3:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }

h1 { font-family: 'Playfair Display', serif !important; font-size: 2.4rem !important;
     color: #1a1a2e !important; margin-bottom: 0.2rem !important; }

.subtitle { color: #555; font-size: 1.05rem; margin-bottom: 2rem; }

.result-box { background: #f0f4ff; border-left: 5px solid #3a5bd9;
    border-radius: 8px; padding: 1.2rem 1.5rem; margin-top: 1.5rem; }

.result-label { font-size: 0.85rem; text-transform: uppercase;
    letter-spacing: 1px; color: #888; margin-bottom: 0.3rem; }

.result-value { font-family: 'Playfair Display', serif; font-size: 2rem;
    color: #1a1a2e; font-weight: 700; }

.category-emoji { font-size: 2.5rem; margin-bottom: 0.5rem; }

.stTextArea textarea { border-radius: 8px !important; font-size: 1rem !important;
    font-family: 'Source Sans 3', sans-serif !important; }

.stButton > button { background-color: #1a1a2e !important; color: white !important;
    border: none !important; border-radius: 8px !important; padding: 0.6rem 2rem !important;
    font-size: 1rem !important; font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important; }

.stButton > button:hover { background-color: #3a5bd9 !important; }

.warning-box { background: #fff8e1; border-left: 4px solid #f5a623;
    border-radius: 6px; padding: 0.8rem 1rem; font-size: 0.9rem; color: #7a5700; }
</style>
""", unsafe_allow_html=True)

# ── Função de limpeza (Igual ao notebook) ──────────────────────────────────
def limpar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r'[^a-z\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# ── Mapeamento de categorias ────────────────────────────────────────────────
CATEGORY_INFO = {
    "POLITICS":        {"emoji": "🏛️", "desc": "Política e governo"},
    "WELLNESS":        {"emoji": "🌿", "desc": "Saúde e bem-estar"},
    "ENTERTAINMENT":   {"emoji": "🎬", "desc": "Entretenimento e cultura"},
    "TRAVEL":          {"emoji": "✈️", "desc": "Viagens e turismo"},
    "STYLE & BEAUTY":  {"emoji": "💅", "desc": "Moda e beleza"},
}

# ── Carregamento do modelo e vectorizer ─────────────────────────────────────
@st.cache_resource
def load_model():
    vec_path   = os.path.join("model", "tfidf_vectorizer.joblib")
    model_path = os.path.join("model", "modelo_final.joblib")
    if not os.path.exists(vec_path) or not os.path.exists(model_path):
        return None, None
    vectorizer = joblib.load(vec_path)
    model      = joblib.load(model_path)
    return vectorizer, model

vectorizer, model = load_model()

# ── Interface principal ─────────────────────────────────────────────────────
st.markdown("<h1>📰 Classificador de Notícias</h1>", unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Insira uma manchete e o modelo irá identificar a categoria da notícia automaticamente.</p>',
    unsafe_allow_html=True
)

# Aviso se arquivos não encontrados
if model is None or vectorizer is None:
    st.markdown("""
    <div class="warning-box">
    ⚠️ <strong>Arquivos do modelo não encontrados.</strong><br>
    Certifique-se de que existem na pasta <code>model/</code>:<br>
    • <code>model/tfidf_vectorizer.joblib</code><br>
    • <code>model/modelo_final.joblib</code><br>
    Veja o arquivo <code>salvar_modelo_notebook.py</code> para instruções.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Campo de entrada ────────────────────────────────────────────────────────
headline = st.text_area(
    "Manchete da notícia (em inglês)",
    placeholder='Ex: "Scientists discover new treatment for diabetes"',
    height=110,
    key="headline_input"
)

# Exemplos
st.markdown("**Exemplos rápidos:**")
col1, col2, col3, col4, col5 = st.columns(5)
examples = {
    "pol":  "Senate votes on new immigration bill amid partisan debate",
    "well": "10 simple habits that will improve your mental health",
    "ent":  "Beyoncé announces world tour dates for next year",
    "trav": "Hidden beaches in Southeast Asia you need to visit",
    "sty":  "Top skincare trends taking over social media this season",
}

with col1:
    if st.button("🏛️ Política"):     st.session_state["ex"] = examples["pol"]
with col2:
    if st.button("🌿 Wellness"):      st.session_state["ex"] = examples["well"]
with col3:
    if st.button("🎬 Entretenim."):   st.session_state["ex"] = examples["ent"]
with col4:
    if st.button("✈️ Viagem"):        st.session_state["ex"] = examples["trav"]
with col5:
    if st.button("💅 Beleza"):        st.session_state["ex"] = examples["sty"]

# Preenche com o exemplo escolhido
if not headline and "ex" in st.session_state:
    headline = st.session_state["ex"]
    st.info(f'Exemplo carregado: *"{headline}"*')

st.markdown("---")

# ── Botão de classificação ──────────────────────────────────────────────────
if st.button("🔍 Classificar manchete", use_container_width=True):
    if not headline.strip():
        st.warning("Por favor, insira uma manchete antes de classificar.")
    else:
        with st.spinner("Analisando..."):
            # Aplica a mesma limpeza feita no notebook
            headline_limpa = limpar_texto(headline)
            # Vetoriza com o TF-IDF salvo
            X = vectorizer.transform([headline_limpa])
            # Predição
            prediction = model.predict(X)[0]

        info = CATEGORY_INFO.get(prediction, {"emoji": "📌", "desc": prediction})

        st.markdown(f"""
        <div class="result-box">
            <div class="category-emoji">{info['emoji']}</div>
            <div class="result-label">Categoria prevista</div>
            <div class="result-value">{prediction}</div>
            <p style="color:#555; margin-top:0.5rem;">{info['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Confiança (LinearSVC usa decision_function; LR usa predict_proba)
        st.markdown("#### Confiança por categoria")
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X)[0]
            # Converte para pseudo-probabilidade com softmax para exibição
            e = np.exp(scores - scores.max())
            probs = e / e.sum()
            classes = model.classes_
        else:
            probs = model.predict_proba(X)[0]
            classes = model.classes_

        prob_dict = dict(zip(classes, probs))
        prob_sorted = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        for cat, prob in prob_sorted:
            emoji = CATEGORY_INFO.get(cat, {}).get("emoji", "📌")
            st.progress(float(prob), text=f"{emoji} {cat}: {prob*100:.1f}%")

# ── Rodapé ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:0.85rem;'>"
    "Projeto P2 · ADS UNIMAR · Classificação de Notícias com Machine Learning"
    "</p>",
    unsafe_allow_html=True
)
