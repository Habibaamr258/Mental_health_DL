import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import pickle
import json
from PIL import Image
import io

# ═══════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MediFusion — AI Diagnosis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════
#  CUSTOM CSS
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --teal-dark:   #0F6E56;
    --teal-mid:    #1D9E75;
    --teal-light:  #E1F5EE;
    --coral:       #D85A30;
    --coral-light: #FAECE7;
    --gray-text:   #2C2C2A;
    --gray-muted:  #888780;
    --gray-border: #D3D1C7;
    --gray-bg:     #F1EFE8;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }

/* ── Header ── */
.app-header {
    display: flex;
    align-items: flex-end;
    gap: 1.5rem;
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--gray-border);
}
.app-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: var(--gray-text);
    line-height: 1;
    margin: 0;
}
.app-title span { color: var(--teal-mid); }
.app-subtitle {
    font-size: 0.85rem;
    color: var(--gray-muted);
    font-weight: 400;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--teal-dark);
    margin-bottom: 0.5rem;
    display: block;
}

/* ── Input card ── */
.input-card {
    background: white;
    border: 1px solid var(--gray-border);
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1rem;
}

/* ── Streamlit widgets ── */
.stTextArea textarea {
    border-radius: 10px !important;
    border: 1.5px solid var(--gray-border) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1rem !important;
    resize: vertical !important;
    transition: border-color 0.2s;
}
.stTextArea textarea:focus {
    border-color: var(--teal-mid) !important;
    box-shadow: 0 0 0 3px rgba(29,158,117,0.15) !important;
}

.stNumberInput input, .stSelectbox select {
    border-radius: 10px !important;
    border: 1.5px solid var(--gray-border) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--gray-border) !important;
    border-radius: 12px !important;
    background: var(--gray-bg) !important;
    padding: 1rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--teal-mid) !important;
}

/* ── Analyze button ── */
.stButton > button {
    background: var(--teal-dark) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2.5rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    transition: background 0.2s, transform 0.1s !important;
    width: 100%;
}
.stButton > button:hover {
    background: var(--teal-mid) !important;
    transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0); }

/* ── Result card ── */
.result-main {
    background: var(--teal-light);
    border: 1.5px solid var(--teal-mid);
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.25rem;
}
.result-disease {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: var(--teal-dark);
    margin: 0.25rem 0 0;
}
.result-conf {
    font-size: 0.85rem;
    color: var(--teal-mid);
    font-weight: 500;
    margin-top: 0.4rem;
}

/* ── Drug pills ── */
.drug-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.75rem;
}
.drug-pill {
    background: white;
    border: 1.5px solid var(--teal-mid);
    color: var(--teal-dark);
    border-radius: 999px;
    padding: 0.3rem 1rem;
    font-size: 0.88rem;
    font-weight: 500;
}

/* ── Probability bars ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.6rem;
}
.prob-label {
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--gray-text);
    width: 110px;
    flex-shrink: 0;
}
.prob-bar-bg {
    flex: 1;
    height: 8px;
    background: var(--gray-bg);
    border-radius: 99px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: var(--teal-mid);
    transition: width 0.5s ease;
}
.prob-bar-fill.top { background: var(--teal-dark); }
.prob-pct {
    font-size: 0.82rem;
    color: var(--gray-muted);
    width: 38px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Warning ── */
.warn-box {
    background: #FAEEDA;
    border: 1px solid #EF9F27;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #633806;
    margin-top: 1rem;
}
.info-box {
    background: #E6F1FB;
    border: 1px solid #85B7EB;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #0C447C;
    margin-top: 0.75rem;
}

/* ── Divider ── */
.thin-hr { border: none; border-top: 1px solid var(--gray-border); margin: 1.5rem 0; }

/* ── Demo mode badge ── */
.demo-badge {
    display: inline-block;
    background: #FAC775;
    color: #412402;
    border-radius: 999px;
    padding: 0.2rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-left: 1rem;
    vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════
CLASSES = ['covid-19', 'lung cancer', 'pneumonia', 'tuberculosis']

DRUG_DB = {
    'covid-19':      ['Remdesivir', 'Dexamethasone', 'Baricitinib', 'Tocilizumab', 'Molnupiravir'],
    'lung cancer':   ['Carboplatin', 'Pemetrexed', 'Osimertinib', 'Nivolumab', 'Bevacizumab'],
    'pneumonia':     ['Amoxicillin', 'Azithromycin', 'Levofloxacin', 'Ceftriaxone', 'Doxycycline'],
    'tuberculosis':  ['Isoniazid', 'Rifampicin', 'Pyrazinamide', 'Ethambutol', 'Streptomycin'],
}

DISEASE_INFO = {
    'covid-19':     'Viral respiratory illness caused by SARS-CoV-2. Ranges from mild to severe.',
    'lung cancer':  'Malignant lung tumor. Early detection significantly improves outcomes.',
    'pneumonia':    'Lung infection that inflames air sacs. Can be bacterial, viral, or fungal.',
    'tuberculosis': 'Bacterial infection (M. tuberculosis) primarily affecting the lungs.',
}

# Common symptoms for autocomplete hints
SYMPTOM_HINTS = [
    "fever", "cough", "shortness of breath", "fatigue", "chest pain",
    "weight loss", "night sweats", "hemoptysis", "wheezing", "chills",
    "loss of taste", "loss of smell", "sore throat", "headache",
    "muscle aches", "difficulty breathing", "persistent cough"
]

# ═══════════════════════════════════════════════════════════
#  MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════
class MetaLearner(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 4)
        )
    def forward(self, x): return self.net(x)

# ═══════════════════════════════════════════════════════════
#  LOAD MODELS (cached)
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_meta_learner(path="meta_learner.pt"):
    model = MetaLearner()
    try:
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model, True
    except FileNotFoundError:
        return model, False

@st.cache_resource
def load_bert_model(model_path="best_bert_model.pt"):
    """Load DistilBERT — returns (tokenizer, model, loaded_ok)"""
    try:
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=4
        )
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return tokenizer, model, True
    except Exception:
        return None, None, False

@st.cache_resource
def load_effnet_model(model_path="EfficientNetB3_MultiClass_BEST_FINE_TUNED.keras"):
    """Load EfficientNetB3 — returns (model, loaded_ok)"""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        return model, True
    except Exception:
        return None, False

# ═══════════════════════════════════════════════════════════
#  INFERENCE HELPERS
# ═══════════════════════════════════════════════════════════
def get_bert_probs(symptoms_text, age, gender, tokenizer, bert_model):
    """Run DistilBERT on symptoms → 4-class probs"""
    gender_str = "male" if gender == "Male" else "female"
    text = f"Patient is {age} years old {gender_str}. Symptoms: {symptoms_text}"
    enc = tokenizer(text, return_tensors="pt", max_length=256,
                    truncation=True, padding="max_length")
    with torch.no_grad():
        logits = bert_model(**enc).logits
    probs = torch.softmax(logits, dim=-1).squeeze().numpy()
    return probs  # shape (4,) — [covid, lung_cancer, pneumonia, tuberculosis]

def get_effnet_probs(image: Image.Image, effnet_model):
    """Run EfficientNetB3 on X-Ray → 4 aligned probs (no normal)"""
    import tensorflow as tf
    img = image.convert("RGB").resize((300, 300))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    probs_5 = effnet_model.predict(arr, verbose=0)[0]
    # order: normal=0, covid=1, pneumonia=2, tuberculosis=3, lung_cancer=4
    probs_4 = probs_5[1:]  # drop normal
    probs_4 = probs_4 / probs_4.sum()
    # reorder → [covid, lung_cancer, pneumonia, tuberculosis]
    REORDER = [0, 3, 1, 2]
    return probs_4[REORDER]

def demo_bert_probs(symptoms_text):
    """Simulate DistilBERT output for demo mode"""
    text_lower = symptoms_text.lower()
    if any(w in text_lower for w in ["fever","cough","taste","smell","covid"]):
        base = [0.65, 0.05, 0.20, 0.10]
    elif any(w in text_lower for w in ["weight loss","hemoptysis","cancer","smoking"]):
        base = [0.05, 0.70, 0.10, 0.15]
    elif any(w in text_lower for w in ["pneumonia","breath","infection"]):
        base = [0.10, 0.05, 0.75, 0.10]
    elif any(w in text_lower for w in ["night sweat","tuberc","tb"]):
        base = [0.05, 0.10, 0.10, 0.75]
    else:
        base = [0.25, 0.25, 0.25, 0.25]
    arr = np.array(base, dtype=np.float32)
    arr += np.random.dirichlet(np.ones(4)) * 0.05
    return arr / arr.sum()

def demo_effnet_probs():
    """Simulate EfficientNet output for demo mode"""
    arr = np.random.dirichlet(np.array([3, 1, 1, 1], dtype=float))
    return arr.astype(np.float32)

def fuse_and_predict(text_probs, image_probs, meta_model):
    """Feed 8-dim vector through Meta-Learner → probs + predicted class"""
    x = np.concatenate([text_probs, image_probs]).astype(np.float32)
    x_t = torch.tensor(x).unsqueeze(0)
    meta_model.eval()
    with torch.no_grad():
        logits = meta_model(x_t)
    probs = torch.softmax(logits, dim=-1).squeeze().numpy()
    pred_idx = int(np.argmax(probs))
    return probs, pred_idx

# ═══════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════
meta_model, meta_loaded = load_meta_learner()
_, _, bert_loaded = load_bert_model() if False else (None, None, False)  # lazy
_, effnet_loaded = load_effnet_model() if False else (None, False)       # lazy

demo_mode = not (meta_loaded)  # if no meta_learner.pt, run in demo mode

st.markdown(f"""
<div class="app-header">
  <div>
    <div class="app-subtitle">AI-powered pulmonary diagnostics</div>
    <h1 class="app-title">Medi<span>Fusion</span></h1>
  </div>
  {'<span class="demo-badge">⚡ DEMO MODE</span>' if demo_mode else ''}
</div>
""", unsafe_allow_html=True)

if demo_mode:
    st.markdown("""
    <div class="info-box">
    ℹ️ <strong>Demo mode active.</strong> Place <code>meta_learner.pt</code>,
    <code>best_bert_model.pt</code>, and
    <code>EfficientNetB3_MultiClass_BEST_FINE_TUNED.keras</code>
    in the same folder as <code>app.py</code> to run with real models.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ═══════════════════════════════════════════════════════════
col_input, col_result = st.columns([1.1, 0.9], gap="large")

# ───────────────── LEFT: INPUT ─────────────────
with col_input:

    # — Patient info —
    st.markdown('<span class="section-label">Patient information</span>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=1, max_value=110, value=45, step=1)
    with c2:
        gender = st.selectbox("Gender", ["Male", "Female"])

    st.markdown('<div class="thin-hr"></div>', unsafe_allow_html=True)

    # — Symptoms —
    st.markdown('<span class="section-label">Symptoms</span>', unsafe_allow_html=True)
    symptoms_text = st.text_area(
        "Describe patient symptoms",
        placeholder="e.g., persistent dry cough, high fever, shortness of breath, fatigue, loss of taste and smell...",
        height=140,
        label_visibility="collapsed"
    )

    # Hint chips
    st.markdown("""
    <div style="margin-top:0.4rem; margin-bottom:0.75rem;">
      <span style="font-size:0.75rem; color:#888780;">Common: </span>
      <span style="font-size:0.78rem; color:#0F6E56;">fever · cough · shortness of breath · weight loss · night sweats · hemoptysis · chest pain</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="thin-hr"></div>', unsafe_allow_html=True)

    # — X-Ray upload —
    st.markdown('<span class="section-label">Chest X-Ray image</span>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload chest X-Ray",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded X-Ray", use_container_width=True,
                 output_format="JPEG")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # — Analyze button —
    analyze_clicked = st.button("🔬 Analyze Patient", use_container_width=True)

# ───────────────── RIGHT: RESULT ─────────────────
with col_result:
    st.markdown('<span class="section-label">Diagnosis</span>', unsafe_allow_html=True)

    if not analyze_clicked:
        st.markdown("""
        <div style="padding: 3rem 2rem; text-align:center; color:#888780; border: 1.5px dashed #D3D1C7; border-radius:16px;">
          <div style="font-size:2.5rem; margin-bottom:0.75rem;">🫁</div>
          <div style="font-size:0.9rem; line-height:1.6;">
            Fill in patient information,<br>enter symptoms, and upload<br>an X-Ray to begin analysis.
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Validate inputs ──
        if not symptoms_text.strip():
            st.error("⚠️ Please enter patient symptoms before analyzing.")
            st.stop()

        with st.spinner("Running AI models…"):

            # ── Get text probs ──
            _, bert_model_obj, bert_ok = load_bert_model()
            if bert_ok and not demo_mode:
                tokenizer, bert_model_obj, _ = load_bert_model()
                text_probs = get_bert_probs(symptoms_text, age, gender, tokenizer, bert_model_obj)
            else:
                text_probs = demo_bert_probs(symptoms_text)

            # ── Get image probs ──
            effnet_model_obj, eff_ok = load_effnet_model()
            if eff_ok and uploaded_file and not demo_mode:
                img_pil = Image.open(uploaded_file)
                image_probs = get_effnet_probs(img_pil, effnet_model_obj)
            elif uploaded_file:
                image_probs = demo_effnet_probs()
            else:
                # No image — use uniform prior
                image_probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

            # ── Fuse ──
            if meta_loaded:
                fusion_probs, pred_idx = fuse_and_predict(text_probs, image_probs, meta_model)
            else:
                # Demo: simple weighted average
                fusion_probs = 0.6 * text_probs + 0.4 * image_probs
                fusion_probs = fusion_probs / fusion_probs.sum()
                pred_idx = int(np.argmax(fusion_probs))

        # ── Disease result ──
        disease = CLASSES[pred_idx]
        confidence = float(fusion_probs[pred_idx]) * 100

        st.markdown(f"""
        <div class="result-main">
          <div style="font-size:0.72rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#0F6E56;">
            Predicted diagnosis
          </div>
          <div class="result-disease">{disease.title()}</div>
          <div class="result-conf">Confidence: {confidence:.1f}%</div>
          <hr style="border:none; border-top:1px solid #9FE1CB; margin:1rem 0;">
          <div style="font-size:0.83rem; color:#0F6E56; line-height:1.55;">
            {DISEASE_INFO[disease]}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability bars ──
        st.markdown('<span class="section-label" style="margin-top:1rem;">Class probabilities</span>', unsafe_allow_html=True)

        sorted_idx = np.argsort(fusion_probs)[::-1]
        bars_html = '<div style="margin-top:0.5rem;">'
        for i, idx in enumerate(sorted_idx):
            pct = float(fusion_probs[idx]) * 100
            is_top = (i == 0)
            fill_class = "prob-bar-fill top" if is_top else "prob-bar-fill"
            bars_html += f"""
            <div class="prob-row">
              <span class="prob-label">{CLASSES[idx].title()}</span>
              <div class="prob-bar-bg">
                <div class="{fill_class}" style="width:{pct:.1f}%"></div>
              </div>
              <span class="prob-pct">{pct:.1f}%</span>
            </div>"""
        bars_html += "</div>"
        st.markdown(bars_html, unsafe_allow_html=True)

        st.markdown('<div class="thin-hr"></div>', unsafe_allow_html=True)

        # ── Drug recommendations ──
        st.markdown('<span class="section-label">Recommended medications</span>', unsafe_allow_html=True)
        drugs = DRUG_DB.get(disease, [])
        pills_html = '<div class="drug-grid">'
        for d in drugs:
            pills_html += f'<span class="drug-pill">💊 {d}</span>'
        pills_html += '</div>'
        st.markdown(pills_html, unsafe_allow_html=True)

        # ── Model breakdown (expander) ──
        with st.expander("🔍 Model breakdown"):
            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown("**DistilBERT (text)**")
                for i, cls in enumerate(CLASSES):
                    st.markdown(f"`{cls}` — {text_probs[i]*100:.1f}%")
            with mc2:
                st.markdown("**EfficientNetB3 (image)**")
                for i, cls in enumerate(CLASSES):
                    st.markdown(f"`{cls}` — {image_probs[i]*100:.1f}%")

            if not uploaded_file:
                st.markdown("""
                <div class="info-box">
                ℹ️ No X-Ray uploaded — uniform image prior used (0.25 each).
                For best accuracy, upload a chest X-Ray.
                </div>
                """, unsafe_allow_html=True)

        # ── Disclaimer ──
        st.markdown("""
        <div class="warn-box">
        ⚠️ <strong>Clinical disclaimer:</strong> This tool is for research purposes only.
        Always consult a licensed physician before making any medical decisions.
        </div>
        """, unsafe_allow_html=True)
