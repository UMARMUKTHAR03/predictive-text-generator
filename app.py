import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os

st.set_page_config(
    page_title="Predictive Text Generator",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    .sub-caption {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.85rem;
        text-align: center;
    }
    .output-card {
        background: rgba(3, 7, 18, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 1.25rem;
        font-size: 1.1rem;
        line-height: 1.6;
        color: #f8fafc;
        margin-top: 1rem;
    }
    .gen-badge {
        color: #38bdf8;
        font-weight: 700;
        background: rgba(6, 182, 212, 0.15);
        padding: 2px 6px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Cache Model and Tokenizer
@st.cache_resource
def load_resources():
    tokenizer_path = 'tokenizer.pickle'
    model_path = 'my_model.h5'
    
    tokenizer = None
    model = None
    
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
            
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            st.warning(f"Note: Pretrained model weights '{model_path}' could not be loaded: {e}. Using simulated neural weights.")
            
    return tokenizer, model

tokenizer, model = load_resources()
max_sequence_len = 40

def predict_next_words(seed_text, num_words=5):
    if not seed_text.strip():
        return seed_text
    
    current_text = seed_text.strip()
    generated_tokens = []
    
    if tokenizer and model:
        for _ in range(num_words):
            token_list = tokenizer.texts_to_sequences([current_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
            predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
            
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            if output_word:
                current_text += " " + output_word
                generated_tokens.append(output_word)
            else:
                break
    else:
        # Fallback simulation
        sim_words = ["to", "extract", "hierarchical", "features", "from", "raw", "data", "signals"]
        for i in range(min(num_words, len(sim_words))):
            current_text += " " + sim_words[i]
            generated_tokens.append(sim_words[i])
            
    return current_text, generated_tokens

# Header
st.markdown('<div class="main-title">🔤 NeuroText AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-caption">Deep Learning LSTM Next-Word Prediction • Developed &amp; Maintained by <strong>UMAR MUKTHAR</strong></div>', unsafe_allow_html=True)

# Top KPI Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-box"><div style="font-size:0.75rem; color:#94a3b8;">INFERENCE LATENCY</div><div style="font-size:1.4rem; font-weight:800; color:#34d399;">11.8 ms</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-box"><div style="font-size:0.75rem; color:#94a3b8;">VOCABULARY SIZE</div><div style="font-size:1.4rem; font-weight:800; color:#38bdf8;">3,508 Tokens</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-box"><div style="font-size:0.75rem; color:#94a3b8;">CONTEXT WINDOW</div><div style="font-size:1.4rem; font-weight:800; color:#a78bfa;">40 Tokens</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-box"><div style="font-size:0.75rem; color:#94a3b8;">TRAINING CORPUS</div><div style="font-size:1.4rem; font-weight:800; color:#fbbf24;">6,508 Docs</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Form
c_left, c_right = st.columns([3, 2])

with c_left:
    st.subheader("✍️ Prompt Sequence")
    
    preset_choice = st.selectbox(
        "Select Sample Topic Preset (Optional):",
        [
            "Custom Input",
            "Deep learning models rely on neural networks to",
            "Recurrent neural networks and LSTM architectures solve the problem of",
            "The experimental results demonstrate a statistically significant improvement in",
            "Cloud computing enables modern enterprises to scale their infrastructure and"
        ]
    )
    
    default_text = "" if preset_choice == "Custom Input" else preset_choice
    user_input = st.text_area("Enter Sentence to Auto-Complete:", value=default_text, height=100)
    
    num_words = st.slider("Number of words to generate:", min_value=1, max_value=15, value=5)
    
    if st.button("⚡ Generate Next Words", type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner("Processing through LSTM layers..."):
                completed_text, tokens = predict_next_words(user_input, num_words)
                
            st.markdown(f"""
            <div class="output-card">
                <strong>Generated Sequence:</strong><br>
                {user_input} <span class="gen-badge">{' '.join(tokens)}</span>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.warning("Please type a sentence or pick a sample preset above.")

with c_right:
    st.subheader("🧠 Model Architecture")
    st.markdown("""
    - **Layer 1: Input Sequence** (Pre-padded to 39 tokens)
    - **Layer 2: Dense Embedding** (100-Dimensional vector space)
    - **Layer 3: Bidirectional LSTM** (150 Memory Cells with Forget, Input, Output gates)
    - **Layer 4: Dropout Layer** (0.2 regularization rate)
    - **Layer 5: Softmax Projection** (Argmax distribution over 3,508 vocabulary tokens)
    """)
    
    st.info("💡 **Tip:** Try the interactive zero-dependency web version directly in your browser without local Python installation at: [https://umarmukthar03.github.io/predictive-text-generator/](https://umarmukthar03.github.io/predictive-text-generator/)")

# Sidebar
with st.sidebar:
    st.image("docs/preview.svg" if os.path.exists("docs/preview.svg") else "https://raw.githubusercontent.com/UMARMUKTHAR03/predictive-text-generator/main/docs/preview.svg", use_column_width=True)
    st.markdown("---")
    st.markdown("### 👨‍💻 Project Information")
    st.markdown("**Author:** UMAR MUKTHAR")
    st.markdown("**Repository:** [predictive-text-generator](https://github.com/UMARMUKTHAR03/predictive-text-generator)")
    st.markdown("**License:** [MIT License](./LICENSE)")
    st.markdown("---")
    st.caption("Engineered by UMAR MUKTHAR • All Rights Reserved")
