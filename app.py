"""
Mining Flotation — Silica Concentrate Predictor
Run: streamlit run app.py
Requires: best_model.pkl + model_meta.json (from notebook)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flotation Predictor",
    page_icon="⛏️",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

.app-header {
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid #21262d;
    margin-bottom: 2rem;
}
.app-title {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #f0c040, #f97316);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.app-subtitle {
    color: #8b949e;
    font-size: 0.95rem;
    margin-top: 6px;
    font-weight: 400;
}
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #f0c040;
    margin-bottom: 1rem;
    padding-bottom: 6px;
    border-bottom: 1px solid #21262d;
}
.input-group-title {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 0.5rem;
}
div[data-testid="stNumberInput"] label {
    font-size: 0.78rem !important;
    color: #8b949e !important;
    letter-spacing: 0.5px;
}
div[data-testid="stNumberInput"] input {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.92rem !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: #f0c040 !important;
    box-shadow: 0 0 0 3px rgba(240,192,64,0.1) !important;
}
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #f0c040, #f97316) !important;
    color: #0d1117 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(240,192,64,0.3) !important;
}
.result-main {
    background: linear-gradient(135deg, #1c2b1a, #162314);
    border: 1.5px solid #238636;
    border-radius: 14px;
    padding: 2.5rem 1.5rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.result-eyebrow {
    font-size: 0.68rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #3fb950;
    margin-bottom: 10px;
}
.result-number {
    font-family: 'DM Mono', monospace;
    font-size: 4rem;
    font-weight: 500;
    color: #ffffff;
    line-height: 1;
}
.result-unit { font-size: 1.2rem; color: #8b949e; margin-top: 4px; }
.quality-badge {
    display: inline-block;
    margin-top: 14px;
    padding: 5px 16px;
    border-radius: 99px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 1px;
}
.quality-excellent { background: #0d4429; color: #3fb950; border: 1px solid #238636; }
.quality-good      { background: #2d2a0f; color: #d29922; border: 1px solid #9e6a03; }
.quality-high      { background: #3d1212; color: #f85149; border: 1px solid #da3633; }
.stat-item {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.stat-name { font-size: 0.72rem; color: #8b949e; letter-spacing: 1px; text-transform: uppercase; }
.stat-val  { font-family: 'DM Mono', monospace; font-size: 1.1rem; color: #e6edf3; margin-top: 2px; }
.idle-box {
    background: #161b22;
    border: 1px dashed #30363d;
    border-radius: 14px;
    padding: 3rem 2rem;
    text-align: center;
    color: #484f58;
}
.idle-icon { font-size: 3rem; margin-bottom: 12px; }
.idle-text { font-size: 0.88rem; line-height: 1.6; }
.model-badge {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 0.78rem;
    color: #8b949e;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 1.5rem;
}
.model-badge span { color: #f0c040; font-weight: 700; font-family: 'DM Mono', monospace; }
</style>
""", unsafe_allow_html=True)


# ── Load artefacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl")
    with open("model_meta.json") as f:
        meta = json.load(f)
    return model, meta


missing = [f for f in ["best_model.pkl", "model_meta.json"] if not os.path.exists(f)]
if missing:
    st.error(
        f"**Missing files:** `{'`, `'.join(missing)}`\n\n"
        "Run `mining_ml_project.ipynb` first to generate the model artefacts."
    )
    st.stop()

model, meta = load_model()
features   = meta["features"]
stats      = meta["feature_stats"]
model_name = meta["model_name"]
r2         = meta["r2_score"]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <p class="app-title">⛏️ Flotation Predictor</p>
    <p class="app-subtitle">Enter process parameters and click <strong>Predict</strong> to estimate % Silica Concentrate</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="model-badge">
    Active model &nbsp;→&nbsp; <span>{model_name}</span>
    &nbsp;|&nbsp; R² <span>{r2}</span>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown('<div class="section-label">Process Parameters</div>', unsafe_allow_html=True)

    all_inputs = {}

    # Categorise features
    feed_feats    = [f for f in features if 'Feed' in f]
    reagent_feats = [f for f in features if any(k in f for k in ['Starch', 'Amina', 'pH', 'Density', 'Ore Pulp Flow'])]
    air_feats     = [f for f in features if 'Air Flow' in f]
    level_feats   = [f for f in features if 'Level' in f]
    grouped       = set(feed_feats + reagent_feats + air_feats + level_feats)
    other_feats   = [f for f in features if f not in grouped]

    def render_inputs(feat_list, ncols=3):
        vals = {}
        cols = st.columns(ncols)
        for i, feat in enumerate(feat_list):
            fmean = stats["mean"][feat]
            fmin  = stats["min"][feat]
            fmax  = stats["max"][feat]
            step  = float(round((fmax - fmin) / 200, 4)) or 0.001
            short = feat.replace("Flotation Column ", "Col ")
            with cols[i % ncols]:
                vals[feat] = st.number_input(
                    short,
                    value=float(round(fmean, 3)),
                    min_value=float(round(fmin, 3)),
                    max_value=float(round(fmax, 3)),
                    step=step,
                    format="%.3f",
                    key=feat,
                )
        return vals

    if feed_feats:
        st.markdown('<div class="input-group-title">Feed Quality</div>', unsafe_allow_html=True)
        all_inputs.update(render_inputs(feed_feats, ncols=min(len(feed_feats), 3)))
        st.markdown("<br>", unsafe_allow_html=True)

    if reagent_feats:
        st.markdown('<div class="input-group-title">Reagents & Pulp</div>', unsafe_allow_html=True)
        all_inputs.update(render_inputs(reagent_feats, ncols=min(len(reagent_feats), 3)))
        st.markdown("<br>", unsafe_allow_html=True)

    if air_feats:
        st.markdown('<div class="input-group-title">Column Air Flow</div>', unsafe_allow_html=True)
        all_inputs.update(render_inputs(air_feats, ncols=min(len(air_feats), 4)))
        st.markdown("<br>", unsafe_allow_html=True)

    if level_feats:
        st.markdown('<div class="input-group-title">Column Level</div>', unsafe_allow_html=True)
        all_inputs.update(render_inputs(level_feats, ncols=min(len(level_feats), 4)))
        st.markdown("<br>", unsafe_allow_html=True)

    if other_feats:
        st.markdown('<div class="input-group-title">Other Parameters</div>', unsafe_allow_html=True)
        all_inputs.update(render_inputs(other_feats, ncols=3))
        st.markdown("<br>", unsafe_allow_html=True)

    btn_col, reset_col = st.columns([3, 1])
    with btn_col:
        predict_clicked = st.button("⚡  Predict Silica Concentrate", use_container_width=True)
    with reset_col:
        if st.button("↺  Reset", use_container_width=True):
            for k in list(st.session_state.keys()):
                if k in features:
                    del st.session_state[k]
            st.rerun()

# ── Result panel ──────────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="section-label">Prediction Output</div>', unsafe_allow_html=True)

    if predict_clicked:
        X_input    = pd.DataFrame([[all_inputs[f] for f in features]], columns=features)
        prediction = float(model.predict(X_input)[0])

        if prediction < 2.0:
            quality_label, quality_class = "Excellent Quality",     "quality-excellent"
        elif prediction < 4.0:
            quality_label, quality_class = "Acceptable Quality",    "quality-good"
        else:
            quality_label, quality_class = "High Silica — Review",  "quality-high"

        st.markdown(f"""
        <div class="result-main">
            <div class="result-eyebrow">% Silica Concentrate</div>
            <div class="result-number">{prediction:.3f}</div>
            <div class="result-unit">percent</div>
            <div class="quality-badge {quality_class}">{quality_label}</div>
        </div>
        """, unsafe_allow_html=True)

        # Key inputs vs average
        st.markdown('<div class="input-group-title" style="margin-top:1rem">Key Inputs vs Average</div>', unsafe_allow_html=True)
        highlight = ['% Iron Feed', '% Silica Feed', 'Starch Flow', 'Amina Flow', 'Ore Pulp pH', 'Ore Pulp Density']
        highlight = [f for f in highlight if f in all_inputs]

        for feat in highlight:
            val  = all_inputs[feat]
            mean = stats["mean"][feat]
            delta_pct = ((val - mean) / mean * 100) if mean != 0 else 0
            arrow = "↑" if delta_pct > 1 else ("↓" if delta_pct < -1 else "→")
            color = "#f85149" if delta_pct > 1 else ("#3fb950" if delta_pct < -1 else "#8b949e")
            st.markdown(f"""
            <div class="stat-item">
                <div class="stat-name">{feat}</div>
                <div class="stat-val">{val:.3f}
                    <span style="color:{color};font-size:0.75rem;font-family:'DM Mono',monospace">
                        {arrow} {delta_pct:+.1f}% vs avg
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:1rem;padding:10px 14px;background:#0d1117;border:1px solid #21262d;
                    border-radius:8px;font-size:0.75rem;color:#484f58;line-height:1.6">
            Model: <span style="color:#f0c040;font-family:'DM Mono',monospace">{model_name}</span>
            &nbsp;|&nbsp; R² = <span style="color:#f0c040;font-family:'DM Mono',monospace">{r2}</span>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="idle-box">
            <div class="idle-icon">🎯</div>
            <div class="idle-text">
                Set your process parameters on the left,<br>
                then click <strong>⚡ Predict</strong> to see the<br>
                estimated <strong>% Silica Concentrate</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="input-group-title">Model Leaderboard</div>', unsafe_allow_html=True)
        for mname, scores in sorted(meta["all_results"].items(), key=lambda x: -x[1]["R2"]):
            crown = "🏆" if mname == model_name else "　"
            c1, c2, c3 = st.columns(3)
            name_color = "#f0c040" if mname == model_name else "#8b949e"
            c1.markdown(f"<span style='font-size:0.8rem;color:{name_color}'>{crown} {mname}</span>", unsafe_allow_html=True)
            c2.markdown(f"<span style='font-family:DM Mono,monospace;font-size:0.8rem;color:#3fb950'>R² {scores['R2']:.4f}</span>", unsafe_allow_html=True)
            c3.markdown(f"<span style='font-family:DM Mono,monospace;font-size:0.8rem;color:#8b949e'>RMSE {scores['RMSE']:.4f}</span>", unsafe_allow_html=True)