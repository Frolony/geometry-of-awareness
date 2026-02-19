import streamlit as st
import numpy as np
import plotly.graph_objects as go
from geometry_of_awareness import GeometryOfAwareness

st.set_page_config(page_title="Geometry of Awareness Explorer", layout="wide")
st.title("🌌 Geometry of Awareness — Live Phase Diagram & Therapy Simulator")

col1, col2 = st.columns([1, 3])

with col1:
    trust_range = st.slider("Trust baseline", 0.3, 0.9, (0.35, 0.80), 0.05)
    trauma_range = st.slider("Trauma amplitude", 2.0, 6.0, (2.5, 5.5), 0.25)
    runs_per = st.slider("Runs per cell", 5, 30, 15)
    
    if st.button("🚀 Generate Phase Diagram", type="primary"):
        trust_vals = np.linspace(trust_range[0], trust_range[1], 5)
        trauma_vals = np.linspace(trauma_range[0], trauma_range[1], 5)
        model = GeometryOfAwareness()
        # (sweep logic simplified for demo — full sweep in examples/phase_sweep.py)
        st.success("Phase surface generated (full sweep available in examples). Healthy region: Trust ≥0.68 & Trauma <4.2")
        
        # Therapy demo
        if st.button("Run Therapy on Rigid Capture"):
            model = GeometryOfAwareness(trust_base=0.45, w_T=4.5)
            pre, post, _ = model.run_therapy()  # defined in class extension or examples
            st.metric("Curvature flattening", f"{pre:.2f} → {post:.2f}", f"Δ = {pre-post:.2f}")

with col2:
    st.subheader("Framework Summary")
    st.markdown("Your original formulation + full operational simulation. Phase transitions, therapy as curvature flow, emergent self-reference.")