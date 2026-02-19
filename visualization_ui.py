import streamlit as st
import os
from geometry_of_awareness import GeometryOfAwareness
import numpy as np

st.set_page_config(page_title="Geometry of Awareness Visuals", layout="wide")
st.title("Geometry of Awareness — Visualizations & Metrics")

viz_dir = os.path.join(os.path.dirname(__file__), 'visualizations')
images = sorted([f for f in os.listdir(viz_dir) if f.lower().endswith('.png')])

with st.sidebar:
    st.header("Controls")
    show_all = st.checkbox("Show all images", value=True)
    model_n = st.selectbox("Model dimension n", options=[7, 15], index=0)
    compute_lyapunov = st.button("Compute Lyapunov Matrix")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Saved Visualizations")
    if show_all:
        for img in images:
            st.markdown(f"**{img}**")
            st.image(os.path.join(viz_dir, img), use_column_width=True)
    else:
        sel = st.selectbox("Image", images)
        st.image(os.path.join(viz_dir, sel), use_column_width=True)

with col2:
    st.subheader("Quick Stability Matrix")
    st.markdown("This runs `lyapunov_analysis` for common basins/trusts and shows spectral radius (ρ(J)).")
    if compute_lyapunov:
        model = GeometryOfAwareness(n=model_n)
        basins = ['H', 'R']
        trusts = [0.4, 0.8]
        rows = []
        for b in basins:
            for t in trusts:
                res = model.lyapunov_analysis(basin=b, trust=t, n_steps=200)
                rho = res['max_abs_eigenvalue']
                stable = res['stable']
                rows.append({'basin': b, 'trust': t, 'rho(J)': float(rho), 'stable': bool(stable)})
        st.table(rows)
    else:
        st.info("Click 'Compute Lyapunov Matrix' to run a quick stability check.")

st.markdown("---")
st.markdown("**Notes:** Lyapunov here refers to the discrete-time map J = I - dt * g^{-1} * Hess(V). See VISUALIZATION_GUIDE.py for details.")

# Display the mathematical appendix (external markdown file)
try:
    appendix_path = os.path.join(os.path.dirname(__file__), 'MATHEMATICAL_APPENDIX.md')
    if os.path.exists(appendix_path):
        with open(appendix_path, 'r', encoding='utf-8') as f:
            appendix_md = f.read()
        with st.expander('MATHEMATICAL APPENDIX (expand)'):
            st.markdown(appendix_md, unsafe_allow_html=False)
    else:
        st.info('MATHEMATICAL_APPENDIX.md not found in project root.')
except Exception as e:
    st.error(f'Error loading appendix: {e}')
