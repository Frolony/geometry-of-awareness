import streamlit as st
import os
from geometry_of_awareness import GeometryOfAwareness
from riemannian_leapfrog import create_integrator
import numpy as np
import matplotlib.pyplot as plt

# Import visualization functions from dashboard
from visualization_dashboard import (
    plot_dynamics_trajectory,
    plot_potential_landscape,
    plot_therapy_intervention,
    plot_phase_diagram,
    plot_metric_geometry,
    plot_lyapunov_stability,
    plot_signed_coherence_analysis,
    plot_numerical_dashboard
)

st.set_page_config(page_title="Geometry of Awareness Visuals", layout="wide")
st.title("🧠 Geometry of Awareness — Interactive Visualizations & Analysis")
st.markdown("""
Comprehensive visualization suite for the Geometry of Awareness framework v1.3.
Includes dynamic simulations, basin analysis, therapy interventions, and signed coherence metrics.
""")

viz_dir = os.path.join(os.path.dirname(__file__), 'visualizations')
images = sorted([f for f in os.listdir(viz_dir) if f.lower().endswith('.png')])

# Define visualization categories with descriptions
viz_metadata = {
    '01_dynamics_trajectory.png': {
        'title': '1. Dynamics Trajectory',
        'category': 'Core Dynamics',
        'description': '''
**4-panel display of fundamental system behavior:**
- **Phase Space (Top-Left):** 2D projection of state trajectory showing how awareness 
  evolves through Emotion-Memory space. Green dot = start, Red X = end.
- **Energy Landscape (Top-Right):** Potential V(x) descending over time. System "rolls" 
  down the energy surface via metric-weighted gradient descent.
- **Salience Gate (Bottom-Left):** Learning rate λ(t) modulation. Controls how strongly 
  Hebbian coherence updates reinforce. Higher λ = faster learning.
- **Metric Stability (Bottom-Right):** Condition number κ(g) showing metric anisotropy. 
  Increasing κ means some dimensions become "shorter" in the metric geometry.
        '''
    },
    '02_potential_landscape.png': {
        'title': '2. Potential Landscape & Basin Geometry',
        'category': 'Basin Structure',
        'description': '''
**Equilibrium landscape visualization:**
- **3D Surface (Left):** Potential V(x) as a function of Emotion and Memory coordinates. 
  Three attracting wells (Healthy, Rigid, Trauma) and one repulsive bump (Trauma).
- **Contour Map (Right):** 2D view with basin locations:
  - ⭐ **Green star:** Healthy basin (positive, integrated state)
  - ✖️ **Red X:** Rigid basin (constraint-dominated)
  - 🟪 **Purple square:** Trauma basin (dissociative/avoidant)
  
The contours show equipotential lines. Dynamics follow gradient descent on this surface.
        '''
    },
    '03_therapy_intervention.png': {
        'title': '3. Therapy Intervention Protocol',
        'category': 'Therapeutic Applications',
        'description': '''
**Pre/during/post therapy effect analysis:**
- **Energy (Top-Left):** Shows V(x) descent interrupted by intervention (red dashed line). 
  Post-therapy energy may oscillate or stabilize differently.
- **Salience (Top-Right):** Pre-therapy λ is free-ranging; during therapy (green shaded), 
  λ is constrained to [0.4, 0.78] for guided, moderate learning.
- **Metric Anisotropy (Bottom-Left):** Quantifies how therapy changes κ(g). 
  Increasing κ = metric becomes more "specialized" (anisotropic). 
  This represents learning-induced structural reorganization.
- **Basin Migration (Bottom-Right):** Trajectory in state space shifts toward 
  Healthy basin (green star) during therapy.
        '''
    },
    '04_phase_diagram.png': {
        'title': '4. Phase Diagram: Trust vs Trauma Parameter Space',
        'category': 'Parameter Sensitivity',
        'description': '''
**6×6 grid exploring basin probabilities across parameter ranges:**
- **Left Panel (Healthy):** Probability P(H) of reaching Healthy basin. 
  Greener = higher P(H). Peak at high trust, low trauma.
- **Middle Panel (Rigid):** Probability P(R) reaches Rigid basin. 
  Redder = higher P(R). Complements P(H) to suggest rigid trapping.
- **Right Panel (Trauma):** Probability P(T) falls into Trauma basin. 
  Purpler = higher P(T). Shows trauma susceptibility under low trust + high trauma amplitude.

**Axes:**
- X-axis: Trust Base (τ) ∈ [0.3, 0.85]. Higher trust biases toward Healthy.
- Y-axis: Trauma Amplitude (w_T) ∈ [1.5, 8.0]. Higher amplitude deepens Trauma well.
        '''
    },
    '05_metric_geometry.png': {
        'title': '5. Metric Geometry: Anisotropy & Curvature',
        'category': 'Riemannian Structure',
        'description': '''
**Manifold properties arising from learned coherence:**
- **Left (Condition Number):** κ(g) evolution along trajectory. 
  High κ indicates metric anisotropy—some dimensions move faster/slower. 
  This encodes which dimension pairs are tightly coupled.
- **Right (Christoffel Norm):** ||Γ|| quantifies geodesic deviation. 
  Measures how "curved" the manifold is locally. High curvature regions 
  are where straight-line paths become curved when pulled back to the manifold.

Increasing ||Γ|| suggests strong coherence gradients = steep learning regions.
        '''
    },
    '06_lyapunov_stability.png': {
        'title': '6. Lyapunov Stability: Eigenvalue Distributions',
        'category': 'Stability Analysis',
        'description': '''
**Discrete-time stability for basin attractors:**
- **Each of 4 panels:** Basin (Healthy or Rigid) × Trust level (0.4 or 0.8)
- **Bar heights:** Magnitude of Jacobian eigenvalues |λ_i|, sorted descending.
- **Green bars:** Eigenvalues |λ| < 1 (stable contribution).
- **Red bars:** Eigenvalues |λ| > 1 (would cause divergence—if present, basin unstable).
- **Red dashed line:** Stability boundary at ρ(J)=1. Title shows spectral radius ρ(J).

**Interpretation:** ρ(J) < 1 ⟹ basin is attracting (nearby trajectories spiral inward).
ρ(J) > 1 ⟹ basin is repelling (unstable).
        '''
    },
    '07_signed_coherence_v13.png': {
        'title': '7. Signed Coherence Analysis (v1.3)',
        'category': 'v1.3 Inhibitory Dynamics',
        'description': '''
**NEW in v1.3: Inhibitory couplings and repulsion dynamics.**

**Top Row (Metrics Over Time):**
- **Left:** Signed Fraction = % of C_ij < 0 (negative/inhibitory). 
  Tracks how much of the coupling network becomes repulsive.
- **Middle:** Inhibitory Strength = Σ|C_ij<0|. Total magnitude of inhibitory interactions.
- **Right:** V_inhib trajectory. Inhibitory potential energy from repulsive couplings.

**Middle Row (Coherence Heatmaps):**
- **Left:** C(t=0)—initial coherence with demo seed C[0,2]=-0.45 (Emotion ↔ Narrative repulsion).
- **Middle:** C(t=final)—learned coherence after 800 steps.
- **Right:** ΔC—change in couplings. Red/blue indicates where repulsion reinforced/weakened.

**Bottom Row:**
- **Left:** Emotion-Narrative repulsion wedge. Phase space shows x₀ vs x₂ trajectory.
  Repulsion forces them apart.
- **Middle:** Count of negative couplings over time.
- **Right:** Summary box with key v1.3 metrics (signed_frac, inhib_strength, V_inhib, counts).
        '''
    },
    '08_numerical_dashboard_v13.png': {
        'title': '8. Numerical Dashboard (v1.3)',
        'category': 'v1.3 Integrated Metrics',
        'description': '''
**NEW in v1.3: Comprehensive 12-plot dashboard integrating all key metrics.**

**Row 1 (Energy & Forces - 4 plots):**
- Basin potentials (V_H, V_R, V_T) + total
- Inhibitory potential V_inhib with fill-in
- Salience gate λ(t) learning modulation
- Metric condition number κ(g) anisotropy

**Row 2 (v1.3 Metrics - 4 plots):**
- Signed fraction evolution (% negative C_ij)
- Inhibitory strength evolution (Σ|C_ij<0|)
- State norm ||x(t)|| magnitude
- Gradient magnitude ||∇V|| steepness

**Row 3 (Correlation & Summary):**
- Dual-axis plot: V_total vs Signed Fraction showing energy-repulsion coupling
- **Formatted statistics table** with 10 summary metrics:
  Mean V_total, Final signed_frac, Mean inhib_strength, Max λ, Mean κ(g), 
  Total V_inhib, Mean ||x||, Mean ||∇V||, Final ||x||, Basin stability status
        '''
    },
    '09_leapfrog_result.png': {
        'title': '9. Riemannian Leapfrog Integrator Result',
        'category': 'Advanced Integration Methods',
        'description': '''
**Phase space exploration via symplectic Riemannian leapfrog integrator.**

Same 4-panel display as Fig. 1, but generated using velocity-Verlet integration 
on a learned Riemannian manifold (metric g = I + α·L from signed coherence matrix).

**Key Differences from Standard Dynamics (Fig. 1):**
- **Integration Method:** Velocity-Verlet (symplectic) vs. gradient descent (dissipative)
- **Metric Learning:** Uses dynamic metric g(t) derived from coherence C(t) via Laplacian
- **Phase Space:** Preserves more structure due to momentum v(t)—trajectories explore 
  more extensively via kinetic energy, not just descending gradient
- **Coherence Updates:** Hebbian learning Δ C = (1-ρ)C + η·λ·(x⊗x) after each position step
- **Stochastic Exploration:** Includes Brownian noise for escaping local minima

**Physical Interpretation:**
The leapfrog integrator is like "rolling a ball" (with momentum) down an anisotropic 
terrain (metric-weighted landscape), whereas gradient descent is like "sliding downhill" 
directly. The leapfrog method better captures oscillatory and basin-hopping dynamics.

**Statistics Panels:**
- **Phase-Space Trajectory:** May show wider excursions due to kinetic energy
- **Energy:** Exhibits symplectic behavior—may oscillate but preserves phase volume
- **Salience λ(t):** Learning gate controlling coherence update strength
- **Metric Stability κ(g):** How anisotropic the learned metric becomes during integration
        '''
    }
}

# Organize by category
categories = {}
for img, meta in viz_metadata.items():
    cat = meta['category']
    if cat not in categories:
        categories[cat] = []
    categories[cat].append((img, meta))

with st.sidebar:
    st.header("📋 Navigation & Controls")
    st.markdown("---")
    
    st.subheader("Display Mode")
    display_mode = st.radio("Choose view:", 
        options=["All Visualizations", "By Category", "Single Select", "Metrics & Analysis"],
        index=0)
    
    st.markdown("---")
    st.subheader("Model Configuration")
    model_n = st.selectbox("Model dimension (n)", options=[7, 15], index=0)
    st.caption("n=7: core emotions + memories. n=15: extended with somatic/cognitive dimensions")
    
    model_seed = st.number_input("Random seed", value=42, min_value=0, step=1)
    
    st.markdown("---")
    st.subheader("Visualization Parameters")
    dynamics_steps = st.slider("Dynamics steps", min_value=100, max_value=2000, value=1000, step=100)
    
    st.markdown("---")
    st.subheader("🎬 Generation Control")
    col_gen, col_stop = st.columns(2)
    with col_gen:
        if st.button("▶️ Generate Visualizations", key='generate_button', use_container_width=True):
            st.session_state.generate_viz = True
    with col_stop:
        if st.button("⏹️ Stop", key='stop_button', use_container_width=True):
            st.session_state.generate_viz = False
    
    # Initialize generate state if not exists
    if 'generate_viz' not in st.session_state:
        st.session_state.generate_viz = False
    
    st.markdown("---")
    st.subheader("Stability Analysis")
    if st.button("🔧 Compute Lyapunov Matrix", key='lyapunov_button'):
        st.session_state.run_lyapunov = True
    
    st.markdown("---")
    st.subheader("Help")
    if st.checkbox("Show metric definitions"):
        st.info("""
**Key Metrics Explained:**
- **κ(g):** Metric condition number. Higher = more anisotropic (specialized dimensions).
- **ρ(J):** Spectral radius of Jacobian. ρ<1 = stable/attracting. ρ>1 = unstable/repelling.
- **λ(t):** Salience gate. Controls learning rate; higher λ = stronger Hebbian updates.
- **V(x):** Total potential = V_H + V_R + V_T + β·V_inhib. Lower = more stable state.
- **Signed_frac (v1.3):** % of couplings that are negative (inhibitory).
- **Inhib_strength (v1.3):** Σ|C_ij| for negative C_ij. Measures total repulsion.
        """)

# ============================================================================
# Helper Function: Generate Leapfrog Plot
# ============================================================================
def generate_leapfrog_plot(n=7, seed=42, n_steps=1000):
    """Generate dynamics plot using Riemannian leapfrog integrator."""
    model = GeometryOfAwareness(n=n, seed=seed)
    integrator = create_integrator(model, use_christoffel=False)
    x, v = np.random.uniform(-0.3, 0.3, n), 0.05*np.random.randn(n)
    for _ in range(n_steps):
        x, v = integrator.step(x, v)
    return plot_dynamics_trajectory(model, n_steps=n_steps)

# Main display
if display_mode == "All Visualizations":
    st.subheader("All 9 Visualizations (v1.2 + v1.3 + Advanced Integration)")
    
    if not st.session_state.generate_viz:
        st.info("👆 Click **Generate Visualizations** button in sidebar to start")
    else:
        st.markdown("**Generating visualizations dynamically...**")
        
        progress_bar = st.progress(0)
        
        # Initialize shared model
        model = GeometryOfAwareness(n=model_n, seed=model_seed)
        
        visualizations = [
            ("1. Dynamics Trajectory", lambda: plot_dynamics_trajectory(model, n_steps=dynamics_steps)),
            ("2. Potential Landscape", lambda: plot_potential_landscape(model)),
            ("3. Therapy Intervention", lambda: plot_therapy_intervention(model)),
            ("4. Phase Diagram", lambda: plot_phase_diagram(model, n_trust=6, n_trauma=6)),
            ("5. Metric Geometry", lambda: plot_metric_geometry(model, n_steps=500)),
            ("6. Lyapunov Stability", lambda: plot_lyapunov_stability(model)),
            ("7. Signed Coherence (v1.3)", lambda: plot_signed_coherence_analysis(model, n_steps=800)),
            ("8. Numerical Dashboard (v1.3)", lambda: plot_numerical_dashboard(model, n_steps=500)),
        ]
        
        total = len(visualizations)
        for idx, (title, plot_func) in enumerate(visualizations):
            if st.session_state.generate_viz:
                with st.spinner(f"Generating {title}..."):
                    fig = plot_func()
                    st.markdown(f"#### {title}")
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                progress_bar.progress((idx + 1) / (total + 1))
            else:
                st.warning("Generation stopped by user.")
                break
        
        # Leapfrog integrator visualization
        if st.session_state.generate_viz:
            with st.spinner("Generating 9. Riemannian Leapfrog Integrator Result..."):
                model_lf = GeometryOfAwareness(n=model_n, seed=model_seed)
                integrator = create_integrator(model_lf, use_christoffel=False)
                x, v = np.random.uniform(-0.3, 0.3, model_n), 0.05*np.random.randn(model_n)
                for _ in range(dynamics_steps):
                    x, v = integrator.step(x, v)
                fig9 = plot_dynamics_trajectory(model_lf, n_steps=dynamics_steps)
                st.markdown("#### 9. Riemannian Leapfrog Integrator Result")
                st.pyplot(fig9, use_container_width=True)
                plt.close(fig9)
        
        progress_bar.progress(1.0)
        st.success("✅ All visualizations generated!")
        st.markdown("---")

elif display_mode == "By Category":
    st.subheader("Browse by Category")
    selected_cat = st.radio("Select category:", list(categories.keys()))
    
    if not st.session_state.generate_viz:
        st.info("👆 Click **Generate Visualizations** button in sidebar to start")
    else:
        # Initialize model for this view
        model = GeometryOfAwareness(n=model_n, seed=model_seed)
        
        # Mapping of categories to plot functions
        plot_functions = {
            'Core Dynamics': [
                ("1. Dynamics Trajectory", lambda m=model: plot_dynamics_trajectory(m, n_steps=dynamics_steps)),
            ],
            'Basin Structure': [
                ("2. Potential Landscape", lambda m=model: plot_potential_landscape(m)),
            ],
            'Therapeutic Applications': [
                ("3. Therapy Intervention", lambda m=model: plot_therapy_intervention(m)),
            ],
            'Parameter Sensitivity': [
                ("4. Phase Diagram", lambda m=model: plot_phase_diagram(m, n_trust=6, n_trauma=6)),
            ],
            'Riemannian Structure': [
                ("5. Metric Geometry", lambda m=model: plot_metric_geometry(m, n_steps=500)),
            ],
            'Stability Analysis': [
                ("6. Lyapunov Stability", lambda m=model: plot_lyapunov_stability(m)),
            ],
            'v1.3 Inhibitory Dynamics': [
                ("7. Signed Coherence Analysis", lambda m=model: plot_signed_coherence_analysis(m, n_steps=800)),
            ],
            'v1.3 Integrated Metrics': [
                ("8. Numerical Dashboard", lambda m=model: plot_numerical_dashboard(m, n_steps=500)),
            ],
            'Advanced Integration Methods': [
                ("9. Riemannian Leapfrog", lambda: generate_leapfrog_plot(model_n, model_seed, dynamics_steps)),
            ]
        }
        
        if selected_cat in plot_functions:
            for title, plot_func in plot_functions[selected_cat]:
                st.markdown(f"### {title}")
                if title in viz_metadata:
                    meta = viz_metadata[list([m for m in viz_metadata.items() if m[1]['title'] == title])[0][0]]
                    st.markdown(meta['description'])
                with st.spinner(f"Generating {title}..."):
                    fig = plot_func()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                st.markdown("---")
        
        st.success("✅ Category visualizations generated!")

elif display_mode == "Single Select":
    st.subheader("Select a Single Visualization")
    
    # Create a mapping of titles to filenames
    title_to_file = {}
    for img, meta in viz_metadata.items():
        title_to_file[meta['title']] = img
    
    selected_title = st.selectbox("Choose visualization:", sorted(title_to_file.keys()))
    selected_img = title_to_file[selected_title]
    selected_meta = viz_metadata[selected_img]
    
    st.markdown(f"### {selected_meta['title']}")
    st.markdown(f"**Category:** {selected_meta['category']}")
    st.markdown(selected_meta['description'])
    
    if not st.session_state.generate_viz:
        st.info("👆 Click **Generate Visualizations** button in sidebar to start")
    else:
        # Generate the selected visualization
        with st.spinner(f"Generating {selected_meta['title']}..."):
            model = GeometryOfAwareness(n=model_n, seed=model_seed)
            
            # Map title to plot function
            if "Dynamics Trajectory" in selected_title and "Leapfrog" not in selected_title:
                fig = plot_dynamics_trajectory(model, n_steps=dynamics_steps)
            elif "Potential Landscape" in selected_title:
                fig = plot_potential_landscape(model)
            elif "Therapy" in selected_title:
                fig = plot_therapy_intervention(model)
            elif "Phase Diagram" in selected_title:
                fig = plot_phase_diagram(model, n_trust=6, n_trauma=6)
            elif "Metric Geometry" in selected_title:
                fig = plot_metric_geometry(model, n_steps=500)
            elif "Lyapunov" in selected_title:
                fig = plot_lyapunov_stability(model)
            elif "Signed Coherence" in selected_title:
                fig = plot_signed_coherence_analysis(model, n_steps=800)
            elif "Numerical Dashboard" in selected_title:
                fig = plot_numerical_dashboard(model, n_steps=500)
            elif "Leapfrog" in selected_title:
                fig = generate_leapfrog_plot(model_n, model_seed, dynamics_steps)
            else:
                st.error("Visualization not found")
                fig = None
            
            if fig is not None:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        
        st.success("✅ Visualization generated!")

elif display_mode == "Metrics & Analysis":
    st.subheader("📊 Interactive Metrics & Stability Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Lyapunov Stability Matrix")
        st.markdown("""
Compute the discrete-time Jacobian eigenvalues (spectral radius ρ(J)) 
at basin equilibria under various trust levels.
        """)
        
        if st.button("🔄 Compute Now"):
            with st.spinner("Computing Lyapunov analysis..."):
                model = GeometryOfAwareness(n=model_n, seed=model_seed)
                basins = ['H', 'R']
                trusts = [0.4, 0.8]
                rows = []
                for b in basins:
                    for t in trusts:
                        res = model.lyapunov_analysis(basin=b, trust=t, n_steps=200)
                        rho = res['max_abs_eigenvalue']
                        stable_str = "✅ STABLE" if res['stable'] else "❌ UNSTABLE"
                        rows.append({
                            'Basin': {'H': 'Healthy', 'R': 'Rigid'}[b],
                            'Trust': f"{t:.1f}",
                            'ρ(J)': f"{rho:.4f}",
                            'Status': stable_str
                        })
                
                st.session_state.lyapunov_results = rows
        
        if 'lyapunov_results' in st.session_state:
            st.table(st.session_state.lyapunov_results)
            st.success("Analysis complete. ρ(J) < 1.0 indicates basin stability.")
    
    with col2:
        st.markdown("### Model Statistics")
        
        if st.button("🎲 Initialize Model & Summary"):
            model = GeometryOfAwareness(n=model_n, seed=model_seed)
            
            st.metric("Dimension", f"n = {model_n}")
            st.metric("Coherence Matrix Shape", f"{model_n} × {model_n}")
            
            # Compute initial conditions
            g_cond = np.linalg.cond(model.g)
            st.metric("Initial κ(g)", f"{g_cond:.3f}")
            
            V_h, _ = model.potential(model.mu_H)
            V_r, _ = model.potential(model.mu_R)
            V_t, _ = model.potential(model.mu_T)
            
            st.metric("V at Healthy basin", f"{V_h:.4f}")
            st.metric("V at Rigid basin", f"{V_r:.4f}")
            st.metric("V at Trauma basin", f"{V_t:.4f}")
            
            # v1.3 metrics
            st.markdown("**v1.3 Signed Coherence Metrics:**")
            st.metric("Signed Fraction (initial)", f"{model.signed_fraction():.4f}")
            st.metric("Inhibitory Strength (initial)", f"{model.inhibitory_strength():.4f}")

st.markdown("---")
st.markdown("""
## 📚 Documentation & Theory

For technical details on the framework, visualizations, and stability analysis, 
consult these guides:
- **VISUALIZATION_GUIDE.py** — Understanding the plots and discrete-time stability
- **MATHEMATICAL_APPENDIX.md** — Full Riemannian geometry and v1.3 signed couplings (expandable below)
- **VISUALIZATION_DASHBOARD_UPDATE.md** — v1.3 integration details
""")

# Display the mathematical appendix (external markdown file)
with st.expander('📖 MATHEMATICAL APPENDIX (v1.3 Complete Framework)', expanded=False):
    try:
        appendix_path = os.path.join(os.path.dirname(__file__), 'MATHEMATICAL_APPENDIX.md')
        if os.path.exists(appendix_path):
            with open(appendix_path, 'r', encoding='utf-8') as f:
                appendix_md = f.read()
            st.markdown(appendix_md, unsafe_allow_html=False)
        else:
            st.error('MATHEMATICAL_APPENDIX.md not found in project root.')
    except Exception as e:
        st.error(f'Error loading appendix: {e}')
