"""
VISUALIZATION GUIDE - Geometry of Awareness v1.2
==============================================

All visualization outputs are saved in: visualizations/

Each PNG file is ~300-450 KB and can be opened with:
  - Windows built-in image viewer
  - Any browser (drag & drop)
  - Adobe Preview, Paint, or any image viewer

Below is a description of each visualization:
"""

VISUALIZATIONS = {
    "01_dynamics_trajectory.png": {
        "file_size": "451 KB",
        "plots": [
            {
                "name": "Phase-Space Trajectory (2D Projection)",
                "position": "Top-Left",
                "shows": [
                    "- X-axis: Emotion (x0) dimension",
                    "- Y-axis: Memory (x1) dimension", 
                    "- Blue line: path through state space over 1000 steps",
                    "- Green dot: Starting position",
                    "- Red dot: Ending position"
                ],
                "interpret": "Shows how the system flows through the awareness manifold"
            },
            {
                "name": "Energy Landscape Descent",
                "position": "Top-Right",
                "shows": [
                    "- X-axis: Step number (0-1000)",
                    "- Y-axis: Total potential energy V(x)",
                    "- Black curve: Energy over time",
                    "- Shaded area: Shows energy reduction"
                ],
                "interpret": "System descends potential landscape; lower energy = more stable state"
            },
            {
                "name": "Learning Rate Modulation (Salience)",
                "position": "Bottom-Left",
                "shows": [
                    "- X-axis: Step number",
                    "- Y-axis: Salience gate λ(t) in [0, 1]",
                    "- Purple curve: Modulation strength",
                    "- Higher λ = stronger learning"
                ],
                "interpret": "Salience gates determine when metric learning happens"
            },
            {
                "name": "Metric Stability",
                "position": "Bottom-Right",
                "shows": [
                    "- X-axis: Step number",
                    "- Y-axis: cond(g) - condition number (log scale)",
                    "- Orange curve: Metric health",
                    "- Lower = better conditioned (more stable)"
                ],
                "interpret": "Laplacian metric remains well-conditioned throughout"
            }
        ]
    },
    
    "02_potential_landscape.png": {
        "file_size": "308 KB",
        "plots": [
            {
                "name": "3D Potential Surface",
                "position": "Left",
                "shows": [
                    "- X-axis: Emotion dimension",
                    "- Y-axis: Memory dimension",
                    "- Z-axis (height): Potential V(x)",
                    "- Colors: Viridis colormap (low=dark, high=bright)",
                    "- Three basin structure visible"
                ],
                "interpret": "3D visualization of the competing potential wells"
            },
            {
                "name": "Basin Geometry (Contour Map)",
                "position": "Right",
                "shows": [
                    "- Gray contour lines: Iso-potential curves",
                    "- Green star (*): Healthy basin center (μH)",
                    "- Red X: Rigid basin center (μR)",
                    "- Purple square: Trauma repulsor center (μT)",
                    "- Numbers on contours: Potential values"
                ],
                "interpret": "Shows three competing attractors and their basins of attraction"
            }
        ]
    },
    
    "03_therapy_intervention.png": {
        "file_size": "388 KB",
        "plots": [
            {
                "name": "Energy Landscape Before & After Therapy",
                "position": "Top-Left",
                "shows": [
                    "- Blue line: Pre-therapy equilibration (first 300 steps)",
                    "- Green line: Post-therapy intervention (next 150 steps)",
                    "- Red dashed line: Intervention start marker",
                    "- Energy typically increases then stabilizes at higher baseline"
                ],
                "interpret": "Therapy perturbs system, trust lift explores different regions"
            },
            {
                "name": "Salience Gating During Therapy",
                "position": "Top-Right",
                "shows": [
                    "- Navy curve: Pre-therapy salience",
                    "- Lime curve: Post-therapy salience",
                    "- Light lime band: [0.4, 0.78] therapy-guided band",
                    "- Shows how therapy constrains learning rate"
                ],
                "interpret": "Therapy_mode clips salience to moderate [0.4-0.78] band"
            },
            {
                "name": "Metric Stability Improvement",
                "position": "Bottom-Left",
                "shows": [
                    "- Blue dots: Pre-therapy condition numbers",
                    "- Green dots: Post-therapy condition numbers",
                    "- Title shows % improvement",
                    "- Log scale reveals orders of magnitude"
                ],
                "interpret": "Good therapy often reduces metric condition > 20-40%"
            },
            {
                "name": "Basin Migration (State-Space Shift)",
                "position": "Bottom-Right",
                "shows": [
                    "- Blue line: Pre-therapy trajectory towards rigid basin",
                    "- Green line: Post-therapy trajectory shifts healthward",
                    "- Blue/Lime dots: Start/end positions",
                    "- Green star: Healthy basin target"
                ],
                "interpret": "Therapy induces state-space migration toward healthier basin"
            }
        ]
    },
    
    "04_phase_diagram.png": {
        "file_size": "52 KB",
        "plots": [
            {
                "name": "Healthy Basin Probability (Green)",
                "position": "Left",
                "shows": [
                    "- X-axis: Trust Base τ ∈ [0.3, 0.85]",
                    "- Y-axis: Trauma Amplitude w_T ∈ [1.5, 8.0]",
                    "- Brightness: Probability of ending in Healthy basin",
                    "- Each cell: average of 10 Monte Carlo runs, 300 steps"
                ],
                "interpret": "High trust + low trauma favor Healthy basin"
            },
            {
                "name": "Rigid Basin Probability (Red)",
                "position": "Center",
                "shows": [
                    "- X-axis: Trust Base τ",
                    "- Y-axis: Trauma Amplitude w_T",
                    "- Brightness: Probability of ending in Rigid basin",
                    "- Complements Healthy: sum of all basins = 100%"
                ],
                "interpret": "Low trust favors Rigid basin (paradoxical belief system)"
            },
            {
                "name": "Trauma Basin Probability (Purple)",
                "position": "Right",
                "shows": [
                    "- X-axis: Trust Base τ",
                    "- Y-axis: Trauma Amplitude w_T",
                    "- Brightness: Probability of repulsion by trauma",
                    "- Higher w_T (trauma) increases repulsion"
                ],
                "interpret": "Very high trauma creates repulsion; system avoids trauma center"
            }
        ]
    },
    
    "05_metric_geometry.png": {
        "file_size": "80 KB",
        "plots": [
            {
                "name": "Metric Conditioning Along Trajectory",
                "position": "Left",
                "shows": [
                    "- X-axis: Step number (0-500)",
                    "- Y-axis: cond(g) condition number (log scale)",
                    "- Blue curve: Condition number trajectory",
                    "- Fluctuations show metric adaptation"
                ],
                "interpret": "Metric remains well-conditioned (cond <10) throughout dynamics"
            },
            {
                "name": "Riemannian Curvature Proxy R(x)",
                "position": "Right",
                "shows": [
                    "- X-axis: Step number (sampled every 50 steps)",
                    "- Y-axis: Scalar curvature R(x)",
                    "- Red points: Curvature samples",
                    "- Periodic recomputation during long trajectories"
                ],
                "interpret": "R(x) quantifies local manifold curvature; higher = more distorted"
            }
        ]
    },
    
    "06_lyapunov_stability.png": {
        "file_size": "129 KB",
        "plots": [
            {
                "name": "Healthy Basin (Trust=0.4)",
                "position": "Top-Left",
                "shows": [
                    "- Bar height = |λ| magnitude of Jacobian eigenvalues",
                    "- Green bars: |λ| < 1 (stable eigenvalue)",
                    "- Red bars: |λ| >= 1 (unstable)",
                    "- Red dashed line: Stability boundary",
                    "- Title shows max|λ| and STABLE/UNSTABLE"
                ],
                "interpret": "All eigenvalues typically < 1; healthy basin locally attractive"
            },
            {
                "name": "Healthy Basin (Trust=0.8)",
                "position": "Top-Right",
                "shows": [
                    "- Same layout as Top-Left",
                    "- Typically *more* stable at higher trust",
                    "- max|λ| often < 0.9 under high trust"
                ],
                "interpret": "Higher trust reinforces Healthy basin stability"
            },
            {
                "name": "Rigid Basin (Trust=0.4)",
                "position": "Bottom-Left",
                "shows": [
                    "- Eigenvalue spectrum at Rigid (high Belief/Identity) state",
                    "- Typically stable for low-trust scenarios",
                    "- More complex spectrum than Healthy"
                ],
                "interpret": "Rigid basin is equilibrium under low-trust conditions"
            },
            {
                "name": "Rigid Basin (Trust=0.8)",
                "position": "Bottom-Right",
                "shows": [
                    "- Same as Bottom-Left but at high trust",
                    "- Should show *less* stability (good for therapy)",
                    "- max|λ| may approach 1 as trust increases"
                ],
                "interpret": "High trust destabilizes Rigid basin; pushes toward Healthy"
            }
        ]
    }
}

# ============================================================================
# HOW TO VIEW VISUALIZATIONS
# ============================================================================

VIEWING_INSTRUCTIONS = """

Option 1: Windows Explorer (easiest)
====================================
1. Open Windows Explorer
2. Navigate to: C:\\Users\\<YourUsername>\\Desktop\\geometry-of-awareness\\visualizations\\
3. Double-click any .png file
4. Image opens in default viewer (Paint, Photos, etc.)

Option 2: Web Browser
=====================
1. Open any browser (Chrome, Edge, Firefox)
2. Drag & drop PNG file into browser window
3. Image displays in full screen

Option 3: Python (programmatic viewing)
========================================
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('visualizations/01_dynamics_trajectory.png')
plt.figure(figsize=(16, 12))
plt.imshow(img)
plt.axis('off')
plt.show()

Option 4: Generate NEW visualizations
======================================
python visualization_dashboard.py
  (Regenerates all 6 visualizations)

"""

# ============================================================================
# INTERPRETATION GUIDE
# ============================================================================

INTERPRETATION_GUIDE = """

1. DYNAMICS TRAJECTORY (01_*)
=============================
Key metrics to watch:
  - Does energy decrease? (trajectory convergence)
  - Is salience bounded [0, 1]? (learn rate sanity check)
  - Does cond(g) stay < 10? (numerical stability)
  - Do patterns show basin structure? (attractors present)

2. POTENTIAL LANDSCAPE (02_*)
==============================
Key observations:
  - Are three attractors clearly separated?
  - Is trauma repulsor (purple) well-defined?
  - Do contours show smooth manifold (no discontinuities)?
  - Do basins have realistic psychology interpretation?

3. THERAPY INTERVENTION (03_*)
===============================
Key metrics for success:
  - Energy change pre→post? (should stabilize)
  - Does salience shift into [0.4, 0.78] band?
  - Is cond(g) improvement > 0% ? (positive delta)
  - Does trajectory shift healthward? (toward μH)

4. PHASE DIAGRAM (04_*)
========================
Key patterns:
  - Does Healthy dominate high-trust regions?
  - Does Rigid dominate low-trust regions?
  - Is transition sharp or smooth?
  - Does Trauma form repelling region?

5. METRIC GEOMETRY (05_*)
==========================
Key properties:
  - Is cond(g) always > 1? (yes = SPD metric)
  - Does cond(g) stay < 10? (yes = well-conditioned)
  - Is R(x) positive? (positive curvature)
  - Are there spikes in R? (near trauma center)

6. LYAPUNOV STABILITY (06_*)
=============================
Key interpretation:
  - Do all eigenvalues satisfy |λ| < 1? (Lyapunov stable)
  - Which basin is more stable?
  - How does trust affect stability?
  - Are there zero or negative eigenvalues?

"""

# ============================================================================
# QUICK REFERENCE
# ============================================================================

DIMENSIONS_EXPLAINED = """

n=7 Dimensions (Original):
=========================
0. Emotion          (affective valence)
1. Memory           (autobiographical + semantic)
2. Narrative        (life story coherence)
3. Belief           (epistemic + religious)
4. Identity         (self-concept)
5. Archetypal       (universal symbols)
6. Sensory          (embodied experience)

n=15 Dimensions (Extended):
===========================
[0-6]: Original 7 above
7.  Somatic         (body awareness)
8.  Cognitive       (reasoning capacity)
9.  Social          (relational capacity)
10. Spiritual       (transcendence)
11. Motor           (action capacity)
12. Aesthetic       (beauty appreciation)
13. Temporal        (time perception)
14. Spatial         (spatial reasoning)

Basin Centers (μH, μR, μT):
=============================
Healthy (μH):       Balanced across all dimensions
Rigid (μR):         High on Belief/Identity (fixed beliefs)
Trauma (μT):        Negated on most dimensions (fragmentation)

"""

if __name__ == '__main__':
    print("\n" + "="*80)
    print("VISUALIZATION GUIDE - Geometry of Awareness v1.2")
    print("="*80)
    
    print("\nGENERATED FILES:")
    print("-" * 80)
    for fname, info in VISUALIZATIONS.items():
        print(f"\n{fname} ({info['file_size']})")
        for plot in info['plots']:
            print(f"  [{plot['position']}] {plot['name']}")
    
    print("\n" + "-"*80)
    print(VIEWING_INSTRUCTIONS)
    
    print("\n" + "-"*80)
    print(INTERPRETATION_GUIDE)
    
    print("\n" + "-"*80)
    print(DIMENSIONS_EXPLAINED)
    
    print("="*80 + "\n")
