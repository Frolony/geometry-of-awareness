# v1.3 Visualization Integration - Quick Reference

## New Files Generated

All files are automatically created in `visualizations/` directory when you run:
```bash
python visualization_dashboard.py
```

---

## The 2 New Visualizations (v1.3 Specific)

### **Figure 7: Signed Coherence Analysis** (`07_signed_coherence_v13.png`)

**What it shows:** Signed coherence dynamics, inhibitory potentials, and coupling evolution

```
┌─────────────────────────────────────────────────────────────────┐
│  SIGNED FRACTION        INHIB. STRENGTH       POTENTIAL COMPARE │
│  (% negative C_ij)      (sum |C_ij<0|)        (V_total vs V_in) │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  │ Red line rising  │  │ Orange line      │  │ Blue: V_total    │
│  │ = learning       │  │ growing = more   │  │ Red --: V_inhib  │
│  │   repulsion      │  │ repulsion        │  │ Shows how much   │
│  │                  │  │                  │  │ inhibition       │
│  │                  │  │                  │  │ contributes      │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  │ INIT. C MATRIX   │  │ FINAL C MATRIX   │  │ CHANGE ΔC        │
│  │ (demo seed)      │  │ (after learning) │  │ (final-initial)  │
│  │ Heatmap shows    │  │ Heatmap shows    │  │ Shows which      │
│  │ C[0,2]=-0.45     │  │ evolved couplings│  │ couplings grew/  │
│  │ (Emotion-Narra.)  │  │ including new    │  │ weakened         │
│  │                  │  │ negative ones    │  │                  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  │ E-N REPULSION    │  │ NEGATIVE COUNT   │  │ SUMMARY TABLE    │
│  │ (x₀ vs x₂)       │  │ (inhibitory      │  │ ╔════════════════╗│
│  │ Phase space:     │  │  links)          │  │ ║ Signed Frac   ││
│  │ Shows trajectory │  │ Evolution of #   │  │ ║ Inhib Str     ││
│  │ avoiding dual    │  │ negative C_ij    │  │ ║ V_inhib final ││
│  │ high state       │  │                  │  │ ║ Neg links     ││
│  │                  │  │                  │  │ ║ Mean C_neg    ││
│  └──────────────────┘  └──────────────────┘  └──────────────────┘
└─────────────────────────────────────────────────────────────────┘
```

**Key Metrics Tracked:**
- Signed Fraction: % of C_ij that are negative
- Inhibitory Strength: Σ|C_ij| where C_ij < 0
- V_inhib: Inhibitory potential over time
- Coherence change: Which couplings evolved

---

### **Figure 8: Numerical Dashboard** (`08_numerical_dashboard_v13.png`)

**What it shows:** Comprehensive multi-metric integration with statistical summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ENERGY & POTENTIALS                                 │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────┐│
│  │ Basin         │  │ Inhibitory    │  │ Salience      │  │ Metric   ││
│  │ Potentials    │  │ Potential     │  │ Gate λ        │  │ Cond(g)  ││
│  │ V_H/R/T       │  │ V_inhib (v1.3)│  │ Learning Gate │  │ κ(g)     ││
│  │ tracking      │  │ Repulsion     │  │ Modulation    │  │ Anis-    ││
│  │ descent       │  │ over time     │  │ salience      │  │ otropy   ││
│  └───────────────┘  └───────────────┘  └───────────────┘  └──────────┘│
│                  V1.3 SPECIFIC METRICS                                   │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────┐│
│  │ Signed        │  │ Inhibitory    │  │ State Norm    │  │ Gradient ││
│  │ Fraction      │  │ Strength      │  │ ||x(t)||      │  │ ||∇V||   ││
│  │ frac_sig (%)  │  │ S_inhib       │  │ Magnitude     │  │ Steep-   ││
│  │ % negative    │  │ Total repul.  │  │ of config     │  │ ness     ││
│  │ couplings     │  │ magnitude     │  │               │  │          ││
│  └───────────────┘  └───────────────┘  └───────────────┘  └──────────┘│
│                          CORRELATIONS & SUMMARY                          │
│  ┌──────────────────────────────────┐  ┌──────────────────────────────┐│
│  │ V_total vs Signed Fraction       │  │ NUMERICAL SUMMARY TABLE      ││
│  │ (Dual-axis plot)                 │  │ ╔════════════════════════════╗││
│  │ Shows relationship between       │  │ ║ Mean V_total        : 0.xxx ║││
│  │ potential energy and inhibitory  │  │ ║ Final Signed Frac   : 0.xxx ║││
│  │ coupling evolution               │  │ ║ Mean Inhib Strength : 0.xxx ║││
│  │                                  │  │ ║ Max Salience        : 0.xxx ║││
│  │ ─ = Energy trajectory            │  │ ║ Mean κ(g)           : x.xxx ║││
│  │ - - = Inhibitory evolution       │  │ ║ Total V_inhib       : 0.xxx ║││
│  │                                  │  │ ║ Mean ||x||          : 0.xxx ║││
│  │                                  │  │ ║ Mean ||∇V||         : 0.xxx ║││
│  │                                  │  │ ║ Final ||x||         : 0.xxx ║││
│  │                                  │  │ ║ Basin Stability     : Valid ║││
│  │                                  │  │ ╚════════════════════════════╝││
│  └──────────────────────────────────┘  └──────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

**Components:**
- **Row 1:** Basin potentials, inhibitory dynamics, salience gating, metric conditioning
- **Row 2:** v1.3-specific metrics (signed fraction, inhibitory strength, state/gradient norms)
- **Row 3:** Correlation plot (V vs Signed Fraction) + formatted numerical summary table

---

## Data Flow: From Model → Visualizations

```
geometry_of_awareness.py (v1.3)
         ↓
    step(x) ────────────→ x, λ, κ(g)
         ↓
  signed_fraction() ──→ % negative couplings
         ↓
inhibitory_strength()→ Σ|C_ij<0|
         ↓
  potential(x) ───────→ (V_total, V_H, V_R, V_T, V_inhib)
         ↓
  history dict ────────→ tracked over time
         ↓
visualization_dashboard.py
         ↓
    Figure 7 (Signed Coherence)
    Figure 8 (Numerical Dashboard)
         ↓
   PNG output (150 DPI)
```

---

## Sample Output: What to Expect

### When you run `python visualization_dashboard.py`:

```
======================================================================
VISUALIZATION DASHBOARD: Geometry of Awareness v1.3
======================================================================

[1] Generating dynamics trajectory...
    [OK] Saved: visualizations/01_dynamics_trajectory.png

[2] Generating potential landscape...
    [OK] Saved: visualizations/02_potential_landscape.png

[3] Generating therapy intervention...
    [OK] Saved: visualizations/03_therapy_intervention.png

[4] Generating phase diagram...
    Computing phase diagram (6×6 grid)...
    [OK] Saved: visualizations/04_phase_diagram.png

[5] Generating metric geometry properties...
    [OK] Saved: visualizations/05_metric_geometry.png

[6] Generating Lyapunov stability analysis...
    [OK] Saved: visualizations/06_lyapunov_stability.png

[7] Generating signed coherence analysis (v1.3)...
    [OK] Saved: visualizations/07_signed_coherence_v13.png

[8] Generating numerical dashboard (v1.3)...
    [OK] Saved: visualizations/08_numerical_dashboard_v13.png

To view: Open PNG files in Windows Explorer or your image viewer
======================================================================
```

**Total Execution Time:** ~55 seconds  
**Total Output Size:** ~2.0 MB (8 PNG files)

---

## Integration: What v1.3 Features Are Displayed?

| v1.3 Feature | Figure 7 | Figure 8 | How |
|--------------|----------|----------|-----|
| Signed coherence C_ij ∈ ℝ | ✅ Heatmaps | ✅ Tracked | Matrices + evolution |
| Inhibitory potentials V_inhib | ✅ Trajectory | ✅ Plot | Component energy |
| Negative couplings (C_ij < 0) | ✅ Fraction | ✅ Stats | Count + strength |
| Emotion-Narrative repulsion (C[0,2]=-0.45) | ✅ Phase space | — | Repulsion wedge |
| signed_fraction() method | ✅ Metric plot | ✅ Summary | Real-time % negative |
| inhibitory_strength() method | ✅ Metric plot | ✅ Summary | Real-time Σ\|C<0\| |

---

## Where Are the Files?

**Generated visualizations located at:**
```
c:\Users\Guestie\Desktop\geometry-of-awareness\visualizations\
├── 01_dynamics_trajectory.png
├── 02_potential_landscape.png
├── 03_therapy_intervention.png
├── 04_phase_diagram.png
├── 05_metric_geometry.png
├── 06_lyapunov_stability.png
├── 07_signed_coherence_v13.png          ← NEW (v1.3 metrics)
└── 08_numerical_dashboard_v13.png       ← NEW (v1.3 integration)
```

---

## Customization

To modify the simulations, edit parameters in `main()`:

```python
# Change model dimensionality
model = GeometryOfAwareness(n=7, seed=42)  # n=7 or n=15

# Change simulation steps
fig7 = plot_signed_coherence_analysis(model, n_steps=800)      # 800 steps
fig8 = plot_numerical_dashboard(model, n_steps=500)            # 500 steps
```

---

## Summary

✅ **2 new comprehensive v1.3 visualizations created**  
✅ **8 total PNG files generated with numerical data integration**  
✅ **All v1.3 metrics captured: signed coherence, inhibitory potentials, negative couplings**  
✅ **Publication-quality output at 150 DPI**  
✅ **Backward compatible: original 6 plots still included**

