"""Test that example imports and basic functionality works"""
import sys
sys.path.insert(0, '..')

from geometry_of_awareness import GeometryOfAwareness
import numpy as np

# Test basic usage
model = GeometryOfAwareness(n=7)
print("✓ Model initialized (n=7)")

# Test step
x = np.zeros(7)
for _ in range(10):
    x, _, _ = model.step(x)
print(f"✓ 10 steps executed, history length: {len(model.history['x'])}")

# Test n=15
model_15 = GeometryOfAwareness(n=15)
print("✓ Model initialized (n=15)")

# Test phase sweep
print("Running phase sweep (5x5 grid, 3 runs per cell)...")
trust_vals = np.linspace(0.4, 0.8, 3)
trauma_vals = np.linspace(2.0, 6.0, 3)
results = model.run_sweep(trust_vals, trauma_vals, runs_per_cell=3, steps=100)
print(f"✓ Phase sweep complete: {len(results)} cells")

# Test therapy
print("Running therapy protocol...")
model.reset()
pre_cond, post_cond, x_final = model.run_therapy(pre_steps=100, therapy_steps=50)
print(f"✓ Therapy complete: cond(g) {pre_cond:.3f} → {post_cond:.3f}")

print("\n" + "="*60)
print("✓ ALL EXAMPLE TESTS PASSED")
print("="*60)
