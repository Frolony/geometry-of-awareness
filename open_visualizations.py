"""
Quick viewer to open all visualization PNGs
"""
import os
import subprocess
from pathlib import Path

viz_dir = Path("visualizations")
if not viz_dir.exists():
    print("ERROR: visualizations/ directory not found. Run visualization_dashboard.py first.")
    exit(1)

png_files = sorted(viz_dir.glob("*.png"))

if not png_files:
    print("ERROR: No PNG files found in visualizations/")
    exit(1)

print("\n" + "="*70)
print("VISUALIZATION VIEWER - Geometry of Awareness v1.2")
print("="*70)
print(f"\nFound {len(png_files)} visualizations:\n")

for i, png in enumerate(png_files, 1):
    size_mb = png.stat().st_size / 1024
    print(f"  [{i}] {png.name} ({size_mb:.0f} KB)")

print("\nOpening all visualizations...\n")

# Open each PNG with default viewer
for i, png in enumerate(png_files, 1):
    try:
        # Start in non-blocking mode
        os.startfile(str(png.absolute()))
        print(f"  [Opening {i}/{len(png_files)}] {png.name}")
    except Exception as e:
        print(f"  [ERROR] Could not open {png.name}: {e}")

print("\n" + "="*70)
print("All visualizations should now be opening in your default viewer.")
print("If not, open manually: visualizations/01_*.png, etc.")
print("="*70 + "\n")
