import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set clean scientific styling
fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
ax.set_facecolor('#f8fafc')
fig.patch.set_facecolor('#ffffff')

# Remove axis ticks and borders
ax.axis('off')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

def draw_box(x, y, w, h, title, subtitle="", color="#e2e8f0", border="#94a3b8", title_color="#0f172a"):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1.2", 
                                facecolor=color, edgecolor=border, linewidth=1.5, zorder=2)
    ax.add_patch(box)
    
    if subtitle:
        ax.text(x + w/2, y + h/2 + 1.8, title, ha='center', va='center', 
                fontsize=11, fontweight='bold', color=title_color, zorder=3)
        ax.text(x + w/2, y + h/2 - 2.0, subtitle, ha='center', va='center', 
                fontsize=8.5, color="#334155", zorder=3, linespacing=1.3)
    else:
        ax.text(x + w/2, y + h/2, title, ha='center', va='center', 
                fontsize=11, fontweight='bold', color=title_color, zorder=3)

def draw_arrow(x1, y1, x2, y2, color="#64748b"):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0, mutation_scale=15),
                zorder=1)

# Title Header
ax.text(50, 96, "PROJECT WORKFLOW BLOCK DIAGRAM", ha='center', va='center', 
        fontsize=16, fontweight='bold', color='#0f172a')
ax.text(50, 93, "Grain Quality Prediction & Spatial Chemical Visualization Using Hyperspectral Imaging", 
        ha='center', va='center', fontsize=10, color='#64748b')

# 1. DATA INGESTION
draw_box(8, 79, 38, 9, "Dataset A: sensAIfood Wheat NIR", 
         "150 Wheat Samples (950-1650 nm)\nLab Wet-Chemistry: Protein % & Moisture %", 
         color="#e0f2fe", border="#0284c7", title_color="#0369a1")

draw_box(54, 79, 38, 9, "Dataset B: WheatDual-4V (Mendeley)", 
         "1,600 3D Hyperspectral Cubes (400-1000 nm)\n4 Commercial Cultivars on Black Foam Trays", 
         color="#fef3c7", border="#d97706", title_color="#b45309")

# 2. PARTITIONING & COMPUTER VISION
draw_box(8, 63, 38, 8.5, "SPXY Sample Partitioning", 
         "80% Calibration (Train: 120 samples)\n20% Prediction (Test: 30 samples)", 
         color="#f1f5f9", border="#64748b")

draw_box(54, 63, 38, 8.5, "3D ROI Segmentation (Otsu)", 
         "Near-Infrared 800 nm Slice Filtering\nAutomatic Seed Boundary & Mask Extraction", 
         color="#f1f5f9", border="#64748b")

# Arrows Level 1 -> Level 2
draw_arrow(27, 79, 27, 72.5)
draw_arrow(73, 79, 73, 72.5)

# 3. SPECTRAL PREPROCESSING
draw_box(8, 48, 38, 8.5, "Spectral Preprocessing", 
         "SNV: Curvature Light-Scatter Correction (Protein)\n1st Derivative: Baseline Drift Removal (Moisture)", 
         color="#ede9fe", border="#7c3aed", title_color="#6d28d9")

draw_arrow(27, 63, 27, 57.5)

# 4. FEATURE SELECTION
draw_box(8, 33, 38, 8.5, "Wavelength Selection (SPA & CARS)", 
         "Compress 141 bands -> 15 Critical Features\nTargeting N-H (1480 nm) & O-H (1450 nm) Bonds", 
         color="#fce7f3", border="#db2777", title_color="#be185d")

draw_arrow(27, 48, 27, 42.5)

# 5. REGRESSION ENGINES
draw_box(8, 18, 38, 8.5, "Quantitative Regression Models", 
         "Chemometric Baseline: PLSR & MLR\nDeep Learning: 1D-CNN & Hybrid CLNet (1D-CNN + LSTM)", 
         color="#dcfce7", border="#16a34a", title_color="#15803d")

draw_arrow(27, 33, 27, 27.5)

# 6. DELIVERABLES / OUTPUTS
draw_box(8, 3, 38, 8.5, "Quantitative Evaluation Metrics", 
         "Protein: Rp² = 0.990, RMSEP = 0.25%, RPD = 10.14\nMoisture: Rp² = 0.982, RMSEP = 0.28%, RPD = 7.55", 
         color="#dcfce7", border="#16a34a", title_color="#15803d")

draw_box(54, 3, 38, 8.5, "Pixel-by-Pixel Chemical Heatmaps", 
         "Spatial Protein & Moisture Distribution Maps\nPseudo-color Gradients across Wheat Cultivars", 
         color="#ffedd5", border="#ea580c", title_color="#c2410c")

# Final Connections
draw_arrow(27, 18, 27, 12.5)
# Model to Heatmap arrow
draw_arrow(47, 22.25, 73, 12.5, color="#2563eb")
# Mask to Heatmap arrow
draw_arrow(73, 63, 73, 12.5, color="#ea580c")

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
out_file = "outputs/project_block_diagram.png"
plt.savefig(out_file, bbox_inches='tight', dpi=300)
plt.close()
print(f"Generated clean diagram at: {out_file}")
