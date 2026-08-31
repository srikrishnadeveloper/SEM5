"""Generate a clean architecture diagram for the SIH26067 prototype."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis('off')
fig.patch.set_facecolor('#0f172a')
ax.set_facecolor('#0f172a')

# Title
ax.text(6, 6.5, 'OceanViz 3D — SIH26067 Architecture', fontsize=18, ha='center',
        color='#e2e8f0', fontweight='bold')

# Color palette
c_user = '#1e293b'
c_front = '#0ea5e9'
c_back = '#22c55e'
c_data = '#f59e0b'
c_real = '#a855f7'
c_line = '#64748b'

# Helper to draw box
def box(x, y, w, h, color, label, sub=None):
    r = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", 
                               facecolor=color, edgecolor='#334155', linewidth=1.5,
                               alpha=0.18)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2 + 0.12, label, fontsize=11, ha='center', color='#e2e8f0',
            fontweight='bold')
    if sub:
        ax.text(x + w/2, y + h/2 - 0.18, sub, fontsize=8, ha='center', color='#94a3b8')

# User
box(0.3, 4.0, 1.6, 1.2, c_user, 'User', 'Browser/Edge')

# Frontend
box(2.6, 4.0, 2.6, 1.2, c_front, 'Frontend', 'Cesium.js + Chart.js')

# Backend
box(5.8, 4.0, 2.4, 1.2, c_back, 'FastAPI', 'Python + NumPy + Pillow')

# Data Adapter
box(8.8, 4.0, 2.4, 1.2, c_data, 'DataAdapter', 'SyntheticSource')

# Real source
box(8.8, 1.5, 2.4, 1.2, c_real, 'Future Source', 'NetCDF / xarray')

# Arrows
arrow_style = dict(arrowstyle='->', color=c_line, lw=1.8, mutation_scale=12)
ax.annotate('', xy=(2.5, 4.6), xytext=(1.9, 4.6), arrowprops=arrow_style)
ax.annotate('', xy=(5.7, 4.6), xytext=(5.2, 4.6), arrowprops=arrow_style)
ax.annotate('', xy=(8.7, 4.6), xytext=(8.2, 4.6), arrowprops=arrow_style)
ax.annotate('', xy=(10.0, 2.7), xytext=(10.0, 4.0), arrowprops=arrow_style)

# Labels below arrows
ax.text(2.2, 4.9, 'interacts', fontsize=8, ha='center', color='#94a3b8')
ax.text(5.45, 4.9, 'REST API', fontsize=8, ha='center', color='#94a3b8')
ax.text(8.45, 4.9, 'fetch data', fontsize=8, ha='center', color='#94a3b8')
ax.text(9.4, 3.35, 'plug-in later', fontsize=8, ha='center', color='#94a3b8')

# Feature boxes
features = [
    (2.6, 1.5, '3D Globe'),
    (3.8, 1.5, 'Data Layer'),
    (5.0, 1.5, 'Time Slider'),
    (6.2, 1.5, 'Current Vectors'),
    (7.4, 1.5, 'Instrument Profiles'),
]
for x, y, label in features:
    box(x, y, 1.1, 0.55, c_front, label)

# Bracket
ax.plot([2.6, 8.3], [2.3, 2.3], color=c_line, lw=1)
ax.plot([2.6, 2.6], [2.1, 2.3], color=c_line, lw=1)
ax.plot([8.3, 8.3], [2.1, 2.3], color=c_line, lw=1)
ax.text(5.45, 2.45, 'Frontend Features', fontsize=9, ha='center', color='#94a3b8')

# Backend endpoints
box(5.8, 1.5, 2.4, 1.2, c_back, 'API Endpoints', '/image, /slice, /vectors')

out = Path(__file__).parent / 'architecture.png'
plt.tight_layout()
plt.savefig(out, dpi=150, facecolor='#0f172a', edgecolor='none', bbox_inches='tight')
print(f'Saved {out}')
