import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'Liberation Sans', 'DejaVu Sans']

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis('off')

# Colors
c_input = '#E1F5EE'    # teal light
c_model = '#EEEDFE'    # purple light
c_target = '#FAECE7'   # coral light
c_eval = '#E6F1FB'     # blue light
c_spec = '#F1EFE8'     # gray light

b_input = '#0F6E56'
b_model = '#534AB7'
b_target = '#993C1D'
b_eval = '#185FA5'
b_spec = '#5F5E5A'

bw = 0.4  # box width (half)
bh = 0.28

def draw_box(ax, cx, cy, w, h, title, subtitle, fc, ec, fontsize=12):
    rect = mpatches.FancyBboxPatch((cx-w, cy-h), 2*w, 2*h,
            boxstyle="round,pad=0.05", facecolor=fc, edgecolor=ec, linewidth=0.8)
    ax.add_patch(rect)
    if subtitle:
        ax.text(cx, cy+0.08, title, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=ec)
        ax.text(cx, cy-0.15, subtitle, ha='center', va='center', fontsize=fontsize-2, color='#444')
    else:
        ax.text(cx, cy, title, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=ec)

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#888', lw=1.0))

# === Section labels ===
ax.text(1.5, 6.6, 'Inputs', ha='center', va='center', fontsize=20, fontweight='bold', color='#333')
ax.text(5.0, 6.6, 'Model', ha='center', va='center', fontsize=20, fontweight='bold', color='#333')
ax.text(8.2, 6.6, 'Targets', ha='center', va='center', fontsize=20, fontweight='bold', color='#333')

# === INPUT boxes ===
draw_box(ax, 1.5, 5.9, 0.9, bh, 'Sentinel-1 SAR', 'VV diff, VH backscatter', c_input, b_input)
draw_box(ax, 1.5, 5.0, 0.9, bh, 'VIIRS NTL', 'Mean, std, lit area ratio', c_input, b_input)
draw_box(ax, 1.5, 4.1, 0.9, bh, 'World Port Index', '91 port attributes', c_input, b_input)

# === MODEL box ===
draw_box(ax, 5.0, 5.0, 0.9, bh, 'XGBoost', 'Supervised learning', c_model, b_model, fontsize=14)

# === TARGET boxes ===
draw_box(ax, 8.2, 5.4, 0.9, bh, 'Trade value (log)', '', c_target, b_target)
draw_box(ax, 8.2, 4.6, 0.9, bh, 'Trade weight (log)', '', c_target, b_target)

# === Arrows: inputs → model ===
draw_arrow(ax, 2.45, 5.9, 3.95, 5.15)
draw_arrow(ax, 2.45, 5.0, 3.95, 5.0)
draw_arrow(ax, 2.45, 4.1, 3.95, 4.85)

# === Arrows: model → targets ===
draw_arrow(ax, 5.95, 5.15, 7.15, 5.4)
draw_arrow(ax, 5.95, 4.85, 7.15, 4.6)

# === Separator ===
ax.plot([0.3, 9.7], [3.35, 3.35], '--', color='#ccc', linewidth=0.8)

# === Evaluation exercises ===
ax.text(5.0, 3.05, 'Evaluation exercises', ha='center', va='center', fontsize=20, fontweight='bold', color='#333')

draw_box(ax, 1.8, 2.2, 1.05, bh, 'Nowcasting', 'U.S. ports, 2022\u20132024', c_eval, b_eval)
draw_box(ax, 5.0, 2.2, 1.05, bh, 'Spatial extrapolation', 'Hawaii (leave-one-out)', c_eval, b_eval)
draw_box(ax, 8.2, 2.2, 1.05, bh, 'Application', 'Russian ports, post-2022', c_eval, b_eval)

# === Specifications ===
ax.text(5.0, 1.35, 'Specifications', ha='center', va='center', fontsize=20, fontweight='bold', color='#333')

draw_box(ax, 1.8, 0.6, 1.05, 0.24, 'Satellite + port chars', '', c_spec, b_spec, fontsize=13)
draw_box(ax, 5.0, 0.6, 1.05, 0.24, 'Satellite only', '', c_spec, b_spec, fontsize=13)
draw_box(ax, 8.2, 0.6, 1.05, 0.24, 'Port chars only', '', c_spec, b_spec, fontsize=13)

plt.tight_layout()
plt.savefig('watching_trade/output_figures/pipeline_schematic.pdf', bbox_inches='tight', dpi=300)
print("Saved: pipeline_schematic.pdf")