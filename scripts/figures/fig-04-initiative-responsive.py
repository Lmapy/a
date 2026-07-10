import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['svg.hashsalt'] = 'vp-guide'
np.random.seed(104)
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# Palette
BG = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
BASE = '#c3c2b7'
POC = '#eb6834'
BULL = '#1baf7a'
BEAR = '#e34948'
BAND_FILL = '#cde2fb'

fig, ax = plt.subplots(figsize=(8.4, 6.0))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Value area band geometry (conceptual, computed from constants)
val_y = 4.0
vah_y = 6.0
poc_y = (val_y + vah_y) / 2.0

# Band fill
ax.axhspan(val_y, vah_y, xmin=0.03, xmax=0.97, color=BAND_FILL, alpha=0.5, zorder=1)

x0, x1 = 0.3, 9.7
# Dashed boundary lines
ax.hlines(vah_y, x0, x1, colors=INK2, linestyles=(0, (5, 3)), linewidth=1.2, zorder=2)
ax.hlines(val_y, x0, x1, colors=INK2, linestyles=(0, (5, 3)), linewidth=1.2, zorder=2)
ax.hlines(poc_y, x0, x1, colors=POC, linestyles=(0, (5, 3)), linewidth=1.2, zorder=2)

# Line labels (right side)
ax.text(x1 + 0.05, vah_y, 'VAH', fontsize=10, fontweight='bold', color=INK,
        va='center', ha='left')
ax.text(x1 + 0.05, val_y, 'VAL', fontsize=10, fontweight='bold', color=INK,
        va='center', ha='left')
ax.text(x1 + 0.05, poc_y, 'POC', fontsize=10, fontweight='bold', color=POC,
        va='center', ha='left')

# Band interior label
ax.text(5.0, poc_y - 0.55, 'Inside value: ambiguous / neutral',
        fontsize=10, color=MUTED, ha='center', va='center', style='italic', zorder=3)
ax.text(x0, vah_y + 0.12, "yesterday's value area", fontsize=9, color=MUTED,
        ha='left', va='bottom')

def swatch_text(x, y, color, bold_part, rest):
    """Small colored square swatch followed by text."""
    ax.add_patch(plt.Rectangle((x, y - 0.11), 0.22, 0.22, facecolor=color,
                               edgecolor='none', zorder=3))
    ax.text(x + 0.38, y, bold_part, fontsize=10, color=INK, fontweight='bold',
            va='center', ha='left', zorder=3)

def swatch_line(x, y, color, main, sub):
    ax.add_patch(plt.Rectangle((x, y - 0.11), 0.22, 0.22, facecolor=color,
                               edgecolor='none', zorder=3))
    ax.text(x + 0.38, y, main, fontsize=10, color=INK, fontweight='bold',
            va='center', ha='left', zorder=3)
    ax.text(x + 0.38, y - 0.45, sub, fontsize=9, color=INK2,
            va='center', ha='left', zorder=3)

# Above the band
swatch_line(0.45, 9.0, BULL, 'BUYING above value = INITIATIVE',
            '(conviction: sees higher future value)')
swatch_line(0.45, 7.7, BEAR, 'SELLING above value = RESPONSIVE',
            '(fading the excursion back to value)')

# Below the band (mirrored)
swatch_line(0.45, 2.9, BEAR, 'SELLING below value = INITIATIVE',
            '(conviction: sees lower future value)')
swatch_line(0.45, 1.6, BULL, 'BUYING below value = RESPONSIVE',
            '(fading the excursion back to value)')

# Arrows
# Responsive: from above pointing down toward the band
ax.annotate('', xy=(7.1, vah_y + 0.12), xytext=(7.1, vah_y + 2.5),
            arrowprops=dict(arrowstyle='->', color=INK2, linewidth=1.4), zorder=3)
ax.text(7.1, vah_y + 3.0, 'responsive pushes\nprice back to value',
        fontsize=9, color=INK2, ha='center', va='center')

# Initiative: from the band pointing up/away
ax.annotate('', xy=(9.0, vah_y + 2.5), xytext=(9.0, vah_y + 0.12),
            arrowprops=dict(arrowstyle='->', color=INK2, linewidth=1.4), zorder=3)
ax.text(9.0, vah_y + 3.0, 'initiative\nrelocates value',
        fontsize=9, color=INK2, ha='center', va='center')

ax.set_title('Initiative vs responsive - judged against the prior value area',
             fontsize=13, fontweight='bold', color=INK, pad=14)

svg_path = '/home/user/a/docs/figures/fig-04-initiative-responsive.svg'
png_path = '/tmp/claude-0/-home-user-a/a79ba79b-564a-5a92-9188-47a1d21553f5/scratchpad/figpng/fig-04-initiative-responsive.png'
fig.savefig(svg_path, format='svg', bbox_inches='tight', pad_inches=0.15,
            facecolor=BG)
fig.savefig(png_path, dpi=200, bbox_inches='tight', pad_inches=0.15,
            facecolor=BG)
print('saved', svg_path, png_path)
