import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['svg.hashsalt'] = 'vp-guide'
np.random.seed(111)
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ---------------------------------------------------------------- palette
BG      = '#fcfcfb'
INK     = '#0b0b0b'
INK2    = '#52514e'
MUTED   = '#898781'
GRID    = '#e1e0d9'
AXIS    = '#c3c2b7'
VA_IN   = '#2a78d6'
POC_C   = '#eb6834'
BULL    = '#1baf7a'
BEAR    = '#e34948'
ATTN    = '#eda100'

# ---------------------------------------------------------------- synthetic composite profile
bin_w  = 0.2
prices = np.arange(90.0, 110.0 + bin_w / 2, bin_w)          # bin centers
bulges = (42.0 * np.exp(-0.5 * ((prices - 105.2) / 2.0) ** 2) +
          38.0 * np.exp(-0.5 * ((prices - 95.8)  / 2.2) ** 2))
vol = bulges * (1.0 + 0.18 * np.random.uniform(-1, 1, prices.size)) \
      + 1.2 + 0.8 * np.random.uniform(0, 1, prices.size)

# --- structural quantities, all COMPUTED -------------------------------
i_poc1 = int(np.argmax(vol))                                # first bulge POC
mask = vol.copy()
lo_m, hi_m = max(0, i_poc1 - 25), min(vol.size, i_poc1 + 26)
mask[lo_m:hi_m] = -np.inf
i_poc2 = int(np.argmax(mask))                               # second bulge POC

i_lo, i_hi = sorted((i_poc1, i_poc2))
i_lvn = i_lo + int(np.argmin(vol[i_lo:i_hi + 1]))           # LVN = min between the POCs
lvn_p = prices[i_lvn]

i_up, i_dn = (i_poc1, i_poc2) if prices[i_poc1] > prices[i_poc2] else (i_poc2, i_poc1)
upper_poc = prices[i_up]                                    # POC of upper HVN bulge
lower_poc = prices[i_dn]                                    # POC of lower HVN bulge

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(9.6, 5.4))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

X_MAX = 10.45
ax.set_xlim(-0.12, X_MAX)
ax.set_ylim(89.3, 110.6)

# gridlines
for gy in range(90, 111, 5):
    ax.axhline(gy, color=GRID, lw=0.7, zorder=0)

# ------------------------------------------------ LEFT: composite profile
w = 2.3 * vol / vol.max()
bar_colors = [VA_IN] * prices.size
bar_colors[i_up] = POC_C
bar_colors[i_dn] = POC_C
ax.barh(prices, w, height=bin_w * 0.85, left=0.0,
        color=bar_colors, edgecolor='none', zorder=2)

ax.text(w[i_up] + 0.15, upper_poc, 'HVN', fontsize=9, color=MUTED,
        va='center', ha='left')
ax.text(w[i_dn] + 0.15, lower_poc, 'HVN', fontsize=9, color=MUTED,
        va='center', ha='left')
ax.text(0.02, 89.52, 'volume', fontsize=9, color=MUTED, ha='left',
        va='center')

# LVN dashed line across the whole figure
ax.axhline(lvn_p, color=ATTN, lw=1.2, ls=(0, (5, 3)), zorder=1)
ax.text(X_MAX - 0.05, lvn_p + 0.25, 'LVN', fontsize=10, fontweight='bold',
        color=INK, ha='right', va='bottom')

# ------------------------------------------------ RIGHT: scenario lanes
ax.text(4.85, 109.55, 'Scenario A — Rejection', fontsize=10,
        fontweight='bold', color=INK, ha='center')
ax.text(8.15, 109.55, 'Scenario B — Traverse', fontsize=10,
        fontweight='bold', color=INK, ha='center')

# --- Scenario A: descend to the LVN, touch, sharp reversal up ----------
touch_y = lvn_p - 0.12                                      # slight penetration
t1 = np.linspace(0, 1, 60)
xa1 = 3.70 + t1 * (5.00 - 3.70)
ya1 = (upper_poc - 0.3) + (touch_y - (upper_poc - 0.3)) * t1 ** 1.3 \
      + 0.30 * np.sin(3 * np.pi * t1 + 0.5) * np.sin(np.pi * t1)
t2 = np.linspace(0, 1, 45)
xa2 = 5.00 + t2 * (5.90 - 5.00)
ya2 = touch_y + ((lvn_p + 3.0) - touch_y) * t2 ** 0.75
xa = np.concatenate([xa1, xa2])
ya = np.concatenate([ya1, ya2])
ax.plot(xa, ya, color=INK, lw=1.8, solid_capstyle='round', zorder=4)

# entry dot on the way back up through the node
i_ent = len(xa1) + int(np.argmax(ya2 >= lvn_p + 0.40))
ax.plot(xa[i_ent], ya[i_ent], marker='o', ms=7.5, color=BULL,
        markeredgecolor=BG, markeredgewidth=0.9, zorder=6)
ax.text(5.45, lvn_p + 0.55, 'entry on rejection', fontsize=9,
        color=INK2, ha='left', va='center')

# stop: short red dash just below the LVN
ax.hlines(lvn_p - 0.70, 4.70, 5.30, color=BEAR, lw=2.4, zorder=5)
ax.text(5.45, lvn_p - 0.70, 'stop beyond the node', fontsize=9,
        color=INK2, ha='left', va='center')

# target: dash at the upper HVN POC
ax.hlines(upper_poc, 4.70, 5.30, color=BULL, lw=2.4,
          linestyle=(0, (4, 2)), zorder=5)
ax.text(5.45, upper_poc, 'target: next HVN', fontsize=9,
        color=INK2, ha='left', va='center')

# --- Scenario B: acceptance through the LVN, fast traverse down --------
t3 = np.linspace(0, 1, 55)
xb1 = 6.90 + t3 * (7.90 - 6.90)
yb1 = (lvn_p + 3.2) + ((lvn_p + 0.10) - (lvn_p + 3.2)) * t3 \
      + 0.28 * np.sin(3 * np.pi * t3 + 1.2) * np.sin(np.pi * t3)
t4 = np.linspace(0, 1, 45)
xb2 = 7.90 + t4 * (9.25 - 7.90)
yb2 = (lvn_p + 0.10) + (lower_poc - (lvn_p + 0.10)) * t4 ** 1.15
xb = np.concatenate([xb1, xb2])
yb = np.concatenate([yb1, yb2])
ax.plot(xb, yb, color=INK, lw=1.8, solid_capstyle='round', zorder=4)
ax.annotate('', xy=(xb[-1] + 0.02, yb[-1] - 0.04),
            xytext=(xb[-7], yb[-7]),
            arrowprops=dict(arrowstyle='->', color=INK, lw=1.8), zorder=4)

ax.text(6.55, 92.15,
        'acceptance through the node\n→ fast traverse to next HVN',
        fontsize=9, color=INK2, ha='left', va='center')
i_mid = len(xb1) + int(np.argmax(yb2 <= lvn_p - 2.0))
ax.annotate('', xy=(xb[i_mid], yb[i_mid]),
            xytext=(8.35, 93.05),
            arrowprops=dict(arrowstyle='->', color=INK2, lw=1.0), zorder=3)

# ---------------------------------------------------------------- cosmetics
for s in ('top', 'right', 'bottom'):
    ax.spines[s].set_visible(False)
ax.spines['left'].set_color(AXIS)
ax.set_xticks([])
ax.set_yticks(range(90, 111, 5))
ax.tick_params(axis='y', colors=MUTED, labelsize=8.5, length=3)
ax.set_ylabel('price', fontsize=10, color=INK2)

ax.set_title('LVN behavior is binary: sharp rejection or fast traverse',
             fontsize=13, fontweight='bold', color=INK, pad=16)
fig.text(0.5, 0.012, 'Decide both branches before price arrives.',
         fontsize=9, color=INK2, ha='center')

fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.09)

fig.savefig('/home/user/a/docs/figures/fig-11-lvn-behavior.svg',
            format='svg', facecolor=BG, bbox_inches='tight', pad_inches=0.15)
fig.savefig('/tmp/claude-0/-home-user-a/a79ba79b-564a-5a92-9188-47a1d21553f5/'
            'scratchpad/figpng/fig-11-lvn-behavior.png',
            dpi=200, facecolor=BG, bbox_inches='tight', pad_inches=0.15)

print(f'upper_poc={upper_poc:.2f} lower_poc={lower_poc:.2f} lvn={lvn_p:.2f}')
