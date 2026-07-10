import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['svg.hashsalt'] = 'vp-guide'
np.random.seed(103)
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ---------------------------------------------------------------- palette ---
BG        = '#fcfcfb'
INK       = '#0b0b0b'
INK2      = '#52514e'
MUTED     = '#898781'
GRID      = '#e1e0d9'
BASELINE  = '#c3c2b7'
VA_IN     = '#2a78d6'
VA_OUT    = '#9ec5f4'
POC       = '#eb6834'
BULL      = '#1baf7a'
BAND      = '#cde2fb'

# ------------------------------------------------------------- price path ---
def smooth_noise(n, sd, k=7):
    raw = np.random.normal(0.0, sd, n + k - 1)
    return np.convolve(raw, np.ones(k) / k, mode='valid')

# segment 1: rotation inside lower balance range
n1 = 130
t1 = np.arange(n1)
seg1 = 103.0 + 2.3 * np.sin(2 * np.pi * t1 / 58.0 + 0.6) + smooth_noise(n1, 0.55)

# segment 2: steep breakout leg upward (imbalance)
n2 = 28
start2, end2 = seg1[-1], 119.0
seg2 = np.linspace(start2, end2, n2) + smooth_noise(n2, 0.18)
seg2[0] = start2  # keep the join continuous

# segment 3: rotation inside new higher balance range
n3 = 142
t3 = np.arange(n3)
seg3 = 119.0 + 2.1 * np.sin(2 * np.pi * t3 / 62.0 + 2.4) + smooth_noise(n3, 0.5)
seg3 += (seg2[-1] - seg3[0]) * np.exp(-t3 / 6.0)  # blend the join

price = np.concatenate([seg1, seg2, seg3])
x = np.arange(len(price))
x1_end = n1                 # right edge of segment 1
x2_end = n1 + n2            # right edge of breakout leg
x3_end = len(price)         # right edge of segment 3

# ------------------------------------------- time-at-price profiles (TPO) ---
BINW = 0.4
lo_edge = np.floor(price.min() / BINW) * BINW - BINW
hi_edge = np.ceil(price.max() / BINW) * BINW + BINW
edges = np.arange(lo_edge, hi_edge + BINW / 2, BINW)
centers = (edges[:-1] + edges[1:]) / 2

hist1, _ = np.histogram(seg1, bins=edges)
hist2, _ = np.histogram(seg2, bins=edges)
hist3, _ = np.histogram(seg3, bins=edges)

def value_area(vol, frac=0.70):
    """Classic 70% value-area expansion: start at POC, add the larger of the
    two-row blocks above/below until the target fraction is contained."""
    poc = int(np.argmax(vol))
    total = vol.sum()
    target = frac * total
    lo = hi = poc
    cum = float(vol[poc])
    n = len(vol)
    while cum < target and (lo > 0 or hi < n - 1):
        up = vol[hi + 1:hi + 3].sum() if hi < n - 1 else -1.0
        dn = vol[max(lo - 2, 0):lo].sum() if lo > 0 else -1.0
        if up >= dn:
            take = min(2, n - 1 - hi)
            cum += vol[hi + 1:hi + 1 + take].sum()
            hi += take
        else:
            take = min(2, lo)
            cum += vol[lo - take:lo].sum()
            lo -= take
    return poc, lo, hi

poc1, va1_lo, va1_hi = value_area(hist1)
poc3, va3_lo, va3_hi = value_area(hist3)

# one shared scale so the corridor profile is honestly thin
SCALE = 26.0 / max(hist1.max(), hist3.max())   # x-units per sample

# ------------------------------------------------------------------ figure --
fig, ax = plt.subplots(figsize=(10.0, 6.0))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

ax.set_xlim(-4, x3_end + 12)
ylim_lo, ylim_hi = price.min() - 2.2, price.max() + 2.8
ax.set_ylim(ylim_lo, ylim_hi)

ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)

# balance bands (behind everything)
from matplotlib.patches import Rectangle
b1_lo, b1_hi = seg1.min() - 0.3, seg1.max() + 0.3
b3_lo, b3_hi = seg3.min() - 0.3, seg3.max() + 0.3
ax.add_patch(Rectangle((-4, b1_lo), x1_end + 4, b1_hi - b1_lo,
                       facecolor=BAND, alpha=0.3, edgecolor='none', zorder=0.5))
ax.add_patch(Rectangle((x2_end, b3_lo), x3_end - x2_end, b3_hi - b3_lo,
                       facecolor=BAND, alpha=0.3, edgecolor='none', zorder=0.5))

# profiles: bars extend LEFT from the right edge of their segment
def draw_profile(hist, anchor, poc=None, va=None):
    for i, c in enumerate(hist):
        if c == 0:
            continue
        if poc is not None and i == poc:
            col = POC
        elif va is not None and va[0] <= i <= va[1]:
            col = VA_IN
        else:
            col = VA_OUT
        ax.barh(centers[i], -c * SCALE, left=anchor, height=BINW * 0.82,
                color=col, edgecolor='none', zorder=2)

draw_profile(hist1, x1_end, poc=poc1, va=(va1_lo, va1_hi))
draw_profile(hist3, x3_end, poc=poc3, va=(va3_lo, va3_hi))
# breakout-leg profile: same scale -> nearly empty rows
draw_profile(hist2, x2_end, poc=None, va=None)

# price path on top
ax.plot(x, price, color=INK, linewidth=1.8, zorder=3,
        solid_capstyle='round')

# --------------------------------------------------------------- labelling --
ax.text(2, b1_hi + 0.7, 'Balance: value building',
        fontsize=9, color=INK2, ha='left', va='bottom', zorder=4)
ax.text(x2_end + 4, b3_hi + 0.7, 'Balance: value building',
        fontsize=9, color=INK2, ha='left', va='bottom', zorder=4)

# imbalance arrow along the breakout leg (offset slightly left of the path)
ax.annotate('',
            xy=(x2_end - 3.5, seg2[-4] + 0.4),
            xytext=(x1_end - 0.5, seg2[1] + 0.6),
            arrowprops=dict(arrowstyle='->', color=BULL, lw=2.0,
                            shrinkA=2, shrinkB=2), zorder=4)
ax.text(x1_end - 6, 112.6, 'Imbalance: price discovery\n(initiative buying)',
        fontsize=9, color=INK2, ha='right', va='center', zorder=4)

# LVN corridor annotation: points at the thin profile between the two bulges
corridor_mid = (b1_hi + b3_lo) / 2
ax.annotate('LVN corridor left by the imbalance leg',
            xy=(x2_end - 1.0, corridor_mid),
            xytext=(x2_end + 46, corridor_mid - 1.2),
            fontsize=9, color=INK2, ha='left', va='center', zorder=4,
            arrowprops=dict(arrowstyle='->', color=INK2, lw=1.0,
                            shrinkA=3, shrinkB=3))

# ------------------------------------------------------------------- axes ---
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(BASELINE)
ax.spines['bottom'].set_color(BASELINE)
ax.tick_params(colors=MUTED, labelsize=9)
for lab in ax.get_xticklabels() + ax.get_yticklabels():
    lab.set_color(MUTED)

ax.set_xlabel('Time (bars)', fontsize=10, color=INK2)
ax.set_ylabel('Price', fontsize=10, color=INK2)
ax.set_title('The auction cycle: balance → imbalance → new balance',
             fontsize=13, fontweight='bold', color=INK, pad=12)

# caption / legend line under the plot
fig.text(0.5, -0.015,
         'Sideways histograms show time-at-price for each segment: '
         'POC row in orange, computed 70% value area in solid blue, tails in light blue.',
         fontsize=9, color=INK2, ha='center')

fig.savefig('/home/user/a/docs/figures/fig-03-auction-cycle.svg',
            format='svg', bbox_inches='tight', pad_inches=0.15,
            facecolor=BG)
fig.savefig('/tmp/claude-0/-home-user-a/a79ba79b-564a-5a92-9188-47a1d21553f5/'
            'scratchpad/figpng/fig-03-auction-cycle.png',
            dpi=200, bbox_inches='tight', pad_inches=0.15, facecolor=BG)
print('done')
print('poc1 price', centers[poc1], 'VA1', centers[va1_lo], centers[va1_hi])
print('poc3 price', centers[poc3], 'VA3', centers[va3_lo], centers[va3_hi])
print('max hist1/3', hist1.max(), hist3.max(), 'max hist2', hist2.max())
