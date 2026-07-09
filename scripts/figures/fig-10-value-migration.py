import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams['svg.hashsalt'] = 'vp-guide'
np.random.seed(110)
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

from matplotlib.patches import FancyBboxPatch

# ---------------- palette ----------------
BG      = '#fcfcfb'
INK     = '#0b0b0b'
INK2    = '#52514e'
MUTED   = '#898781'
GRID    = '#e1e0d9'
BASE    = '#c3c2b7'
VA_IN   = '#2a78d6'
VA_OUT  = '#9ec5f4'
POC     = '#eb6834'

TICK = 0.10  # price tick size for the synthetic profiles


def day_profile(center, sd):
    """Synthetic single-day volume profile: gaussian shape * noise."""
    prices = np.arange(center - 4.0 * sd, center + 4.0 * sd + TICK / 2, TICK)
    shape = np.exp(-0.5 * ((prices - center) / sd) ** 2)
    noise = 1.0 + 0.12 * np.random.randn(prices.size)
    vols = shape * np.clip(noise, 0.5, 1.5)
    return prices, vols


def value_area(prices, vols, frac=0.70):
    """Real 70% expansion: start at POC, grow toward the larger neighbor row
    until >= frac of total volume is inside."""
    poc_i = int(np.argmax(vols))
    total = vols.sum()
    lo = hi = poc_i
    acc = vols[poc_i]
    while acc < frac * total:
        up = vols[hi + 1] if hi + 1 < vols.size else -np.inf
        dn = vols[lo - 1] if lo - 1 >= 0 else -np.inf
        if up >= dn:
            hi += 1
            acc += vols[hi]
        else:
            lo -= 1
            acc += vols[lo]
    return prices[lo], prices[hi], prices[poc_i]


# ---------------- six days: (center, sd) chosen to produce the relationships,
# but VAL/VAH/POC are COMPUTED from the volume data ----------------
params = [
    (100.0, 1.9),   # D1 baseline
    (104.6, 1.9),   # D2 higher value
    (109.2, 1.9),   # D3 higher value again
    (109.7, 2.1),   # D4 overlapping D3 ~ balance
    (109.6, 0.9),   # D5 inside day (narrow)
    (103.2, 1.9),   # D6 lower value
]

days = []
for c, s in params:
    p, v = day_profile(c, s)
    val, vah, poc = value_area(p, v)
    days.append(dict(val=val, vah=vah, poc=poc))

VAL = [d['val'] for d in days]
VAH = [d['vah'] for d in days]
PC  = [d['poc'] for d in days]

# ---- geometric assertions (relationships must be true of computed boxes) ----
ol_12 = max(0.0, min(VAH[0], VAH[1]) - max(VAL[0], VAL[1]))
ol_23 = max(0.0, min(VAH[1], VAH[2]) - max(VAL[1], VAL[2]))
assert VAL[1] > VAL[0] and VAH[1] > VAH[0] and ol_12 < 0.25 * (VAH[0] - VAL[0]), 'D2 higher'
assert VAL[2] > VAL[1] and VAH[2] > VAH[1] and ol_23 < 0.25 * (VAH[1] - VAL[1]), 'D3 higher'
ol_34 = max(0.0, min(VAH[2], VAH[3]) - max(VAL[2], VAL[3]))
assert ol_34 > 0.7 * (VAH[2] - VAL[2]), 'D4 heavily overlaps D3'
assert VAL[4] > VAL[3] and VAH[4] < VAH[3], 'D5 strictly inside D4'
assert (VAH[4] - VAL[4]) < (VAH[3] - VAL[3]), 'D5 shorter'
assert VAH[5] < VAL[4], 'D6 clearly below D5'
for i in range(6):
    assert VAL[i] < PC[i] < VAH[i], 'POC inside box'

# ---------------- figure ----------------
fig, ax = plt.subplots(figsize=(9.2, 5.6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

xs = np.arange(6)
BOXW = 0.56

for i, d in enumerate(days):
    x0 = xs[i] - BOXW / 2
    h = d['vah'] - d['val']
    box = FancyBboxPatch(
        (x0, d['val']), BOXW, h,
        boxstyle='round,pad=0,rounding_size=0.12',
        mutation_aspect=0.16,
        facecolor=VA_OUT, alpha=0.8,
        edgecolor='#7ba8dc', linewidth=0.6, zorder=3)
    ax.add_patch(box)
    # POC tick inside the box
    ax.hlines(d['poc'], xs[i] - 0.20, xs[i] + 0.20,
              color=POC, linewidth=2.4, zorder=4)

# relationship labels, above/below each box
lab = [
    ('baseline', 'below'),
    ('higher value', 'below'),
    ('higher value', 'above'),
    ('overlapping value ~ balance', 'above'),
    ('inside day - coiling', 'below'),
    ('lower value', 'below'),
]
PAD = 0.55
for i, (txt, where) in enumerate(lab):
    if where == 'above':
        y, va = VAH[i] + PAD, 'bottom'
    else:
        y, va = VAL[i] - PAD, 'top'
    ax.text(xs[i], y, txt, ha='center', va=va, fontsize=9, color=INK)

# staircase arrow D1 -> D3
ax.annotate('',
            xy=(xs[2] - 0.42, VAL[2] + 0.8),
            xytext=(xs[0] + 0.10, VAH[0] + 0.4),
            arrowprops=dict(arrowstyle='->', color=INK2, linewidth=1.3,
                            shrinkA=2, shrinkB=2), zorder=5)
ax.text(xs[0] - 0.42, VAH[2] + 1.1,
        'staircase of value = trend\n(auction-theory definition)',
        fontsize=9, color=INK2, ha='left', va='bottom')

# axes cosmetics
ax.set_xlim(-0.75, 5.75)
ymin = min(VAL) - 3.4
ymax = max(VAH) + 3.2
ax.set_ylim(ymin, ymax)
ax.set_xticks(xs)
ax.set_xticklabels([f'D{i+1}' for i in range(6)], fontsize=10, color=INK2)
ax.set_ylabel('price', fontsize=10, color=INK2)
ax.tick_params(axis='y', labelsize=9, colors=MUTED, length=0)
ax.tick_params(axis='x', colors=INK2, length=0)
ax.grid(axis='y', color=GRID, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color(BASE)

ax.set_title('Read the market day-over-day by value relationships',
             fontsize=13, fontweight='bold', color=INK, loc='left', pad=12)

fig.text(0.02, 0.015,
         'Each box = one day’s value area (70% of volume, computed by POC expansion); '
         'orange tick = that day’s point of control.',
         fontsize=9, color=INK2, ha='left')

fig.subplots_adjust(bottom=0.13)

fig.savefig('/home/user/a/docs/figures/fig-10-value-migration.svg',
            format='svg', bbox_inches='tight', pad_inches=0.15,
            facecolor=BG)
fig.savefig('/tmp/claude-0/-home-user-a/a79ba79b-564a-5a92-9188-47a1d21553f5/scratchpad/figpng/fig-10-value-migration.png',
            dpi=200, bbox_inches='tight', pad_inches=0.15, facecolor=BG)
print('VAL', np.round(VAL, 2))
print('VAH', np.round(VAH, 2))
print('POC', np.round(PC, 2))
