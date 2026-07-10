import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['svg.hashsalt'] = 'vp-guide'
np.random.seed(102)
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ---------------------------------------------------------------- palette
BG      = '#fcfcfb'
INK     = '#0b0b0b'
INK2    = '#52514e'
MUTED   = '#898781'
BASE    = '#c3c2b7'
VA_IN   = '#2a78d6'
VA_OUT  = '#9ec5f4'
POC_C   = '#eb6834'

# ---------------------------------------------------------------- helpers
def value_area(vol, frac=0.70):
    """Classic 70% value-area expansion from the POC.

    Start at the POC row; repeatedly compare the combined volume of the
    next two rows above vs the next two rows below and absorb the larger,
    until the accumulated volume reaches `frac` of total.
    Returns (poc_index, lo_index, hi_index) inclusive.
    """
    vol = np.asarray(vol, dtype=float)
    n = len(vol)
    poc = int(np.argmax(vol))
    target = frac * vol.sum()
    lo = hi = poc
    acc = vol[poc]
    while acc < target and (lo > 0 or hi < n - 1):
        up = vol[hi + 1: hi + 3].sum() if hi < n - 1 else -1.0
        dn = vol[max(0, lo - 2): lo].sum() if lo > 0 else -1.0
        if up >= dn:
            take = min(2, n - 1 - hi)
            acc += vol[hi + 1: hi + 1 + take].sum()
            hi += take
        else:
            take = min(2, lo)
            acc += vol[lo - take: lo].sum()
            lo -= take
    return poc, lo, hi

def noisy(base):
    """Multiplicative noise, strictly positive."""
    return np.clip(base * (1.0 + 0.10 * np.random.randn(len(base))), 1e-4, None)

def gauss(y, mu, sig):
    return np.exp(-0.5 * ((y - mu) / sig) ** 2)

# ---------------------------------------------------------------- synthetic profiles
# common vertical (price) coordinate 0..100; each profile owns a sub-range
profiles = []

# (1) D-shape: symmetric bell, POC central
y1 = np.arange(22, 79)
v1 = noisy(gauss(y1, 50, 9.0) + 0.02)

# (2) P-shape: bulge in UPPER half (POC upper third), thin tail DOWN
y2 = np.arange(15, 81)
v2 = noisy(gauss(y2, 69, 7.0) + 0.055 * np.exp(-(69 - y2) / 45.0))

# (3) b-shape: mirror of P — bulge LOWER half, thin tail UP
y3 = np.arange(15, 81)
v3 = noisy(gauss(y3, 27, 7.0) + 0.055 * np.exp(-(y3 - 27) / 45.0))

# (4) B-shape: two similar bulges separated by a thin LVN neck
y4 = np.arange(17, 84)
v4 = noisy(gauss(y4, 32, 6.0) + 0.96 * gauss(y4, 68, 6.0) + 0.03)

# (5) Trend: elongated, thin, much taller range, no dominant bulge,
#     POC near one extreme (bottom -> a down-trending one-timeframe day)
y5 = np.arange(3, 98)
ramp = 0.6 + 0.5 * (y5.max() - y5) / (y5.max() - y5.min())   # mild ramp toward lows
v5 = noisy(ramp + 0.35 * gauss(y5, 8, 4.0))                  # modest bump pins POC low

names    = ['D-shape', 'P-shape', 'b-shape', 'B-shape', 'Trend']
subs     = ['balance', 'rally into balance', 'decline into balance',
            'double distribution', 'one-timeframe control']
captions = ['fade the edges', 'old business: finite fuel', 'mirror of P',
            'neck = line in the sand', 'do not fade']

for y, v in [(y1, v1), (y2, v2), (y3, v3), (y4, v4), (y5, v5)]:
    profiles.append((y, v / v.sum()))          # equal total volume -> trend is thin

xmax = max(v.max() for _, v in profiles) * 1.12

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 5, figsize=(12.6, 5.8), facecolor=BG)
fig.subplots_adjust(left=0.03, right=0.985, top=0.855, bottom=0.155, wspace=0.28)

for i, (ax, (y, v)) in enumerate(zip(axes, profiles)):
    ax.set_facecolor(BG)
    poc, lo, hi = value_area(v)

    colors = [VA_IN if lo <= k <= hi else VA_OUT for k in range(len(v))]
    colors[poc] = POC_C
    ax.barh(y, v, height=0.92, color=colors, linewidth=0)

    # POC reference dash extending across the panel
    ax.plot([0, xmax], [y[poc], y[poc]], color=POC_C, lw=1.2, ls=(0, (4, 3)),
            zorder=0, alpha=0.85)

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ('top', 'right', 'bottom'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color(BASE)
    ax.spines['left'].set_linewidth(1.0)

    # labels under each panel
    ax.text(0.5, -0.055, names[i], transform=ax.transAxes, ha='center', va='top',
            fontsize=10, fontweight='bold', color=INK)
    ax.text(0.5, -0.115, subs[i], transform=ax.transAxes, ha='center', va='top',
            fontsize=9, color=MUTED)
    ax.text(0.5, -0.175, u'“%s”' % captions[i], transform=ax.transAxes,
            ha='center', va='top', fontsize=9, color=INK2, style='italic')

# LVN neck annotation on the B-shape panel (computed: local minimum between bulges)
ax4 = axes[3]
y4v, v4v = profiles[3]
p1 = int(np.argmax(v4v))                                  # dominant bulge peak
# other bulge peak: argmax on the far half
half = len(v4v) // 2
p2 = int(np.argmax(v4v[half:])) + half if p1 < half else int(np.argmax(v4v[:half]))
a, b = sorted((p1, p2))
neck = a + int(np.argmin(v4v[a:b + 1]))                   # thinnest row between peaks
ax4.annotate('LVN neck',
             xy=(v4v[neck] + 0.0006, y4v[neck]),
             xytext=(xmax * 0.52, y4v[neck] + 0.5),
             fontsize=9, color=INK2, va='center',
             arrowprops=dict(arrowstyle='->', color=INK2, lw=1.0))

# title + legend
fig.suptitle('The shape alphabet', fontsize=13, fontweight='bold', color=INK,
             x=0.03, y=0.965, ha='left')
fig.text(0.03, 0.905, 'Five ways an auction can print — volume vs. price, POC computed as the highest-volume row',
         fontsize=9, color=INK2, ha='left')

from matplotlib.patches import Patch
handles = [Patch(facecolor=VA_IN, label='value area (70%)'),
           Patch(facecolor=VA_OUT, label='outside value area'),
           Patch(facecolor=POC_C, label='POC')]
leg = fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.985, 0.99),
                 ncol=3, frameon=False, fontsize=9, handlelength=1.1,
                 handleheight=0.9, columnspacing=1.2)
for t in leg.get_texts():
    t.set_color(INK2)

# ---------------------------------------------------------------- sanity assertions
def span(y):
    return y.max() - y.min()

# (a) P: POC in upper third, tail down; b mirrored
poc2 = y2[int(np.argmax(v2))]
assert poc2 > y2.min() + 2 * span(y2) / 3, 'P POC not in upper third'
poc3 = y3[int(np.argmax(v3))]
assert poc3 < y3.min() + span(y3) / 3, 'b POC not in lower third'
# (b) B: neck visibly thin vs both peaks
assert v4v[neck] < 0.35 * min(v4v[p1], v4v[p2]), 'B neck not thin'
# (c) trend taller & thinner
assert span(y5) > 1.35 * max(span(y1), span(y2), span(y3), span(y4))
assert profiles[4][1].max() < 0.55 * min(p[1].max() for p in profiles[:4])
# (e) trend POC near an extreme
poc5 = int(np.argmax(v5))
assert poc5 < 0.12 * len(v5) or poc5 > 0.88 * len(v5), 'trend POC not near extreme'

# ---------------------------------------------------------------- save
fig.savefig('/home/user/a/docs/figures/fig-02-profile-shapes.svg', format='svg',
            bbox_inches='tight', pad_inches=0.15, facecolor=BG)
fig.savefig('/tmp/claude-0/-home-user-a/a79ba79b-564a-5a92-9188-47a1d21553f5/'
            'scratchpad/figpng/fig-02-profile-shapes.png', dpi=200,
            bbox_inches='tight', pad_inches=0.15, facecolor=BG)
print('ok')
