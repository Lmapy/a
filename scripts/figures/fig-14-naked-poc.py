import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams['svg.hashsalt'] = 'vp-guide'
np.random.seed(114)
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ---------------- palette ----------------
BG      = '#fcfcfb'
INK     = '#0b0b0b'
INK2    = '#52514e'
MUTED   = '#898781'
GRID    = '#e1e0d9'
BASE    = '#c3c2b7'
VA_IN   = '#2a78d6'
VA_OUT  = '#9ec5f4'
POC_C   = '#eb6834'
ATTN    = '#eda100'

# ---------------- synthetic daily volume profiles ----------------
BIN = 0.25
edges   = np.arange(96.5, 111.5 + BIN, BIN)
centers = 0.5 * (edges[:-1] + edges[1:])

day_center = [100.2, 104.1, 107.3, 106.4, 105.3]
day_sd     = [0.90, 1.05, 0.85, 0.80, 0.90]

profiles, pocs, vas = [], [], []

def value_area(vol, poc_idx, frac=0.70):
    """Classic two-row 70% expansion around the POC."""
    total  = vol.sum()
    target = frac * total
    lo = hi = poc_idx
    acc = vol[poc_idx]
    n = len(vol)
    while acc < target:
        if hi + 2 < n:
            up = vol[hi + 1] + vol[hi + 2]; ustep = 2
        elif hi + 1 < n:
            up = vol[hi + 1]; ustep = 1
        else:
            up = -1.0; ustep = 0
        if lo - 2 >= 0:
            dn = vol[lo - 2] + vol[lo - 1]; dstep = 2
        elif lo - 1 >= 0:
            dn = vol[lo - 1]; dstep = 1
        else:
            dn = -1.0; dstep = 0
        if up < 0 and dn < 0:
            break
        if up >= dn:
            acc += up; hi += ustep
        else:
            acc += dn; lo -= dstep
    return lo, hi

for c, s in zip(day_center, day_sd):
    samples = np.random.normal(c, s, 4200)
    vol, _ = np.histogram(samples, bins=edges)
    vol = vol.astype(float)
    p = int(np.argmax(vol))                    # POC = argmax of the data
    lo, hi = value_area(vol, p)
    profiles.append(vol); pocs.append(p); vas.append((lo, hi))

poc_price = [centers[p] for p in pocs]
poc1, poc2, poc3, poc4, poc5 = poc_price

# ---------------- multi-day price path ----------------
DAYW    = 1.0
BARFRAC = 0.55            # POC bar (longest) reaches 55% of the day width
tips    = [i + BARFRAC for i in range(5)]  # right edge of each POC bar

def build_day(cps, x0, n=240, amp=0.10, quiet=()):
    """Piecewise-linear control points + Brownian-bridge noise (pinned)."""
    xs, ys = [], []
    for k in range(len(cps) - 1):
        f0, v0 = cps[k]; f1, v1 = cps[k + 1]
        m = max(int(round(n * (f1 - f0))), 4)
        t = np.arange(m) / m
        base = v0 + (v1 - v0) * t
        if k in quiet:
            nz = np.zeros(m)
        else:
            w = np.cumsum(np.random.normal(0.0, amp / np.sqrt(m), m))
            nz = w - (np.arange(m) / max(m - 1, 1)) * w[-1]
        xs.append(x0 + f0 + (f1 - f0) * t)
        ys.append(base + nz)
    xs.append(np.array([x0 + cps[-1][0]]))
    ys.append(np.array([cps[-1][1]]))
    return np.concatenate(xs), np.concatenate(ys)

cps1 = [(0.00, poc1 + 1.20), (0.12, poc1 + 1.70), (0.25, poc1 + 0.50),
        (0.36, poc1 + 1.05), (0.44, poc1 + 0.45), (0.50, poc1 - 0.50),
        (0.62, poc1 - 0.75), (0.78, poc1 - 0.45), (0.90, poc1 - 0.85),
        (1.00, poc1 - 0.70)]
cps2 = [(0.00, poc1 - 0.70), (0.08, poc1 - 0.35), (0.18, poc1 + 0.70),
        (0.30, poc2 - 1.60), (0.40, poc2 - 0.35), (0.48, poc2 + 0.55),
        (0.58, poc2 + 0.35), (0.70, poc2 + 0.85), (0.82, poc2 + 0.40),
        (1.00, poc2 + 0.60)]
cps3 = [(0.00, poc2 + 0.60), (0.10, poc2 + 1.30), (0.22, poc3 - 1.10),
        (0.34, poc3 - 0.35), (0.45, poc3 + 0.90), (0.56, poc3 + 0.40),
        (0.68, poc3 + 0.95), (0.80, poc3 + 0.45), (0.90, poc3 + 0.80),
        (1.00, poc3 + 0.55)]
cps4 = [(0.00, poc3 + 0.55), (0.10, poc3 + 0.20), (0.22, poc3 - 0.80),
        (0.32, poc4 - 0.55), (0.42, poc4 - 0.20), (0.52, poc4 + 0.45),
        (0.64, poc4 + 0.75), (0.76, poc4 + 0.35), (0.88, poc4 + 0.70),
        (1.00, poc4 + 0.50)]
cps5 = [(0.00, poc4 + 0.50), (0.10, poc4 - 0.35), (0.20, poc2 + 1.50),
        (0.32, poc2 + 0.85), (0.50, poc2), (0.62, poc5 - 0.60),
        (0.72, poc5 - 0.30), (0.84, poc5 - 0.65), (1.00, poc5 - 0.35)]

seg = []
seg.append(build_day(cps1, 0.0))
seg.append(build_day(cps2, 1.0))
seg.append(build_day(cps3, 2.0))
seg.append(build_day(cps4, 3.0))
seg.append(build_day(cps5, 4.0, quiet=(3, 4)))  # clean V into the nPOC touch
px = np.concatenate([s[0] for s in seg])
py = np.concatenate([s[1] for s in seg])

def first_cross(xs, ys, level, x_start):
    """First x >= x_start where the path crosses or touches `level`."""
    for i in range(len(xs) - 1):
        if xs[i + 1] < x_start:
            continue
        y0, y1 = ys[i] - level, ys[i + 1] - level
        if y0 * y1 < 0:
            xc = xs[i] + (xs[i + 1] - xs[i]) * (-y0) / (y1 - y0)
            if xc >= x_start:
                return xc
        if y1 == 0 and xs[i + 1] >= x_start:
            return xs[i + 1]
    return None

x_fill1 = first_cross(px, py, poc1, tips[0])
x_touch = first_cross(px, py, poc2, tips[1])
x_fill3 = first_cross(px, py, poc3, tips[2])
x_fill4 = first_cross(px, py, poc4, tips[3])
x_fill5 = first_cross(px, py, poc5, tips[4])

# ---------------- determinism / accuracy assertions ----------------
assert 1.0 < x_fill1 < 1.5, f'D1 POC must fill early on D2, got {x_fill1}'
assert abs(x_touch - 4.5) < 1e-9, f'nPOC touch must be exactly mid-D5, got {x_touch}'
assert 3.0 < x_fill3 < 3.4, f'D3 POC must fill early on D4, got {x_fill3}'
assert 4.0 < x_fill4 < 4.3, f'D4 POC must fill early on D5, got {x_fill4}'
assert x_fill5 is None, 'D5 POC must stay naked to the right edge'
m = (px >= tips[1]) & (px < x_touch - 1e-9)
assert py[m].min() > poc2, 'path must not cross the nPOC before the D5 touch'
m = (px >= tips[1]) & (px < x_touch - 0.05)
assert py[m].min() > poc2 + 0.05, 'path must stay clear of the nPOC until the D5 approach'
m = (px >= tips[0]) & (px <= 1.0)
assert py[m].max() < poc1 - 0.02, 'D1 tail must close below its own POC'
m = (px >= tips[2]) & (px <= 3.0)
assert py[m].min() > poc3 + 0.02, 'D3 tail must hold above its own POC'
m = (px >= tips[3]) & (px <= 4.0)
assert py[m].min() > poc4 + 0.02, 'D4 tail must hold above its own POC'

# ---------------- figure ----------------
XMAX = 6.05
fig, ax = plt.subplots(figsize=(11.5, 6.4))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

for b in range(1, 5):
    ax.axvline(b, color=GRID, lw=0.8, zorder=1)

# profiles (horizontal histograms anchored at each day's left edge)
for d in range(5):
    vol = profiles[d]
    scale = BARFRAC / vol.max()
    lo, hi = vas[d]
    for i, v in enumerate(vol):
        if v <= 0:
            continue
        if i == pocs[d]:
            col = POC_C
        elif lo <= i <= hi:
            col = VA_IN
        else:
            col = VA_OUT
        ax.barh(centers[i], v * scale, height=BIN * 0.82, left=d,
                color=col, edgecolor='none', zorder=2)

# POC extension lines
ax.plot([tips[0], x_fill1], [poc1, poc1], color=POC_C, lw=1.4, zorder=3,
        solid_capstyle='butt')
ax.plot([tips[1], 2.0], [poc2, poc2], color=POC_C, lw=1.4, zorder=3,
        solid_capstyle='butt')
ax.plot([2.0, x_touch], [poc2, poc2], color=POC_C, lw=1.2, ls=(0, (5, 3)),
        zorder=3)
ax.plot([tips[2], x_fill3], [poc3, poc3], color=POC_C, lw=1.4, zorder=3,
        solid_capstyle='butt')
ax.plot([tips[3], x_fill4], [poc4, poc4], color=POC_C, lw=1.4, zorder=3,
        solid_capstyle='butt')
ax.plot([tips[4], XMAX], [poc5, poc5], color=POC_C, lw=1.4, zorder=3,
        solid_capstyle='butt')

# price path
ax.plot(px, py, color=INK, lw=1.7, zorder=4, solid_capstyle='round')

# the nPOC revisit
ax.plot([x_touch], [poc2], marker='o', ms=9, mfc=ATTN, mec=BG, mew=1.2,
        zorder=6, ls='none')

# annotations
for xc, yc, dy in [(x_fill1, poc1, -1), (x_fill3, poc3, 1), (x_fill4, poc4, 1)]:
    ax.annotate('filled', xy=(xc, yc), xytext=(xc + 0.20, yc + dy * 0.55),
                fontsize=9, color=INK2, ha='left',
                va='bottom' if dy > 0 else 'top',
                arrowprops=dict(arrowstyle='->', color=INK2, lw=0.9))

ax.text(3.22, poc2 - 0.42, 'naked POC (nPOC) - unfinished business',
        fontsize=9, color=INK2, ha='center', va='top')
ax.annotate('revisit: magnet resolved', xy=(x_touch, poc2 - 0.06),
            xytext=(x_touch + 0.18, poc2 - 1.15), fontsize=9, color=INK2,
            ha='left', va='top',
            arrowprops=dict(arrowstyle='->', color=INK2, lw=0.9))
ax.text(tips[4] + 0.10, poc5 + 0.14, "today's POC - still open", fontsize=9,
        color=MUTED, ha='left', va='bottom')

# swatch legend (identity via markers, text stays ink-colored)
ly = 110.6
for lx, col, lab in [(0.02, VA_IN, 'value area (70%)'),
                     (1.32, VA_OUT, 'outside value'),
                     (2.50, POC_C, 'POC / POC line')]:
    ax.add_patch(plt.Rectangle((lx, ly - 0.11), 0.14, 0.26, color=col,
                               ec='none', zorder=5, clip_on=False))
    ax.text(lx + 0.21, ly, lab, fontsize=9, color=INK2, va='center')

# axes cosmetics
ax.set_xlim(-0.15, XMAX)
ax.set_ylim(96.6, 111.4)
ax.set_xticks([i + 0.5 for i in range(5)])
ax.set_xticklabels([f'Day {i + 1}' for i in range(5)], fontsize=10,
                   color=MUTED)
ax.tick_params(axis='x', length=0)
ax.tick_params(axis='y', colors=MUTED, labelsize=9, width=0.8)
ax.set_ylabel('Price', fontsize=10, color=INK2)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
for side in ('left', 'bottom'):
    ax.spines[side].set_color(BASE)

ax.set_title('Naked POCs act as magnets', fontsize=13, fontweight='bold',
             color=INK, loc='left', pad=14)

fig.text(0.005, -0.02,
         'Track outstanding nPOCs as targets; the circulating ~80%-revisit '
         'statistic is vendor folklore - measure your own market.',
         fontsize=9, color=INK2, ha='left', va='top')

fig.savefig('/home/user/a/docs/figures/fig-14-naked-poc.svg', format='svg',
            bbox_inches='tight', pad_inches=0.15, facecolor=BG)
fig.savefig('/tmp/claude-0/-home-user-a/a79ba79b-564a-5a92-9188-47a1d21553f5/'
            'scratchpad/figpng/fig-14-naked-poc.png', dpi=200,
            bbox_inches='tight', pad_inches=0.15, facecolor=BG)
print('POCs:', [round(p, 3) for p in poc_price])
print('fills:', x_fill1, x_fill3, x_fill4, 'touch:', x_touch, 'D5:', x_fill5)
