import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['svg.hashsalt'] = 'vp-guide'
np.random.seed(105)
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

N_MIN = 390          # minutes in the session
IB_MIN = 60          # first hour defines the initial balance


def ou(n, mu, sigma, theta, x0):
    """Mean-reverting (Ornstein-Uhlenbeck-ish) minute path."""
    x = np.empty(n)
    x[0] = x0
    eps = np.random.normal(0.0, sigma, n)
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (mu - x[i - 1]) + eps[i]
    return x


# ------------------------------------------------- synthetic minute paths
def path_normal():
    t = np.linspace(0, 1, IB_MIN)
    first = 100 + 8.5 * np.sin(2 * np.pi * 1.25 * t + 0.3) + np.random.normal(0, 0.30, IB_MIN)
    rest = ou(N_MIN - IB_MIN, 100.0, 0.55, 0.05, first[-1])
    return np.concatenate([first, rest])


def path_normal_variation():
    t = np.linspace(0, 1, IB_MIN)
    first = 100 + 4.3 * np.sin(2 * np.pi * 1.25 * t + 0.5) + np.random.normal(0, 0.25, IB_MIN)
    a = ou(140, 99.8, 0.35, 0.09, first[-1])
    b = np.linspace(a[-1], 106.8, 70) + np.random.normal(0, 0.22, 70)
    c = ou(120, 106.3, 0.40, 0.09, b[-1])
    return np.concatenate([first, a, b, c])


def path_trend():
    t = np.linspace(0, 1, IB_MIN)
    first = 95.8 + 1.5 * np.sin(2 * np.pi * 1.5 * t) + np.random.normal(0, 0.20, IB_MIN)
    tt = np.linspace(0, 1, N_MIN - IB_MIN)
    rest = (np.linspace(first[-1], 115.0, N_MIN - IB_MIN)
            + 0.55 * np.sin(2 * np.pi * 5 * tt)
            + np.random.normal(0, 0.18, N_MIN - IB_MIN))
    return np.concatenate([first, rest])


def path_double():
    t = np.linspace(0, 1, IB_MIN)
    first = 97 + 1.7 * np.sin(2 * np.pi * 1.3 * t + 0.4) + np.random.normal(0, 0.22, IB_MIN)
    a = ou(110, 97.0, 0.38, 0.09, first[-1])                       # lower distribution
    b = np.linspace(a[-1], 108.0, 18) + np.random.normal(0, 0.15, 18)  # fast repricing
    c = ou(N_MIN - IB_MIN - 110 - 18, 108.2, 0.42, 0.09, b[-1])    # upper distribution
    return np.concatenate([first, a, b, c])


def path_nontrend():
    t = np.linspace(0, 1, IB_MIN)
    first = 100 + 1.8 * np.sin(2 * np.pi * 1.25 * t + 0.2) + np.random.normal(0, 0.18, IB_MIN)
    rest = ou(N_MIN - IB_MIN, 100.0, 0.42, 0.09, first[-1])
    return np.concatenate([first, rest])


def path_neutral():
    t = np.linspace(0, 1, IB_MIN)
    first = 100 + 4.2 * np.sin(2 * np.pi * 1.4 * t + 0.3) + np.random.normal(0, 0.25, IB_MIN)
    a = np.linspace(first[-1], 107.5, 80) + np.random.normal(0, 0.28, 80)   # above the IB
    b = np.linspace(107.5, 93.0, 130) + np.random.normal(0, 0.28, 130)      # below the IB
    c = np.linspace(93.0, 100.3, 120) + np.random.normal(0, 0.25, 120)      # back to middle
    return np.concatenate([first, a, b, c])


# ------------------------------------------- value-area (70%) expansion
def value_area(vol):
    """Classic POC-out expansion: repeatedly add the larger of the
    two-row sums above vs below until >= 70% of total volume."""
    poc = int(np.argmax(vol))
    total = vol.sum()
    target = 0.70 * total
    lo = hi = poc
    cum = vol[poc]
    n = len(vol)
    while cum < target and (lo > 0 or hi < n - 1):
        up = vol[hi + 1:hi + 3].sum() if hi < n - 1 else -1.0
        dn = vol[max(lo - 2, 0):lo].sum() if lo > 0 else -1.0
        if up >= dn:
            cum += up
            hi = min(hi + 2, n - 1)
        else:
            cum += dn
            lo = max(lo - 2, 0)
    return lo, hi, poc


# ------------------------------------------------------------- assemble
panels = [
    ('Normal day',          'range ~= IB',                 path_normal(),           False),
    ('Normal variation',    'extension < 2x IB',           path_normal_variation(), False),
    ('Trend day',           'one-timeframing, do not fade', path_trend(),           True),
    ('Double-distribution', 'neck = reference',            path_double(),           False),
    ('Nontrend day',        'stores energy',               path_nontrend(),         False),
    ('Neutral day',         'both sides active',           path_neutral(),          True),
]

# shared vertical scale so day-type shapes are comparable
ranges = [p[2].max() - p[2].min() for p in panels]
span = max(ranges) * 1.14
binw = span / 56.0

# ------------------------------------------------ sanity/accuracy checks
fracs = {}
for name, _, path, _ in panels:
    ib_lo, ib_hi = path[:IB_MIN].min(), path[:IB_MIN].max()
    fracs[name] = (ib_hi - ib_lo) / (path.max() - path.min())
assert min(fracs, key=fracs.get) == 'Trend day', fracs
assert max(fracs, key=fracs.get) == 'Normal day', fracs
assert fracs['Normal day'] > fracs['Nontrend day'] + 0.04, fracs

p_nv = panels[1][2]
nv_ib = p_nv[:IB_MIN].max() - p_nv[:IB_MIN].min()
assert (p_nv.max() - p_nv.min()) < 2.0 * nv_ib                      # extension < 2x IB
assert p_nv.max() > p_nv[:IB_MIN].max() + 1.0                       # one-sided: extends up...
assert p_nv.min() > p_nv[:IB_MIN].min() - 0.6                       # ...but not down

p_tr = panels[2][2]
assert p_tr[-1] > p_tr.min() + 0.88 * (p_tr.max() - p_tr.min())     # trend closes at extreme

p_ne = panels[5][2]
assert p_ne.max() > p_ne[:IB_MIN].max() + 1.0                       # neutral: beyond IB high
assert p_ne.min() < p_ne[:IB_MIN].min() - 1.0                       # neutral: beyond IB low
mid_lo = p_ne.min() + 0.30 * (p_ne.max() - p_ne.min())
mid_hi = p_ne.min() + 0.70 * (p_ne.max() - p_ne.min())
assert mid_lo < p_ne[-1] < mid_hi                                   # close mid-range

# ------------------------------------------------------------------ plot
fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.6))
fig.set_facecolor(BG)
fig.subplots_adjust(left=0.045, right=0.975, top=0.865, bottom=0.075,
                    wspace=0.30, hspace=0.42)

for ax, (name, tell, path, show_close) in zip(axes.flat, panels):
    ax.set_facecolor(BG)
    vols = np.random.lognormal(0.0, 0.35, N_MIN)

    dlo, dhi = path.min(), path.max()
    nb = max(int(np.ceil((dhi - dlo) / binw)), 8)
    edges = np.linspace(dlo, dlo + nb * binw, nb + 1)
    hist, _ = np.histogram(path, bins=edges, weights=vols)
    centers = 0.5 * (edges[:-1] + edges[1:])

    va_lo, va_hi, poc = value_area(hist)
    colors = [VA_IN if va_lo <= i <= va_hi else VA_OUT for i in range(nb)]
    colors[poc] = POC_C
    ax.barh(centers, hist, height=binw * 0.88, color=colors, linewidth=0)

    xmax = hist.max()
    ib_lo, ib_hi = path[:IB_MIN].min(), path[:IB_MIN].max()

    # baseline of the profile
    ax.plot([0, 0], [dlo - 0.25, dhi + 0.25], color=BASE, lw=1.0, zorder=1)

    # IB bracket: spans exactly the first-hour range
    xb = -0.11 * xmax
    cap = 0.05 * xmax
    ax.plot([xb, xb], [ib_lo, ib_hi], color=INK, lw=1.8, solid_capstyle='butt')
    ax.plot([xb, xb + cap], [ib_lo, ib_lo], color=INK, lw=1.8, solid_capstyle='butt')
    ax.plot([xb, xb + cap], [ib_hi, ib_hi], color=INK, lw=1.8, solid_capstyle='butt')
    ax.text(xb - 0.06 * xmax, 0.5 * (ib_lo + ib_hi), 'IB',
            fontsize=9, color=INK, ha='right', va='center')

    # close tick where the spec calls it out
    if show_close:
        close = path[-1]
        ax.plot([1.05 * xmax, 1.16 * xmax], [close, close],
                color=INK, lw=1.6, solid_capstyle='butt')
        ax.text(1.21 * xmax, close, 'close',
                fontsize=9, color=INK2, ha='left', va='center')
        ax.set_xlim(-0.34 * xmax, 1.62 * xmax)
    else:
        ax.set_xlim(-0.34 * xmax, 1.14 * xmax)

    mid = 0.5 * (dlo + dhi)
    ax.set_ylim(mid - span / 2, mid + span / 2)

    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(name, fontsize=10, fontweight='bold', color=INK,
                 loc='left', pad=17)
    ax.text(0.0, 1.035, tell, transform=ax.transAxes,
            fontsize=9, color=INK2, ha='left', va='bottom')

fig.suptitle('Day types and the initial balance tell',
             fontsize=13, fontweight='bold', color=INK, x=0.045, ha='left', y=0.965)
fig.text(0.045, 0.018,
         'Profiles built from synthetic minute paths (same vertical scale). Blue rows = 70% value area '
         'around the point of control (orange row); the bracket marks the first-hour initial balance.',
         fontsize=9, color=INK2, ha='left')

fig.savefig('/home/user/a/docs/figures/fig-05-day-types.svg',
            format='svg', bbox_inches='tight', pad_inches=0.15, facecolor=BG)
fig.savefig('/tmp/claude-0/-home-user-a/a79ba79b-564a-5a92-9188-47a1d21553f5/scratchpad/figpng/fig-05-day-types.png',
            dpi=200, bbox_inches='tight', pad_inches=0.15, facecolor=BG)

print({k: round(v, 3) for k, v in fracs.items()})
print('ok')
