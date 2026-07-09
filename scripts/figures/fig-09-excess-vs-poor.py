import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['svg.hashsalt'] = 'vp-guide'
np.random.seed(109)
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

# ---------------- synthetic profiles ----------------
n_rows = 34
prices = np.linspace(4480.0, 4513.0, n_rows)          # 1-pt rows
row_h  = prices[1] - prices[0]

# LEFT: excess high (finished auction) -> long thin tail at the top.
# Gaussian body centred low-mid, plus an exponential taper over the top rows.
center_L = prices[13]
body_L = np.exp(-0.5 * ((prices - center_L) / 8.0) ** 2)
vol_L = 100.0 * body_L * (1.0 + 0.13 * np.random.randn(n_rows))
vol_L = np.convolve(vol_L, [0.25, 0.5, 0.25], mode='same')  # mild smoothing
vol_L = np.clip(vol_L, 0.5, None)
# enforce a monotone taper to near-zero over the top 7 rows: geometric decay
# starting from the body's level at the join, ending at a thin tip
tail_n = 7
tail_start = 0.85 * vol_L[n_rows - tail_n - 1]
tail_end = 0.015 * vol_L.max()
vol_L[-tail_n:] = np.geomspace(tail_start, tail_end, tail_n)

# RIGHT: poor high (unfinished auction) -> substantial, near-equal rows up to
# the very top row, squared-off (no taper).
center_R = prices[13]
body_R = np.exp(-0.5 * ((prices - center_R) / 7.5) ** 2)
vol_R = 100.0 * body_R * (1.0 + 0.15 * np.random.randn(n_rows))
vol_R = np.clip(vol_R, 0.5, None)
flat_n = 8
plateau = 0.62 * vol_R.max()
vol_R[-flat_n:] = plateau * (1.0 + 0.02 * np.random.randn(flat_n))  # near-equal
vol_R[-flat_n:] = np.clip(vol_R[-flat_n:], 0.97 * plateau, 1.03 * plateau)
# smooth join just below the plateau
vol_R[n_rows - flat_n - 2] = 0.5 * (vol_R[n_rows - flat_n - 3] + plateau)
vol_R[n_rows - flat_n - 1] = 0.5 * (vol_R[n_rows - flat_n - 2] + plateau)

# ---------------- structural quantities (computed) ----------------
def poc_and_value_area(vol):
    """POC = argmax; value area = standard 70% two-row expansion."""
    poc = int(np.argmax(vol))
    total = vol.sum()
    target = 0.70 * total
    lo = hi = poc
    acc = vol[poc]
    while acc < target:
        up = vol[hi + 1] + (vol[hi + 2] if hi + 2 < len(vol) else 0.0) if hi + 1 < len(vol) else -1.0
        dn = vol[lo - 1] + (vol[lo - 2] if lo - 2 >= 0 else 0.0) if lo - 1 >= 0 else -1.0
        if up >= dn and up >= 0.0:
            take = min(hi + 2, len(vol) - 1)
            acc += vol[hi + 1:take + 1].sum()
            hi = take
        elif dn >= 0.0:
            take = max(lo - 2, 0)
            acc += vol[take:lo].sum()
            lo = take
        else:
            break
    return poc, lo, hi

poc_L, lo_L, hi_L = poc_and_value_area(vol_L)
poc_R, lo_R, hi_R = poc_and_value_area(vol_R)

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 2, figsize=(9.2, 6.2), sharey=True)
fig.patch.set_facecolor(BG)

max_vol = max(vol_L.max(), vol_R.max())

specs = [
    (axes[0], vol_L, poc_L, lo_L, hi_L, 'Excess high (finished auction)',
     'durable reference — less likely revisited'),
    (axes[1], vol_R, poc_R, lo_R, hi_R, 'Poor high (unfinished auction)',
     'magnet — odds favor revisit and repair'),
]

for ax, vol, poc, lo, hi, title, caption in specs:
    ax.set_facecolor(BG)
    colors = [VA_IN if lo <= i <= hi else VA_OUT for i in range(n_rows)]
    colors[poc] = POC_C
    ax.barh(prices, vol, height=row_h * 0.82, color=colors, linewidth=0)
    # POC reference line
    ax.axhline(prices[poc], color=POC_C, linewidth=1.2, linestyle='--',
               alpha=0.85, zorder=0)
    ax.text(max_vol * 1.025, prices[poc], 'POC', color=POC_C, fontsize=9,
            va='center', ha='left', fontweight='bold', zorder=5,
            bbox=dict(facecolor=BG, edgecolor='none', pad=1.2))
    ax.set_title(title, fontsize=11, color=INK, fontweight='bold', pad=10)
    ax.set_xlim(0, max_vol * 1.14)
    ax.set_ylim(prices[0] - 1.2, prices[-1] + 2.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(BASE)
    ax.spines['bottom'].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel('volume', fontsize=9, color=MUTED)
    ax.grid(axis='x', color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.text(0.5, -0.115, caption, transform=ax.transAxes, fontsize=9,
            color=INK2, ha='center', style='italic')

axes[0].set_ylabel('price', fontsize=10, color=MUTED)

# ---------------- annotations ----------------
# LEFT: tail / single prints (point at the middle of the taper region)
tail_mid_i = n_rows - 3
axes[0].annotate('tail / single prints:\naggressive rejection',
                 xy=(vol_L[tail_mid_i] + max_vol * 0.015, prices[tail_mid_i]),
                 xytext=(max_vol * 0.42, prices[-1] + 0.4),
                 fontsize=9, color=INK,
                 arrowprops=dict(arrowstyle='->', color=INK2, linewidth=1.0),
                 ha='left', va='top')

# RIGHT: flat top (point at the squared-off edge of the top rows, with the
# text sitting in the empty space to the right of the plateau)
axes[1].annotate('flat top: no excess,\nday-timeframe\ntraders only',
                 xy=(vol_R[-1] + max_vol * 0.015, prices[-1] - 0.5 * row_h),
                 xytext=(max_vol * 0.72, prices[-5]),
                 fontsize=9, color=INK,
                 arrowprops=dict(arrowstyle='->', color=INK2, linewidth=1.0,
                                 connectionstyle='arc3,rad=-0.18'),
                 ha='left', va='top')

fig.suptitle('Excess vs poor structure at an extreme', fontsize=13,
             fontweight='bold', color=INK, y=0.985)

# small legend swatches under the title
fig.text(0.335, 0.925, ' ', fontsize=9)
legend_items = [(VA_IN, 'value area'), (VA_OUT, 'outside VA'), (POC_C, 'POC')]
x0 = 0.33
for c, lab in legend_items:
    fig.text(x0, 0.928, '■', color=c, fontsize=9, ha='left', va='center')
    fig.text(x0 + 0.017, 0.928, lab, color=INK2, fontsize=9, ha='left',
             va='center')
    x0 += 0.017 + 0.011 * len(lab) + 0.025

fig.subplots_adjust(top=0.85, bottom=0.14, left=0.09, right=0.97, wspace=0.14)

fig.savefig('/home/user/a/docs/figures/fig-09-excess-vs-poor.svg',
            format='svg', bbox_inches='tight', pad_inches=0.15,
            facecolor=BG)
fig.savefig('/tmp/claude-0/-home-user-a/a79ba79b-564a-5a92-9188-47a1d21553f5/'
            'scratchpad/figpng/fig-09-excess-vs-poor.png',
            dpi=200, bbox_inches='tight', pad_inches=0.15, facecolor=BG)

# quick self-checks
top4_L = vol_L[-4:]
assert np.all(np.diff(top4_L) < 0), 'left top rows must shrink monotonically'
assert vol_L[-1] < 0.05 * vol_L.max(), 'left tip must be near-zero'
top4_R = vol_R[-4:]
assert top4_R.min() > 0.9 * top4_R.max(), 'right top rows must be near-equal'
assert top4_R.min() > 0.5 * vol_R.max(), 'right top rows must be substantial'
print('POC L:', prices[poc_L], 'VA L:', prices[lo_L], '-', prices[hi_L])
print('POC R:', prices[poc_R], 'VA R:', prices[lo_R], '-', prices[hi_R])
print('ok')
