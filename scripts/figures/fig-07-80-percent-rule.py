import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import numpy as np
matplotlib.rcParams['svg.hashsalt'] = 'vp-guide'; np.random.seed(107)
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
from matplotlib.patches import Rectangle

# ---------------- palette ----------------
BG      = '#fcfcfb'
INK     = '#0b0b0b'
INK2    = '#52514e'
MUTED   = '#898781'
GRID    = '#e1e0d9'
AXIS    = '#c3c2b7'
VA_IN   = '#2a78d6'
VA_OUT  = '#9ec5f4'
POC_C   = '#eb6834'
BULL    = '#1baf7a'
BEAR    = '#e34948'
ATTN    = '#eda100'
VA_FILL = '#cde2fb'

# ---------------- yesterday's profile (synthetic) ----------------
step = 0.5
centers = np.arange(100.0, 120.0 + step, step)
n = len(centers)
vol = (1.00 * np.exp(-0.5 * ((centers - 110.2) / 2.8) ** 2)
       + 0.35 * np.exp(-0.5 * ((centers - 106.5) / 1.5) ** 2)
       + 0.05 * np.random.rand(n))
vol = np.clip(vol, 0.02, None)

poc = int(np.argmax(vol))                      # POC = argmax, computed

# --- real 70% value-area expansion (contiguous around POC) ---
total = vol.sum()
target = 0.70 * total
lo = hi = poc
acc = vol[poc]
while acc < target:
    up = vol[hi + 1] if hi + 1 < n else -np.inf
    dn = vol[lo - 1] if lo - 1 >= 0 else -np.inf
    if up >= dn:
        hi += 1; acc += up
    else:
        lo -= 1; acc += dn
VAH = centers[hi] + step / 2.0
VAL = centers[lo] - step / 2.0
POC_PRICE = centers[poc]
H, L = VAH, VAL
R = H - L

# ---------------- today's 30-min bars (relative to computed VA) ----------------
opens_f  = np.array([0.35, 0.22, 0.08, -0.14, -0.28, -0.20, -0.45, -0.62, -0.55, -0.80])
closes_f = np.array([0.22, 0.08, -0.14, -0.28, -0.20, -0.45, -0.62, -0.55, -0.80, -1.00])
opens  = H + opens_f * R
closes = H + closes_f * R
wick_hi = np.abs(np.random.randn(10)) * 0.030 * R + 0.02 * R
wick_lo = np.abs(np.random.randn(10)) * 0.030 * R + 0.02 * R
highs = np.maximum(opens, closes) + wick_hi
lows  = np.minimum(opens, closes) - wick_lo

# enforce structural constraints (still computed against real VAH/VAL)
lows[0] = max(lows[0], H + 0.10 * R)           # first bar fully above value
lows[1] = max(lows[1], H + 0.02 * R)
for i in (3, 4):                               # acceptance bars fully inside VA
    highs[i] = min(highs[i], H - 0.03 * R)
    lows[i]  = max(lows[i],  L + 0.05 * R)
closes[9] = L                                   # session ends at the VAL target
lows[9] = min(lows[9], L)

assert opens[0] > VAH, 'first bar must open above VAH'

# find FIRST two consecutive bars fully inside the value area (computed)
inside = (highs < VAH) & (lows > VAL)
acc_i = next(i for i in range(9) if inside[i] and inside[i + 1])

# ---------------- figure ----------------
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(10.5, 6.3), sharey=True,
    gridspec_kw={'width_ratios': [1.0, 1.9], 'wspace': 0.04})
fig.patch.set_facecolor(BG)

for ax in (ax1, ax2):
    ax.set_facecolor(BG)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.axhspan(VAL, VAH, color=VA_FILL, alpha=0.45, zorder=0)
    ax.axhline(VAH, ls='--', lw=1.2, color=INK2, zorder=2)
    ax.axhline(VAL, ls='--', lw=1.2, color=INK2, zorder=2)

ax1.set_ylim(99.0, 120.8)

# left: profile
colors = [VA_OUT] * n
for i in range(lo, hi + 1):
    colors[i] = VA_IN
colors[poc] = POC_C
ax1.barh(centers, vol, height=step * 0.82, color=colors, zorder=3)
vmax = vol.max()
ax1.set_xlim(0, vmax * 1.12)
ax1.set_xticks([])
ax1.set_xlabel('volume', fontsize=9, color=MUTED)
ax1.set_title("yesterday's profile", fontsize=10, color=INK2, loc='left', pad=8)
ax1.text(vmax * 1.08, VAH + 0.06 * R, 'VAH', fontsize=9, color=INK2,
         ha='right', va='bottom')
ax1.text(vmax * 1.08, VAL - 0.06 * R, 'VAL', fontsize=9, color=INK2,
         ha='right', va='top')
ax1.text(vol[poc] + vmax * 0.03, POC_PRICE, 'POC', fontsize=9, color=POC_C,
         ha='left', va='center')

# right: today's session
ax2.set_xlim(-1.2, 12.8)
ax2.set_title('today: 30-minute bars', fontsize=10, color=INK2, loc='left', pad=8)
ax2.set_xticks([0, 3, 6, 9])
ax2.set_xticklabels(['09:30', '11:00', '12:30', '14:00'])
ax2.tick_params(axis='y', left=False)

tick = 0.28
for i in range(10):
    c = BULL if closes[i] >= opens[i] else BEAR
    ax2.vlines(i, lows[i], highs[i], color=c, lw=1.7, zorder=4)
    ax2.hlines(opens[i], i - tick, i, color=c, lw=1.7, zorder=4)
    ax2.hlines(closes[i], i, i + tick, color=c, lw=1.7, zorder=4)

# open dot + label (open of first bar, above extended VAH)
ax2.plot(-tick, opens[0], 'o', ms=6.5, color=INK, zorder=6)
ax2.annotate('open outside value',
             xy=(-tick, opens[0]), xytext=(1.1, H + 0.50 * R),
             fontsize=9, color=INK2, ha='left', va='center',
             arrowprops=dict(arrowstyle='->', color=INK2, lw=0.9))

# acceptance box around first two consecutive inside bars (computed)
pad = 0.03 * R
box_lo = min(lows[acc_i], lows[acc_i + 1]) - pad
box_hi = max(highs[acc_i], highs[acc_i + 1]) + pad
ax2.add_patch(Rectangle((acc_i - 0.45, box_lo), 1.9, box_hi - box_lo,
                        fill=False, edgecolor=ATTN, lw=1.8, zorder=5))
ax2.annotate('acceptance: two 30-min\nperiods inside',
             xy=(acc_i + 0.5, box_lo), xytext=(-0.6, H - 0.78 * R),
             fontsize=9, color=INK2, ha='left', va='center',
             arrowprops=dict(arrowstyle='->', color=INK2, lw=0.9,
                             connectionstyle='arc3,rad=-0.15'))

# target arrow: down to VAL
ax2.annotate('', xy=(10.9, VAL), xytext=(10.9, box_lo),
             arrowprops=dict(arrowstyle='->', color=BEAR, lw=2.2))
ax2.text(11.2, L - 0.16 * R, 'target: traverse the full value area',
         fontsize=9, color=INK2, ha='right', va='top')

# stop marker: short red dash above VAH after re-entry
stop_y = H + 0.08 * R
ax2.hlines(stop_y, 5.3, 6.5, color=BEAR, lw=2.6, zorder=5)
ax2.text(6.8, stop_y, 'stop: re-rejection above VAH',
         fontsize=9, color=INK2, ha='left', va='center')

ax2.grid(axis='y', color=GRID, lw=0.6, zorder=0)
ax1.grid(axis='y', color=GRID, lw=0.6, zorder=0)

fig.suptitle('The 80% rule (value-area rule)', fontsize=13, fontweight='bold',
             color=INK, x=0.06, ha='left', y=0.985)

fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.13)
fig.text(0.06, 0.025,
         'Claimed ~80% (Profile Reports, 1987-91); independent tests on ES '
         'measure ~60-67%. Size for the measured number.',
         fontsize=9, color=INK2, ha='left')

fig.savefig('/home/user/a/docs/figures/fig-07-80-percent-rule.svg',
            format='svg', bbox_inches='tight', pad_inches=0.15)
fig.savefig('/tmp/claude-0/-home-user-a/a79ba79b-564a-5a92-9188-47a1d21553f5/'
            'scratchpad/figpng/fig-07-80-percent-rule.png',
            dpi=200, bbox_inches='tight', pad_inches=0.15)
print('POC', POC_PRICE, 'VAH', VAH, 'VAL', VAL, 'acceptance bars', acc_i, acc_i + 1)
print('open0', opens[0], '> VAH:', opens[0] > VAH)
