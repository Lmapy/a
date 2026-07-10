import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['svg.hashsalt'] = 'vp-guide'
np.random.seed(106)
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ---- palette ----------------------------------------------------------
BG      = '#fcfcfb'
INK     = '#0b0b0b'
INK2    = '#52514e'
MUTED   = '#898781'
GRID    = '#e1e0d9'
BASE    = '#c3c2b7'
BULL    = '#1baf7a'
BEAR    = '#e34948'
ATTN    = '#eda100'

N    = 91                      # minutes 0..90
t    = np.arange(N, dtype=float)
OPEN = 100.0

# ---- panel 1: Open-Drive ----------------------------------------------
# leaves the open immediately and never comes back
drift1 = 4.0 * np.linspace(0, 1, N) ** 0.9
noise1 = np.cumsum(np.random.randn(N) * 0.14)
noise1 -= np.linspace(noise1[0], noise1[-1], N)     # pin both ends
p1 = OPEN + drift1 + noise1
p1[0] = OPEN
# enforce: after the first step the path stays strictly above the open
floor = OPEN + 0.4
p1[1:] = floor + np.abs(p1[1:] - floor)             # reflect around the floor
assert p1[1:].min() > OPEN + 0.2

# ---- panel 2: Open-Test-Drive -----------------------------------------
PRIOR_LOW = OPEN - 1.3
p2 = np.empty(N)
# probe down through prior low (minutes 0..17), low at ~minute 17
probe_end = 17
down = OPEN + (PRIOR_LOW - 0.9 - OPEN) * np.linspace(0, 1, probe_end + 1) ** 1.2
down += np.cumsum(np.random.randn(probe_end + 1) * 0.10) * np.linspace(0, 1, probe_end + 1)
down[0] = OPEN
p2[:probe_end + 1] = down
low_idx2 = int(np.argmin(p2[:probe_end + 1]))
# drive up from the failed test, never returning to the open
drive = p2[probe_end] + (OPEN + 3.6 - p2[probe_end]) * np.linspace(0, 1, N - probe_end) ** 0.85
dn = np.cumsum(np.random.randn(N - probe_end) * 0.12)
drive += dn - np.linspace(dn[0], dn[-1], N - probe_end)
p2[probe_end:] = drive
cap2 = OPEN + 4.6
p2 = np.where(p2 > cap2, cap2 - (p2 - cap2) * 0.5, p2)   # keep inside panel
# index where the drive first crosses back above the open
cross2 = probe_end + int(np.argmax(p2[probe_end:] > OPEN))
# enforce: once back above the open it stays above
p2[cross2:] = np.maximum(p2[cross2:], OPEN + 0.35)
low_idx2 = int(np.argmin(p2))
assert p2[low_idx2] < PRIOR_LOW - 0.3          # visibly breaks prior low
assert p2[cross2:].min() > OPEN + 0.2          # stays above open afterwards

# ---- panel 3: Open-Rejection-Reverse ----------------------------------
peak_i = 34
up3 = OPEN + 2.6 * np.linspace(0, 1, peak_i + 1) ** 0.9
up3 += np.cumsum(np.random.randn(peak_i + 1) * 0.12) * np.linspace(0, 1, peak_i + 1)
up3[0] = OPEN
up3[1:] = np.maximum(up3[1:], OPEN + 0.15)
p3 = np.empty(N)
p3[:peak_i + 1] = up3
down3 = up3[-1] + (OPEN - 3.2 - up3[-1]) * np.linspace(0, 1, N - peak_i) ** 1.05
down3 += np.cumsum(np.random.randn(N - peak_i) * 0.12)
down3 -= down3[0] - up3[-1]
p3[peak_i:] = down3
# exact crossing index on the way down (computed, not hand placed)
below = np.nonzero(p3[peak_i:] < OPEN)[0]
cross3 = peak_i + below[0]
# enforce it keeps going lower afterwards
p3[cross3:] = np.minimum(p3[cross3:], OPEN - 0.05 - 0.04 * (np.arange(N - cross3)))
p3[cross3:] = np.minimum.accumulate(p3[cross3:] + np.abs(np.random.randn(N - cross3)) * 0.05)
assert p3[cross3 - 1] >= OPEN and p3[cross3] < OPEN
# interpolated crossing point for the annotation
frac = (OPEN - p3[cross3 - 1]) / (p3[cross3] - p3[cross3 - 1])
x_cross3 = (cross3 - 1) + frac

# ---- panel 4: Open-Auction --------------------------------------------
wob4 = np.cumsum(np.random.randn(N) * 0.06)
wob4 -= np.linspace(wob4[0], wob4[-1], N)       # detrend so it keeps rotating
p4 = OPEN + 1.05 * np.sin(t / 90 * 2 * np.pi * 3.4) + wob4
p4[0] = OPEN
n_cross4 = int(np.sum(np.diff(np.sign(p4[1:] - OPEN)) != 0))
assert n_cross4 >= 4, n_cross4

# ---- figure ------------------------------------------------------------
fig, axs = plt.subplots(1, 4, figsize=(14.0, 4.6), sharex=True)
fig.patch.set_facecolor(BG)

titles = ['Open-Drive\n(highest conviction)',
          'Open-Test-Drive',
          'Open-Rejection-Reverse',
          'Open-Auction\n(lowest conviction)']
ranks = ['conviction rank 1 — strongest',
         'conviction rank 2',
         'conviction rank 3',
         'conviction rank 4 — weakest']

YLIM = (OPEN - 4.6, OPEN + 5.2)

for ax, title, rank in zip(axs, titles, ranks):
    ax.set_facecolor(BG)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9, left=False, labelleft=False)
    ax.set_xticks([0, 30, 60, 90])
    ax.set_xlim(-4, 94)
    ax.set_ylim(*YLIM)
    ax.grid(axis='x', color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel('minutes since open', fontsize=9, color=MUTED)
    ax.set_title(title, fontsize=10, fontweight='bold', color=INK, pad=8)
    ax.text(0.5, -0.24, rank, transform=ax.transAxes, ha='center',
            fontsize=9, color=INK2)
    # dashed open reference line
    ax.axhline(OPEN, color=MUTED, linestyle='--', linewidth=1.2, zorder=1)
    # open dot + label
    ax.plot(0, OPEN, 'o', ms=6, color=INK, zorder=5)

axs[0].text(2.5, OPEN - 0.75, 'open', fontsize=9, color=INK2, va='top')
axs[1].text(2.5, OPEN + 0.45, 'open', fontsize=9, color=INK2, va='bottom')
axs[2].text(2.5, OPEN - 0.75, 'open', fontsize=9, color=INK2, va='top')
axs[3].text(2.5, OPEN - 0.75, 'open', fontsize=9, color=INK2, va='top')

# panel 1: whole path is the up-drive -> aqua
axs[0].plot(t, p1, color=BULL, linewidth=1.8, solid_capstyle='round', zorder=3)
axs[0].annotate('never returns\nto the open', xy=(45, np.interp(45, t, p1) - 0.3),
                xytext=(52, OPEN - 3.1), fontsize=9, color=INK2,
                arrowprops=dict(arrowstyle='->', color=INK2, lw=0.9))

# panel 2: prior-low reference, ink test leg, aqua drive leg
axs[1].axhline(PRIOR_LOW, color=BASE, linestyle='--', linewidth=1.2, zorder=1)
axs[1].text(93, PRIOR_LOW - 0.15, 'prior low', fontsize=9, color=MUTED,
            ha='right', va='top')
axs[1].plot(t[:low_idx2 + 1], p2[:low_idx2 + 1], color=INK, linewidth=1.8,
            solid_capstyle='round', zorder=3)
axs[1].plot(t[low_idx2:], p2[low_idx2:], color=BULL, linewidth=1.8,
            solid_capstyle='round', zorder=3)
axs[1].plot(t[low_idx2], p2[low_idx2], 'o', ms=8, mfc='none', mec=ATTN,
            mew=1.6, zorder=6)
axs[1].annotate('failed test =\npremium location',
                xy=(t[low_idx2] + 1.5, p2[low_idx2] - 0.15),
                xytext=(34, OPEN - 4.35), fontsize=9, color=INK2, va='bottom',
                arrowprops=dict(arrowstyle='->', color=INK2, lw=0.9))

# panel 3: ink up-leg, red down-drive, annotated re-cross of the open
axs[2].plot(t[:peak_i + 1], p3[:peak_i + 1], color=INK, linewidth=1.8,
            solid_capstyle='round', zorder=3)
axs[2].plot(t[peak_i:], p3[peak_i:], color=BEAR, linewidth=1.8,
            solid_capstyle='round', zorder=3)
axs[2].annotate('back through the open\n= failing conviction',
                xy=(x_cross3, OPEN), xytext=(24, OPEN + 3.6),
                fontsize=9, color=INK2,
                arrowprops=dict(arrowstyle='->', color=INK2, lw=0.9))

# panel 4: rotational two-sided auction, all ink
axs[3].plot(t, p4, color=INK, linewidth=1.8, solid_capstyle='round', zorder=3)
axs[3].annotate('rotates across the open\n(%d crossings)' % n_cross4,
                xy=(60, np.interp(60, t, p4)), xytext=(28, OPEN - 3.9),
                fontsize=9, color=INK2,
                arrowprops=dict(arrowstyle='->', color=INK2, lw=0.9))

fig.suptitle('The open types: conviction ladder for the first hour',
             fontsize=13, fontweight='bold', color=INK, y=1.005)
fig.text(0.5, -0.045,
         'Aggressive one-sided opens (drive, test-drive) signal high directional conviction; '
         'a rejected drive or a rotating open signals a balanced, lower-odds session.',
         ha='center', fontsize=9, color=INK2)

fig.subplots_adjust(left=0.03, right=0.985, top=0.82, bottom=0.22, wspace=0.18)

fig.savefig('/home/user/a/docs/figures/fig-06-open-types.svg', format='svg',
            facecolor=BG, bbox_inches='tight', pad_inches=0.15)
fig.savefig('/tmp/claude-0/-home-user-a/a79ba79b-564a-5a92-9188-47a1d21553f5/'
            'scratchpad/figpng/fig-06-open-types.png', dpi=200,
            facecolor=BG, bbox_inches='tight', pad_inches=0.15)

# ---- pixel-level accuracy checks on the data ---------------------------
print('p1 min after open:', p1[1:].min() - OPEN)
print('p2 low vs prior low:', p2[low_idx2] - PRIOR_LOW,
      '| after cross min vs open:', p2[cross2:].min() - OPEN)
print('p3 crossing at minute', round(x_cross3, 1),
      '| after cross max vs open:', p3[cross3:].max() - OPEN)
print('p4 open-line crossings:', n_cross4)
