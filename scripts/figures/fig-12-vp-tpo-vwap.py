import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['svg.hashsalt'] = 'vp-guide'
np.random.seed(112)
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
VWAP_C  = '#4a3aa7'

# ---------------- synthetic 1-minute session ----------------
# 390 minutes = 13 x 30-min TPO brackets (A..M).
# Design: heavy volume early while price chops LOW (volume POC low),
# then a drift up and a long, quiet, tight balance HIGH (TPO POC high),
# VWAP (volume-weighted mean) lands in between.
N = 390
t = np.arange(N)

price = np.empty(N)
price[0] = 99.6
for i in range(1, N):
    if i < 120:                      # early low balance, wide + noisy
        anchor, sigma, k = 99.5, 0.16, 0.10
    elif i < 210:                    # transition: drift up
        frac = (i - 120) / 90.0
        anchor, sigma, k = 99.5 + 2.85 * frac, 0.13, 0.12
    else:                            # late high balance, tight + quiet
        anchor, sigma, k = 102.35, 0.07, 0.18
    price[i] = price[i-1] + k * (anchor - price[i-1]) + sigma * np.random.randn()

# volume: heavy early, decaying through the day, small noise floor
vol = 1300.0 * np.exp(-t / 130.0) + 140.0
vol *= (1.0 + 0.25 * np.random.rand(N))
vol = vol.astype(float)

# ---------------- price rows ----------------
ROW = 0.25
lo = np.floor(price.min() / ROW) * ROW
hi = np.ceil(price.max() / ROW) * ROW
edges = np.arange(lo, hi + ROW / 2, ROW)
centers = (edges[:-1] + edges[1:]) / 2.0
nrows = len(centers)

def row_of(p):
    return int(np.clip(np.floor((p - lo) / ROW), 0, nrows - 1))

# ---------------- volume-at-price ----------------
vap = np.zeros(nrows)
for p, v in zip(price, vol):
    vap[row_of(p)] += v

poc_idx = int(np.argmax(vap))          # volume POC = mode of volume-at-price
poc_price = centers[poc_idx]

# real 70% value-area expansion from the POC
target = 0.70 * vap.sum()
va = {poc_idx}
acc = vap[poc_idx]
up, dn = poc_idx + 1, poc_idx - 1
while acc < target and (up < nrows or dn >= 0):
    up_sum = vap[up:up+2].sum() if up < nrows else -1.0
    dn_sum = vap[max(dn-1, 0):dn+1].sum() if dn >= 0 else -1.0
    if up_sum >= dn_sum:
        for j in (up, up + 1):
            if j < nrows and acc < target:
                va.add(j); acc += vap[j]
        up += 2
    else:
        for j in (dn, dn - 1):
            if j >= 0 and acc < target:
                va.add(j); acc += vap[j]
        dn -= 2
vah = centers[max(va)] + ROW / 2
val = centers[min(va)] - ROW / 2

# ---------------- TPO (time-at-price) ----------------
BRK = 30
nbr = N // BRK
letters = [chr(ord('A') + i) for i in range(nbr)]
tpo_rows = [[] for _ in range(nrows)]          # letters stacked per row
for b in range(nbr):
    seg = price[b*BRK:(b+1)*BRK]
    r0, r1 = row_of(seg.min()), row_of(seg.max())
    for r in range(r0, r1 + 1):
        tpo_rows[r].append(letters[b])

tpo_counts = np.array([len(r) for r in tpo_rows])
tpo_poc_idx = int(np.argmax(tpo_counts))       # TPO POC = mode of time-at-price
tpo_poc_price = centers[tpo_poc_idx]
# the TPO mode must be unique so the widest stack is unambiguous
assert (tpo_counts == tpo_counts.max()).sum() == 1

# ---------------- running VWAP + volume-weighted stdev bands ----------------
cv = np.cumsum(vol)
vwap = np.cumsum(price * vol) / cv
var = np.cumsum(vol * price**2) / cv - vwap**2
sd = np.sqrt(np.maximum(var, 0.0))
vwap_end = vwap[-1]

# sanity: the three references must be distinct rows
assert row_of(poc_price) != row_of(tpo_poc_price)
assert row_of(vwap_end) not in (row_of(poc_price), row_of(tpo_poc_price))

# ---------------- figure ----------------
fig, axes = plt.subplots(
    1, 3, figsize=(11.6, 6.4), sharey=True,
    gridspec_kw={'width_ratios': [1.0, 1.35, 2.1], 'wspace': 0.06})
fig.patch.set_facecolor(BG)

# y-limits must contain the full +-2 sigma envelope so the bands are
# never clipped asymmetrically
ylo = min(lo, (vwap - 2*sd).min()) - 0.3
yhi = max(hi, (vwap + 2*sd).max()) + 0.3
for ax in axes:
    ax.set_facecolor(BG)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.set_ylim(ylo, yhi)

ax1, ax2, ax3 = axes
ax1.set_ylabel('Price', fontsize=10, color=INK2)

# ----- panel 1: volume profile -----
colors = [VA_OUT] * nrows
for j in va:
    colors[j] = VA_IN
colors[poc_idx] = POC_C
ax1.barh(centers, vap, height=ROW * 0.82, color=colors, edgecolor='none')
ax1.set_title('Volume profile\n(volume-at-price)', fontsize=10, color=INK, pad=8)
ax1.set_xlabel('Volume', fontsize=9, color=MUTED)
ax1.set_xlim(0, vap.max() * 1.42)
ax1.set_xticks([])
ax1.spines['bottom'].set_visible(False)
ax1.annotate('POC = mode', xy=(vap[poc_idx] * 1.01, poc_price),
             xytext=(vap.max() * 0.68, poc_price - 1.05),
             fontsize=9, color=INK, ha='center', va='center',
             arrowprops=dict(arrowstyle='->', color=INK2, lw=0.9))
for yv, lab in ((vah, 'VAH'), (val, 'VAL')):
    ax1.axhline(yv, color=BASE, lw=1.0, ls=(0, (4, 3)), zorder=0)
    ax1.text(vap.max() * 1.38, yv + 0.07, lab, fontsize=8, color=MUTED,
             ha='right', va='bottom')

# ----- panel 2: TPO / market profile -----
max_stack = tpo_counts.max()
for r in range(nrows):
    for k, L in enumerate(tpo_rows[r]):
        ax2.text(k + 0.62, centers[r], L, fontsize=7.5,
                 family='DejaVu Sans Mono', color=INK2,
                 ha='center', va='center')
ax2.set_title('TPO / Market Profile\n(time-at-price)', fontsize=10, color=INK, pad=8)
ax2.set_xlim(-1.6, max_stack + 3.4)
ax2.set_xticks([])
ax2.set_xlabel('30-min brackets A–%s' % letters[-1], fontsize=9, color=MUTED)
ax2.spines['bottom'].set_visible(False)
ax2.plot([-0.55], [tpo_poc_price], marker='o', ms=6, color=POC_C, clip_on=False)
ax2.annotate('TPO POC', xy=(tpo_counts[tpo_poc_idx] + 0.4, tpo_poc_price),
             xytext=(tpo_counts[tpo_poc_idx] + 1.1, tpo_poc_price + 0.75),
             fontsize=9, color=INK, ha='left', va='center',
             arrowprops=dict(arrowstyle='->', color=INK2, lw=0.9))

# ----- panel 3: VWAP with bands -----
ax3.fill_between(t, vwap - 2*sd, vwap + 2*sd, color=VWAP_C, alpha=0.08, lw=0)
ax3.fill_between(t, vwap - sd, vwap + sd, color=VWAP_C, alpha=0.16, lw=0)
ax3.plot(t, price, color=INK, lw=1.6, solid_capstyle='round')
ax3.plot(t, vwap, color=VWAP_C, lw=2.0)
ax3.set_title('VWAP\n(volume-weighted mean)', fontsize=10, color=INK, pad=8)
ax3.set_xlabel('Time (minutes)', fontsize=9, color=MUTED)
ax3.set_xlim(0, N + 88)
ax3.set_xticks([0, 120, 240, 360])
ax3.grid(axis='y', color=GRID, lw=0.7)
ax3.set_axisbelow(True)
ax3.annotate('VWAP = mean', xy=(N - 1, vwap_end), xytext=(N + 12, vwap_end),
             fontsize=9, color=INK, ha='left', va='center', zorder=5,
             bbox=dict(facecolor=BG, edgecolor='none', pad=1.2))
ax3.plot([N - 1], [vwap_end], marker='o', ms=4, color=VWAP_C)
for mult, lab in ((-2, '−2σ'), (-1, '−1σ'), (1, '+1σ'), (2, '+2σ')):
    ax3.text(N + 8, vwap_end + mult * sd[-1], lab, fontsize=8,
             color=INK2, ha='left', va='center')

# dotted cross-panel guides at the three (distinct) reference levels
for ax in axes:
    ax.axhline(poc_price, color=POC_C, lw=1.2, ls=(0, (3, 3)), alpha=0.55, zorder=0)
    ax.axhline(tpo_poc_price, color=POC_C, lw=1.2, ls=(0, (1, 2.4)), alpha=0.45, zorder=0)
    ax.axhline(vwap_end, color=VWAP_C, lw=1.2, ls=(0, (3, 3)), alpha=0.45, zorder=0)

fig.suptitle('One session, three lenses: volume profile, TPO and VWAP',
             fontsize=13, fontweight='bold', color=INK, x=0.5, y=0.985)
fig.text(0.5, 0.015,
         'Same session, three lenses: mode (volume), time consensus (TPO), '
         'mean (VWAP) — they need not agree.',
         fontsize=9, color=INK2, ha='center')

fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.115)

fig.savefig('/home/user/a/docs/figures/fig-12-vp-tpo-vwap.svg',
            format='svg', bbox_inches='tight', pad_inches=0.15,
            facecolor=BG)
fig.savefig('/tmp/claude-0/-home-user-a/a79ba79b-564a-5a92-9188-47a1d21553f5/'
            'scratchpad/figpng/fig-12-vp-tpo-vwap.png',
            dpi=200, bbox_inches='tight', pad_inches=0.15, facecolor=BG)

print('volume POC %.2f | TPO POC %.2f | VWAP end %.2f' %
      (poc_price, tpo_poc_price, vwap_end))
print('rows:', nrows, 'VA: %.2f-%.2f' % (val, vah),
      'max TPO stack:', max_stack)
