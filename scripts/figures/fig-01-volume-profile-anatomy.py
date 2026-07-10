import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams['svg.hashsalt'] = 'vp-guide'
np.random.seed(101)
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ---- palette --------------------------------------------------------------
BG = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXISC = '#c3c2b7'
VA_IN = '#2a78d6'
VA_OUT = '#9ec5f4'
POC_C = '#eb6834'
BAND = '#cde2fb'

# ---- synthetic volume-at-price data ---------------------------------------
prices = np.arange(4980, 5061)            # 1-point rows, 4980..5060
vol = (1100.0 * np.exp(-0.5 * ((prices - 5033) / 7.0) ** 2)    # large bulge
       + 400.0 * np.exp(-0.5 * ((prices - 4995) / 4.2) ** 2)   # small bulge
       + 20.0)                                                  # thin base
vol *= (1.0 + 0.10 * np.random.randn(prices.size))
vol = np.clip(vol, 5.0, None)
vol[np.argmax(vol)] *= 1.12          # make the single max row unambiguous

# ---- structural quantities (computed, never hand-placed) -------------------
poc_i = int(np.argmax(vol))                # POC = argmax row
poc_p = int(prices[poc_i])

# Value area: standard 70% expansion from the POC
total = vol.sum()
target = 0.70 * total
lo = hi = poc_i
acc = vol[poc_i]
while acc < target:
    up = vol[hi + 1] if hi + 1 < len(vol) else -np.inf
    dn = vol[lo - 1] if lo - 1 >= 0 else -np.inf
    if up >= dn:
        hi += 1
        acc += up
    else:
        lo -= 1
        acc += dn
val_p = int(prices[lo])                    # value area low
vah_p = int(prices[hi])                    # value area high

# HVN centers = the two bulge peaks; LVN = min volume between them
small_peak_i = int(np.argmax(np.where(prices < 5015, vol, -np.inf)))
large_peak_i = poc_i
between = slice(min(small_peak_i, large_peak_i) + 1,
                max(small_peak_i, large_peak_i))
lvn_i = between.start + int(np.argmin(vol[between]))
lvn_p = int(prices[lvn_i])
hvn_p = int(prices[small_peak_i])

# ---- figure ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.2, 7.0))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

xmax = vol.max() * 1.62
ax.set_xlim(0, xmax)
ax.set_ylim(prices[0] - 2.5, prices[-1] + 2.5)

# value-area shaded band (behind everything)
ax.axhspan(val_p - 0.5, vah_p + 0.5, color=BAND, alpha=0.35, zorder=0)

# bars
in_va = (prices >= val_p) & (prices <= vah_p)
colors = np.where(in_va, VA_IN, VA_OUT).astype(object)
colors[poc_i] = POC_C
ax.barh(prices, vol, height=0.88, color=list(colors), zorder=2,
        edgecolor='none')

# POC line extended across the axis + label
ax.plot([0, xmax], [poc_p, poc_p], color=POC_C, lw=1.4, zorder=1.5)
ax.text(xmax * 0.985, poc_p + 1.0, 'POC (point of control)',
        ha='right', va='bottom', fontsize=9.5, color=INK, zorder=4)

# VAH / VAL dashed reference lines + labels
for yv, lab, va, dy in ((vah_p, 'VAH', 'bottom', 0.9),
                        (val_p, 'VAL', 'top', -0.9)):
    ax.plot([0, xmax], [yv, yv], ls=(0, (5, 4)), lw=1.2, color=INK2,
            zorder=1.6)
    ax.text(xmax * 0.985, yv + dy, lab, ha='right', va=va,
            fontsize=9.5, color=INK, zorder=4)

# ---- annotations -----------------------------------------------------------
ann_kw = dict(fontsize=9.5, color=INK,
              arrowprops=dict(arrowstyle='->', color=INK2, lw=1.1,
                              shrinkA=3, shrinkB=2))

# HVN (smaller bulge)
ax.annotate('HVN - acceptance: price slows, rotates',
            xy=(vol[small_peak_i] + 12, hvn_p),
            xytext=(xmax * 0.52, prices[0] + 4.5),
            ha='left', va='center', **ann_kw)

# LVN (thin gap between the bulges)
ax.annotate('LVN - rejection: price moves fast',
            xy=(vol[lvn_i] + 12, lvn_p),
            xytext=(xmax * 0.46, lvn_p - 2.5),
            ha='left', va='center', **ann_kw)

# Value area band
ax.annotate('Value area ≈ 70% of volume',
            xy=(xmax * 0.88, val_p + (poc_p - val_p) * 0.45),
            xytext=(xmax * 0.985, val_p - 6.5),
            ha='right', va='center', **ann_kw)

# ---- axes cosmetics --------------------------------------------------------
ax.set_title('Anatomy of a volume profile', fontsize=13,
             fontweight='bold', color=INK, loc='left', pad=12)
ax.set_xlabel('Volume traded at price', fontsize=9.5, color=MUTED)
ax.set_ylabel('Price', fontsize=9.5, color=MUTED)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for s in ('left', 'bottom'):
    ax.spines[s].set_color(AXISC)
ax.tick_params(colors=MUTED, labelsize=9, length=3)
ax.set_yticks(np.arange(4980, 5061, 10))
ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
ax.set_axisbelow(True)

# caption line
fig.text(0.065, 0.015,
         'Value area computed with the standard 70% expansion from the POC.',
         fontsize=9, color=INK2, ha='left')
fig.subplots_adjust(bottom=0.12)

# ---- save ------------------------------------------------------------------
svg_path = '/home/user/a/docs/figures/fig-01-volume-profile-anatomy.svg'
png_path = ('/tmp/claude-0/-home-user-a/a79ba79b-564a-5a92-9188-47a1d21553f5/'
            'scratchpad/figpng/fig-01-volume-profile-anatomy.png')
fig.savefig(svg_path, format='svg', facecolor=BG,
            bbox_inches='tight', pad_inches=0.15)
fig.savefig(png_path, dpi=200, facecolor=BG,
            bbox_inches='tight', pad_inches=0.15)

print('POC', poc_p, 'VAL', val_p, 'VAH', vah_p, 'LVN', lvn_p,
      'HVN2', hvn_p, 'VA%', round(acc / total * 100, 2))
