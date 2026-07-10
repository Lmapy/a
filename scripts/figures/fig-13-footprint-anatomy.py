import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams['svg.hashsalt'] = 'vp-guide'
np.random.seed(113)
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

from matplotlib.patches import Rectangle

# ---------------------------------------------------------------- palette
BG      = '#fcfcfb'
INK     = '#0b0b0b'
INK2    = '#52514e'
MUTED   = '#898781'
BASELN  = '#c3c2b7'
BUY     = '#1baf7a'
SELL    = '#e34948'
ATTN    = '#eda100'

# ---------------------------------------------------------------- data
# 10 price rows, top (highest) to bottom (lowest); tick = 0.25
tick   = 0.25
prices = 5301.25 - tick * np.arange(10)          # descending
bid    = np.array([  0,  96, 143, 118, 104,  87,  64,  52,  41,  38])
ask    = np.array([ 87, 412, 158, 234, 189, 205, 178, 141,  96,  57])

# ---- computed structural quantities (never hand-placed) ----------------
# diagonal ask-side imbalance: ask at price P vs bid one tick lower (P-1)
RATIO = 3.0
imb = np.zeros(len(prices), dtype=bool)
for i in range(len(prices) - 1):
    if bid[i + 1] > 0 and ask[i] >= RATIO * bid[i + 1]:
        imb[i] = True
imb_rows = np.flatnonzero(imb)

# stacked imbalances = run of exactly three consecutive qualifying rows
assert list(imb_rows) == [5, 6, 7], f'expected rows 5,6,7 to qualify, got {imb_rows}'
stack = imb_rows                                  # rows 5,6,7

# absorption row = top-area row with the largest ask number
absorb_row = int(np.argmax(ask))
assert absorb_row == 1 and absorb_row <= 2        # top area of the bar

# finished auction: very top price has bid == 0, ask > 0 (one side exhausted)
assert bid[0] == 0 and ask[0] > 0

# delta computed from the printed numbers
delta = int(ask.sum() - bid.sum())
assert delta == ask.sum() - bid.sum()

# bar geometry (schematic candle aligned to rows)
high, low = prices[0], prices[-1]                 # 5301.25 / 5299.00
o, c = 5299.25, 5300.00                           # closes off the highs -> failed advance

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(9.5, 6.8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 10.5)
ax.set_ylim(5297.95, 5302.0)
ax.axis('off')

# ---------------------------------------------------------------- table
H  = 0.23                                          # row height (price units)
BX0, BX1 = 1.10, 2.42                              # bid cell x-span
AX0, AX1 = 2.58, 3.90                              # ask cell x-span

for i, p in enumerate(prices):
    y0 = p - H / 2
    # price label
    ax.text(0.95, p, f'{p:.2f}', ha='right', va='center',
            fontsize=9, color=MUTED)
    # cell tints
    ax.add_patch(Rectangle((BX0, y0), BX1 - BX0, H, facecolor=SELL,
                           alpha=0.10, edgecolor='none', zorder=1))
    ax.add_patch(Rectangle((AX0, y0), AX1 - AX0, H, facecolor=BUY,
                           alpha=0.10, edgecolor='none', zorder=1))
    # absorption highlight over the whole row
    if i == absorb_row:
        ax.add_patch(Rectangle((BX0, y0), AX1 - BX0, H, facecolor=ATTN,
                               alpha=0.32, edgecolor='none', zorder=2))
    # numbers
    ax.text((BX0 + BX1) / 2, p, f'{bid[i]:d}', ha='center', va='center',
            fontsize=10, color=INK, zorder=4)
    ax.text((AX0 + AX1) / 2, p, f'{ask[i]:d}', ha='center', va='center',
            fontsize=10, color=INK, zorder=4)
    ax.text(2.50, p, '×', ha='center', va='center',
            fontsize=8, color=MUTED, zorder=4)

# column headers with swatches
hy = prices[0] + H / 2 + 0.13
for cx, col, lab in (((BX0 + BX1) / 2, SELL, 'bid (sell)'),
                     ((AX0 + AX1) / 2, BUY, 'ask (buy)')):
    ax.add_patch(Rectangle((cx - 0.60, hy - 0.035), 0.13, 0.07,
                           facecolor=col, edgecolor='none', zorder=3))
    ax.text(cx - 0.42, hy, lab, ha='left', va='center',
            fontsize=9, color=INK2)

# ---------------------------------------------------------------- outlines
# each qualifying diagonal pair: ask cell at P and bid cell at P-1
for i in stack:
    ax.add_patch(Rectangle((AX0, prices[i] - H / 2), AX1 - AX0, H,
                           facecolor='none', edgecolor=BUY,
                           linewidth=1.7, zorder=5))
    ax.add_patch(Rectangle((BX0, prices[i + 1] - H / 2), BX1 - BX0, H,
                           facecolor='none', edgecolor=BUY,
                           linewidth=1.7, zorder=5))

# ---------------------------------------------------------------- bracket
by_top = prices[stack[0]] + H / 2
by_bot = prices[stack[-1]] - H / 2
bx = 4.10
ax.plot([bx, bx], [by_bot, by_top], color=INK2, lw=1.2)
ax.plot([bx - 0.10, bx], [by_top, by_top], color=INK2, lw=1.2)
ax.plot([bx - 0.10, bx], [by_bot, by_bot], color=INK2, lw=1.2)
ax.text(bx + 0.14, (by_top + by_bot) / 2,
        'stacked imbalances\n(three in a row)',
        ha='left', va='center', fontsize=9, color=INK)

# ---------------------------------------------------------------- annotations
def note(text, xytext, xy, fs=9):
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=fs, color=INK,
                ha='left', va='center',
                arrowprops=dict(arrowstyle='->', color=INK2, lw=1.2))

note('finished auction at the high (one-sided print: bid = 0\n= buying exhausted, clean rejection; both sides still\nprinting at an extreme = unfinished, revisit magnet)',
     xytext=(4.45, 5301.62), xy=(3.98, 5301.30))

note('absorption: heavy aggressive buying,\nno progress (passive seller)',
     xytext=(4.45, 5300.95), xy=(3.98, prices[absorb_row]))

r_top = ask[stack[0]] / bid[stack[0] + 1]
note(f'diagonal imbalance ≥ 3:1 (buyers aggressive)\n'
     f'ask {ask[stack[0]]} vs bid {bid[stack[0]+1]} one tick below',
     xytext=(4.85, 5300.38), xy=(3.55, prices[stack[0]] + H / 2 + 0.01))

# ---------------------------------------------------------------- candle
xc = 9.30
bw = 0.20
ax.plot([xc, xc], [low, high], color=INK2, lw=1.8, zorder=2,
        solid_capstyle='butt')
ax.add_patch(Rectangle((xc - bw / 2, min(o, c)), bw, abs(c - o),
                       facecolor=BUY, edgecolor='none', zorder=3))
ax.text(xc + 0.28, high, 'high', ha='left', va='center', fontsize=8, color=MUTED)
ax.text(xc + 0.28, c, 'close', ha='left', va='center', fontsize=8, color=MUTED)
ax.text(xc + 0.28, low, 'low', ha='left', va='center', fontsize=8, color=MUTED)
ax.text(xc, low - 0.24, 'same bar,\ncandle view', ha='center', va='top',
        fontsize=8, color=MUTED)
# faint alignment guides from table to candle range
for yy in (high, low):
    ax.plot([AX1 + 0.05, xc - 0.35], [yy, yy], color=BASELN, lw=0.8,
            ls=(0, (2, 3)), zorder=1)

# ---------------------------------------------------------------- delta line
d_sign_col = BUY if delta > 0 else SELL
t1 = ax.text(BX0, 5298.52,
             f'delta = total ask − total bid = {int(ask.sum())} − '
             f'{int(bid.sum())} = ',
             ha='left', va='center', fontsize=10, color=INK)
fig.canvas.draw()
bb = t1.get_window_extent(fig.canvas.get_renderer())
x_end = ax.transData.inverted().transform((bb.x1, bb.y0))[0]
ax.text(x_end + 0.05, 5298.52, f'{delta:+d}', ha='left', va='center',
        fontsize=10, fontweight='bold', color=d_sign_col)

# caption
ax.text(BX0, 5298.18,
        'Each row prints contracts traded at the bid (sellers hit) × at the ask (buyers lifted).\n'
        'Diagonal reading compares the ask at price P with the bid one tick below, at P − 1 tick.',
        ha='left', va='top', fontsize=9, color=INK2)

ax.set_title('Reading a footprint bar', fontsize=13, fontweight='bold',
             color=INK, loc='left', pad=14)

# ---------------------------------------------------------------- save
svg_path = '/home/user/a/docs/figures/fig-13-footprint-anatomy.svg'
png_path = ('/tmp/claude-0/-home-user-a/a79ba79b-564a-5a92-9188-47a1d21553f5/'
            'scratchpad/figpng/fig-13-footprint-anatomy.png')
fig.savefig(svg_path, format='svg', facecolor=BG,
            bbox_inches='tight', pad_inches=0.15)
fig.savefig(png_path, dpi=200, facecolor=BG,
            bbox_inches='tight', pad_inches=0.15)
print('delta', delta, '| imbalance rows', list(imb_rows),
      '| absorption row', absorb_row, '| ratios',
      [round(ask[i] / bid[i + 1], 2) for i in stack])
