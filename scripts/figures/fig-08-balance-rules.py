import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams['svg.hashsalt'] = 'vp-guide'
np.random.seed(108)
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ---- palette -------------------------------------------------------------
BG      = '#fcfcfb'
INK     = '#0b0b0b'
INK2    = '#52514e'
MUTED   = '#898781'
GRID    = '#e1e0d9'
BASE    = '#c3c2b7'
VA_IN   = '#2a78d6'
VA_OUT  = '#9ec5f4'
POC     = '#eb6834'
BULL    = '#1baf7a'
BEAR    = '#e34948'
SPECIAL = '#4a3aa7'
ATTN    = '#eda100'
BAL_FILL = '#cde2fb'

# ---- balance-area geometry (everything derives from these) ---------------
BAL_LO, BAL_HI = 42.0, 58.0            # balance low / high
BAL_X0, BAL_X1 = 0.4, 11.6             # horizontal extent of the balance box
MID = 0.5 * (BAL_LO + BAL_HI)

# ---- smooth path builder --------------------------------------------------
def smooth_path(waypoints, pts_per_seg=40, wiggle_amp=0.0, wiggle_cycles=3.0):
    """Cosine-eased interpolation through waypoints, with an optional gentle
    wiggle that is tapered to ZERO at both endpoints so structural endings
    (exact edge touches) are preserved."""
    xs, ys = [], []
    for (x0, y0), (x1, y1) in zip(waypoints[:-1], waypoints[1:]):
        t = np.linspace(0.0, 1.0, pts_per_seg, endpoint=False)
        e = (1.0 - np.cos(np.pi * t)) / 2.0
        xs.append(x0 + (x1 - x0) * t)
        ys.append(y0 + (y1 - y0) * e)
    xs.append(np.array([waypoints[-1][0]]))
    ys.append(np.array([waypoints[-1][1]]))
    x = np.concatenate(xs); y = np.concatenate(ys)
    if wiggle_amp > 0:
        s = np.linspace(0.0, 1.0, len(x))
        taper = np.sin(np.pi * s) ** 2                     # 0 at both ends
        phase = np.random.uniform(0, 2 * np.pi)
        y = y + wiggle_amp * taper * np.sin(2 * np.pi * wiggle_cycles * s + phase)
    return x, y

# ---- figure ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.0, 6.8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(-0.2, 12.4)
ax.set_ylim(26.5, 74.5)
ax.axis('off')

# balance rectangle
ax.add_patch(plt.Rectangle((BAL_X0, BAL_LO), BAL_X1 - BAL_X0, BAL_HI - BAL_LO,
                           facecolor=BAL_FILL, alpha=0.30,
                           edgecolor=VA_OUT, linewidth=1.2, zorder=1))
ax.text(BAL_X0 + 0.12, BAL_HI + 0.7, 'balance high', fontsize=9,
        color=MUTED, ha='left', va='bottom', zorder=3)
ax.text(BAL_X0 + 0.12, BAL_LO - 0.7, 'balance low', fontsize=9,
        color=MUTED, ha='left', va='top', zorder=3)

def draw_path(waypoints, color, wiggle=0.25, cycles=3.0, lw=1.9):
    x, y = smooth_path(waypoints, wiggle_amp=wiggle, wiggle_cycles=cycles)
    ax.plot(x, y, color=color, linewidth=lw, solid_capstyle='round', zorder=4)
    # start dot
    ax.plot([x[0]], [y[0]], marker='o', ms=4.5, color=color, zorder=5)
    # arrowhead at the end, oriented along the final segment
    ax.annotate('', xy=(x[-1], y[-1]), xytext=(x[-14], y[-14]),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw),
                zorder=5)
    return x, y

# ---- scenario 2 (leftmost band): look above and FAIL ----------------------
p2 = [(1.0, 50.5), (1.6, 54.0), (2.1, BAL_HI + 2.6),     # poke just above
      (2.5, 55.5), (2.9, 49.0), (3.2, 45.0), (3.5, BAL_LO)]  # ends AT low
x2, y2 = draw_path(p2, ATTN, wiggle=0.18, cycles=2.0)
poke2_i = int(np.argmax(y2))                              # computed poke apex
ax.annotate('trapped breakout traders',
            xy=(x2[poke2_i], y2[poke2_i] + 0.4), xytext=(2.3, 66.6),
            fontsize=8.5, color=INK2, ha='left', va='center',
            arrowprops=dict(arrowstyle='->', color=INK2, lw=1.0), zorder=6)
ax.text(0.4, 38.4, 'Look above and FAIL → rotate to opposite extreme',
        fontsize=9.5, color=INK, ha='left', va='top', zorder=6)

# ---- scenario 1: look above and GO ----------------------------------------
p1 = [(4.0, 49.0), (4.35, 53.0), (4.75, BAL_HI), (5.1, 63.0), (5.5, 70.5)]
x1, y1 = draw_path(p1, BULL, wiggle=0.22, cycles=2.5)
ax.text(5.78, 70.5, 'Look above and GO → destination trade',
        fontsize=9.5, color=INK, ha='left', va='center', zorder=6)

# ---- scenario 5: stay in balance -------------------------------------------
p5 = [(5.95, 50.0), (6.25, 54.5), (6.55, 46.0), (6.9, 55.0),
      (7.25, 45.5), (7.5, 53.5), (7.75, 49.5)]
x5, y5 = draw_path(p5, MUTED, wiggle=0.12, cycles=2.0)
ax.text(6.75, 44.0, 'Stay in balance →\ntrade the rotation',
        fontsize=9.5, color=INK, ha='center', va='top', zorder=6)

# ---- scenario 3: look below and GO -----------------------------------------
p3 = [(8.1, 51.0), (8.5, 47.0), (8.9, BAL_LO), (9.3, 37.0), (9.65, 31.5)]
x3, y3 = draw_path(p3, BEAR, wiggle=0.22, cycles=2.5)
ax.text(9.65, 29.4, 'Look below and GO', fontsize=9.5, color=INK,
        ha='center', va='top', zorder=6)

# ---- scenario 4 (rightmost band): look below and FAIL ----------------------
p4 = [(10.15, 49.5), (10.5, 45.0), (10.8, BAL_LO - 2.6),  # poke just below
      (11.1, 47.0), (11.35, 53.0), (11.5, BAL_HI)]        # ends AT high
x4, y4 = draw_path(p4, ATTN, wiggle=0.15, cycles=2.0)
ax.text(11.5, 60.6, 'Look below and FAIL', fontsize=9.5, color=INK,
        ha='right', va='bottom', zorder=6)

# ---- title & caption --------------------------------------------------------
ax.set_title('Balance rules: prepare all five scenarios in advance',
             fontsize=13, fontweight='bold', color=INK, pad=14, loc='left')
fig.text(0.045, 0.035,
         'Every break of a balance edge is a test: acceptance outside means GO; '
         'rejection means FAIL and price rotates to the opposite extreme.',
         fontsize=9, color=INK2, ha='left')

fig.savefig('/home/user/a/docs/figures/fig-08-balance-rules.svg',
            format='svg', bbox_inches='tight', pad_inches=0.15,
            facecolor=BG)
fig.savefig('/tmp/claude-0/-home-user-a/a79ba79b-564a-5a92-9188-47a1d21553f5/scratchpad/figpng/fig-08-balance-rules.png',
            dpi=200, bbox_inches='tight', pad_inches=0.15, facecolor=BG)
print('done')
