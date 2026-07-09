# THE AUCTION — Game Design Document (Final, v1.0)

**Status:** Final synthesis. Base: the "arcade-first" proposal (judge panel winner, 3/3). Grafted: every judged-worth-stealing idea from "sim-first" and "curriculum-first" that survives the conflict-resolution rule — **learning efficacy first, then engagement, then feasibility.**

**Domain source of truth:** `/home/user/a/docs/volume-profiles-and-auction-market-theory.md`
**Algorithm reference implementations:** `/home/user/a/scripts/figures/` (fig-01 … fig-14 Python scripts)

---

## 1. Vision & Thesis

**The Auction is a video-game-grade trainer for volume profile reading and auction market theory.** It exists because real markets are a wicked learning environment (Hogarth; Kahneman–Klein): feedback is slow, confounded by luck, and never explains itself — which is why 97% of day traders lose and learning-by-trading is a documented near-zero. The Auction inverts every one of those properties **by construction**: the simulator *generates* every session from a labeled model (day type, open type, planted nPOCs, poor extremes, LVN necks), so every player decision gets an objective, instant, explainable verdict — the thing no amount of real-market screen time can provide.

**The seven thesis pillars, and where each lives in this document:**

1. **Ground truth by construction** → §7 (engine spec: labels true by construction AND by measurement).
2. **Sub-second feedback loops** → §2 (the second-by-second core loop; verdict ≤100ms, explanation ≤250ms, full loop <400ms).
3. **Train calibration, not prediction** → §4 Drill E and §5 (Brier scoring wrapped as Bot Points, calibration curves, base rates revealed after every block).
4. **One skill per drill, interleave later** → §3–4 (blocked introduction, interleaved forever; atomic drills → speed rounds → boss sessions).
5. **Boss levels that punish the classic mistakes** → §6 (three bosses: trend-day fade, 80%-rule through news, P-shape "bottom").
6. **Expectancy dashboard endgame** → §5 (measured win%/R per setup × regime vs. folklore columns).
7. **Hyper-learning scaffolding** → §3 (spaced repetition of misses, Warm-Up streaks, adaptive ~85% difficulty, 3-minute drills, mastery-gated tree).

**Design stance (from the winning proposal, unchanged):** the drill loop *is* the game. Everything else — boss sessions, dashboards, the skill tree — exists to feed reps into a sub-second tap→verdict→explanation cycle that feels like Puzzle Rush and grades like GTO Wizard. The rep economics are the learning-science bet: 8–15 graded retrievals per minute in speed modes versus a handful per ten minutes in a free-play sim is an order-of-magnitude difference in testing-effect exposures per player-hour (testing effect g≈0.6; Kellman PLM template: many short trials, forced fast classification, mastery = accuracy AND latency).

**The shallowness defense, shipped as mechanics, not promises:**
1. Mastery is granted **only** by delayed, interleaved checkpoints (§3) — arcade scores unlock nothing.
2. Boss sessions and the sim Read Score are the certification layer; **drill-rating vs. sim Read-Score divergence** is a first-class product-health metric from day one (the aim-trainer failure detector).
3. **No naked verdicts anywhere.** Every ✓/✗ carries a one-line auction-logic explanation quoting the guide. A verdict number alone does not improve judgment; the explanation is load-bearing.
4. Grades compare the decision to the **planted ground truth or generative distribution, never the single realized path.** Realized P&L is shown always, scored never, labeled "outcome ≠ decision quality."

**Voice of the product:** we train *recognition and calibration*, weather-forecaster style — not market-beating prediction. In-sim expectancy is certifiable; real-market expectancy is not, and the UI says so.

---

## 2. Core Loop (Second by Second)

The atomic rep — the unit everything else is assembled from:

| Time | Event |
|---|---|
| **t = 0.0s** | Item appears: a volume profile (or a replaying tape), one question, 2–4 large tap targets or a snap-to-level tap zone (≥44px targets, snap-to-row). A thin ring shows *par time* — informational in untimed modes, a deadline only in Rush. |
| **t = 0.0–3.0s** | Player answers by tapping. Retrieval-first: the answer is never visible before commitment. On binary drills, an optional confidence tap (sure / lean / guess) rides along at zero extra UI cost. |
| **t + ≤100ms** | **Verdict.** The tapped level hit-flashes; the verdict marker squash-stretches in with a ✓/✗ icon **plus** color (never color alone); correct/incorrect earcon. Juice lives on the verdict layer only — the chart itself never shakes or emits particles (over-embellishment measurably hurts performance). |
| **t + ≤250ms** | **One-line auction-logic explanation** slides in under the chart. Task-referenced, never person-referenced (Kluger & DeNisi: ego-directed feedback backfires). Score counts up; within-round streak ticker increments. |
| **t + 250–400ms** | **NEXT is live.** Total loop <400ms (Doherty threshold). A small ⟲ replay chip sits beside the explanation — player-invoked micro-replay with play/pause/step/speed controls showing the deciding structure form. Novices get the replay auto-offered; the offer fades with mastery (guidance hypothesis). |
| **t + 400ms** | Next item appears — **pre-generated and pre-rendered during the previous answer window**, so the rhythm never stutters. Target cadence: 8–15 reps/min in speed modes, 4–6 in Rated. |
| **Every ~10 reps** | A 5-second **base-rate interstitial** (stolen from curriculum-first): *"In this set, highlighted extremes were poor 47% of the time."* Scoring feedback alone does not improve calibration; revealed reference classes do (Lichtenstein & Fischhoff). |

**Permanence:** past calls stay pinned as tiny markers when a profile recurs, and every round ends with a **film strip** of every profile judged — each thumbnail tappable to replay. Permanence plus delayed review in one artifact.

**Scaffolding fade (grafted from curriculum-first, explicit schedule):**
- **T0 nodes:** annotated overlays on (VA shading, HVN/LVN highlights); verdict + explanation + auto-offered micro-replay.
- **T1 nodes:** overlays on request; verdict + explanation.
- **T2+:** raw profiles; terse verdict, explanation one tap away.
- **Two tiers past mastery:** end-of-round batch summary only. Expertise reversal is real; scaffolding that helps novices hurts experts.

**Sound:** two earcons (correct/incorrect) + one milestone sound. No music. One-tap mute. `prefers-reduced-motion` swaps squash-stretch for fades; no more than 3 flashes/sec even in Rush.

---

## 3. Skill Tree & Progression

The curriculum brief's T0→T5 tree, rendered as a **subway map** (one line per tier). Each node = one skill, one drill family, one Glicko rating, four states: **locked / learning / checkpoint-armed / mastered** (plus **rusty**, below).

### The tree (v1 ships T0–T2 complete, T3 partial; T4/T5 visible as locked silhouettes)

- **T0 — Read the Map:** read-POC → shade-VA → mark-HVN-LVN → excess-vs-poor → classify-shape (D/P/b/B/thin-trend). *(track-references deferred to v1.1.)*
- **T1 — Session Structure:** classify-open-type + read-open-location → read-IB → detect-one-timeframing → classify-day-type; initiative-vs-responsive → acceptance-vs-rejection → read-value-migration. *(spike-rules folded into Acceptance Clock feedback, not a standalone node.)*
- **T2 — Regime & Calibration (the hinge):** regime-call → probability-calibration → folklore-audit. Gated by the **<1-hour calibration intro module** (base rates, reference classes, overconfidence — 6–11% Brier improvement lasting a year in GJP; the single best-evidenced hour in the product).
- **T3 — Playbook (v1: setup-picker only):** setup-selection with NO TRADE across the 9 playbook setups. Stop-placement, target-placement, kill-switch, and individual setup drills ship v1.1.
- **T4 — Order Flow:** locked silhouette ("needs footprint engine — coming"). 
- **T5 — Integration:** boss stations (skull icons) + the Expectancy Dashboard as the visible endgame station.

### Mastery mechanics

- **Unlock** = prerequisite mastered.
- **Arming the gate** (Kellman compliance, grafted from curriculum-first): in-drill rolling accuracy implies ≥85% at target difficulty **AND** median latency under the node's criterion (e.g., <2.5s POC snap, <3s shape calls, ≤1 bracket for the one-timeframing buzzer). Latency criteria ship with placeholder values and are tuned empirically.
- **Mastery** = pass a **delayed, interleaved checkpoint**: 10 items, this skill mixed 40/60 with confusable items from mastered siblings (shape checkpoint interleaves P vs b vs D vs thin-trend), at gate difficulty, ≥8/10, available **no sooner than the next calendar day** (fights the illusion of competence; in-drill accuracy never grants mastery).
- **Checkpoint failure UX** (curriculum-first steal): no punishment text. The node shows "checkpoint retry available in 2 days" and the miss-set enters spaced repetition automatically.
- **Demoted "rusty" mastery:** a mastered node whose spaced-review performance decays below 70% flips to *rusty*, flickers on the map, and re-enters the Warm-Up rotation.
- **Leech rule:** 8 lapses on one pattern → drilling stops; a 60–90s taught micro-lesson serves instead.
- **Sim-permission gating** (sim-first steal, applied to bosses): tree mastery gates **boss-session permissions**, not just content. T0 mastery unlocks bosses in observer mode (forecast taps only); T2 mastery unlocks the order ticket; an unmastered setup **literally is not on your ticket**. Progression is felt as capability.
- **Boss gates:** T2→T3 requires Boss 1; T3 completion requires Bosses 2 and 3 (§6).
- **Interleaving is explained, not smuggled:** blocked introduction (5 items per new category), then interleaved forever, with an in-UI card: *"Mixed reps feel harder — that's the point. Blocked practice fakes mastery."*
- **Spacing:** misses re-queue on fixed expanding intervals (1d / 3d / 7d — 10–20% of the retention interval), plus a same-session retry. Half-life regression is deferred until there is data to regress on.
- **No XP anywhere.** Progression currency is per-node rating, Brier trend, and expectancy. Nothing on screen goes up unless a measured number went up.

### Mode roster (v1)

1. **Rated Drill** — the default. Untimed, per-skill Glicko rating, adaptive to ~85% expected success (Wilson setpoint), 20–30 min soft cap with a "fatigue kills value" nudge. New skills always debut here untimed.
2. **Rush** — 3:00 clock, escalating difficulty, time bonuses, 3 strikes. Never touches rating.
3. **Streak** — untimed, one miss ends it, difficulty ramps. Never touches rating.
4. **Woodpecker** — your personal miss-set at shrinking time budgets on the expanding-interval schedule. **Re-serves a missed item's *sibling*** (same seed, nudged knob via named PRNG substreams), never the identical item — kills answer memorization.
5. **Calibration Range** — probability sliders, Brier-scored in blocks of 25 (§4, Drill E).
6. **Checkpoint** — the gate exam. Visually distinct, **juice-free** (dark slate frame, no per-item feedback, full debrief at the end) so certification feels categorically different from practice.
7. **Boss Session** — §6. Explicitly labeled *"Boss — expect ~40% success. Errors are the content."*
8. **Daily Warm-Up** — the streak unit: a 3-minute assembled set of 2 due Woodpecker items + 1 new-skill block + 1 interleaved mixed block. Always adaptive-difficulty, so the streak cannot be farmed on trivial content.
9. **Daily Recap** — 2-minute next-session review of yesterday's misses (delayed-feedback consolidation), served as the Warm-Up's opening card when misses exist.

---

## 4. v1 Drill Catalog

Common contract for every drill: verdict <1s → one-line auction-logic explanation quoting the guide → optional micro-replay of the deciding structure → visible stat update. Adaptive difficulty holds ~85% success (bosses excepted, deliberately). Misses enter spaced repetition. Feedback templates below are canonical; the feedback-text audit (§10) checks every rendered line against the guide.

### Drill A — POC + VA Snap (T0.read-POC / T0.shade-VA)
- **Input:** a completed generated session's profile, 60–120 price rows. Prompt cycles: "Tap the POC" / "Place VAH" / "Place VAL." Single tap, snap-to-row (nearest row center within 22px). VAH/VAL items show the revealed POC.
- **Ground truth:** POC = argmax volume row; VA = the actual 70% two-row expansion algorithm ported from `fig-01-volume-profile-anatomy.py`. The same library renders and grades — disagreement is structurally impossible.
- **Feedback template (miss):** *"✗ VAH is {n} rows higher — the 70% expansion swallowed the upper HVN bulge before your row. Value is fatter above the POC than it looks."* (Hit): *"✓ Exact. The expansion ran {k} rounds; both edges are where the 70% mass ends."*
- **Scoring:** exact row = 100; ±1 row = 60; else 0. Latency bonus in Rush only.
- **Mastery:** 92% within ±1 row AND median latency <2.5s, verified at the delayed interleaved checkpoint.
- **Difficulty knobs:** `noiseGain` up (spawns secondary HVN decoys at 85–95% of POC volume), Student-t ν down (4 = spiky), double-distribution scripts with near-equal box mass.

### Drill B — Excess or Poor? (T0.excess-vs-poor)
- **Input:** profile with one extreme highlighted. Two buttons — EXCESS / POOR — plus a 3-notch confidence tap (sure/lean/guess → implicit 90/75/55%).
- **Ground truth:** the planted `excessTicks` label (0–1 = flat 2–3-touch poor extreme; 4–8 = rejection tail). Items in the ambiguous band (2–3 ticks) are graded probabilistically, marked "judgment call," never binary.
- **Feedback template (miss, poor called excess):** *"✗ Poor high — two flat touches, no tail. Excess ends the auction; a poor extreme is unfinished business and a repair magnet, not resistance."* (Punishes misconceptions 4 and 9.)
- **Mastery:** ≥85% at gate difficulty AND latency <2.0s at checkpoint.
- **Difficulty knobs:** adaptive selection serves the `excessTicks` value holding the player near 85%; boundary shrinks as rating rises. Confidence taps feed the calibration ledger (§5) at zero extra UI cost.
- **Reference:** `fig-09-excess-vs-poor.py`.

### Drill C — HVN/LVN Marker (T0.mark-HVN-LVN)
- **Input:** raw profile; prompt: "Tap the LVN neck" or "Tap the dominant HVN." Snap-to-row.
- **Ground truth:** planted LVN neck coordinates and HVN registry from the generator's structure labels, verified by the classifier pass (`fig-11-lvn-behavior.py` reference).
- **Feedback template (miss — tapped HVN when asked for LVN):** *"✗ That's an HVN — an acceptance bulge, a magnet with friction. The LVN air pocket sits {n} rows below: price traverses it fast or rejects at it."* (Punishes misconception 1.)
- **Mastery:** 90% within ±1 row, latency <2.5s at checkpoint.
- **Difficulty knobs:** LVN depth (shallower = harder), decoy near-LVNs (0–2), `noiseGain`.

### Drill D — Shape Alphabet (T0.classify-shape; the signature Rush content)
- **Input:** profile flashes; 5 letter buttons in a thumb-reachable arc: D / P / b / B / thin-trend. Blocked introduction (5 items per shape), interleaved forever after.
- **Ground truth:** the day-type script that generated the session, verified by the game's own classifier in the rejection loop (label true by construction AND measurement). Reference: `fig-02-profile-shapes.py`, `fig-05-day-types.py`.
- **Feedback template (P/b confusion):** *"✗ That's a b, not a P — the bulge sits low with a thin upper tail. P is short-covering above; b is liquidation below. Mirror-image, opposite story."*
- **Mastery:** ≥85% interleaved AND latency <3s at checkpoint.
- **Difficulty knobs:** ambiguity blend weight, ν, reveal level (full profile → developing profile).
- **Confusion matrix** off-diagonals (P↔b, trend↔B) map 1:1 to misconception counters and drive priority re-queue.

### Drill E — Calibration Range (T2.probability-calibration; the hinge)
- **Input:** mid-session snapshot + one question: "Probability this poor high repairs by close?" / "…this gap fills?" / "…value traverses to the far edge?" Slider, 5–95% (cap taught explicitly in the intro module: the 99% asymmetry).
- **Ground truth:** the generator's true conditional probability for this question under the item pool's **declared regime mixture** — graded against the *distribution*, never the single realized path. The realized path plays afterward under a persistent banner: **"Outcome ≠ decision quality."**
- **Feedback:** per item, only points vs. the base-rate bot. Every 25 items, a **block reveal**: *"This block's regime: the two-period hold filled value 63% of the time. Folklore says 80%. When you said 80%, it happened 66% — you're anchored on the folklore number."*
- **Mastery:** Brier ≤ 0.18 over a 50-item rolling window, verified at checkpoint.
- **Difficulty knobs:** question type mix, ambiguity weight, distance-to-resolution.
- **Reference:** `fig-07-80-percent-rule.py`, `fig-14-naked-poc.py`.

### Drill F — Open-Type Conviction Ladder (T1.classify-open-type)
- **Input:** first-30-minutes replay at 8×; four chips (DRIVE / TEST-DRIVE / REJECTION-REVERSE / AUCTION) + a conviction dot (1–3). Open location (in value / out of value / out of range) is shown, then quizzed separately on alternating items.
- **Ground truth:** the open-type script segment; open-drive items enforce "never re-trades the open" via rejection-resampling. Reference: `fig-06-open-types.py`.
- **Feedback template:** *"✗ Open-drive: price never re-traded the {price} open and one-timeframed from bracket 1. Conviction opens don't get faded."*
- **Mastery:** ≥85% AND median call before minute 20 of replay time, at checkpoint.
- **Difficulty knobs:** ambiguity blend (drive vs test-drive is the designed confusion), `noiseGain`, jump intensity.

### Drill G — One-Timeframing Buzzer (T1.detect-one-timeframing; latency-native)
- **Input:** 30-min brackets replay at 8×; player **holds the screen and releases the instant control breaks** (a bracket fails to make a higher low / lower high).
- **Ground truth:** the scripted bar index where the skeleton's one-timeframing segment ends.
- **Scoring:** full credit within 1 bracket; linear decay to 0 over 3; early release = false alarm, −50.
- **Feedback template:** *"✗ Released 3 brackets late — the 11:00 bracket already printed a lower high. One-timeframing broke there; every fade before that point was fighting initiative."*
- **Mastery:** 85% within ±1 bracket at checkpoint.
- **Difficulty knobs:** break subtlety (how marginal the failed bracket is), ν, decoy near-breaks.

### Drill H — Acceptance Clock (T1.acceptance-vs-rejection; bridges drill → sim)
- **Input:** replay begins as price leaves prior value; player presses ACCEPTED or REJECTED whenever ready. Earlier correct answers score more; wrong early answers score least (correctness × lag curve).
- **Ground truth:** the script's acceptance flag AND the game's own classifier run on the generated path (trainer–grader agreement invariant). Reference: `fig-10-value-migration.py`.
- **Feedback template:** *"✗ You called rejection, but this was acceptance: the second consecutive bracket built value outside and the dPOC ratcheted higher. Two 30-minute periods outside + value migration = acceptance."* Spike-rule items append: *"Next open above the spike confirms it; below it, the spike was exhaustion."*
- **Mastery:** ≥85% correct with mean lag ≤2 brackets at checkpoint.
- **Difficulty knobs:** how long the evidence stays mixed (ambiguity window length), `noiseGain`.

### Drill I — Regime Gate (T2.regime-call; the tree's hinge node)
- **Input:** mid-session snapshot (profile-so-far + last 6 brackets of tape). Two-stage forced answer: (1) BALANCE / IMBALANCE; (2) which playbook is live: RESPONSIVE / INITIATIVE-ONLY / STAND ASIDE.
- **Ground truth:** the day-type script segment active at snapshot time; blended-ambiguity items (weight >0.3) grade stage 1 probabilistically against the blend weight.
- **Scoring:** stage 1 = 60 pts, stage 2 = 40 pts. **Cardinal error (curriculum-first steal): a scripted trend day called "balance" costs 3× rating and force-queues a micro-replay of the specific tells** — IB width 0.3–0.5× normal, pullbacks ≤40% of the prior impulse, rising impulse volume.
- **Feedback template (cardinal miss):** *"✗✗ One-timeframing since 10:00, value stair-stepping up, pullbacks under 40% — this is imbalance. Every responsive fade looks perfect on a trend day, and every one loses."*
- **Mastery:** ≥85% with zero cardinal errors in the checkpoint set; latency <5s.
- **Difficulty knobs:** ambiguity blend, snapshot timing (earlier = harder), decoy balance-lookalikes.

### Drill J — Setup Picker with NO TRADE (T3.setup-selection; v1's whole-task layer)
- **Input:** a dealt scenario (session-so-far + references + a news-pending flag when applicable). Choose one of the 9 playbook setups or **NO TRADE**.
- **Ground truth:** setup **legality** graded against the active script segment and planted structures (sim-first's tractable grading — labels only, no EV oracle): a fade is illegal on a scripted trend day; any setup is illegal through the scheduled-news window; NO-TRADE scenarios (news pending, nontrend chop) score highest when passed.
- **Feedback template:** *"✗ Regime error: this is a scripted trend day (one-timeframing since B period). The VA-edge fade is a balance tactic — structural error regardless of execution."* / *"✓ NO TRADE. CPI in 12 minutes: the auction is about to be repriced by information, not inventory."*
- **Mastery:** ≥85% including ≥90% NO-TRADE compliance at checkpoint.
- **Difficulty knobs:** decoy attractiveness (how photogenic the illegal setup looks), ambiguity, news-flag subtlety.

---

## 5. Scoring Math

**Per-item drill score (Rush/Streak display only):**
`S = base(correctness) × difficultyMult × speedMult`
where `difficultyMult = clamp(itemRating / playerRating, 0.5, 2.0)` and `speedMult` (Rush only) `= 1 + max(0, (par − t)/par) × 0.5`. Streak mode: score = streak length, nothing else.

**Rating (the honest number):** per-skill Glicko-2. **Launch with item ratings derived deterministically from generator difficulty knobs** (`noiseGain`, ν, ambiguity, decoys, reveal level); migrate to full player–item co-rating once n > 10,000 answers per pool, with recalibration jobs planned from launch (chess.com's puzzle-rating inflation is the known failure). Adaptive selection picks items where expected score ≈ 0.85. Rush, Streak, Woodpecker, and Warm-Up never move rating. Cardinal errors (Drill I) carry 3× rating weight.

**Calibration ledger:** every probability answer p on outcome o scores Brier `B = (p − o)²`. Sources: Calibration Range sliders **plus confidence taps on binary drills folded in as implicit 55/75/90% answers** — calibration sample size multiplies at zero extra UI cost, reaching the 100–200-judgment normalization threshold faster. Surfaced three ways (the Metaculus-trap rule):
1. **Headline = Bot Points:** `AP = Σ (Brier_bot − Brier_you) × 100` per block of 25, where the bot always answers the item pool's declared true base rate. Positive = beating the base-rate bot; sums are mostly positive for competent play.
2. **Reliability curve every block of 25–50** (never per item): binned dots sized by N, confidence bands for small N, behind a sentence headline: *"When you say 80%, it happens 64% — overconfident at high confidence."*
3. **Raw Brier trend** in Stats, with expectations set in the FTUE: *"Most players normalize within 100–200 scored judgments."*

Confidence input is capped at 95%; the intro module teaches the asymmetry explicitly.

**Boss/sim scoring — Read Score and the luck channel:** every boss decision is graded against planted truth and legality labels (no EV oracle in v1). The debrief prints two numbers plus a decomposition sentence (sim-first steal, adapted to label-based grading):

> **Read Score 78 · P&L −0.8R.** *"Your reads earned a top-quartile session; variance handed you −0.8R. Right process, unlucky outcome."*

The decomposition is computed from the ledger of graded decisions: legal, correctly-read decisions that lost money are attributed to the luck channel; illegal reads that made money earn zero credit — **no XP for wrong-process lucky wins** (bright line). P&L is shown always, scored never.

**Expectancy Dashboard (endgame):** per setup × regime × day type: n, win%, avg win/avg loss, expectancy `E = p·W − (1−p)·L`, R-distribution — rendered beside folklore columns. **Cells gray until n ≥ 30, watermarked "unmeasured = folklore"** (curriculum-first steal). Caption: in-sim expectancy is certifiable; real-market expectancy is not.

**Streak (meta):** counts **completed Daily Warm-Ups only** — practice days, never accuracy — with auto-granted freezes and grace periods. Warm-Ups are always adaptive, so the streak is unfarmable by design. No near-miss dramatization anywhere; no appointment mechanics; no decaying rewards.

**Health metric:** drill-rating vs. boss Read-Score divergence per node, plotted on an internal dashboard from day one. Sustained divergence = players are learning the drill, not the skill — the trigger for item-pool rotation or drill redesign.

---

## 6. Boss Levels

Bosses are **error-management training** (d ≈ 0.80 on adaptive transfer, the largest effect size in the briefs): engineer the classic mistake, let the player fail, frame the error as informative, then micro-replay the missed tell. Bosses deliberately break the 85% band and **say so up front: "Boss — expect ~40% success."** All three ship in v1 (resolving curriculum-first's one-boss cut in favor of thesis pillar 5). Bosses run as full generated sessions with the declared-read ticket (§8, screen 6) and **prop-desk constraints: a daily-loss limit and trailing drawdown** — 70% of real prop failures are limit breaches, so punishing them is authentic, and breaching the limit ends the boss regardless of open positions.

**Boss 1 — Trend-Day Fade Gauntlet** (gates T2→T3). *Punishes misconceptions 2 and 6.*
A scripted trend day where every responsive fade looks textbook-perfect — clean VA edge, poor high overhead, tidy structural stop — and every one loses, because one-timeframing never breaks. The winning line is initiative-only or standing aside. Post-boss micro-replay steps through the tells the player overrode: IB 0.4× normal width, pullbacks never exceeding 40%, impulse volume rising. Debrief line: *"Every fade looked perfect. Every fade lost. That is what a trend day does."*

**Boss 2 — The 80% Rule Through News** (T3 gate). *Punishes misconceptions 3 and 8.*
A textbook two-period hold inside prior value fires the 80%-rule trigger — with scheduled CPI 15 minutes out. Taking the trade gets repriced through the level; the read verdict fires instantly: *"The auction is repriced by information, not inventory — no level survives scheduled news."* Players who wait are then quizzed on the probability they'd size for; the block reveal shows this pool's true fill rate (~63%, declared by the item pool) versus the folklore 80%. The folklore number is the trap; the measured number is the loot.

**Boss 3 — The P-Shape "Bottom"** (T3 gate). *Punishes misconceptions 7 and 9.*
After an extended scripted rally plus fabricated prior sessions showing the run, a stubby P-profile forms and a poor low sits below. The seductive read: "P is bullish, buy the dip; the poor low is support." Ground truth: the P is short-covering — old business, not new buying — and the poor low is unfinished business that gets repaired through the "support." Debrief: *"A P after a long rally is shorts leaving, not buyers arriving. And a poor extreme is a repair magnet — never support."*

**Boss mechanics shared by all three:** every decision is attributed back to its atomic skill's stat pool; 5-second **KILL/HOLD reflex interrupts** fire at scripted plantable moments for position holders, scored on latency (sim-first steal), with slept-through interrupts surfaced in the debrief; the debrief timeline scrubber (§8) ends with a **"one thing to drill" button** that queues tomorrow's Woodpecker.

---

## 7. Synthetic Market Engine Spec

**Generation model: scripted skeleton, not learned.** Day type + open type compile to a segment script (per-segment drift, volatility, volume-intensity, anchors); Brownian-bridge noise fills between anchors; a verification pass runs the game's own classifiers and regenerates failing segments. Labels are true **by construction and by measurement** — trainer and grader can never disagree. GANs/diffusion (uncontrollable, unreplayable) and agent-based models (emergent = undirectable) are rejected for v1; a small Hawkes process for footprint print timing is a v2 add for T4.

**One latent intensity drives everything (MDH):** `λ_t = U-shaped seasonal × GARCH(1,1) variance state × script boost`. Variance uses it, bar volume uses it, print arrivals use it. Profile mass = dwell time × λ, so D/P/b/B shapes, HVNs, and LVN necks **emerge** from the script with no special-case profile code.

**Kernel numbers:** 1-min bars; GARCH(1,1) α≈0.10, β≈0.85; standardized Student-t innovations, ν as difficulty knob (ν=4 hard/spiky, ν=8 easy/tame); seasonal vol curve open ≈2.5–3× midday, close ≈1.5–2×; jump (news) intensity morning-weighted; closes get diffusive volatility, not gaps. Trend-day script: IB width 0.3–0.5× normal, pullbacks ≤40% of prior impulse, close in the extreme 15% of range, volume rising on impulses. Double-distribution: two balance boxes joined by a fast traverse — the LVN neck appears automatically. Open-drive: "never re-trades the open" enforced by rejection-resampling the first hour.

**Label schema (shipped with every item, consumed by the local grader):**

```
ItemLabels {
  dayType, blendWeight            // 6-type taxonomy + ambiguity mixture
  openType, openLocation
  script: Segment[]               // per-segment regime timeline w/ bar ranges
  ibWidthRatio
  oneTimeframingBreakBar          // exact bar index
  extremes: { side, excessTicks }[]
  planted: { nPOCs[], poorExtremes[], lvnNecks[], priorVAEdges[] }  // off-round coords
  decoys: Structure[]             // 0–2 near-structures per session
  acceptanceFlags: per-bracket
  conditionalProbs: { question -> p }   // per calibration question, per declared mixture
  resolutionState: per-structure (nPOC touched? poor high repaired?)  // for boss grading
  ambiguityWeight
  poolBaseRates: { question -> p }      // the bot's answers & interstitial reveals
}
```

**Planted structures via context, not paint:** fabricate **3–5 prior sessions per sim day from the same generator** (sim-first steal) so nPOCs, prior VA edges, and multi-day balances exist with ground-truth ancestry. Anti-obviousness rules: no zero-volume bars, off-round-number plant levels, 1–2 decoy near-structures per session, jittered structure timing.

**Difficulty knobs (all continuous, all mapped to adaptive selection):** `noiseGain` (master signal-to-noise dial), ν (4–8), ambiguity blend weight, `excessTicks` continuum, decoy count (0–2), jump intensity, LVN depth, reveal level (full profile → developing profile → price-only), replay speed (independent of generation).

**Determinism & replay:** 64-bit master seed → splitmix64 → **named PRNG substreams** (skeleton / noise / volume / prints / decoys) of PCG32, so changing one knob never reshuffles the others — "same day, harder" works, which Woodpecker's sibling re-serve requires. PRNG state = one integer per stream → free save/scrub/micro-replay. `(seed, paramsVersion)` persisted with every scored decision so any verdict is re-derivable forever (this is also the anti-cheat: the server can re-simulate any client verdict). Avoid engine-dependent transcendental bits in the hot path.

**Throughput:** Rush needs <50ms generation per single-profile item (cheap); full sessions come from a **background pool of 20 warm items per active skill**, generated in a Web Worker.

**Verification loop:** the game's own classifiers must recover labels ≥99% post-rejection; acceptance rates logged; **alarm below 20% acceptance** (script and predicate disagree = bug).

**Anti-overfit policy (the Rocksmith lesson):** parameter packs **rotate monthly**; regularities contaminated with realistic noise; a **sim-only-tell audit** (can a shallow classifier distinguish generated from real sessions on trivial statistics?) runs as a recurring engineering task; folklore numbers never hard-coded — each item pool declares its own mixtures and resolution probabilities, which double as the bot's answers and the interstitial reveals.

**Ensemble invariants (unit-tested in CI, not a product feature):** excess kurtosis >1 at 1-min; |r| autocorrelation positive to lag 30+; open/midday volume ratio 2–4×; volume-clock-sampled returns near-Gaussian; classifier label recovery ≥99%.

**Real-data replay:** v2. Curated static library of a few hundred CME sessions (trades-level for true volume-at-price), labeled by the same classifiers with **soft labels** (classifier confidence), used as out-of-distribution checks.

---

## 8. UI — Screen-by-Screen Wireframes

**1. Home (portrait)**

```
┌─────────────────────────────┐
│ 🔥 12  (2 freezes)     ⚙︎   │
│                             │
│ ┌─────────────────────────┐ │
│ │  DAILY WARM-UP  · 3 min │ │
│ │  2 misses due · shapes  │ │
│ │  + mixed block          │ │
│ │        [ START ]        │ │
│ └─────────────────────────┘ │
│                             │
│  Skills                     │
│  POC/VA      ▁▂▄▅  1420 ↑8 │
│  Excess/Poor ▃▃▅▆  1510 ↑2 │
│  Shapes      ▂▄▄▃  1180 ↓4 │
│                             │
│ ┌─────────────────────────┐ │
│ │ ▶ CONTINUE TREE:        │ │
│ │   Regime Gate (T2)      │ │
│ └─────────────────────────┘ │
│  [Tree] [Drill] [Stats]     │
└─────────────────────────────┘
```
No feed. No leaderboard. One primary CTA.

**2. Drill screen (portrait-native)**

```
┌─────────────────────────────┐
│ Rated · Excess/Poor    ◔par │
│ ┌─────────────────────────┐ │
│ │        ▓▓              │ │
│ │      ▓▓▓▓▓▓            │ │
│ │    ▓▓▓▓▓▓▓▓▓▓          │ │
│ │      ▓▓▓▓▓▓▓           │ │
│ │        ▓▓▓   ←(hilite) │ │  profile = 70% height
│ │         ▓              │ │
│ └─────────────────────────┘ │
│  Excess or poor?            │
│ ┌───────────┐ ┌───────────┐ │
│ │  EXCESS   │ │   POOR    │ │  ≥44px targets
│ └───────────┘ └───────────┘ │
│  conf: (sure)(lean)(guess)  │
└─────────────────────────────┘
   after answer (≤250ms):
│ ✗ POOR — two flat touches,  │
│ no tail. Unfinished business│
│ = repair magnet.   ⟲ replay │
│ ────────────  [ NEXT ▸ ]    │
```

**3. Result screen (per round)**

```
┌─────────────────────────────┐
│  "When you said sure, you   │
│   were right 84% — well     │
│   calibrated this round."   │
│  Rating 1510 → 1518  (+8)   │
│  Bot Points this block: +37 │
│                             │
│  Film strip (tap to replay) │
│  [▓][▓][▓][▓][▓][▓][▓][▓]  │
│   ✓  ✓  ✗  ✓  ✓  ✗  ✓  ✓  │
│                             │
│  2 misses queued → Thu      │
│  [ AGAIN ]  [ HOME ]        │
└─────────────────────────────┘
```
Sentence first, numbers second, spacing made visible.

**4. Skill Tree (subway map, portrait)**

```
┌─────────────────────────────┐
│ T0 ●━●━●━●━●  (lit=mastered)│
│    POC VA HVN EX  SHAPE     │
│         │       ╲           │
│ T1 ●━●━◐━○      ~●~ (rusty, │
│    OPEN IB 1TF DAY  flicker)│
│         │                   │
│ T2 ◎━━━○━━━○   ◎=checkpoint │
│    REGIME CAL FOLK   armed  │
│         │                   │
│    ☠ BOSS 1 "expect ~40%"   │
│ T3 ○━░━░  (T4/T5 silhouettes│
│ T4 ░░░░░   locked, visible) │
│ T5 ░░☠☠░━[EXPECTANCY]       │
└─────────────────────────────┘
```

**5. Checkpoint screen** — deliberately juice-free:

```
┌─────────────────────────────┐
│ ▓ CHECKPOINT · Shapes ▓     │  dark slate frame
│   Item 4 of 10              │  no per-item feedback
│   [profile]                 │  no earcons, no streaks
│   D  P  b  B  TREND         │  full debrief at end
└─────────────────────────────┘
   on fail: "Retry available
   Thursday. Misses queued
   for review." (no red, no
   punishment copy)
```

**6. Boss / Sim screen (desktop-first)**

```
┌──────────────────────────────────────────────┬───────────────┐
│  chart + live-building profile (right edge)  │ ORDER TICKET  │
│  prior-day profiles ghosted left             │ setup: [VA-fade▾]│
│  ── planted structures drawn, never labeled  │ regime: (BAL/IMB)│
│                                              │ stop:  tap level │
│  ⏸ ▶ ⏩ 10×   bracket: E (5/13)              │ target:tap level │
│                                              │ [SUBMIT READ]    │
│  READ FEED                    Read Score 74  │──────────────── │
│  ✓ open-drive called (B1)     P&L +0.4R      │ desk limits:     │
│  ✗ fade illegal — trend day   luck  −0.3R    │ day −$500 ▓▓░░  │
│  ✓ KILL in 1.2s                              │ trail  ▓▓▓░     │
└──────────────────────────────────────────────┴───────────────┘
```
The ticket **forces the declared read** — setup tag + regime tag + structural stop/target tapped on profile levels — before any action exists. Verdict ≤100ms on submit. Past calls pin as chart markers.

**7. Debrief (after boss/sim)**

```
┌──────────────────────────────────────────────┐
│ READ SCORE 78 · P&L −0.8R · LUCK −1.4R       │
│ "Your reads earned +0.6R of decisions;       │
│  variance handed you −1.4R."                 │
│ timeline ────●───●───✗───●───●───✗──── close │
│        (tap node → micro-replay of the tell) │
│ [ ONE THING TO DRILL → queues Woodpecker ]   │
└──────────────────────────────────────────────┘
```

**8. Stats (desktop-first, sentence-led)** — calibration curve behind its headline sentence; confusion matrices labeled by misconception (*"You confuse P with b — 7 times"*); discipline counters (fades-vs-one-timeframing, news violations, NO-TRADE compliance); the expectancy ledger with grayed n<30 cells and the "unmeasured = folklore" watermark; drill-vs-boss divergence sparkline.

**FTUE (<3 minutes):** no account, no lecture. Screen one is "tap the poor high" — glow, one-liner, three reps, then *"You just read an auction"* + first rating. One mechanic per drill, ever.

**Accessibility:** verdict color always paired with ✓/✗; direction with arrows; `prefers-reduced-motion` honored; ≤3 flashes/sec; replays pausable; snap-to-level so no pixel precision is ever required.

---

## 9. Architecture & Accuracy Requirements

**TypeScript monorepo, no backend required for v1.** Local-first; all grading client-side; optional thin sync later.

- **`@auction/core` (pure TS, zero dependencies):** the profile builder, 70% VA expansion, day/open-type classifiers, one-timeframing detector, acceptance heuristics — **ported once from `/home/user/a/scripts/figures/`** and used identically by generation-verification, rendering, and grading. The single-library rule makes grader-vs-renderer disagreement structurally impossible. Golden-file tests pin TS outputs to the Python reference outputs.
- **`@auction/gen` (pure TS):** script compiler (day/open type → segments) → Brownian-bridge filler → λ engine (seasonal × GARCH × boost) → structure planter → rejection-resampling verifier calling `core` classifiers. Seeded PCG32 named substreams. Runs in a Web Worker; keeps the 20-item warm pool per active skill; single-profile items generate in <50ms.
- **`@auction/schedule` (pure TS):** expanding-interval queue (1d/3d/7d), Warm-Up assembler, leech detector, checkpoint arming logic.
- **Client:** React + **canvas** chart renderer (SVG won't hold 60fps replays); verdict layer as a separate compositor so juice never repaints data; state machine per drill (`ITEM → ARMED → ANSWERED → VERDICT → NEXT`); **verdicts computed locally from labels shipped with the item, so the ≤100ms budget never touches a network.** IndexedDB persistence of the decision ledger: `(decisionId, nodeId, seed, paramsVersion, answer, latency, verdict, brier)`.
- **v1 has no server.** Ratings, scheduling, and stats are all local. The ledger format is designed for later sync: because every record carries `(seed, paramsVersion)`, a future server can re-derive any verdict by re-simulation (which is also the anti-cheat story when leaderboard-free sync arrives).
- **Analytics (local, exportable):** node-mastery funnels, item-pool difficulty drift, and the drill-rating vs. boss Read-Score divergence dashboard — first-class from day one.

**Accuracy requirements (non-negotiable):**
1. **Every displayed quantity is computed by the real algorithm.** POC, VA, HVN/LVN, IB, dPOC, day/open classifications — no hand-drawn profiles, no painted structures, no approximations in the render path. Reference implementations: `fig-01` (profile anatomy/VA), `fig-02`/`fig-05` (shapes/day types), `fig-06` (open types), `fig-09` (excess vs poor), `fig-10` (value migration), `fig-11` (LVN), `fig-14` (nPOC).
2. **One library** (`@auction/core`) for generation-verification, rendering, and grading.
3. Golden-file parity with the Python reference on shared fixtures (bit-identical row indices for POC/VA; identical classification labels).
4. No displayed quantity may bypass `core` — enforced by lint rule (renderer imports only from `core` types) and code review.

---

## 10. Test Plan

**A. Engine reference cases (CI, blocking):**
- Golden-file tests: fixture price/volume paths → `@auction/core` outputs must match the Python `/scripts/figures/` outputs exactly (POC row, VA rows after each expansion round, shape label, open-type label, one-timeframing break index).
- Property tests: VA always contains POC; VA mass ∈ [69%, 71%]; profile mass equals path dwell × λ within tolerance; snap-to-row inverse-maps correctly at every zoom.
- Ensemble invariants (nightly, 1,000-session batches): excess kurtosis >1; |r| autocorrelation positive to lag 30+; open/midday volume 2–4×; volume-clock returns near-Gaussian; classifier label recovery ≥99% post-rejection; rejection acceptance rate ≥20% per script type (alarm below).
- Determinism: same `(seed, paramsVersion)` → byte-identical session; knob nudge on one substream leaves other substreams' output unchanged (the Woodpecker-sibling guarantee).
- Anti-artifact audit as a unit test: shallow classifier must not separate generated sessions from the real-data holdout on trivial statistics beyond a set AUC threshold.
- Conditional-probability labels: Monte-Carlo re-estimate each pool's declared `conditionalProbs` from 10k fresh draws; declared vs. measured must agree within CI bounds.

**B. Feedback-text audit (vs. the guide):**
- Every feedback template string lives in a single registry keyed by (drill, verdict-class, misconception). A doc test walks the registry and asserts each template's claim set against a curated assertion list extracted from `/home/user/a/docs/volume-profiles-and-auction-market-theory.md` (e.g., "excess ends the auction," "poor extremes are repair magnets," "two 30-min periods outside + value migration = acceptance," "P after extended rally = short covering / exhaustion risk"). Any new template requires a guide citation (section anchor) in the registry entry.
- Style lint on templates: task-referenced only (reject second-person trait language: "you always," "you're bad"), one sentence to first period ≤160 chars, no folklore number stated as fact (the literal string "80%" may only appear in folklore-audit contexts).
- Human pass: domain review of all templates against the guide before each parameter-pack rotation.

**C. Loop-latency budget tests:** automated performance test asserting verdict paint ≤100ms, explanation ≤250ms, NEXT interactive ≤400ms on a mid-tier phone profile; Rush item generation <50ms p95.

**D. Playtest checklist (per build):**
- [ ] FTUE: fresh player reaches first "You just read an auction" in <3 minutes, no account.
- [ ] A full Daily Warm-Up completes in ≤4 minutes and contains ≥1 adaptive-difficulty block.
- [ ] Miss an item → it (or its sibling) reappears on schedule; sibling is visibly different but tests the same discrimination.
- [ ] Checkpoint arms only after accuracy AND latency criteria; unlocks next calendar day; failure shows the no-punishment copy and queues misses.
- [ ] Cardinal error in Regime Gate triggers 3× rating hit AND the forced micro-replay of the trend tells.
- [ ] Boss 1: a player who fades everything busts the loss limit; the debrief replay highlights the IB/pullback/volume tells; "one thing to drill" queues correctly.
- [ ] Calibration block of 25 ends with the base-rate reveal; the bot's answer equals the pool's declared rate.
- [ ] Confidence taps show up in the calibration ledger.
- [ ] No naked verdict found anywhere (spot-check 50 random reps).
- [ ] Reduced-motion mode: no squash-stretch, no flashes; color-blind pass: every verdict readable by icon alone.
- [ ] Wrong-process lucky win in a boss earns zero Read-Score credit and the debrief says so.
- [ ] Streak survives a skipped day when a freeze is available; streak does not increment on Rush-only days.

**E. Learning-health monitoring (post-launch, from day one):** drill-rating vs. boss Read-Score divergence per node; checkpoint pass rates (target 60–80% first attempt); calibration normalization curve vs. the 100–200-judgment expectation; leech frequency per item pool.

---

## 11. Ruthless v1 Scope

**IN (ship this, exactly this):**
- T0–T2 complete: Drills A–I (POC+VA Snap, Excess-or-Poor, HVN/LVN Marker, Shape Alphabet, Calibration Range, Open-Type Ladder, One-Timeframing Buzzer, Acceptance Clock, Regime Gate).
- T3 as **Setup Picker with NO TRADE only** (Drill J).
- **Three bosses** (trend-day fade, 80%-rule through news, P-shape "bottom") with declared-read ticket, prop-desk limits, KILL/HOLD interrupts, and the debrief scrubber.
- Modes: Rated, Rush, Streak, Woodpecker (sibling re-serve), Calibration Range, Checkpoint, Daily Warm-Up, Daily Recap.
- Calibration intro module (<1 hour) as the T2 gateway.
- Skill tree with delayed interleaved checkpoints, rusty demotion, leech rule, cardinal-error weighting, scaffolding fade schedule.
- Fixed expanding-interval spacing (1d/3d/7d).
- Knob-derived item ratings; Glicko-2 player ratings per node.
- Stats screen: sentence-led calibration, confusion matrices → misconception counters, discipline counters, expectancy ledger (grayed <30n).
- Generator per §7 including fabricated prior sessions, decoys, ambiguity blending, determinism spec, CI invariants, monthly parameter-pack rotation.
- Local-first web app (portrait phone drills + desktop boss/stats), no account, no backend.
- Divergence health dashboard (internal).

**OUT (explicitly cut from v1):**
- T4 order-flow/footprint drills entirely (needs Hawkes prints + a new renderer — v2).
- T3 beyond the setup picker: individual setup drills (VA-edge-fade, 80%-rule honest edition, look-above-and-fail, LVN both-branches ticket, structural-stop sniper, kill-switch drill as standalone), stop/target placement drills → v1.1.
- track-references and spike-rules as standalone nodes (spike logic folded into Acceptance Clock feedback).
- Real-data replay library (v2, soft labels).
- Any conditional-EV oracle / Monte-Carlo rollout grading (the sim-first research project). Boss grading is label/legality-based in v1.
- Multiplayer, leaderboards, social of any kind.
- Half-life-regression scheduling (fixed intervals until there's data).
- Glicko item co-rating (until n >10k answers per pool).
- Accounts, cloud sync, server anything.
- Native mobile apps (responsive web only).
- Narrative fiction/desk-promotion skin beyond the minimal boss framing.
- Sound beyond three earcons; music; cosmetics; XP; badges.
- Bosses 4+ (the out-of-range open-auction boss → v1.1).
- Position-sizing, multi-timeframe confluence, multi-day nPOC tracking drills.

**If the schedule slips further,** cut in this order: Daily Recap (fold into Warm-Up) → Streak mode → HVN/LVN Marker (fold into VA Snap feedback) → Boss 3 → Boss 2. Never cut: delayed interleaved checkpoints, the calibration module, the determinism spec, the outcome/decision split, the one-library accuracy rule, streak freezes. Those are the difference between a trainer and homework with candlesticks.

---

## 12. Open Questions for the User

1. **Latency criteria calibration:** the per-node response-time thresholds (<2.5s POC, <3s shapes, etc.) are placeholders. Do you want a pre-launch tuning study (internal playtesting cohort), or ship with placeholders and tune from live data?
2. **Ambiguous-item grading formula:** for blended/judgment-call items we grade "probabilistically" — proposed: score = 1 − |answer − blendWeight| mapped onto the item's point scale, with rating K-factor halved. Acceptable, or do you prefer excluding ambiguous items from rating entirely (feedback-only)?
3. **Boss luck-residual math:** without an EV oracle, the decomposition sentence attributes P&L minus a legality/read-weighted expectation. How rough may this approximation be before you'd rather drop the sentence and show only Read Score vs. P&L side by side?
4. **Platform priority:** v1 is a responsive web app, portrait-first. Is desktop boss-mode a launch requirement, or can bosses ship 2–4 weeks after the mobile drill loop?
5. **Content of the calibration intro module:** license/adapt existing GJP-style training text, or write an original <1-hour module from the guide's folklore-audit material? (Original is more work; it's also the on-brand choice.)
6. **Real-data holdout for the anti-artifact audit:** the CI test needs a small set of real CME sessions to discriminate against. Do you have data access (e.g., Databento GLBX.MDP3) budgeted for v1, or should the audit launch with public 1-min OHLCV and an approximation caveat?
7. **Monetization posture:** nothing in this GDD assumes payment. Free-with-local-data forever, one-time purchase, or subscription-after-T1? This affects whether an account system moves up from v2.
8. **Parameter-pack rotation cadence:** monthly is proposed. Who owns authoring/validating new packs post-launch — is that a standing content role or automated generation with spot review?
9. **Naming check:** drill and mode names ("Woodpecker," "Rush," "Bot Points," "The Auction" itself) — any trademark/brand review needed before UI copy freezes?

---

*Word count: ~5,300. Base: arcade-first. Grafts: declared-read ticket, sim-permission gating, luck decomposition, prior-session fabrication, KILL/HOLD interrupts, Read Feed, debrief scrubber, prop limits, monthly rotation (sim-first); cardinal-error 3×, scaffolding fade schedule, juice-free checkpoints, base-rate interstitials, grayed expectancy cells, leech rule, latency criteria, checkpoint-failure UX, knob-seeded ratings, locked silhouettes, in-UI interleaving explainer, re-simulation anti-cheat (curriculum-first). Conflicts resolved per rule: EV oracle cut for legality grading (feasibility, all three judges); three bosses kept over curriculum-first's one (learning efficacy, thesis pillar 5); fixed intervals over half-life regression (feasibility); checkpoints-only mastery kept absolute (learning efficacy over engagement).*
