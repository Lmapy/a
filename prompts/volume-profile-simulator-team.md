# PROMPT — Build "The Auction": a video-game-grade volume profile simulator

*Paste everything below this line as the task prompt. It is designed for multi-agent
orchestration: it opts in explicitly, phases the work, and puts a human checkpoint
between design and build.*

---

Use a workflow (multi-agent orchestration) for this. You are the orchestrator of a
team of agents building **The Auction** — a game-like volume profile / auction market
theory simulator whose single purpose is to make a trader *fluent* in reading profiles,
faster than any book could. Run it in four phases. Do not skip Phase A, and stop for
my approval at the end of Phase B before writing any product code.

## Ground truth and source material

- The domain curriculum is `docs/volume-profiles-and-auction-market-theory.md` in this
  repo — treat it as the single source of truth for every concept, definition, setup,
  and (critically) every caveat. The simulator must never teach a folklore statistic
  as fact; the guide labels which is which.
- The rendering/accuracy bar is set by `scripts/figures/` in this repo: every
  structural quantity shown to the player (POC, VAH/VAL, day type, delta, imbalance
  ratios) must be **computed by the real algorithm, never hand-placed**, and covered
  by unit tests against reference cases.

## The design thesis (bake these in unless Phase A research overturns them)

1. **Ground truth by construction.** The simulator *generates* its market sessions
   from labeled generative models (day type, open type, planted structures like
   naked POCs, poor highs, LVNs). Because the engine knows what it built, every
   player decision gets **objective, instant, explainable feedback** — the thing
   real-market practice can never give a beginner. This is the core advantage;
   protect it.
2. **Instant feedback loops, sub-second.** Decision → verdict → one-line auction-logic
   explanation → optional micro-replay. Kahneman/Klein: intuition is trainable only in
   high-validity environments with fast feedback — the simulator manufactures exactly
   that environment.
3. **Train calibration, not prediction.** Players answer with probabilities
   ("70% this open-drive holds"), scored by Brier score / calibration curves, with
   base rates revealed after. This teaches the guide's meta-lesson: measure, don't
   believe folklore.
4. **One skill per drill; interleave later.** Atomic drills (spot the POC; shade the
   value area; classify this open in 30 seconds; 80%-rule trigger or no trigger;
   fade or follow?) → mixed speed rounds → full simulated sessions with P&L.
5. **Punish the classic failure modes on purpose.** Boss levels: a trend day that
   bleeds anyone who fades the edges; a news session that voids the 80% rule; a
   P-shape that looks like a bottom and isn't. Losing to these IS the curriculum.
6. **Expectancy dashboard as the endgame.** R-multiples, per-setup win rates, the
   player's own measured statistics vs the folklore numbers — the game's final lesson
   is the guide's Part VII: your edge is what you measured, sized structurally.
7. **Hyper-learning scaffolding**: spaced repetition queue of missed items, streaks,
   adaptive difficulty (keep the player at ~80% success), 3-minute drill sessions,
   skill tree gated by demonstrated mastery, juice (sound/motion feedback) that never
   outruns correctness.

## Phase A — Research team (parallel agents; each returns sourced structured notes)

1. **Learning science**: deliberate practice (Ericsson), desirable difficulties
   (Bjork), immediate-feedback research, calibration training literature,
   flow/difficulty curves, transfer of simulator skill to the real task. What does
   the evidence actually support?
2. **Trainer prior art**: chess.com puzzles/puzzle rush, Duolingo, Aim Lab/KovaaK's,
   typing trainers, poker trainers, flight sims; existing trading sims and replay
   tools (TradingSim, NinjaTrader/TradingView replay, Jigsaw drills, prop-firm sims,
   exocharts replay). What mechanics demonstrably drive mastery; where do trading
   sims fall short (usually: no ground truth, no feedback, no curriculum).
3. **Synthetic market engine**: how to generate realistic intraday sessions with
   ground-truth labels — regime/day-type-conditioned price paths, volume-at-price
   generation consistent with the shape taxonomy, stylized facts to respect
   (U-shaped intraday volume, fat tails, IB behavior), and how to plant structures
   (nPOCs, poor highs, LVN corridors) without making them obvious. Also: a later
   real-data replay mode's data requirements.
4. **Curriculum mining**: decompose the guide into a skill tree — atomic skills with
   prerequisites, the misconception list (HVN-as-bounce-zone, fading trend days,
   reversed finished/unfinished auction, folklore stats), and per-skill testable
   mastery criteria that a drill can measure.
5. **Game feel / UX**: feedback juice, scoring and streak systems, progression and
   unlock design, dashboard/analytics UX, mobile-vs-desktop, accessibility
   (colorblind-safe — reuse the guide figures' validated palette).

## Phase B — Design (judge panel → GDD → STOP for approval)

- Three agents produce independent design proposals from different angles:
  **arcade-first** (drills and dopamine), **sim-first** (realistic sessions and P&L),
  **curriculum-first** (skill tree and mastery gates).
- A judge panel scores them on: predicted learning efficacy (against Phase A
  evidence), engagement, correctness-feasibility, and scope realism. Synthesize the
  winner plus the best ideas of the losers into a **Game Design Document**:
  core loop, full mode list with drill specs, feedback spec (exact verdict +
  explanation format), scoring math (calibration + expectancy), progression/skill
  tree, synthetic-engine spec with its label schema, UI wireframes, tech
  architecture, accuracy requirements, and a test plan.
- Recommended default architecture (challenge it if Phase B finds better): a
  self-contained web app (TypeScript, no backend required to start), engine as a
  pure, unit-tested library (market generation + profile math) fully decoupled from
  rendering; deterministic seeds so any session is replayable and shareable.
- **Stop here. Present the GDD and wait for my approval before Phase C.**

## Phase C — Build team (after approval)

- **Engine first, tests first**: the 70% value-area expansion (with tie rules), POC,
  TPO letters, VWAP + σ-bands, delta/imbalance math, day-type/open-type generators —
  each validated against reference cases before any UI exists. The figure scripts in
  `scripts/figures/` double as reference implementations.
- Then drills/modes, then UI, in vertical slices: each slice = one playable drill
  with feedback, shipped and verified before the next.
- Parallelize by module with clear interfaces; agents that touch the same files work
  in worktree isolation.

## Phase D — Adversarial QA (no self-grading)

- **Correctness audit**: independent agents recompute every number the UI can display
  from the engine's raw output; any mismatch is a release blocker.
- **Curriculum audit**: every feedback string checked against the guide — no
  contradiction, no unlabeled folklore stat.
- **Playtest audit**: agents actually play each drill (drive the UI), verify the
  feedback loop latency, the difficulty adaptation, and that boss levels punish the
  intended mistake and say why.
- **Learning audit**: score the finished product against the Phase A evidence
  checklist; gaps become the next iteration's backlog.

## Definition of done (v1)

A player with zero background can, in one sitting: complete the profile-anatomy
drills, reach the open-type speed round, lose to the trend-day boss, read the
explanation of *why* fading it was wrong, and see their own calibration curve —
with every number on screen computed by a tested algorithm.
