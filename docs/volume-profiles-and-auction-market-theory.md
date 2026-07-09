# Volume Profiles & Auction Market Theory — The Complete Practitioner's Guide

*A deeply-researched guide to what volume profiles are, how to read them, and how to trade them within the auction market theory framework. Compiled from a multi-agent research sweep across primary texts (Steidlmayer, Dalton), platform documentation, practitioner education, and the academic microstructure literature. All probability figures are labelled with their provenance — folklore is called folklore.*

---

## Table of Contents

1. [How to Use This Guide](#1-how-to-use-this-guide)
2. [Part I — Auction Market Theory: The Foundation](#part-i--auction-market-theory-the-foundation)
3. [Part II — The Volume Profile: Construction & Anatomy](#part-ii--the-volume-profile-construction--anatomy)
4. [Part III — Reading Profiles: Shapes, Day Types, and Context](#part-iii--reading-profiles-shapes-day-types-and-context)
5. [Part IV — The Trading Playbook](#part-iv--the-trading-playbook)
6. [Part V — Combining Profiles with Order Flow and Delta](#part-v--combining-profiles-with-order-flow-and-delta)
7. [Part VI — Instruments, Data, and Platforms](#part-vi--instruments-data-and-platforms)
8. [Part VII — Misconceptions, Failure Modes, and the Honest Evidence](#part-vii--misconceptions-failure-modes-and-the-honest-evidence)
9. [Part VIII — Risk Management and the Daily Process](#part-viii--risk-management-and-the-daily-process)
10. [Glossary](#glossary)
11. [Sources & Further Reading](#sources--further-reading)

---

## 1. How to Use This Guide

Volume profile is not an indicator that gives signals. It is a **map of where the market has done business**, and auction market theory (AMT) is the **language for interpreting that map**. Used together they answer three questions that drive every trade:

1. **Where is value?** (Where has the market accepted price?)
2. **Where is price relative to value?** (Inside it, above it, below it?)
3. **Is the current move being accepted or rejected?** (Is value relocating, or is price stretching away from value and snapping back?)

Everything else in this guide — shapes, setups, statistics, order flow — is machinery for answering those three questions faster and more precisely than other participants.

Read Part I first even if you only care about the setups. The setups in Part IV are trivial to memorize and nearly useless without the auction logic that tells you **when each one applies**. The single most repeated failure mode across every school of profile trading is applying a balance-market setup (fading edges) on an imbalanced (trend) day.

---

# Part I — Auction Market Theory: The Foundation

## 1.1 Where This All Came From

**J. Peter Steidlmayer**, a CBOT pit trader from a California farming family (CBOT member from 1963, Board of Directors 1981–83), wanted an objective, market-generated definition of *value* usable in the day timeframe — as opposed to pattern-based charting. While on the board he initiated two exchange products:

- **Market Profile** (introduced publicly ~1984–85): a chart that organizes the day's trade by *price* on the vertical axis, accumulating activity horizontally, producing a distribution that is typically fatter in the middle and thinner at the extremes — a bell-ish shape Steidlmayer recognized from his Berkeley statistics coursework.
- **The Liquidity Data Bank (LDB)**: end-of-day cleared volume broken down by price and by *class of trader* (CTI codes: CTI1 = local floor traders, CTI2 = commercial clearing members, CTI3 = members filling for members, CTI4 = public orders). Old-school profile traders watched the %CTI2 at a price as a proxy for "commercial" acceptance. The LDB also reported each contract's **value area** — the range containing ~70% of the day's trade.

In 1986 Steidlmayer and **Kevin Koy** founded the Market Logic School in Chicago, and their book ***Markets and Market Logic* (1986)** became the theoretical founding document. Its core axioms (near-verbatim):

> "A marketplace exists solely to facilitate trade. This is its only purpose."
>
> "The market, in facilitating trade, uses price to promote activity. Price does this by advertising opportunity."
>
> "All markets 'auction' or trend up and down in order to fulfill their purpose, to facilitate trade."

**James F. Dalton** — CBOT/CBOE member, early Senior EVP of the CBOE, later an institutional desk head at PaineWebber/UBS — turned the abstract market logic into an operational trader's handbook:

- ***Mind Over Markets* (1990, updated 2013)** codified the day-type taxonomy, opening types, tails/excess, range extension, initiative-vs-responsive classification, and TPO counting. It remains the dominant practitioner reference for day-timeframe profile reading.
- ***Markets in Profile* (2007)** extended the framework to nested long/intermediate/short-term auctions and fused it with behavioral finance — its central instruction: locate current activity within the next-larger auction, and trade in the direction of the timeframe larger than your own.

Two historical notes worth knowing: the "80% rule" and most of the famous probability lore trace back to **The Profile Reports (Dalton Capital Management, 1987–1991)**, not to any audited study. And Steidlmayer himself later moved beyond fixed-session bell curves toward flexible variable-window distributions (his "Steidlmayer Distribution" and Capital Flow software), remarking in later interviews that the popular 1980s presentation had become dated — an implicit self-critique the marketing around Market Profile rarely mentions.

## 1.2 The Core Model: The Market Is a Two-Way Auction

The market's only purpose is to facilitate trade. It does this by running a continuous **two-way auction**: price moves up until the last buyer has bought (cutting off buying), then reverses; down until the last seller has sold. The end of an up auction *is* the beginning of a down auction.

Dalton's canonical triad describes the division of labor:

> **Price advertises** all opportunities; **time regulates** all opportunities; **volume measures** the success or failure of the advertised opportunities.

From this, the operational corollary: **Price × Time = Value**, confirmed (or refuted) by volume.

- **Price ≠ value.** Price is just an advertising mechanism — a number flashed to attract business. *Value* is price that has been **accepted**: traded at over time, with volume. The profile makes value visible.
- If price at a level **facilitates trade** (volume builds, time is spent), the market builds value there.
- If price at a level **fails to facilitate trade** (volume dries up, price leaves quickly), price *must* move — that price was rejected as unfair by one side or both.

### Balance and imbalance

Markets alternate perpetually between two regimes:

- **Balance**: two-sided rotational trade around an agreed fair price. Value areas overlap day to day; the profile fattens into a bell (a "D-shape"); responsive traders fade the edges. Markets spend the majority of their time here.
- **Imbalance**: one side becomes aggressive (initiative activity), price leaves the balance area and **trends — vertically searching for new value** — until it finds a level that again cuts off activity in both directions, where a new balance forms.

The full cycle is **balance → imbalance → new balance**, endlessly. A trading range *is* a balance area. A breakout *is* auction imbalance. This is the regime switch that determines which trades work: fading edges works in balance; following price works in imbalance.

### Acceptance and rejection

- **Acceptance**: price trades at a level long enough (time) and with enough volume to build structure there — the market ratifies the level as value. Rule-of-thumb heuristic used across the Dalton school: **two consecutive 30-minute periods** spent at new price levels constitutes acceptance.
- **Rejection**: price visits a level and rapidly leaves with little time or volume — tails, single prints, low-volume nodes. The market refused to do business there.

Acceptance ⇒ the level will likely matter as *value* in the future (support/resistance by consensus). Rejection ⇒ the level will likely matter as an *extreme* (excess) in the future.

### Initiative vs responsive activity

Every trade is classified relative to the **previous day's value area**:

| Location of activity | Buying is… | Selling is… |
|---|---|---|
| **Above** prior value area | **Initiative** (buying despite "expensive" prices — comparing price to *future* value) | **Responsive** (selling in response to price above value) |
| **Below** prior value area | **Responsive** (buying the discount) | **Initiative** (selling despite "cheap" prices) |
| Inside prior value | Ambiguous / neutral | Ambiguous / neutral |

**Initiative activity carries conviction** — it can relocate value and drive trends. **Responsive activity** pushes price back toward existing value but lacks the conviction to relocate it. The practitioner shorthand: *responsive = fade, initiative = follow.*

### The "other timeframe" participant

Dalton's two-actor model of any session:

- **Day-timeframe traders** (locals, scalpers, day traders): flat by the close, trade around current value, provide liquidity, dominate the value-area churn.
- **Other-timeframe (OTF) traders** (institutions, swing/position players): compare price to a longer-horizon notion of value and act with size. Their footprints on the profile are **range extension, tails, elongated profiles, and value-area migration.**

OTF *initiative* action drives trends. OTF *responsive* action fades price far from value. Reading the profile is largely the art of detecting whether, where, and in which direction the OTF participant is active. (Caveat for modern markets: in 24-hour electronic trade the clean pit-era duality is diluted — but the logic of "is bigger, longer-horizon money active here?" survives.)

## 1.3 Structural Vocabulary You Must Know

- **Excess**: a swift, decisive rejection at an extreme — visible as a **tail** (in TPO terms, ≥2 single prints at the high or low; in volume terms, a low-volume taper/wick at the extreme). "Excess marks the end of one auction and the beginning of a new auction." A longer tail = greater conviction of rejection. Extremes with proper excess are *less* likely to be revisited and act as durable references.
- **Poor (unfinished) high/low**: the opposite — a flat, "mechanically formed" extreme with little or no excess (multiple TPOs/even volume right up to the edge). Usually built by day-timeframe traders with no OTF player finishing the auction. Poor extremes are **unfinished business**: elevated odds the market later revisits and "repairs" them (creates proper excess or extends beyond). One practitioner-published figure: ~78% of poor highs exceeded and ~75% of poor lows broken within five sessions — *practitioner statistics, not peer-reviewed*.
- **Single prints**: stretches of profile only one TPO wide inside the body — the fingerprint of fast, emotional, one-sided repricing. Their volume analog is the **low volume node (LVN)**. Both mark unfair prices the market transited without doing business, and both are future reference/repair zones.
- **Range extension**: any new high or low made after the initial balance — the classical tell that the OTF participant has entered the session.
- **One-timeframing**: consecutive 30-minute brackets making higher lows without violating the prior bracket's low (up case; mirror for down). One side is continuously in control — the hallmark of trend days. *While a market is one-timeframing, do not fade it.*
- **Spike**: a late-session directional push that leaves no time to build structure. Dalton's spike rules for the next open: open **above** the spike = acceptance (bullish, don't fade); open **within** the spike = qualified acceptance, trade off the spike base; open **below** the spike base = the late move was emotional — rejection, expect early trade lower.

## 1.4 The Bell Curve — Useful Metaphor, Not Statistics

Steidlmayer mapped the Gaussian "first standard deviation" idea onto the day's distribution: the **value area ≈ ±1σ ≈ 68.3%**, rounded to the working convention of **70%**. Know the honest status of this analogy:

1. **Returns are not Gaussian** — fat tails, skew, regime dependence (Mandelbrot's critique). Anything leaning on literal 1σ logic understates extremes.
2. The profile is a distribution of *time/volume at price within one bounded session*, not a distribution of returns — defenders say the bell curve is a **metaphor for balance**, not a statistical model. That's the correct reading, but it concedes "1σ" is branding: 70% is a convention, and nothing guarantees a day's histogram is unimodal or symmetric.
3. Profile practitioners themselves teach that **non-normal shapes are the tradable information** (P, b, B shapes) — quietly abandoning the normality premise where it matters.

Practical takeaway: treat the value area as "the belt of prices the market ratified today," not as a statistical confidence interval.

---

# Part II — The Volume Profile: Construction & Anatomy

## 2.1 What a Volume Profile Is and How It's Built

A volume profile is a **volume-at-price histogram**: all volume traded over a chosen window, redistributed along the **vertical price axis** instead of the horizontal time axis. Each row is a price bucket; the row's length is the volume transacted there.

**Data resolution matters more than most retail traders realize:**

- The ideal input is **tick data** (every trade at its exact price and size) — an exact distribution.
- Platforms without tick data approximate from bars: assigning a bar's volume to its close, spreading it evenly across the bar's range, or using statistical density models. **TradingView's built-in profiles are 1-minute-bar approximations, not tick-exact** — TradingView's own docs note the shape "is an estimate… and the Point of Control may differ slightly from a native volume-profile tool."
- **Row size (ticks per row) changes the profile.** Coarser aggregation smooths the histogram and *moves the POC and value area*. Two users on the "same" platform with different row settings will disagree on levels. Pick sensible settings (e.g., Sierra Chart recommends 1–2 ticks/row intraday) and keep them consistent.

**Profile variants of the same histogram**: bid/ask profiles (volume split by aggressor side), delta profiles (net ask-minus-bid per row), and the per-candle version — the **footprint chart** (Part V).

## 2.2 Anatomy: POC, Value Area, HVN, LVN

### Point of Control (POC)

The single price row with the **highest traded volume** — the mode of the distribution and, in auction terms, the session's "fairest price," where the most two-sided business was done. The longest bar on the histogram.

### Value Area (VA), VAH, VAL

The price band containing **~70% of the period's volume**; its top is the **Value Area High (VAH)**, bottom the **Value Area Low (VAL)**. Standard expansion algorithm (TradingView-documented variant):

1. Target volume = total × 70%.
2. Start at the POC; include it.
3. Compare the next untaken row **above** vs the next untaken row **below**; add the **larger**; extend VAH or VAL accordingly.
4. Repeat until the target is reached (ties go to the row closer to POC).

The classic CBOT/TPO procedure differs in details that explain cross-platform disagreement: it expands in **pairs of rows** (two above vs two below), sometimes uses **68%** instead of 70%, and applies specific POC tie-break rules (closest to range midpoint, etc.). Add differing row sizes and RTH-vs-full-session templates, and you get the perennial "why is my VAH different from yours?" Answer: settings. Consistency beats "correctness."

### High Volume Nodes (HVN) and Low Volume Nodes (LVN)

- **HVN** — a local bulge: a zone of heavy two-sided trade = **acceptance, consensus, fair price**. Market logic: HVNs act as **magnets and friction** — price returning to an HVN tends to slow, get sticky, and rotate, because prior inventory and unfinished business live there. HVNs are **arrival targets**, not bounce zones.
- **LVN** — a thin shelf or gap between HVNs: price passed through fast without finding acceptance = **rejection, imbalance**. Market logic: LVNs are "speed zones/air pockets" — on revisit, price either **rejects sharply at the edge** or **traverses fast** to the next HVN, because few participants have inventory there to defend. This binary behavior is precisely what makes LVNs superb *decision* prices: define both outcomes before entry.

## 2.3 Profile Scopes

| Scope | What it is | What it's for |
|---|---|---|
| **Session profile** | One trading session per profile | The day-trading workhorse: yesterday's POC/VAH/VAL are today's reference levels |
| **Composite / multi-day** | Aggregated across a week, month, or an entire balance range | Swing-timeframe structure: the durable HVN shelves and LVN fast-lanes |
| **Fixed range** | User-selected exact start/end (a trend leg, a consolidation, a news reaction) | Locked levels that don't change as the chart moves |
| **Visible range (VPVR)** | Computed over whatever bars are on screen | Instant context only — it **recomputes every scroll/zoom**, so its levels are not stable references. Prefer fixed-range/session for level-marking |
| **Anchored** | Starts at a chosen event (swing high/low, breakout, earnings) extending to now | "Where has value built *since X*?" |
| **Developing POC/VA (dPOC)** | The POC/VAH/VAL plotted live as they evolve during the session | Real-time gauge of where value is being accepted; its migration direction is a trend tell |
| **Naked / virgin POC (nPOC)** | A prior session's POC never yet revisited | Tracked as a magnet target for days or weeks; loses "naked" status on first touch |

## 2.4 Volume Profile vs TPO (Market Profile)

- **TPO profile** stacks letters — one per 30-minute bracket per price touched — measuring **time-at-price**. **Volume profile** measures **volume-at-price** and ignores duration.
- They look similar on rotational days and **diverge when time and volume decouple**: a market can sit at a price for hours on thin volume, or transact enormous volume in minutes. TPO-POC and volume-POC often sit at different prices.
- **The Dalton camp prefers TPO**: time is the market's acceptance mechanism; TPO structure carries the auction narrative and smooths mechanically volume-heavy prices (open, close, settlement); volume then *validates* structure. **The volume camp counters**: volume is actual committed capital — with modern tick data there's no reason to proxy it with time.
- Pragmatic professional consensus: **run both — TPO for structure and context, volume profile for precise levels** — and treat prices where both POCs/VAs align as the highest-confidence zones. A persistent TPO-POC vs volume-POC split is itself information (time-consensus disagreeing with volume-conviction).

## 2.5 Volume Profile vs VWAP

- **VWAP** = Σ(price × volume) / Σ(volume) from the session open (or an anchor) — the volume-weighted **mean** as a line through time. The **POC is the mode** of the same distribution. In a symmetric session they nearly coincide; on trend/double-distribution days they diverge.
- Session VWAP is commonly plotted with **±1σ/±2σ bands** — the VWAP analog of the value area (both are "≈1σ of where volume traded").
- **Used together**: VWAP supplies intraday directional bias (above = bullish, below = bearish) and dynamic mean-reversion rails; the profile supplies *level quality* (HVN/LVN, POC, nPOCs). **VWAP + POC confluence — especially a naked POC — is a standard high-attention reaction zone.** Wide separation between VWAP and the dPOC flags a one-sided, still-repricing market.

## 2.6 Data Caveats by Market (read before trusting any profile)

- **Futures (ES/NQ/CL/GC…)**: the gold standard — one centralized book, complete volume, reliable aggressor data. One decision matters: **RTH-only vs full-session profiles.** 80–90% of index-futures volume prints in regular hours; mixing thin overnight volume into the profile drags POC/VA toward low-participation prices. **RTH-only is the common default for value-area work**; keep a separate overnight profile for gap/overnight-inventory analysis.
- **Stocks**: consolidated-tape volume is complete, but **~40–50% of US equity volume executes off-exchange** (dark pools/OTC) and prints with delay — total volume is right, the *price-level attribution and order-flow character* are muddier than futures.
- **Crypto**: volume is fragmented across venues; a Binance-only profile is not "the market's" profile. Use the dominant venue's **perpetual futures** or an aggregating platform. Per-exchange aggressor data is real and free, but differs by venue.
- **Spot forex**: there is **no real volume** — retail platforms show *tick volume* (count of price changes on one broker's feed). A forex "volume profile" is a proxy from one liquidity pool. Many practitioners skip VP in spot FX entirely or use CME FX futures instead.

---

# Part III — Reading Profiles: Shapes, Day Types, and Context

## 3.1 The Shape Alphabet

The profile's shape is a fossil record of who was active and where. Learn four canonical shapes plus the trend profile:

### D-shape — balance

Symmetric bell; POC central; value area brackets it. Two-sided rotation around an agreed fair price. **Implication**: mean-reversion conditions — fade the edges, POC is a magnet — *while it lasts*. A mature D is also **stored energy**: larger players may be building positions inside it; watch the edges for the eventual initiative breakout.

### P-shape — rally into balance (classic short-covering print)

Long thin low-volume **lower stem**, wide well-developed **upper bulge** with the POC high. Price rallied fast (lower prices rejected), then built two-sided trade higher.

Dalton's key diagnostic: the classic P is **short covering — "old business, not new participants."** Trapped shorts buying back is *finite* fuel; when covering completes, the rally stalls and the top balances. Three context-dependent readings, all legitimate:

- **In an established uptrend**: a stair-step of higher value — continuation. (P-profiles routinely appear early in bull moves.)
- **At the end of a downtrend**: a short-covering rally that often marks the start of bottoming.
- **After an extended rally, into major composite resistance**: exhaustion risk — the buying was inventory correction, not initiative.

Dalton's tell for real initiative buying vs mere inventory adjustment: **elongation**. New-money buying plus covering produces an elongated profile; a stubby P without elongation is inventory adjustment.

### b-shape — decline into balance (long liquidation)

Mirror of the P: thin upper stem, wide bulge at the lows. **Long liquidation** — trapped longs selling out (again, old business). Common as a continuation stair-step in downtrends; appearing *during an uptrend* it's a reversal warning; after a long decline it can precede a bounce (liquidation ≠ new initiative shorts).

A refinement worth internalizing (Bookmap's framing): the thin extreme of a P/b is often not aggressive rejection but **participation exhaustion** — one side simply ran out. That's exactly why such extremes are *weak* (no defenders) and frequently get revisited, unlike true excess.

### B-shape — double distribution

**Two distinct HVN bulges separated by a mid-profile LVN** (single prints in TPO terms). Records a completed *balance → imbalance → new balance* cycle inside one window. Reading:

- The **mid-LVN neck is the line in the sand**: support while price is in the upper distribution, resistance while in the lower; re-entries through it tend to move fast.
- Which bulge holds the dominant POC tells you where the auction's center of gravity ended.
- Generally continuation structure in the direction of the second distribution; a sustained rotation back through the neck negates it.

### Thin elongated trend profile

Long, skinny, minimal horizontal development; POC near one extreme. **One-timeframe control**: the OTF participant is in charge from open to close; the auction never finds an opposing responsive party. Participation *grows* with distance from value (initiative conviction). **Do not fade this shape.**

## 3.2 Day Types (Steidlmayer/Dalton taxonomy)

| Day type | Signature | Auction logic |
|---|---|---|
| **Normal day** | Wide initial balance (IB); range ≈ IB; symmetric D | Early OTF sets the extremes, then two-sided rotation. Rarer than the name implies |
| **Normal variation** | Moderate IB; one-sided range extension; total range < ~2× IB | OTF enters mid-morning, extends one way, value rebuilds. The most common type in many markets |
| **Trend day** | Narrow IB; elongated profile; one-timeframing; close near the extreme; volume expands with the move | Sustained initiative OTF conviction all session |
| **Double-distribution trend day** | Narrow IB → first bulge → LVN/single-print neck → second bulge | Two acceptance zones bridged by rapid repricing |
| **Nontrend day** | Tiny IB and range, thin volume, squat profile | Nobody leads (pre-news/holiday); stores energy |
| **Neutral day** | Range extension **both** sides of IB; close mid-range | OTF buyers *and* sellers both active — tug of war |
| **Neutral-extreme** | Both-side extension, close **on** an extreme | The side that won late; directional information for tomorrow |

Frequency lore varies by source and era (trend days ~5–10%; true normal days rare; normal-variation and neutral days most common) — the taxonomy's base rates were never rigorously established, so treat published percentages as indicative.

**The initial balance (first hour) is the early tell**: narrow IB → vulnerable to range extension → trend/double-distribution candidate; wide IB → harder to break → rotational-day candidate.

## 3.3 Open Types — Dalton's Conviction Ladder

The first 30 minutes, classified, plus *where* the open occurs relative to prior value, gives your earliest read on OTF conviction:

1. **Open-Drive** (highest conviction): opens and drives hard one way; price **never re-trades the open**. OTF decided before the bell. Never fade it; buy/sell pullbacks; the open price is the line in the sand; strongly raises trend-day odds.
2. **Open-Test-Drive**: opens, *tests* beyond a known reference (prior high/low, VA edge), finds no business, reverses hard through the open. Second-highest conviction; the failed-test point usually becomes the day's extreme and is premium trade location.
3. **Open-Rejection-Reverse**: drives one way, met by responsive opposition, returns through the open. Moderate conviction; early extremes hold only ~half the time; expect two-sided trade.
4. **Open-Auction**: quiet rotation around the open. Split by location: **in prior range/value** → genuine apathy, rotational day likely, early extremes unreliable; **out of prior range** → deceptive listlessness — opening out of balance carries unresolved inventory and elevated odds of a large directional resolution.

**Open location rules of thumb**: open inside prior value = balance odds, lowest directional confidence, fade edges; open outside value but inside range = mild imbalance; open outside range (true gap) = clear imbalance — highest opportunity *and* risk; monitor acceptance (value building at gap prices) vs rejection (gap fill, 80%-rule behavior).

## 3.4 Value Migration — the Day-Over-Day Frame

Compare today's developing value area with yesterday's:

- **Clearly higher / lower value** (little overlap): orderly directional migration — trend behavior; a multi-day staircase of value areas *is* the auction-theory definition of trend.
- **Completely non-overlapping value**: strong imbalance; continuation favored.
- **Overlapping (≥80%)**: balance; rotation expected.
- **Inside day (value within yesterday's)**: coiling; breakout watch.
- **Outside day (value engulfing yesterday's)**: expanding volatility, both sides probing.

Powerful divergence tell: **dPOC ratcheting one way while price flatlines = absorption/accumulation** in that direction.

## 3.5 Dalton's Balance Rules (trading a multi-day balance)

When the market has been balanced for ≥2 days, prepare all five scenarios in advance:

1. **Look above and GO** — breaks the balance high with acceptance → destination trade (target ≈ the balance range projected beyond the break).
2. **Look above and FAIL** — pokes above, gets rejected back inside → strong odds of rotation to the **opposite (lower) extreme** of the balance.
3. **Look below and GO** — mirror of 1.
4. **Look below and FAIL** — mirror of 2 → rotation to the balance high.
5. **Stay inside** — keep trading the rotation.

The FAIL variants are prized because the failed breakout **traps the initiative breakout traders**, whose forced exit fuels the traverse. This "look-above-and-fail" template scales down to prior-day highs/lows and the initial balance, and scales up to weekly composites.

---

# Part IV — The Trading Playbook

## 4.0 The Master Principle: Trade Location and Asymmetry

Dalton's framing in *Markets in Profile*: structural edges give **asymmetric opportunity** — you risk a small, structurally-defined amount (beyond the excess/LVN that invalidates the idea) against a much larger structural target (the next HVN, the opposite value edge, a naked POC). The profile's genuine, defensible edge is not any magic probability — it is that it tells you **exactly where your idea is wrong** and **where the market is likely to travel if you're right**, letting you size and select only trades where those two distances are favorably skewed.

Before any setup, answer the regime question: **balanced or imbalanced?** (Overlapping value + D-shape + open in value = balance → responsive trades. Value migrating + elongation + one-timeframing = imbalance → initiative trades only.)

## 4.1 Responsive Fade at the Value-Area Edge

*Regime: balance only.*

- **Context filter (mandatory)**: overlapping value, D-shaped developing profile, open inside prior value/range, no one-timeframing.
- **Setup**: price probes above VAH (below VAL) and **fails to be accepted** — wick-and-reject on your execution timeframe, excess forming outside the edge, no value building out there.
- **Entry**: on the reclaim back inside value after rejection — not blindly at the line.
- **Stop**: beyond the excess/swing that formed outside the edge (structural).
- **Targets**: POC first (scale), opposite value edge for the runner.
- **Kill-switch**: if the market **accepts** price outside value (≈ two consecutive 30-min periods out there, or the dPOC starts migrating out), the mean-reversion idea is dead — flatten, or flip to the initiative side.

## 4.2 Initiative Breakout of Value / Balance

*Regime: transition from balance to imbalance.*

- **Setup**: price leaves the value area / balance range and **builds volume and time outside** (acceptance) instead of snapping back.
- **Confirmation**: Dalton's volume rule — breakouts on **increasing** volume have good odds; on decreasing volume they'll probably fail. Time confirmation: two 30-min periods holding outside.
- **Entry**: the break itself, or the first pullback to the broken edge (which should now behave as support/resistance).
- **Stop**: back inside the broken area by more than a scalp — re-acceptance inside the old value negates the breakout.
- **Target**: destination logic — the next composite HVN, naked POC, or the projected balance-range extension.

## 4.3 The 80% Rule (value-area rule)

*The most famous named setup — presented here with its real statistics.*

- **Rule**: IF the session opens **outside** the prior day's value area, AND price re-enters the VA and **holds inside for two consecutive 30-minute periods**, THEN the claimed probability is ~80% that price traverses the **entire value area** to the opposite side.
- **Entry**: on the second 30-min period holding inside (aggressive: first close back inside, for better location).
- **Stop**: back outside the VA edge where price entered (re-rejection out of value = invalidation).
- **Target**: the opposite VA boundary; POC is the natural partial.
- **Filters**: balanced conditions only; avoid on strong trend days and around major scheduled news.
- **Honest provenance and evidence**: the figure comes from **The Profile Reports (Dalton Capital Management, 1987–91)** — pit-era observation, never an audited study. Independent community testing on the e-mini S&P repeatedly lands **well below 80% — roughly 60–67%** — and results are highly sensitive to how entry/holding conditions are coded. A ~60–67% traverse rate with a structurally capped loss is still a perfectly tradeable proposition; just size for the real number, not the folklore one. Better yet, measure it on your instrument and session.

## 4.4 Open-Drive Continuation

- **Identify** in the first 30 minutes: aggressive one-way auction from the bell, price never re-trading through the open, typically opening outside prior value.
- **Rule**: do not fade. Buy/sell pullbacks in the drive direction. **The open price is the line in the sand** — an "open-drive" that trades back through its open is failing; that's the stop.
- Open-test-drive variant: enter on the drive after the failed test of a known reference; stop beyond the test extreme (that extreme has high odds of holding as the day's high/low).
- Preparation: study the overnight session — open-drives are usually visible in overnight inventory and pre-open imbalance before the bell.

## 4.5 Failed Auction / Look-Above-and-Fail

*The trap-fade. Works at balance-area extremes, prior-day highs/lows, and the initial balance.*

- **Setup**: price pokes beyond a well-defined reference (balance high, prior-day high, IB high), **finds no new business** (no volume expansion, no acceptance, often a delta/absorption tell — see Part V), and returns back inside.
- **Entry**: on re-entry back inside the range.
- **Stop**: beyond the failure point (the new excess).
- **Target**: the **opposite** end of the balance/IB — a destination trade; the trapped breakout traders' exits fuel the traverse.
- IB-scale variant with practitioner statistics: a 30-min period closing back inside after poking out of the IB sets up the rotation to the IB's other side with a claimed ~70–75% hit rate *(vendor statistic — verify per instrument)*.

## 4.6 Naked POC (nPOC) Trades

- **Thesis**: a never-revisited prior POC is maximum past consensus with open inventory — a magnet ("unfinished business").
- **As target**: when price breaks toward an nPOC, it's the natural objective for runners.
- **As reaction level**: the *first touch* of an nPOC frequently produces a reaction — fade the touch **only with order-flow confirmation** (absorption/delta divergence), stop beyond the level.
- **Statistics**: the circulating "~80% of naked POCs are revisited within 10 sessions" is a **vendor claim**; the more careful services model nPOC fill probability per instrument with survival analysis. Track your market's own fill behavior.

## 4.7 LVN Rejection and LVN Break

*The binary level. Decide both branches before price arrives.*

- **Rejection variant**: price pulls back into an LVN left by an impulsive move → **confirm defense with order flow** (passive absorption, aggressors failing) → enter the rejection; stop just beyond the node (+1–2 ticks buffer); target the structural level beyond (day high/low, next HVN). The thin node = tight stop = high R:R.
- **Break variant**: price pushes *through* the LVN on strong volume (rule of thumb: ≥1.5× session average) and conviction delta → enter with the break, **target the next HVN** — LVNs are corridors; HVNs are destinations.
- The B-profile neck (Section 3.1) is the canonical LVN pivot: longs above it target the upper POC; shorts below it target the lower POC.

## 4.8 Anchored-Profile Pullback in Trends (the "accumulation defense" family)

*Trader Dale's three setups, generalized:*

- **Accumulation setup**: rotation → breakout into trend. Anchor a profile over the rotation; the heavy-volume zone is where institutions built inventory. On pullback to that zone, enter **with** the original trend — the thesis is that the players who built the inventory defend it.
- **Trend setup**: anchor over the trend leg itself; find the heaviest cluster inside it; buy/sell the pullback to that cluster, with-trend.
- **Rejection setup**: after an aggressive V-reversal, find the significant volume cluster near the turn; enter on retest of the zone's near edge.
- All three: stop beyond the far side of the volume zone; targets at prior structural extremes.

## 4.9 POC Reversion Scalps and IB Extensions

- **POC reversion**: in confirmed balance, fade extensions away from a *stable* dPOC back toward it. Dalton's caveat: while the market is one-timeframing, the POC "isn't doing meaningful work" — the market is searching, not settling. No POC-magnet trades on trend days.
- **IB extension**: enter on a decisive close outside the initial balance (visibly larger body/volume than the inside bars); stop back inside the IB (at least half its width, or under the breakout structure); targets at 1×/2×/3× IB-range measured moves. Wait for a closed bar outside to filter false breaks.

## 4.10 Multi-Timeframe Playbook

**Composite for context, session profile for execution.** Build a composite over the active bracket (e.g., a 20-day range): its HVNs are durable shelves, its LVNs the fast lanes. Then execute intraday against session levels *interpreted inside* the composite: a session-level fade at a **composite LVN edge** is a completely different (better) trade than the same pattern in the middle of a composite HVN.

- **Scalpers**: session profiles, 1–5 min execution, LVN rejections/breaks with order-flow confirmation, tightest structural stops.
- **Day traders**: prior-day VA + developing profile + IB; 80% rule, VA-edge fades, IB extensions; RTH profiles for level quality.
- **Swing traders**: weekly/monthly composites and anchored profiles from major turns; enter at composite HVN/LVN edges; hold across sessions; combine RTH + overnight data for the full picture.
- **Confluence rule**: independent levels aligning at one price (weekly POC = daily VAL; anchored VWAP = nPOC; composite LVN = prior-day low) = high-confidence zone. Stack independent reasons, not correlated ones.

## 4.11 Stops and Targets Are Structural — Always

- **Why not fixed ticks**: a 10-tick stop on an 80-tick-range day is noise; fixed distances ignore the regime. The profile hands you the exact price at which your thesis is objectively wrong: **beyond the excess, through the LVN, past the POC, outside the balance extreme.** "Structure determines stops, not arbitrary distances."
- **Targets by structure**: next HVN (acceptance attracts), opposite VA edge, nPOC, measured balance projection. Scale at the POC; let runners travel to destinations.
- **When to abandon a mean-reversion idea** — the three acceptance heuristics:
  1. **Time**: ~two consecutive 30-min periods accepted outside value.
  2. **Value migration**: the developing VA/POC follows price (not just price extending).
  3. **One-timeframing**: consecutive brackets refusing to violate prior extremes.
  Any of these printing against your fade = stop fading, reassess for initiative entries.

---

# Part V — Combining Profiles with Order Flow and Delta

**The doctrine: volume profile tells you WHERE, order flow tells you WHEN — and whether.** Order flow is a confirmation layer at pre-chosen profile levels, not a standalone signal generator (a point made explicitly even by the educators who sell it).

## 5.1 The Toolkit, Precisely Defined

- **Footprint (bid/ask) chart**: each bar broken down by price level showing volume at the bid (aggressive selling) vs at the ask (aggressive buying). Read **diagonally** — ask volume at one price vs bid volume one tick lower — because that's how aggressors compete.
- **Delta**: aggressive buys − aggressive sells per bar. **CVD (cumulative volume delta)**: the running sum — is aggression accumulating on one side?
- **Delta/CVD divergence**: new price extreme without net aggressive participation confirming it — a leading reversal warning.
- **Absorption**: heavy aggressive flow into large *passive* limit orders — high volume, **no price progress**. A bigger player is blocking the move. Often implemented via **iceberg orders** (auto-replenishing hidden size; detectable when executed volume keeps exceeding displayed size at one price).
- **Exhaustion**: aggressive flow **dries up** — shrinking prints into the extreme; the aggressors ran out. Both absorption and exhaustion cap moves; the mechanism (and who's in control) differs: absorption = fade *with* the blocker; exhaustion = fade the vacuum.
- **Imbalance / stacked imbalances**: one side exceeding the other diagonally by ≥3:1–4:1 at a price; three-plus consecutive levels = a stacked imbalance, marking a zone of one-sided dominance that often acts as future support/resistance.
- **Finished vs unfinished auction**: a **finished** auction prints volume on only one side at the bar's exact extreme (a zero in one cell at the very high/low) — one side was completely exhausted and the level cleanly rejected. An **unfinished** auction is the opposite: the extreme still prints volume on **both** sides — two-sided business was still being done when price turned, so the auction never completed there; such extremes tend to get revisited ("the market goes back to finish its business"). The unfinished auction is the footprint-scale version of the **poor high/low**.
- **DOM/liquidity caveat**: resting orders can be **spoofed** (placed to be cancelled); executed volume cannot. That's why footprint/tape generally outranks raw DOM reading.

## 5.2 Confluence Recipes (level + trigger)

1. **VA-edge fade, confirmed**: at VAH in balance, require **delta divergence** (new high, CVD not confirming) or **visible absorption** (heavy ask-side prints, price stalled) before shorting toward POC. No confirmation = no fade.
2. **LVN break, confirmed**: breakout through an LVN with **strong aligned delta and stacked imbalances** = continuation to the next HVN. Tapering volume through the level = likely stop-run — stand aside or fade.
3. **POC / nPOC defense**: a tap of the level with a **delta flip plus large passive absorption** (iceberg refills) = trap-and-reversal signature; enter the rejection with the defenders.
4. **Acceptance outside value, confirmed**: for a real value transition, volume should *build* outside, CVD should make new extremes **with** price, imbalances printing in the break direction. Initiative buying above VAH literally *is* market orders lifting offers up there — that's what the delta shows you.
5. **The four-quadrant trend read**: price ↑ + CVD ↑ + dPOC migrating ↑ = healthy initiative trend (buy pullbacks). Price ↑ + CVD flat/↓ = passive squeeze/short-covering — fragile (the P-shape's microstructure). Price flat + CVD ↑ = aggressive buyers being absorbed — bearish warning. Price ↑ + dPOC anchored = extension without acceptance — rotation risk back to the dPOC.
6. **How excess forms, microstructurally**: shrinking aggressive prints into the extreme (exhaustion) → one-sided print at the very high/low (finished auction) → delta flip → responsive aggression the other way → the profile is left with its low-volume tail. Watching this sequence live at a composite edge is the highest-conviction reversal read this toolkit offers.

*(Attribution note: these recipes are taught, in close variants, by Jigsaw/Peter Davies — "a change in order flow comes before a change in price"; confirm pre-chosen levels, don't generate them from the tape — Axia Futures' Footprint Edge curriculum (absorption, initiative drive, exhaustion, failed breaks), ATAS, FuturesTrader71/Convergent (profile statistics + halfback), and Trader Dale (delta + profile at pre-marked zones).)*

## 5.3 What Order Flow Requires

A complete **tick-by-tick feed with per-trade aggressor attribution**: futures via Sierra Chart (Denali/CQG/Rithmic), NinjaTrader Order Flow+, ATAS, Jigsaw, Bookmap; crypto via Exocharts/ATAS per-venue. TradingView-class tools **approximate** delta from 1-minute bars (~75–80% trade-sign accuracy) — fine for context, not for footprint-level decisions. US equities' delta is structurally incomplete (dark pool prints). Spot FX has no tape at all.

---

# Part VI — Instruments, Data, and Platforms

**Where profiles work best, in order:**

1. **Liquid, centralized futures** — ES, NQ, YM, RTY, CL, GC, ZN. One book, complete volume, reliable aggressor data, deep liquidity. ES is the canonical learning instrument.
2. **Large-cap stocks** — real consolidated volume; profiles reliable; order-flow layer weaker (fragmentation + dark pools).
3. **Liquid crypto perps** (BTC/ETH majors on dominant venues, or aggregated) — real per-venue aggressor data; mind fragmentation; per-venue CVD is the honest unit of analysis.
4. **Weak/avoid**: spot forex (tick-volume proxy only), illiquid futures, small-caps, thin alts — sparse profiles produce meaningless levels.

**Timeframes**: the methodology is native to the **30-minute auction bracket** (TPO periods, IB, acceptance heuristics, 80%-rule confirmation). Execution can compress to 1–5 minutes with order flow, but acceptance/rejection judgments made below the 30-minute bracket lose their anchor.

**Platform notes**: Sierra Chart (tick-exact profiles, the futures professional standard), NinjaTrader Order Flow+, ATAS (footprint-rich), Exocharts (crypto-first, aggregated markets), Bookmap (liquidity heatmap), TradingView (approximated profiles — good enough for level context on liquid markets; know its limits). Whatever you use: **fix your row size, value-area %, and session template (RTH vs ETH) and never change them casually** — your levels' consistency is worth more than theoretical precision.

---

# Part VII — Misconceptions, Failure Modes, and the Honest Evidence

## 7.1 The Classic Mistakes

1. **Fading trend days.** The most serious recurring mistake per Dalton himself. On a genuine trend day every fade from the VAH "looks perfect — and every one loses." Antidotes: the one-timeframing check, the value-migration check, the elongating shape, expanding volume with the move. When in doubt: a market accepting price outside value is not a fade.
2. **Treating POC/VA lines as magic support/resistance without context.** Levels are references inside an auction narrative. Ask *who is in control and what regime is this* before asking *what's the level*.
3. **Misreading HVNs as bounce zones.** HVNs are *acceptance* — price arriving there tends to **stay and chop**. They are targets and friction, not rejection levels. LVNs are the rejection candidates.
4. **Wrong profile scope/anchoring.** Profiles over incomplete legs, hand-tuned composite lookbacks that "explain" the past (curve-fitting with extra steps), early-session profiles read as if developed. Anchor to structure (swings, breakouts, events), not to whatever makes the level fit.
5. **Wrong day-type template.** D-day tactics (edge-to-edge) on a trend profile, or trend tactics (pullback-to-cluster) inside a nontrend chop, is a structural error regardless of execution quality.
6. **Data errors.** Tick-volume FX profiles, single-venue crypto profiles, ETH volume polluting RTH value areas, minute-bar approximations treated as tick-exact, row sizes changed mid-week. The map must be drawn from real territory.
7. **Trading profile levels through scheduled news.** The 80% rule and VA fades are explicitly contraindicated around major releases — the auction is being re-priced by information, not by inventory.

## 7.2 What the Evidence Actually Says

Be an adult about this — here is the full epistemic picture:

**Well-supported (academic microstructure literature):**
- **Order flow imbalance moves price.** Cont, Kukanov & Stoikov (2014): short-horizon price changes are driven by order-flow imbalance in a near-linear relation, R² ≈ 70%, with impact inversely proportional to depth. The physics behind "delta matters" is real (though it's contemporaneous short-horizon impact, not a multi-hour edge).
- **Support/resistance coincides with liquidity concentrations.** Kavajecz & Odders-White (2004): technical S/R levels align with peaks of resting depth on the limit order book. The best academic rationale for HVN-as-S/R.
- **Volume shocks carry information.** Gervais, Kaniel & Mingelgrin (2001): unusually high-volume stocks appreciate over following weeks (the high-volume return premium).
- **Hidden liquidity is real and partially detectable** (Hautsch & Huang; CME iceberg-detection literature).

**Unsupported / untested (no peer-reviewed record):**
- The value-area rules, the 80% rule, naked-POC magnetism, footprint patterns, and Market Profile day-type edges **as traded strategies**. Repeated searches find no peer-reviewed validation of any of them. The academic "volume profile" literature studies intraday volume seasonality and order-book shapes — not Steidlmayer profiles.

**Folklore statistics, with their real provenance:**

| Claim | Provenance | Independent testing |
|---|---|---|
| 80% rule (VA traverse) | The Profile Reports, Dalton Capital Mgmt, 1987–91 | ~60–67% on modern ES in community tests; parameter-sensitive |
| ~80% of nPOCs filled within 10 sessions | Vendor/retail education | Unaudited; serious services model it per-instrument instead |
| ~78%/75% poor high/low repair within 5 sessions | Practitioner publication | Unaudited, instrument/period-specific |
| IB failed-break rotation ~70–75% | Vendor education | Unaudited; per-ticker stats services exist precisely because it varies |

**The fair overall verdict**: the *raw materials* are real (flow imbalance moves price; liquidity concentrations act as S/R; volume carries information). What is unproven is that a discretionary human, reading profiles and footprints at retail latency and cost, converts them into positive expectancy. Volume profile is best defended as a **framework for locating asymmetric risk/reward and defining objective invalidation** — not as a signal generator with certified probabilities. The burden of measurement is on you: log 30–50+ occurrences per setup on *your* instrument and session before believing any number in this or any other guide.

## 7.3 Structural-Obsolescence Critique (and the Response)

Critics note the original construct encoded pit ecology: locals building the IB, visible commercials, one primary session. With 24-hour electronic markets and algorithmic flow, session-bounded profiles and IB logic are partly artifacts — Steidlmayer himself called the 1980s methods outdated and moved to variable-window distributions. The Dalton school's response: the *auction logic* — trade facilitation, price/time/volume, balance/imbalance, acceptance/rejection — is participant-agnostic and survives venue changes, even if session conventions must be adapted (hence RTH templates, overnight-inventory analysis, and anchored profiles). Both points are right: keep the logic, hold the conventions loosely.

---

# Part VIII — Risk Management and the Daily Process

## 8.1 Sizing Around Structure

Because stops are structural (and therefore variable), **size is the free variable**:

> Contracts = (Account × Risk%) ÷ (structural stop distance × point value + costs)

Example: $50k account, 1% risk ($500), 8-point ES stop × $50/pt = 1 contract. Wider structural stop ⇒ fewer contracts, same dollar risk. Risk 1–2% per trade; never widen a structural stop to "give it room" — the structure already defined the room.

## 8.2 Expectancy, Not Win Rate

Expectancy = Win% × AvgWin − Loss% × AvgLoss. An "80% setup" loses money if the 20% are full-VA-width losers against POC-scalp winners. Profile trading's asymmetric-location advantage only materializes if you *let* the asymmetric targets pay — scaling everything at the POC converts a positive-expectancy playbook into breakeven-minus-costs. Judge each playbook setup only after 30–50+ logged trades (200+ for real confidence), tagged by regime (balance/imbalance), day type, and session.

## 8.3 A Daily Routine (the Dalton-school preparation ritual)

**Pre-open:**
1. Mark yesterday's **POC, VAH, VAL**; the developing **weekly/composite VA**; all outstanding **naked POCs**; any **poor highs/lows** and **single-print/LVN** zones within reach; the current **balance bracket** extremes.
2. Read the **overnight session**: inventory (net long/short vs yesterday's close), whether overnight trade is inside or outside value/range, any overnight spike.
3. Classify the likely open: **where** will we open relative to prior value/range (in value / out of value / out of range)? Write the if-then scenarios for each (balance rules, gap rules, spike rules, 80%-rule watch).

**First hour:**
4. Classify the **open type** (drive / test-drive / rejection-reverse / auction) → conviction level.
5. Watch the **IB** form: width → day-type odds. Check **one-timeframing** bracket by bracket.

**All session:**
6. Track **value migration** (dPOC direction, developing VA vs yesterday's) and **CVD/delta at your pre-marked levels** — acceptance or rejection is the only question.
7. Execute only the playbook branch that matches the regime. Trend day signals = retire the fade playbook for the day.

**Post-close:**
8. Log the day type, your trades vs the playbook, and update the level map (new nPOC? poor extreme left behind? balance extended?).

## 8.4 The Ten Commandments (a summary you can tape to the monitor)

1. Value first: locate price relative to value before any trade.
2. Regime first: balance = fade edges; imbalance = follow price. Never mix.
3. Never fade a trend day; never fade an open-drive; never fade one-timeframing.
4. Acceptance kills fades: two 30-min periods outside value, or migrating value = stop fighting.
5. LVNs are decision prices; HVNs are destinations. Stops beyond LVN/excess; targets at HVNs/nPOCs.
6. Elongation = initiative conviction; stubby P/b bulges = inventory adjustment (old business).
7. Excess extremes are finished; poor extremes are magnets.
8. Confluence of independent references beats any single level.
9. Order flow confirms at levels; it doesn't generate trades from nothing.
10. Every probability you haven't measured yourself on your instrument is folklore. Size for the measured number.

---

# Glossary

| Term | Definition |
|---|---|
| **AMT** | Auction Market Theory — the framework: markets are continuous two-way auctions facilitating trade, alternating balance and imbalance |
| **POC** | Point of Control — the price with the most volume (or TPOs) in a profile; the "fairest price" |
| **dPOC** | Developing POC — the live, evolving POC during a session |
| **nPOC / naked (virgin) POC** | A prior session POC never revisited since; loses status on first touch |
| **VA / VAH / VAL** | Value Area (≈70% of volume around the POC) and its High/Low boundaries |
| **HVN / LVN** | High/Low Volume Node — acceptance bulge / rejection gap in the profile |
| **TPO** | Time Price Opportunity — one 30-min letter at a touched price; the unit of Market Profile |
| **IB** | Initial Balance — the range of the first hour (first two 30-min brackets) |
| **Range extension** | New high/low beyond the IB — the OTF entry tell |
| **OTF** | Other-timeframe participant — money operating on horizons beyond the day |
| **Initiative / responsive** | Activity driving away from prior value / activity fading back toward it |
| **Excess** | Decisive rejection at an extreme (tail, low-volume taper) — a finished auction |
| **Poor high/low** | Flat extreme without excess — unfinished auction, revisit magnet |
| **Single prints** | One-TPO-wide profile stretches from fast repricing; volume analog = LVN |
| **One-timeframing** | Consecutive brackets holding prior extremes — one side in control |
| **Spike** | Late-session directional push without time to build structure |
| **Balance / bracket** | Multi-day two-sided range; value overlapping |
| **Look above/below and fail** | Failed probe beyond a balance extreme → rotation to the opposite extreme |
| **80% rule** | Open outside prior VA + re-entry held two 30-min periods → traverse of the VA (measured ~60–67%, not 80) |
| **Delta / CVD** | Net aggressive buy−sell volume per bar / its running cumulative sum |
| **Footprint** | Per-bar bid/ask volume ladder — a profile inside every candle |
| **Absorption / exhaustion** | Passive size blocking aggressive flow / aggressive flow drying up |
| **Stacked imbalance** | ≥3 consecutive diagonal bid/ask imbalances — one-sided dominance zone |
| **Unfinished auction** | Bar extreme still printing volume on both sides — the auction didn't complete; footprint-scale poor high/low (a one-sided print at the extreme is a *finished* auction) |
| **Halfback** | 50% of the RTH range — an FT71-school pullback reference |
| **RTH / ETH** | Regular / Extended (overnight) Trading Hours — choose your profile session template deliberately |

---

# Sources & Further Reading

**Primary texts (read in this order):**
1. Steidlmayer & Koy — *Markets and Market Logic* (1986) — the theory
2. Dalton, Jones & Dalton — *Mind Over Markets* (1990; updated 2013) — the operational handbook
3. Dalton, Dalton & Jones — *Markets in Profile* (2007) — context and timeframes
4. CBOT — *A Six-Part Study Guide to Market Profile* (1996) — the exchange's own pedagogy
5. Steidlmayer & Hawkins — *Steidlmayer on Markets* (2nd ed., 2003) — the founder's later evolution

**Key reference documentation:** TradingView volume-profile docs (construction & VA algorithm); Sierra Chart Volume-by-Price docs; CQG/mypivots value-area calculation notes; NinjaTrader Order Flow docs; Exocharts help (cross-platform POC/VA discrepancies).

**Practitioner schools referenced:** Jim Dalton (jimdaltontrading.com); FuturesTrader71 / Convergent Trading; Axia Futures (Footprint Edge, Volume Profiling Edge); Jigsaw Trading / Peter Davies (free order-flow lessons); Trader Dale; ShadowTrader glossary (balance/gap/spike/80% rules); Marketcalls Market Profile tutorial series; mypivots dictionary; Vtrender; TradeZella playbooks; Edgeful (per-instrument setup statistics).

**Academic anchors:** Cont, Kukanov & Stoikov, "The Price Impact of Order Book Events" (J. Fin. Econometrics 2014); Kavajecz & Odders-White, "Technical Analysis and Liquidity Provision" (RFS 2004); Gervais, Kaniel & Mingelgrin, "The High-Volume Return Premium" (J. Finance 2001); Hautsch & Huang on hidden liquidity; Mandelbrot on non-Gaussian returns (context for the bell-curve critique).

**Skeptical reading:** mypivots "The 80% Rule" (the ~60% finding); Trader Dale, "The Dark Side of Order Flow"; MarketTrace on CVD signal-quality problems; the survivorship-bias literature on trading education.

*Research method note: this guide was synthesized from five parallel research agents totaling 80+ web searches across primary texts, platform documentation, practitioner education and academic literature (compiled July 2026). Contested claims were cross-corroborated across independent agents; where sources conflict (P-shape readings, day-type frequencies, all probability lore), the disagreement is presented rather than resolved by fiat. All folklore statistics are labelled as such.*
