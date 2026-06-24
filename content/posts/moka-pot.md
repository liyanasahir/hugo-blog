+++
title = "A Moka Pot, a Flat White Habit, and Some Arithmetic"
date = "2026-06-23T10:00:00Z"
draft = false
protected = true
tags = ["notes"]
description = "What a 240 ml aluminium pot can teach you about coffee, money, and friction."

[_build]
  list = "never"
+++

<div class="moka">

<style>
/* ── Moka pot post ── */
.moka {
  --coffee: #6F4E37;
  --coffee-light: #A0795C;
  --cream: #F5F0E8;
  --espresso: #3C2415;
  --steam: #D4C5B2;
}
.dark .moka {
  --cream: #2A2520;
  --steam: #4A4035;
  --espresso: #E8DDD0;
}

/* Steam animation for hero */
.moka-hero {
  text-align: center;
  padding: 1.5rem 0 0.5rem;
}
.moka-cups {
  display: inline-flex;
  gap: 8px;
  align-items: flex-end;
}
.moka-cup {
  position: relative;
}
.moka-cup svg {
  width: 44px;
  height: 36px;
  fill: var(--coffee);
}
.moka-cup:nth-child(2) svg {
  opacity: 0.65;
}
.steam-wisps {
  position: absolute;
  top: -18px;
  left: 40%;
  transform: translateX(-50%);
  display: flex;
  gap: 5px;
}
.steam {
  width: 2px;
  background: var(--steam);
  border-radius: 2px;
  animation: steam-rise 2.5s ease-in-out infinite;
}
.steam:nth-child(1) { height: 14px; animation-delay: 0s; }
.steam:nth-child(2) { height: 18px; animation-delay: 0.4s; }
.steam:nth-child(3) { height: 12px; animation-delay: 0.8s; }
@keyframes steam-rise {
  0%, 100% { opacity: 0.3; transform: translateY(0) scaleY(1); }
  50% { opacity: 0.7; transform: translateY(-6px) scaleY(1.2); }
}

/* Intro note */
.moka-intro-note {
  font-size: 0.85em;
  font-style: italic;
  color: var(--secondary);
  border-left: 2.5px solid var(--coffee-light);
  padding: 0.6rem 1rem;
  margin: 0 0 1.5rem;
  background: var(--cream);
  border-radius: 0 6px 6px 0;
  line-height: 1.65;
}

/* Section divider: three coffee beans */
.moka .bean-divider {
  text-align: center;
  margin: 2rem 0;
  line-height: 1;
}
.moka .bean-divider svg {
  width: 16px;
  height: 16px;
  fill: var(--coffee-light);
  margin: 0 6px;
  opacity: 0.5;
}

/* Gear card */
.moka .gear-card {
  background: var(--cream);
  border: 1px solid var(--steam);
  border-radius: 8px;
  padding: 1.25rem 1.4rem;
  margin: 1rem 0 1.5rem;
}
.moka .gear-card-label {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--coffee-light);
  margin: 0 0 0.75rem;
}
.moka .gear-card ul {
  list-style: none;
  padding: 0;
  margin: 0;
}
.moka .gear-card li {
  padding: 0.45rem 0;
  border-bottom: 0.5px solid var(--steam);
  font-size: 0.9em;
  line-height: 1.6;
}
.moka .gear-card li:last-child {
  border-bottom: none;
  padding-bottom: 0;
}

/* Smaller headings */
.moka h2 {
  font-size: 1.25em;
}
.moka h3 {
  font-size: 1.05em;
}

/* Blockquotes */
.moka blockquote {
  border-left: 3px solid var(--coffee-light) !important;
  background: var(--cream);
  padding: 0.75rem 1rem;
  border-radius: 0 6px 6px 0;
  margin: 1.25rem 0;
  font-style: italic;
}
.moka blockquote p {
  margin: 0;
}

/* Tables */
.moka table {
  border-collapse: collapse;
  width: 100% !important;
  display: table !important;
  margin: 1rem 0;
  font-size: 0.88em;
  overflow: visible;
}
.moka thead th {
  background: var(--coffee);
  color: #fff;
  font-weight: 500;
  padding: 0.55rem 0.75rem;
  text-align: left;
  font-size: 0.92em;
}
.moka thead th:first-child {
  border-radius: 6px 0 0 0;
}
.moka thead th:last-child {
  border-radius: 0 6px 0 0;
}
.moka tbody td {
  padding: 0.5rem 0.75rem;
  border-bottom: 0.5px solid var(--steam);
}
.moka tbody tr:nth-child(even) {
  background: var(--cream);
}
.moka tbody tr:last-child td:first-child {
  border-radius: 0 0 0 6px;
}
.moka tbody tr:last-child td:last-child {
  border-radius: 0 0 6px 0;
}

/* Cup icon for list items */
.moka .uses-list {
  list-style: none;
  padding: 0;
}
.moka .uses-list li {
  padding: 0.35rem 0 0.35rem 1.6rem;
  position: relative;
  line-height: 1.6;
}
.moka .uses-list li::before {
  content: "☕";
  position: absolute;
  left: 0;
  font-size: 0.85em;
}

/* Timeline for friction section */
.moka .brew-timeline {
  display: flex;
  align-items: stretch;
  gap: 0;
  margin: 1rem 0 1.5rem;
  border-radius: 8px;
  overflow: hidden;
  font-size: 0.8em;
}
.moka .brew-step {
  flex: 1;
  padding: 0.6rem 0.5rem;
  text-align: center;
  color: #fff;
  line-height: 1.35;
  min-width: 0;
}
.moka .brew-step-label {
  font-weight: 500;
  font-size: 0.95em;
}
.moka .brew-step-time {
  font-size: 0.85em;
  opacity: 0.8;
  margin-top: 2px;
}
.moka .brew-step:nth-child(1) { background: #8B6F47; }
.moka .brew-step:nth-child(2) { background: #7A5C35; }
.moka .brew-step:nth-child(3) { background: #6F4E37; }
.moka .brew-step:nth-child(4) { background: #5C3D28; }
.moka .brew-step:nth-child(5) { background: #4A2E1C; }

/* Side annotation */
.moka .side-note {
  font-size: 0.8em;
  color: var(--secondary);
  font-style: italic;
  border-top: 1px solid var(--steam);
  padding-top: 0.5rem;
  margin-top: 1rem;
}

/* Horizontal rules */
.moka hr {
  border: none !important;
  background: none !important;
  background-color: transparent !important;
  margin: 2.25rem 0;
  text-align: center;
  height: auto !important;
  overflow: visible;
}
.moka hr::after {
  content: "· · ·";
  color: var(--coffee-light);
  font-size: 14px;
  letter-spacing: 4px;
  display: block;
}
</style>

<div class="moka-hero">
  <div class="moka-cups">
    <div class="moka-cup">
      <div class="steam-wisps">
        <div class="steam"></div>
        <div class="steam"></div>
        <div class="steam"></div>
      </div>
      <svg viewBox="0 0 40 32" xmlns="http://www.w3.org/2000/svg">
        <path d="M4 8 Q4 4 8 4 L24 4 Q28 4 28 8 L26 24 Q26 28 22 28 L10 28 Q6 28 6 24 Z"/>
        <path d="M28 10 Q34 10 34 16 Q34 22 28 22" fill="none" stroke="currentColor" stroke-width="2.5" style="color: var(--coffee)"/>
      </svg>
    </div>
    <div class="moka-cup">
      <div class="steam-wisps">
        <div class="steam"></div>
        <div class="steam"></div>
        <div class="steam"></div>
      </div>
      <svg viewBox="0 0 40 32" xmlns="http://www.w3.org/2000/svg">
        <path d="M4 8 Q4 4 8 4 L24 4 Q28 4 28 8 L26 24 Q26 28 22 28 L10 28 Q6 28 6 24 Z"/>
        <path d="M28 10 Q34 10 34 16 Q34 22 28 22" fill="none" stroke="currentColor" stroke-width="2.5" style="color: var(--coffee)"/>
      </svg>
    </div>
  </div>
</div>

<div class="moka-intro-note">
This is an abstraction of a long conversation with Claude. She brought the questions and the numbers; Claude did the research, the math, and the writing. What follows is the condensed version.
</div>

She came in with a gift she was not sure what to do with: an **Agaro Classic Moka Pot, 240 ml**, and a long-standing habit of buying flat whites from Blue Tokai. **₹252 with regular milk, ₹272 with oat**. The question underneath was simple and good: *is making it myself actually worth it, or am I just trading rupees for chores?*

So we did the arithmetic. Here is what came out, and where the real answer turned out to have nothing to do with money.

---

<div class="gear-card">
<div class="gear-card-label">What she was working with</div>

<ul>
<li><strong>The pot:</strong> Agaro Classic, 240 ml. A gift, so its cost is <strong>₹0</strong>. (It retails around ₹799.)</li>
<li><strong>The bean:</strong> Blue Tokai <em>Dhak Blend</em>, pre-ground. <strong>₹650 / 250 g</strong>, which is <strong>₹2.60 a gram</strong>. This is the cheapest option on Blue Tokai for 250 g. The range goes up to about ₹800.</li>
<li><strong>The milk:</strong> Alt Co oat (<strong>₹265/L</strong>) or dairy (<strong>₹34 / 450 ml</strong>).</li>
<li><strong>The drinkers:</strong> two, her and her partner, which, as it turns out, quietly solves the pot's biggest flaw.</li>
</ul>
</div>

---

### Wait, how much coffee am I actually using?

She had noticed Blue Tokai recommends **16 g per 120 ml of water**, but also that a moka pot's basket has to be *filled*. So which wins?

The basket wins, every time. On a moka pot **you do not get to choose the dose**. You fill the basket level (no tamping, no heaping) and that fixed amount is what you use on every single brew. Blue Tokai's ratio can not even be honoured here: a full pot would call for ~38 g, and the basket simply can not hold it. For her 240 ml pot, a level basket is about **18 g** (somewhere in the 17-22 g range, Agaro does not list exact grams, so weigh a level basket to be sure).

> The one thing worth doing once: tip a level-full basket onto a kitchen scale and read the number. Then she *knows* her pot instead of trusting an estimate, mine included.

### Okay but can I just make one cup?

Not really, and this is the one piece of bad news about the gift. "240 ml" is the **most** coffee it makes: six little 40 ml espresso-style cups, a classic "6-cup" moka. Under-fill the basket to make a smaller batch and water just channels through the gap. Weak, bitter, uneven. The pot is built to run full.

For one person, that is an oversized-pot problem. For two people brewing together, it is exactly right. **One full pot makes two flat whites with a little left over.** Her habit fit the hardware better than she expected.

---

## The money

One full pot costs **₹47 in grounds**, whatever she does with it. Split across two drinks, that is **₹23.50 of coffee per cup.** Then it is just a question of milk.

| Per flat white | Oat (Alt Co) | Dairy |
|---|---:|---:|
| Coffee (half pot) | ₹23.50 | ₹23.50 |
| Milk (~120 ml) | ₹32.00 | ₹9.00 |
| **Home total** | **~₹56** | **~₹33** |
| Blue Tokai outlet | ₹272 | ₹252 |
| **Saved per cup** | **~₹216** | **~₹219** |
| **Cheaper by** | **~4.9x** | **~7.7x** |

For the two of them in one sitting: **~₹111 (oat) or ~₹65 (dairy)** at home, against **₹544 or ₹504** at the outlet.

### Two things she did not expect

1. **The oat milk costs more than the coffee.** At ₹32 a cup, the milk is the single biggest line item in her home flat white. Oat vs dairy swings the whole drink by ~40%. It is the only real lever she has.
2. **₹650 for 250 g sounds steep, until you do the per-cup math.** ₹23.50 vs ₹252. The money lives in the milk, not the bean.

---

## What to do with the extra coffee

She asked the practical question: what happens to the leftover ~80-120 ml after two cups? The nice answer: the grounds are paid for the moment she brews, so a third drink from the surplus costs only its milk, or nothing at all black. (A third iced cup drops her coffee-per-cup from ₹23.50 to about ₹15.70.)

<ul class="uses-list">
<li><strong>Iced in the afternoon.</strong> Decant into a jar straight away, chill, pour over ice. Moka coffee is lovely cold. The best move.</li>
<li><strong>Coffee ice cubes.</strong> So future iced coffees never get watered down.</li>
<li><strong>Affogato, overnight oats, baking, smoothies.</strong> Anywhere strong cold coffee is an ingredient.</li>
</ul>

Two firm rules: **decant immediately** (do not let it stew on the warm hob, it keeps cooking and turns bitter), and **never reheat it** (reheating flattens and embitters). Leftovers go cold or iced. Sealed in the fridge, good for about 24 hours.

---

## The part she flagged herself: the friction

She was honest that the **prep and cleanup cause friction at the start, and that the habit will be a slow one to build.** That is the right thing to worry about, more than the minutes.

<div class="brew-timeline">
  <div class="brew-step">
    <div class="brew-step-label">Assemble</div>
    <div class="brew-step-time">~2 min</div>
  </div>
  <div class="brew-step">
    <div class="brew-step-label">Brew</div>
    <div class="brew-step-time">~5 min</div>
  </div>
  <div class="brew-step">
    <div class="brew-step-label">Froth</div>
    <div class="brew-step-time">~2 min</div>
  </div>
  <div class="brew-step">
    <div class="brew-step-label">Pour</div>
    <div class="brew-step-time">~1 min</div>
  </div>
  <div class="brew-step">
    <div class="brew-step-label">Clean</div>
    <div class="brew-step-time">~2 min</div>
  </div>
</div>

About 12 minutes total. Roughly 8 minutes hands-on. For two drinks.

**The catches:**
- The moka will not let you walk away. Pull it the second it sputters or it over-extracts. No timer, no wandering off.
- Cleanup is the real tax. Not hard, but daily and recurring. Recurring friction is exactly what erodes a young habit. (Rinse the aluminium only. No soap, it leaves a taste.)
- You are the barista now, and a beginner. The cafe does 9-bar espresso with microfoam. The moka does ~1.5-2 bar, no real crema, and a handheld frother gives decent but coarser foam. The gap is real. It narrows with practice. It will not fully close on this gear. The timings will shift as she gets better, the quality gap will shrink, but the friction stays constant unless she designs it out.

---

## The trade she actually cares about

This is where it stops being about money. The cafe is close by, no travel to account for, and she and her partner genuinely love the **atmosphere, the focus it gives them, and the top-notch quality.** But she also never feels comfortable lingering **more than an hour per drink** there. At home, there is no such clock.

So the real contest is not cafe-coffee vs home-coffee. It is two different goods:

| The Cafe | The Pot |
|---|---|
| Top-tier quality, 9-bar, microfoam | Strong and characterful, ~1.5-2 bar, and *hers to improve* |
| Atmosphere, and the productivity it unlocks | A quiet, meditative ritual she gets better at by the day |
| Served, spotless, zero labour | Prep + cleanup friction, the habit's real hurdle |
| ~₹252-272 a cup | ~₹33-56 a cup |
| A soft **one-hour ceiling** per drink | **No time limit.** Linger, refill, stay all morning |
| She is renting a place to *be* | She is building a craft that compounds |

The cafe time is not overhead she pays to get coffee. It *is* the thing she is buying: the room, the service, the hour of focus. And the home ritual is not just cheaper coffee. It is the slow pleasure of making something well and watching yourself improve. Those are different things. Neither replaces the other.

---

## Where she landed

Home is not really competing with the cafe. It is competing with *not having good coffee at home*, and for the daily, ordinary, start-the-morning cup, it wins outright: six minutes, no leaving, ₹33-56, on demand, and no hourglass on the table.

The cafe keeps the days when **being there is the point**: the atmosphere, the shared focus, an hour together that is not at their own kitchen counter.

So the split writes itself: **brew at home for the everyday cups, where the 5-8x saving lives and compounds, and go to the cafe on purpose, for the room and not the coffee.**

> The single highest-leverage move is not a better bean or a fancier frother. It is lowering the cleanup friction. A drying rack by the hob, a fixed routine, washing up while the milk froths. Get that one barrier down and the home habit sticks. Leave it annoying, and the ₹272 cup quietly wins back.


</div>
