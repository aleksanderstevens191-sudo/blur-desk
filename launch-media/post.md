# Post text for X

Every number here is a Mac figure from `LAUNCH_FACTS.md` (row in brackets). Under 280
characters each.

## Single post (attach the ?record=1 clip)

> An option's price is its payoff, blurred. Blur Desk does the blur on your GPU, in the
> browser: within 1.4e-5 of Black-Scholes [A2], then fitted to real SPY options it reads
> off the market's odds: a 7.6% chance SPY falls more than 10% by Dec 18 [O1].
>
> blur-desk.vercel.app

## Thread

1. (the ?record=1 clip) Same text as the single post.
2. (the ?record=2 clip)
   > Run it backwards and it's a diffusion model. Blur the market's odds into noise, then
   > run 65,536 points of noise back with the exact score, no network: 7.44% rebuilt vs
   > the market's 7.62% [D15].
3. > Python reference, WebGPU shaders checked against it on every push (all under 1e-4
   > [A5]), benchmarks, open source:
   > github.com/aleksanderstevens191-sudo/blur-desk

## Alt text

- ?record=1 clip: "A white wireframe chart on black. A zigzag payoff is drawn, then
  blurred into a smooth price curve as the expiry grows. The blur is swapped for one fitted
  to SPY options, then the market's implied distribution appears: a 7.6% chance SPY falls
  more than 10% by December 18."
- ?record=2 clip: "The market's implied distribution for SPY is blurred into noise with a
  slider. Pressing Play runs 65,536 points of noise back into a histogram that matches the
  original curve: 7.44% rebuilt against the market's 7.62%."

## Don't say

- "~34x faster than QuantLib". The same-Mac barrier figure is 12x on 1 thread [Q1];
  all-cores was never measured [Q3].
- That the odds are a forecast. They are risk-neutral, as the clip says.
