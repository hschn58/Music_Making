# Relative period — the math

The second axis of consonance (the first is *roughness* — beating between partials
inside a critical band). Relative period measures **how soon the combined pressure
wave $p(t)$ repeats itself**, in units of the lowest tone's period. Short = deep,
frequent reinforcement ("locks in", consonant). Long = the pattern only comes back
around after many cycles (the tritone's "spooky, sort of works"). Effectively
infinite = never repeats (inharmonic / dead / hollow).

Code: `scripts/periodicity_study.py` (`relative_period`). Companion roughness axis:
`scripts/consonance_study.py`.

---

## 1. Definition

Take tones with frequencies $f_1, \dots, f_n$. The combined wave

$$p(t) = \sum_i a_i \sin(2\pi f_i t + \varphi_i)$$

repeats with period $T$ exactly when **every component completes a whole number of
cycles in $T$**:

$$f_i\,T \in \mathbb{Z} \qquad \text{for all } i.$$

The smallest such $T$ is the true period $T_{*}$. It exists **only if every ratio
$f_i/f_j$ is rational**. When it does, the combined wave has fundamental frequency

$$F = \gcd(f_1, \dots, f_n), \qquad T_{*} = \frac{1}{F},$$

where $\gcd$ is the largest $F$ that divides every $f_i$ a whole number of times.

**Relative period** = $T_{*}$ measured in units of the lowest tone's period
$T_0 = 1/f_0$, with $f_0 = \min_i f_i$:

$$\boxed{\ \ \text{relperiod} \;=\; \frac{T_{*}}{T_0} \;=\; \frac{f_0}{\gcd(f_1,\dots,f_n)}\ \ }$$

> **In words:** how many cycles of the lowest tone go by before the whole waveform
> returns to where it started.

### Fraction form (what the code computes)

Write each ratio in lowest terms, $f_i/f_0 = p_i/q_i$. Then

$$\frac{\gcd(f_i)}{f_0} = \frac{\gcd(p_i)}{\operatorname{lcm}(q_i)}
\qquad\Longrightarrow\qquad
\text{relperiod} = \frac{\operatorname{lcm}(q_1,\dots,q_n)}{\gcd(p_1,\dots,p_n)}.$$

Because the lowest tone is always in the set, its ratio is $1/1$, so $1$ is always
among the numerators and $\gcd(p_i) = 1$ automatically. That collapses to

$$\text{relperiod} = \operatorname{lcm}(q_1, \dots, q_n).$$

### Worked consonant cases

| interval       | ratios to f0 | denominators q_i | relperiod = lcm |
|----------------|--------------|------------------|-----------------|
| unison         | 1            | 1                | **1**           |
| octave         | 1, 2         | 1, 1             | **1**           |
| perfect fifth  | 1, 3/2       | 1, 2             | **2**           |
| perfect fourth | 1, 4/3       | 1, 3             | **3**           |
| major triad    | 1, 5/4, 3/2  | 1, 4, 2          | **4**           |

Small integers → tiny relperiod → the waveform repeats every few cycles.

---

## 2. Worked example: the tritone = 29

Tones: $f_0 = 440$, $f_1 = 622$. One ratio matters:

$$\frac{622}{440} = 1.4136\ldots$$

**Catch:** $622/440$ is *already exactly rational* — it reduces to $311/220$. So the
**literal** relative period is $\operatorname{lcm}(1, 220) = \mathbf{220}$ cycles. The
wave does repeat, but only after 440 begins its 221st cycle — far too long for the
ear to lock onto. That is *why* it sounds like it "doesn't quite work."

The code does not report 220. It reports **29**, via
`Fraction(r).limit_denominator(64)` — "the closest fraction whose denominator is
$\le 64$." That step is a deliberate **perceptual horizon**: the ear won't track a
repeat hundreds of cycles long, so we ask what simple ratio this is *near*.

Find it with the continued-fraction expansion of $311/220$:

$$\frac{311}{220} = [\,1;\ 2, 2, 2, 1, 1, 7\,]$$

with successive best approximations (convergents)

$$\tfrac{1}{1},\quad \tfrac{3}{2},\quad \tfrac{7}{5},\quad \tfrac{17}{12},\quad
\tfrac{24}{17},\quad \mathbf{\tfrac{41}{29}},\quad \tfrac{311}{220}.$$

The last convergent with denominator $\le 64$ is $\mathbf{41/29}$ (the next jumps to
220). Its value $1.41379$ is off from the true $1.41364$ by **0.011%** (the `err%`
column). Therefore:

$$\operatorname{lcm}(1, 29) = \mathbf{29}, \qquad
\text{period} = \frac{29}{440}\times 1000 = 65.9\ \text{ms}.$$

> "29" means: *to within a 0.01% mistuning, the tritone behaves as if its waveform
> repeats every 29 cycles of the low tone* — long, but graspably long. That is the
> "spooky but sort of works" zone, between the fifth's 2 and the inharmonic clang's
> effectively-infinite.

---

## 3. The tolerance dial (`qmax`)

The key takeaway: **relative period is a property of the frequencies *plus* the
tolerance `qmax`**, not the frequencies alone.

| qmax   | tritone reads as      | interpretation                             |
|--------|-----------------------|--------------------------------------------|
| 5      | 5 (snaps to 7/5)      | "close enough to nice — call it consonant" |
| 64     | 29 (via 41/29)        | graspably long — "spooky but sort of works"|
| ≥ 220  | 220 (literal 311/220) | the true, ungraspable period               |

`qmax` (equivalently, an allowed mistuning in **cents**) models a real perceptual
fact: the ear **rounds near-misses to simple ratios** up to some limit — which is why
a slightly out-of-tune piano still sounds like consonant chords.

For this project it is an **aesthetic parameter** with a clear meaning:

- **loose tolerance** → forgiving, lush, "everything fuses"
- **tight tolerance** → austere, only truly simple ratios survive

When the color→spectrum model hands us messy real-world frequencies, this dial
decides how generously the ear is allowed to hear structure in them.

> **cents:** $\text{cents} = 1200\,\log_2(f_2/f_1)$; 100 cents = one semitone. A
> mistuning tolerance in cents is the perceptually-uniform way to set this knob (it
> means the same thing across the whole frequency range, unlike a raw `qmax`).

---

## 4. The two axes together

|                | axis 1 — **roughness**          | axis 2 — **relative period**       |
|----------------|---------------------------------|------------------------------------|
| measures       | beating between close partials  | how soon p(t) repeats              |
| physical cause | partials within a critical band | (ir)rationality of the freq ratios |
| catches        | beat, semitone cluster, screech | tritone, inharmonic clang          |
| "good" =       | low                             | short                              |

Good-sounding = **lower-left**: smooth (no beating) **and** deeply periodic (short
relperiod). See `demos/consonance/two_axes.png`.
