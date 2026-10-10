<!--
AGENT GUIDELINE

Copy to r<N>-assessment.md when a round closes. Then STOP and wait for the human.

This is the only document that crosses a round boundary, so it is the only place the
next round's plan can come from. Two failure modes it exists to prevent:

  - A win that was really noise gets carried into the next round and everything built
    on top of it inherits the error. Section 2 forces each keep to show its sigma.
  - A direction that merely crashed gets recorded as "refuted" and never retried.
    Section 3 forces the distinction between a real null and an inconclusive run.

Fill every section. "Nothing to report" is a legitimate entry; a missing section is not.
-->

# Round &lt;N&gt; — assessment

**Scale** &lt;1/4 | 1/2 | full&gt; · **Budget** &lt;N&gt;s · **Experiments run** &lt;N&gt; · **Branch** `autoresearch/r<N>`
**σ&lt;N&gt;** &lt;value&gt; · **Round baseline** &lt;val_bpb&gt; → **round best** &lt;val_bpb&gt; (**Δ &lt;value&gt;**, &lt;N&gt;σ)

## 1 · Ledger

| Exp | Direction | val_bpb | Δ | Δ/σ | Status | One-line finding |
|---|---|---|---|---|---|---|
| | | | | | | |

&lt;Every experiment in the round, including crashes. This table is generated from
results.tsv, not retyped — `grep "	<N>	" results.tsv`.&gt;

## 2 · What held

| Change | Owns Δbpb | Seeds | Δ/σ | Confirmed at |
|---|---|---|---|---|
| | | | | |

**Cumulative:** baseline &lt;value&gt; → best &lt;value&gt;, Δ &lt;value&gt;.

&lt;The sum of the individual Δs rarely equals the cumulative Δ. Say by how much it misses
and which interaction most likely explains the gap — that gap is the single most useful
number in this document for planning the next round.&gt;

## 3 · What died

| Direction | Outcome | Real null or inconclusive? | What it rules out |
|---|---|---|---|
| | | | |

A **real null** means the effect was measured and sits inside the noise band — that
direction is closed. **Inconclusive** means confounded, crashed, or under-tested: it is
not refuted and may be re-opened at a larger scale, where the effect may clear a
different σ.

## 4 · Interactions not yet tested

&lt;Pairs that were only tested separately. Each one is a candidate experiment for the next
round. Name the pair and the reason to expect an interaction.&gt;

## 5 · Scale risk

| Kept change | Expected to survive the next round? | Why |
|---|---|---|
| | | |

&lt;LR and schedule wins in particular rarely transfer across a 2.5× budget change: both
were tuned against a step count that the next round changes. Flag those explicitly as
needing re-measurement rather than assuming they carry.&gt;

## 6 · Next round plan

**New baseline:** &lt;the config carried forward, as an ExpConfig literal&gt;

**σ plan:** 3 seed repeats of the above at the next round's scale and budget, before any
experiment runs.

| # | Direction | Why now | Expected Δ | Runs |
|---|---|---|---|---|
| | | | | |

&lt;Ranked by expected Δbpb per run. Carry forward the re-measurements §5 demands, then the
interactions from §4, then new ground.&gt;

## 7 · Open questions for the human

&lt;Anything needing a decision. If the round produced a result that changes what the
project should be optimising, say so here plainly — that is more valuable than the
round's best val_bpb.&gt;
