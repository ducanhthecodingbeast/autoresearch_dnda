<!--
AGENT GUIDELINE — read before filling this in.

This file is the deliverable. A run without it is not done.

Four rules, and they are the whole point of the format:

1. EVERY NUMBER IS COPIED FROM metrics.json. Never retyped from memory, never
   rounded by hand, never recalled from the terminal scrollback. If a number is not
   in metrics.json, it does not go in the table.

2. EVERY ASSESSMENT CLAIM NAMES ITS EVIDENCE. A sentence with no number, artifact or
   named mechanism behind it gets deleted. These three are NOT evidence:
     - a lower val_bpb on its own (that is the claim, not the support)
     - "as expected"
     - a mechanism story with no curve to back it
   Evidence looks like: "3 seeds spread 0.0004 against sigma 0.0055" or
   "loss.png: the gap opens at step ~90 and holds".

3. NULL AND NEGATIVE RESULTS GET THE SAME QUALITY AS WINS. The round assessment is
   built from them. "Discarded, no effect" with no analysis wastes the run — the
   useful content is WHY there was no effect and what that rules out.

4. DELETE THE ANGLE-BRACKET PROMPTS as you fill them. A report shipped with
   <...> placeholders still in it is an unfinished report.

The example numbers in research-program.md section 5 show a filled-in version.
-->

# exp&lt;NN&gt;r&lt;R&gt; — &lt;short title&gt;

**Round** &lt;R&gt; (&lt;scale&gt; data, &lt;N&gt; shards, one pass) · **Parent** &lt;exp id&gt;
**Hypothesis** &lt;one sentence, falsifiable: what should move, and in which direction&gt;
**Change** &lt;the delta against the parent. one axis. nothing else changed.&gt;

## Benchmarks

| Metric | Baseline | This run | Δ | Δ/σ |
|---|---|---|---|---|
| **val_bpb** (primary, ↓) | | | | |
| val_bpb seed spread (n=) | | | | |
| peak_vram_mb | | | | |
| mfu_percent | | | | |
| tok_per_s | | | | |
| training_seconds (outcome, not a budget) | | | | |
| total_tokens_M | | | | |
| num_steps | | | | |
| num_params_M | | | | |
| epoch (one pass by design, ~1.0) | | | | |

σ&lt;R&gt; = &lt;value&gt; (from the round baseline's seed repeats). Threshold to keep: Δ < &lt;−2σ&gt;.

## Score

**val_bpb &lt;value&gt;, Δ/σ = &lt;value&gt; → &lt;KEEP | DISCARD | RETEST | CRASH&gt;.**

No downstream benchmark is reported. At this parameter count and token budget,
HellaSwag / ARC / MMLU sit at chance and carry no signal — reporting them would be
noise dressed as evidence. The valid instruments here are val_bpb, the held-out
TEST_SHARD bpb (Round 3 only), and a qualitative sample.

## Assessment

&lt;The verdict, then WHY. Two or three sentences. Say what the result means for the
hypothesis — confirmed, refuted, or neither — and name the number that decides it.&gt;

## Evidence

1. **&lt;Noise band.&gt;** &lt;The seed numbers, their spread, and how many σ the mean sits
   from baseline. This is what separates a result from a coin flip.&gt;
2. **&lt;Mechanism.&gt;** &lt;What loss.png shows, and at which step. Does the shape match the
   hypothesised mechanism, or does it match a different explanation?&gt;
3. **&lt;Confounds controlled.&gt;** &lt;Did the change buy extra steps or throughput? If so,
   how much of Δbpb does that alone predict, and how much is left over?&gt;
4. **&lt;Cost.&gt;** &lt;VRAM, throughput, lines of code, new concepts. Section 7's simplicity
   criterion is decided here.&gt;
5. **&lt;Raw artifacts.&gt;** &lt;run.log line numbers, metrics.json, loss.png.&gt;

## Threats to validity

&lt;What could still be wrong. Be specific and name what the next round must re-check.
The standing ones worth checking every time: only tested at one depth; untested
interaction with LR or batch size; measured on this round's narrower data slice, so it
may be fitting the slice rather than the problem.&gt;

## Carry forward

&lt;keep / discard / retest-at-R(N+1), and the exact next question this raises.&gt;
