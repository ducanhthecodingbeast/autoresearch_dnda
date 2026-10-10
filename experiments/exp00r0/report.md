<!--
STATUS: NOT YET RUN. Every number below is a blank to fill from metrics.json after
`train.ipynb` executes. Nothing here is a measurement — do not cite this file or carry
any value out of it until the blanks are gone.

Round 0 is instrumentation, not an experiment, so this report is shaped differently
from the others: there is no parent to diff against and no verdict. Its job is to
publish the three numbers the rest of the project depends on.
-->

# exp00r0 — baseline + noise measurement

**Round** 0 (1/4 data, 8 shards, one pass ≈ 0.49B tokens) · **Parent** none — this is the baseline
**Hypothesis** None. Round 0 measures the instrument rather than testing an idea.
**Change** None. Upstream `train.py` defaults, unmodified.

## Benchmarks

Baseline, seed 0:

| Metric | Value |
|---|---|
| **val_bpb** (primary, ↓) | |
| training_seconds | |
| total_seconds | |
| peak_vram_mb | |
| mfu_percent | |
| tok_per_s | |
| total_tokens_M | |
| num_steps | |
| num_params_M | |
| depth | 8 |
| epoch (one pass by design, ~1.0) | |

Seed repeats (this is the deliverable):

| Seed | val_bpb |
|---|---|
| 0 | |
| 1 | |
| 2 | |
| **mean** | |
| **σ₀ (stdev)** | |
| spread (max − min) | |

## Score

**Baseline val_bpb &lt;value&gt;. σ₀ = &lt;value&gt;. Keep threshold for R1: Δ < &lt;−2σ₀&gt;.**

σ₀ is the resolution limit of this entire project. Any val_bpb difference smaller than
roughly 2σ₀ is not measurable here, whatever its sign. Experiments designed to chase an
effect smaller than that are not worth the run.

## Scale and runtime

| Quantity | Value |
|---|---|
| measured throughput (tok/s) | |
| `SHARD_BUDGET` (the compute dial) | |
| epoch reached (should be ~1.0) | |
| R1 runtime/run | |
| R2 runtime/run | |
| R3 runtime/run | |
| total GPU time for the project | |

&lt;There is no time limit, so these are outcomes, not budgets. Two things to say here:
whether `epoch` landed near 1.0 (if not, `SHARD_TOKENS` needs the correction the harness
printed), and whether the total GPU time is one the human is willing to spend — that is
what `SHARD_BUDGET` is for, and it is cheapest to change now.&gt;

## Assessment

&lt;Does the setup work, and is it precise enough to do research with? The question to
answer: given σ₀, what is the smallest effect this project can actually detect, and does
the Round 1 direction list contain anything expected to be smaller than that?&gt;

## Evidence

1. **&lt;Run completes.&gt;** &lt;val_bpb printed, no crash, run.log line numbers.&gt;
2. **&lt;Noise is stationary.&gt;** &lt;The three seed values and their spread. If the spread is
   wide relative to the effects you plan to chase, say so now rather than discovering it
   in round 2.&gt;
3. **&lt;Overhead accounted.&gt;** &lt;total_seconds − training_seconds = startup + compile +
   eval. It does not affect the result (the run is token-bounded) but it does affect how
   many experiments fit in a night, so it belongs in the plan.&gt;
4. **&lt;One pass confirmed.&gt;** &lt;The epoch figure, and the runtime table above.&gt;
5. **&lt;Artifacts.&gt;** &lt;metrics.json holds one object per seed; loss.png per run.&gt;

## Threats to validity

&lt;Standing ones for Round 0: σ₀ from 3 seeds is itself a noisy estimate of σ (a 3-sample
stdev has roughly 40% relative error), so treat the 2σ gate as approximate and re-measure
σ at each round rather than inheriting it. Thermal drift no longer changes the result —
runs are token-bounded, so a slow run is only a longer one — but it does change how much
fits in a night.&gt;

## Carry forward

- σ₀ = &lt;value&gt; → record in `rounds/r1-directions.md`
- `SHARD_BUDGET` = &lt;value&gt; → confirmed with the human in `experiments/_shared/data.py`
- `SHARD_TOKENS` → corrected if `epoch` came back far from 1.0
- Baseline config → the parent for every Round 1 experiment
- **STOP.** Round 1 does not open until the human approves this and the direction list.
