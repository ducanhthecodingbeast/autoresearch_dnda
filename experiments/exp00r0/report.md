<!--
STATUS: NOT YET RUN. Every number below is a blank to fill from metrics.json after
`train.ipynb` executes. Nothing here is a measurement — do not cite this file or carry
any value out of it until the blanks are gone.

Round 0 is instrumentation, not an experiment, so this report is shaped differently
from the others: there is no parent to diff against and no verdict. Its job is to
publish the three numbers the rest of the project depends on.
-->

# exp00r0 — baseline + noise measurement

**Round** 0 (1/4 data, &lt;N&gt; shards, 120 s) · **Parent** none — this is the baseline
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
| epoch (validity, must be < 1.0) | |

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

## Dataset sizing

| Quantity | Value |
|---|---|
| measured throughput (tok/s) | |
| `SHARD_BUDGET` required | |
| R1 epochs (must be < 1.0) | |
| R2 epochs (must be < 1.0) | |
| R3 epochs (must be < 1.0) | |

&lt;If any round is ≥ 1.0, say so here and raise `SHARD_BUDGET` in
`experiments/_shared/data.py` before Round 1 opens. Do not shorten a time budget to
bring a round under the gate — that changes what the round measures.&gt;

## Assessment

&lt;Does the setup work, and is it precise enough to do research with? The question to
answer: given σ₀, what is the smallest effect this project can actually detect, and does
the Round 1 direction list contain anything expected to be smaller than that?&gt;

## Evidence

1. **&lt;Run completes.&gt;** &lt;val_bpb printed, no crash, run.log line numbers.&gt;
2. **&lt;Noise is stationary.&gt;** &lt;The three seed values and their spread. If the spread is
   wide relative to the effects you plan to chase, say so now rather than discovering it
   in round 2.&gt;
3. **&lt;Compilation excluded.&gt;** &lt;total_seconds − training_seconds = startup + eval. Confirm
   the 10 warmup steps the harness skips were enough to keep compile time out of the
   budget, otherwise every run's effective budget differs.&gt;
4. **&lt;Epoch gate clears.&gt;** &lt;The epochs-per-round table above.&gt;
5. **&lt;Artifacts.&gt;** &lt;metrics.json holds one object per seed; loss.png per run.&gt;

## Threats to validity

&lt;Standing ones for Round 0: σ₀ from 3 seeds is itself a noisy estimate of σ (a 3-sample
stdev has roughly 40% relative error), so treat the 2σ gate as approximate and re-measure
σ at each round rather than inheriting it. Throughput may drift with GPU thermals, which
moves steps-per-run between experiments run hours apart.&gt;

## Carry forward

- σ₀ = &lt;value&gt; → record in `rounds/r1-directions.md`
- `SHARD_BUDGET` = &lt;value&gt; → set in `experiments/_shared/data.py`
- Baseline config → the parent for every Round 1 experiment
- **STOP.** Round 1 does not open until the human approves this and the direction list.
