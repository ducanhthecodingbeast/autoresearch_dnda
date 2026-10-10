# Research program — rounds, experiments, reports

Custom protocol. Replaces `program.md` (kept as upstream reference).
Agent reads this file, runs the loop, and stops at every round gate for human review.

---

## 1. Round structure

Three experiment rounds on a cheap→expensive ladder, plus a setup round.
The agent **stops dead** at the end of each round and does not start the next one
until the human approves the round assessment.

| Round | Data scale | Time budget | Purpose |
|---|---|---|---|
| **R0** | 1/4 | 120 s | Baseline + noise measurement. Not an experiment round. |
| **R1** | **1/4** | **120 s** | Wide screening. Many cheap runs, coarse knobs. |
| **R2** | **1/2** | **300 s** | Confirm R1 survivors, tune what held. |
| **R3** | **full** | **900 s** | Final config. Few runs. Test-shard confirmation. |

### 1.1 Why the time budget scales too

Data scale alone does not make a round cheaper or more expensive. The upstream design
fixes wall-clock at 300 s, so a run consumes the same token count whatever the dataset
size — scaling data from 1/4 to full would change only *which* documents are seen and
how often they repeat, at identical cost. The rounds would differ by nothing that shows
up in a budget.

Scaling the budget alongside the data gives the ladder real teeth: R1 buys ~2.5× the
experiments per hour, R3 buys the longest-horizon result. Data scale then also binds,
because a longer run needs more fresh data to stay inside one epoch.

If you want the pure data-only axis instead, set all three budgets to 300 s and keep
everything else. The protocol does not otherwise depend on the ladder.

### 1.2 Sizing the dataset — the epoch rule

**Every round must stay under 1.0 epoch.** A round that repeats data is measuring
memorization, not the change under test, and its verdicts do not transfer upward.

`train.py` already tracks and prints `epoch`; the report treats it as a validity gate.
Size the shard budget from measured throughput, not from a guess:

```
shard_tokens   ~= 61M                       # 400B ClimbMix / 6542 shards
R3_tokens       = throughput * 900          # measured at R0
SHARD_BUDGET    = ceil(1.25 * R3_tokens / shard_tokens)   # 25% headroom
```

Round slices are nested, so R2 and R3 include everything the earlier rounds saw:

| Round | Shards | Scale |
|---|---|---|
| R1 | `0 .. SHARD_BUDGET/4 - 1` | 1/4 |
| R2 | `0 .. SHARD_BUDGET/2 - 1` | 1/2 |
| R3 | `0 .. SHARD_BUDGET - 1` | full |

**Worked default (H100, ~1.67M tok/s from the upstream summary):**
`R3_tokens` = 1.5B → `SHARD_BUDGET` = **32** shards (~1.95B tokens, ~6.4 GB on disk).
R1 = 8 shards / 0.41 epochs, R2 = 16 / 0.51, R3 = 32 / 0.77. All clear.

Recompute this at R0 for the actual platform. If any round lands above 1.0 epoch,
raise `SHARD_BUDGET` — never lower the time budget to hide it.

### 1.3 Gates

**R0 exit gate** — all three must hold before R1 opens:
1. A run completes and prints `val_bpb:` without crashing.
2. Throughput measured, `SHARD_BUDGET` computed, all three rounds under 1.0 epoch.
3. Three repeated baseline runs, seeds 0/1/2 → gives **σ₀**, the noise band (§4).

**Round exit gate** — a round closes when the direction list is exhausted or the run
budget is spent. The agent writes `rounds/rN-assessment.md` (§6), then stops.

---

## 2. Fixed project constants

Pinned once. Changing any of these invalidates cross-round comparison and starts a new
project, not a new experiment.

Live in `experiments/_shared/data.py`:

```
SHARD_BUDGET   = 32           # "full dataset". recompute at R0 per 1.2.
MAX_SEQ_LEN    = 2048         # prepare.py default
EVAL_TOKENS    = 40*524288    # prepare.py default (~21M tokens)
VAL_SHARD      = 6542         # pinned by prepare.py. every val_bpb uses this.
TEST_SHARD     = 6541         # NEVER touched until the Round 3 gate. overfit detector.
```

`prepare.py --num-shards N` only ever fetches the val shard (6542) alongside the train
slice, so **`TEST_SHARD` 6541 must be downloaded separately** before the Round 3 gate.

If a platform needs `MAX_SEQ_LEN` or `EVAL_TOKENS` changed, override the attribute in
the notebook *before* the first dataloader/eval call — never by editing read-only
`prepare.py`:

```python
import prepare
prepare.MAX_SEQ_LEN = 1024
```

Both are read as module globals at call time inside `make_dataloader` / `evaluate_bpb`,
so patching the attribute works. Do it once, in §2, for the whole project.
`evaluate_bpb` itself is the ground-truth metric and is never reimplemented, wrapped,
or adjusted.

---

## 3. Layout and naming

```
research-program.md             this file — the protocol
prepare.py                      READ-ONLY. never edited.
train.py                        upstream reference. not run after R0; kept for diffing.
experiments/
  _shared/
    model.py                    GPT + MuonAdamW, extracted verbatim from train.py
    harness.py                  ExpConfig, run(), run_seeds() — setup/loop/eval/metrics
    data.py                     round -> symlinked shard dir, prepare.DATA_DIR override
    ledger.py                   verdict() (section 4 as code) + results.tsv append
  _template/                    COPY THIS to start an experiment
    preprocessing.py
    train.ipynb
    report.md
  exp00r0/                      R0: baseline, throughput sizing, sigma_0
    preprocessing.py
    train.ipynb
    report.md
  exp01r1/                      one folder per experiment, named for its id
    preprocessing.py            REQUIRED — pins round data scale for this experiment
    train.ipynb                 REQUIRED — the experiment itself
    report.md                   REQUIRED — benchmarks, scores, assessment, evidence
    run.log                     raw stdout
    metrics.json                written by the harness; one object per seed
    loss.png                    written by the harness; cited as evidence
rounds/
  _assessment-template.md       copy at each round gate
  r1-directions.md              agreed with the human before R1 opens
  r1-assessment.md              written at the R1 gate
  r2-directions.md ...
results.tsv                     append-only ledger, all rounds. untracked by git.
```

### 3.0 Quickstart for an agent

```bash
# once, before R0 — download data (SHARD_BUDGET comes out of R0, start with 32)
uv run prepare.py --num-shards 32

# the two logic self-checks. no GPU needed, run them after editing _shared/
uv run python -m experiments._shared.data --check
uv run python -m experiments._shared.ledger --check

# start an experiment
cp -r experiments/_template experiments/exp01r1
uv run python experiments/exp01r1/preprocessing.py       # verifies data is in place
git add -A && git commit -m "exp01r1: <one-line change>"
uv run jupyter nbconvert --to notebook --execute --inplace \
  experiments/exp01r1/train.ipynb > experiments/exp01r1/run.log 2>&1
grep "^val_bpb:\|^epoch:" experiments/exp01r1/run.log
```

Note: importing `_shared/model.py` fetches a FlashAttention kernel from the HF Hub on
first use (upstream `train.py` behaviour, lines 20–24). It needs network access once,
and caches after that.

**Experiment id: `exp<NN>r<R>`** — `NN` = two-digit counter, restarts each round;
`R` = round number. `exp01r1`, `exp02r1`, …, `exp01r2`. Folder name == experiment id.
Git branch per round: `autoresearch/r<R>`. One commit per experiment, message
`exp<NN>r<R>: <one-line change>`.

### 3.1 `preprocessing.py` contract

Pins the data scale for the round. **Does not re-download, does not copy shards** — it
builds a symlink directory and points `prepare.DATA_DIR` at it. Shards download once,
into the real cache, and every round shares them.

```python
"""exp01r1 — data prep. Round 1 = 1/4 scale."""
from experiments._shared.data import prepare_round

ROUND = 1
if __name__ == "__main__":
    print(prepare_round(ROUND))   # -> ~/.cache/autoresearch/data_r1, asserts shard count
```

`_shared/data.py::prepare_round(round)` asserts the round's shards plus the val shard
exist in the real cache (else it prints the exact `prepare.py --num-shards` command and
exits), rebuilds `data_r{N}/` as a directory of symlinks to the round's slice, refuses
to link `TEST_SHARD` before R3, sets `prepare.DATA_DIR`, and returns the path.

`prepare.list_parquet_files()` reads `DATA_DIR` at call time, which is what makes the
symlink-plus-override trick work without touching `prepare.py`. `harness.run()` asserts
`DATA_DIR` points at a `data_r*` directory, so forgetting `prepare_round()` fails loudly
instead of silently training at the wrong scale.

Round slices come from `ROUND_SCALE` in that file, and `required_shard_budget()` /
`epochs_per_round()` implement the §1.2 sizing. Both are covered by
`uv run python -m experiments._shared.data --check`.

### 3.2 `train.ipynb` contract

Cells, in order:

1. **Header** — experiment id, round, one-sentence hypothesis, parent experiment id
   (which config this is a delta against), and the exact delta as a diff or param table.
2. **Setup** — `prepare_round(ROUND)`, any §2 overrides, seed, backend probe. Prints the
   resolved environment.
3. **Config** — every hyperparameter as one flat dict, printed. Nothing hidden in
   defaults; the printed dict is what the report cites.
4. **Train** — the loop from `_shared/harness.py`. Logs step, loss, lr, tok/s, VRAM,
   epoch to a list. Honours the round time budget.
5. **Eval** — `prepare.evaluate_bpb(model, tokenizer, batch_size)`, unmodified.
6. **Metrics** — write `metrics.json`, print the upstream summary block verbatim
   (`val_bpb:`, `peak_vram_mb:`, … — same keys as `train.py`, so `grep` still works).
7. **Curves** — train loss vs step and vs wall-clock, saved as `loss.png`. This is the
   evidence the report cites for "did it converge, or did it just get more steps".

Run headless, log to file, never stream into agent context:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
  experiments/exp01r1/train.ipynb > experiments/exp01r1/run.log 2>&1
```

Kill and mark `crash` if a run exceeds **3× its round time budget**.

---

## 4. The decision rule (noise band)

The upstream loop keeps any run with a lower `val_bpb`. At a 2–15 minute budget part of
that difference is seed noise, and a greedy walk on noise manufactures 100 fake wins
overnight. This project does not do that.

**σ_R** = standard deviation of `val_bpb` across the three seed-repeat runs of the
round's baseline. Measured once per round, recorded in `rounds/rN-directions.md`.

| Δbpb vs round baseline | Verdict | Action |
|---|---|---|
| Δ < −2σ_R | **keep** | Advance the branch. Becomes the new round baseline. |
| −2σ_R ≤ Δ < −1σ_R | **retest** | Rerun at 2 more seeds. Keep only if mean Δ < −1.5σ_R. |
| \|Δ\| ≤ 1σ_R, code simpler | **keep** | Simplification win. Advance. |
| \|Δ\| ≤ 1σ_R, code same or worse | **discard** | Revert. Record as a null result — these are useful. |
| Δ > +1σ_R | **discard** | Revert. |
| OOM / crash | **crash** | Revert. Record the traceback's first line. |

A single-seed result is never promoted across a round boundary. Anything carried from
R1 to R2 is re-measured at the R2 scale before it is trusted.

`results.tsv` (tab-separated, append-only, untracked):

```
exp_id	round	commit	val_bpb	sigma_mult	memory_gb	tok_per_s	epoch	status	description
```

`sigma_mult` = Δbpb / σ_R, the number the verdict was actually made on.

This table is `verdict()` in `_shared/ledger.py`, and the agent calls it rather than
eyeballing the comparison. `run_seeds()` in `_shared/harness.py` measures σ and resolves
a `retest`. Both are covered by `uv run python -m experiments._shared.ledger --check`.

---

## 5. Report contract — `experiments/exp<NN>r<R>/report.md`

Every experiment folder has one. No run is "done" without it. Numbers are copied from
`metrics.json`, never retyped from memory.

```markdown
# exp01r1 — <short title>

**Round** 1 (1/4 data, 8 shards, 120 s budget) · **Parent** exp00r1 (baseline)
**Hypothesis** <one sentence, falsifiable: what should move, and in which direction>
**Change** <the delta, as a param table or diff. nothing else changed.>

## Benchmarks

| Metric | Baseline | This run | Δ | Δ/σ |
|---|---|---|---|---|
| **val_bpb** (primary, ↓) | 1.012000 | 0.994000 | **−0.018000** | **−3.3** |
| val_bpb seed spread (n=3) | ±0.0055 | ±0.0051 | | |
| peak_vram_mb | 44100 | 44600 | +500 | |
| mfu_percent | 39.8 | 41.2 | +1.4 | |
| tok_per_s | 1_664_000 | 1_722_000 | +58_000 | |
| total_tokens_M | 199.7 | 206.6 | +6.9 | |
| num_steps | 381 | 394 | +13 | |
| num_params_M | 50.3 | 50.3 | 0 | |
| epoch (validity, must be < 1.0) | 0.41 | 0.42 | | |

σ₁ = 0.0055 (R0 seed repeats). Threshold to keep: Δ < −0.0110.

## Score

**val_bpb 0.994000, Δ/σ = −3.3 → KEEP.**

No downstream benchmark is reported. At this parameter count and token budget,
HellaSwag / ARC / MMLU sit at chance and carry no signal — reporting them would be
noise dressed as evidence. The valid instruments here are val_bpb, the held-out
TEST_SHARD bpb (Round 3 only), and a qualitative sample.

## Assessment

<verdict, then WHY. every claim names the number or artifact behind it.>

## Evidence

1. **Beats the noise band.** 3 seeds: 0.9938 / 0.9940 / 0.9942, spread 0.0004, well
   inside σ₁ = 0.0055, and the mean sits 3.3σ below baseline. Not a seed artifact.
2. **Mechanism matches the hypothesis.** `loss.png` — the gap opens at step ~90 and
   holds, rather than appearing only during warmdown. Consistent with <mechanism>, not
   with <the alternative explanation>.
3. **Not bought with extra steps.** +13 steps (+3.4%) on +58k tok/s. A pure throughput
   win of that size predicts ≈ −0.003 bpb from the step-count scaling measured in
   exp03r1; the observed −0.018 is 6× that, so most of the gain is the change itself.
4. **Cost is flat.** +500 MB VRAM, no new dependency, net −4 lines.
5. **Raw artifacts.** `run.log` lines 180–195, `metrics.json`, `loss.png`.

## Threats to validity

<what could still be wrong. e.g. only tested at depth 8; interacts with LR, untested;
1/4 data and a 120 s budget, so this may not hold at R3 scale. Say what R2 must
re-check.>

## Carry forward

<keep / discard / retest-at-R2, and the exact next question this raises.>
```

**Evidence rules.** An assessment line with no number, artifact, or named mechanism
behind it is deleted. Three things that are *not* evidence: a lower val_bpb on its own
(that is the claim, not the support); "as expected"; and a mechanism story with no
curve. Null and negative results get the same report quality as wins — the round
assessment depends on them.

---

## 6. Round assessment gate — `rounds/rN-assessment.md`

Written when the round closes. The agent then **stops and waits**.

1. **Ledger** — every experiment in the round, one row, with Δ/σ and verdict.
2. **What held** — the kept changes, the cumulative Δbpb from round baseline to round
   best, and how much of it each change owns.
3. **What died** — discarded directions, and whether each was a *real* null (effect
   below noise) or *inconclusive* (confounded, crashed, under-tested). Inconclusive is
   not refuted, and may be re-opened.
4. **Interactions** — pairs tested only separately. Name them; they are candidate
   R(N+1) experiments.
5. **Scale risk** — which wins look likely to shrink at the next round's larger data
   and longer budget, and why. LR and schedule wins in particular rarely transfer
   across a 2.5× budget change.
6. **Next round plan** — the proposed direction list for R(N+1), ranked by expected
   Δbpb per run, with the new baseline config and the σ measurement plan.
7. **Open questions for the human** — anything needing a decision.

**Round 3 extra gate.** The final config is evaluated once on `TEST_SHARD` (6541).
If `test_bpb − val_bpb` > 2σ₃, the project has overfit the validation shard across
rounds, and the report says so with both numbers. That is the headline result, not a
footnote.

---

## 7. Rules

**Allowed:** edit anything inside `experiments/`; architecture, optimizer, schedule,
batch size, model size, attention backend, precision, data ordering.

**Not allowed:**
- Edit `prepare.py`, or reimplement / wrap / "fix" `evaluate_bpb`.
- Change a §2 project constant mid-project.
- Add a dependency outside `pyproject.toml`.
- Look at `TEST_SHARD` before the Round 3 gate.
- Start round N+1 before the human approves round N's assessment.
- Promote a single-seed result across a round boundary.
- Commit `results.tsv`.

**Simplicity criterion** (upstream, kept): a 0.001 win that adds 20 lines of hack is not
a win. A 0.001 win from deleting code is. Equal result with less code is a keep.

**VRAM** is a soft constraint. Some increase is fine for a real val_bpb gain; it should
not blow up.

---

## 8. Round 1 candidate directions — agree before R1 opens

Ranked by expected Δbpb per run. R0 closes first and sets the baseline these are deltas
against. Upstream defaults for reference: `DEPTH=8`, `ASPECT_RATIO=64` (dim 512),
`HEAD_DIM=128`, `WINDOW_PATTERN="SSSL"`, `TOTAL_BATCH_SIZE=2**19`, `MATRIX_LR=0.04`
(Muon), `EMBEDDING_LR=0.6`, `WARMDOWN_RATIO=0.5`.

| # | Direction | Why it should matter | Cost |
|---|---|---|---|
| 1 | **Learning rate scale**: `MATRIX_LR` ×{0.5, 1, 2}, then `EMBEDDING_LR`, then `UNEMBEDDING_LR` | Historically the highest-yield knob whenever the budget changes. R1 runs at 120 s are ~380 steps against the ~950 the defaults were tuned for, and the optimum LR moves with step count. | 4–6 runs |
| 2 | **Depth vs width at fixed params**: depth 6/8/12 with `ASPECT_RATIO` compensating | `ASPECT_RATIO=64` is one tuned point, not a law. Cheap, and it sets the arena every later experiment runs in — so it goes early or not at all. | 3–4 runs |
| 3 | **Schedule**: `WARMDOWN_RATIO` {0.3, 0.5, 0.7}, `WARMUP_RATIO` {0, 0.02}, `FINAL_LR_FRAC` | Schedule fractions interact directly with total step count, which the round ladder changes by 7.5×. Expect the R1 optimum to differ from R3's — which is itself a useful finding. | 4 runs |
| 4 | **`TOTAL_BATCH_SIZE`**: 2**18 / 2**19 / 2**20 | Critical batch size scales with model size and budget. Interacts strongly with #1, so run it *after* the LR scale is known, not in parallel. | 3 runs |
| 5 | **Attention window**: `WINDOW_PATTERN` "SSSL" vs "L" vs "SSSSSSSL" | Trades context for throughput. The banded pattern's payoff depends on depth and seq len, both of which #2 may move. | 3 runs |
| 6 | **Optimizer ablation**: Muon off (AdamW everywhere), Muon momentum schedule, `WEIGHT_DECAY` | Simplification candidate. If Muon's edge is small at this scale, dropping it deletes ~60 lines — a keep under §7 even at Δ ≈ 0. Measure before assuming. | 3–4 runs |
| 7 | **`HEAD_DIM`** 64 vs 128 at fixed dim | Changes head count and attention FLOPs per param. Small expected effect, cheap to settle, and it unblocks #5. | 2 runs |

Not in R1, and why: `vocab_size` (needs a tokenizer retrain, which moves a §2 constant);
fp8 (debugging cost would eat the round); any downstream eval (chance-level here, per §5).

---

## 9. Loop

```
R0: baseline runs clean -> measure throughput -> compute SHARD_BUDGET
    -> verify every round < 1.0 epoch -> 3 seeds -> sigma_0 -> STOP for approval

for ROUND in 1, 2, 3:
    agree rounds/rROUND-directions.md with the human
    prepare_round(ROUND); measure sigma_ROUND on the round baseline (3 seeds)
    for each direction:
        mkdir experiments/exp<NN>r<ROUND>; write preprocessing.py + train.ipynb
        git commit
        run headless -> run.log, metrics.json, loss.png
        verdict per section 4; 3 seeds for anything near the band
        write report.md; append to results.tsv
        keep -> advance branch | else -> git reset
    write rounds/rROUND-assessment.md
    STOP. wait for human approval.

R3 final: evaluate once on TEST_SHARD; report test_bpb - val_bpb
```
