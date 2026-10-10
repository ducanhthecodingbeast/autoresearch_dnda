<!--
AGENT GUIDELINE

This file is AGREED WITH THE HUMAN before the round opens, and is not edited once
experiments start. It is the round's pre-registration: writing the direction list and
the expected effects down in advance is what stops the round from becoming a post-hoc
story about whichever runs happened to come out low.

Two fields must be filled before the first experiment:
  - sigma for this round, measured on THIS round's baseline (never inherited)
  - the baseline config, so every experiment has a named parent

Copy this file to r2-directions.md / r3-directions.md at each round gate, rebuilt from
the previous round's assessment.
-->

# Round 1 — directions

**Scale** 1/4 data · **Shards** 0..7 · **Run length** one pass (~0.49B tokens, ~930 steps) · **Branch** `autoresearch/r1`
**Status** NOT OPEN — waiting on the exp00r0 report and human approval

## Baseline

| | |
|---|---|
| Parent experiment | `exp00r0` |
| Baseline val_bpb | &lt;from exp00r0, mean of 3 seeds&gt; |
| **σ₁** | &lt;measured on the R1 baseline, 3 seeds&gt; |
| Keep threshold (−2σ₁) | &lt;value&gt; |
| Config | upstream `train.py` defaults (see `ExpConfig` in exp00r0's metrics.json) |

σ₁ is re-measured on this round's own baseline. R0 trains identically to an R1 run, so
σ₀ should transfer — measure anyway, because σ is cheap and a stale σ silently corrupts
every verdict in the round.

## Direction list

Ranked by expected Δbpb per run. Upstream defaults for reference: `depth=8`,
`aspect_ratio=64` (dim 512), `head_dim=128`, `window_pattern="SSSL"`,
`total_batch_size=2**19`, `matrix_lr=0.04`, `embedding_lr=0.6`, `warmdown_ratio=0.5`.

| # | Exp ids | Direction | Why it should matter | Runs |
|---|---|---|---|---|
| 1 | exp01r1–exp04r1 | **Depth vs width at fixed params**: depth 6/8/12, `aspect_ratio` compensating | `aspect_ratio=64` is one tuned point, not a law, and it is the axis upstream explored least. It also sets the arena every later experiment runs in — so it goes early or not at all. | 3–4 |
| 2 | exp05r1–exp07r1 | **Attention window**: `window_pattern` "SSSL" vs "L" vs "SSSSSSSL" | Trades context for throughput, and at a fixed budget throughput converts directly into steps. Depends on depth, so run after #1. | 3 |
| 3 | exp08r1–exp11r1 | **Optimizer ablation**: Muon off (AdamW everywhere), Muon momentum, `weight_decay` | Simplification candidate: if Muon's edge is small here, dropping it deletes ~60 lines — a keep under §7 even at Δ≈0. The one direction where a null is still a win. | 3–4 |
| 4 | exp12r1–exp14r1 | **`total_batch_size`**: 2\*\*18 / 2\*\*19 / 2\*\*20 | Critical batch size scales with model size, and #1 moves model size. Interacts with LR, so pair the winner with #5. | 3 |
| 5 | exp15r1–exp18r1 | **LR scale**: `matrix_lr` ×{0.5, 1, 2}, then `embedding_lr` | Normally first, demoted *in R1* because ~930 steps is upstream's own tuning point. Confirm once here, re-test against what #1 and #4 settle on, and expect it to matter far more in R2/R3. | 4 |
| 6 | exp19r1–exp20r1 | **`head_dim`** 64 vs 128 at fixed dim | Changes head count and attention FLOPs per param. Small expected effect, cheap to settle, feeds #2. | 2 |
| 7 | exp21r1–exp24r1 | **Schedule**: `warmdown_ratio` {0.3, 0.5, 0.7}, `warmup_ratio` {0, 0.02}, `final_lr_frac` | Lowest expected yield *in R1*: at ~930 steps the schedule sits at the step count it was tuned for. Promote to first rank in R2/R3, which run 2–4× longer. | 4 |

**Ordering constraint.** #1 before #2, #4 and #6 — all three depend on the model shape
#1 settles, and testing a dependent knob first measures the interaction rather than the
knob itself.

**A caution on yield.** An R1 run is ~930 steps, within a few percent of what upstream
tuned these defaults at (~950). The schedule- and LR-shaped knobs therefore start at or
near their optimum here, so expect small effects and a low keep rate; skip any direction
whose plausible effect is under ~2σ₁. This reverses in R2/R3, which run 2× and 4× longer
at step counts upstream never tuned for — promote LR and schedule there.

## Out of scope for R1

| Direction | Why not |
|---|---|
| `vocab_size` | Needs a tokenizer retrain, which moves a §2 project constant. |
| fp8 / other precisions | Debugging cost would consume the round. |
| Downstream evals (HellaSwag, ARC, MMLU) | Chance-level at this scale; would be noise presented as evidence. |
| Anything touching `prepare.py` or `evaluate_bpb` | §7. Not negotiable. |

## Expected outcomes, recorded in advance

&lt;For each direction, one line on what you expect and roughly how large. Writing this
before the runs is what makes a surprise legible as a surprise instead of getting
absorbed into the narrative afterwards.&gt;

## Open questions for the human

&lt;Anything needing a decision before the round opens.&gt;
