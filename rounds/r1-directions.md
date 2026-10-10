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

**Scale** 1/4 data · **Shards** 0..&lt;N&gt; · **Budget** 120 s · **Branch** `autoresearch/r1`
**Status** NOT OPEN — waiting on the exp00r0 report and human approval

## Baseline

| | |
|---|---|
| Parent experiment | `exp00r0` |
| Baseline val_bpb | &lt;from exp00r0, mean of 3 seeds&gt; |
| **σ₁** | &lt;measured on the R1 baseline, 3 seeds&gt; |
| Keep threshold (−2σ₁) | &lt;value&gt; |
| Config | upstream `train.py` defaults (see `ExpConfig` in exp00r0's metrics.json) |

σ₁ is re-measured at this round's data scale and budget. It is not inherited from
σ₀ even when the scale matches, because the budget does not.

## Direction list

Ranked by expected Δbpb per run. Upstream defaults for reference: `depth=8`,
`aspect_ratio=64` (dim 512), `head_dim=128`, `window_pattern="SSSL"`,
`total_batch_size=2**19`, `matrix_lr=0.04`, `embedding_lr=0.6`, `warmdown_ratio=0.5`.

| # | Exp ids | Direction | Why it should matter | Runs |
|---|---|---|---|---|
| 1 | exp01r1–exp03r1 | **LR scale**: `matrix_lr` ×{0.5, 1, 2}, then `embedding_lr`, then `unembedding_lr` | Historically the highest-yield knob whenever the budget changes. R1 at 120 s is ~380 steps against the ~950 the defaults were tuned for, and the optimal LR moves with step count. | 4–6 |
| 2 | exp04r1–exp07r1 | **Depth vs width at fixed params**: depth 6/8/12, `aspect_ratio` compensating | `aspect_ratio=64` is one tuned point, not a law. Cheap, and it sets the arena every later experiment runs in — so it goes early or not at all. | 3–4 |
| 3 | exp08r1–exp11r1 | **Schedule**: `warmdown_ratio` {0.3, 0.5, 0.7}, `warmup_ratio` {0, 0.02}, `final_lr_frac` | Schedule fractions interact with total step count, which the round ladder moves by 7.5×. Expect R1's optimum to differ from R3's — which is itself a finding. | 4 |
| 4 | exp12r1–exp14r1 | **`total_batch_size`**: 2\*\*18 / 2\*\*19 / 2\*\*20 | Critical batch size scales with model size and budget; 2\*\*19 is sized for 50M params at 300 s. Interacts strongly with #1, so run it *after* the LR scale is known. | 3 |
| 5 | exp15r1–exp17r1 | **Attention window**: `window_pattern` "SSSL" vs "L" vs "SSSSSSSL" | Trades context for throughput. The banded pattern's payoff depends on depth, which #2 may move. | 3 |
| 6 | exp18r1–exp21r1 | **Optimizer ablation**: Muon off (AdamW everywhere), Muon momentum schedule, `weight_decay` | Simplification candidate. If Muon's edge is small at this scale, dropping it deletes ~60 lines — a keep under §7 even at Δ≈0. Measure before assuming. | 3–4 |
| 7 | exp22r1–exp23r1 | **`head_dim`** 64 vs 128 at fixed dim | Changes head count and attention FLOPs per param. Small expected effect, cheap to settle, and it unblocks #5. | 2 |

**Ordering constraint.** #1 before #4, and #2 before #5 — both pairs interact, and
testing the dependent one first measures the interaction rather than the knob.

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
