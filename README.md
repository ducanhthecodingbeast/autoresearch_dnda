# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

**This is a fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch)** that runs a different research loop. The upstream idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up to a log of experiments and (hopefully) a better model. The training code is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat), and you program the agent by editing Markdown, not Python. Upstream context: [tweet](https://x.com/karpathy/status/2029701092347630069), [this tweet](https://x.com/karpathy/status/2031135152349524125).

**What this fork changes:** the overnight loop is replaced by a **three-round protocol with human gates**. Rounds run at 1/4, 1/2 and full data with 120s/300s/900s budgets, and the agent stops for review at the end of each one. A result is only kept if it clears **2 sigma** of measured seed noise, rather than merely coming out lower. The protocol lives in [`research-program.md`](research-program.md); the agent's entry point is [`program.md`](program.md). Details in [This fork](#this-fork-a-three-round-protocol-with-human-gates) below.

## How it works

The files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). **Read-only.** The `evaluate_bpb` function in it is the ground-truth metric and is never edited, wrapped or reimplemented.
- **`experiments/_shared/`** — what the agent actually edits. `model.py` (GPT + Muon/AdamW, extracted verbatim from `train.py`), `harness.py` (the run loop), `data.py` (round → data scale), `ledger.py` (the keep/discard rule).
- **`experiments/exp<NN>r<R>/`** — one folder per experiment, each with a `preprocessing.py`, a `train.ipynb`, and a `report.md` carrying benchmarks, a score, an assessment, and numbered evidence for every claim.
- **`research-program.md`** — the protocol: rounds, decision rule, folder contracts, report format. **This file is edited and iterated on by the human** — it is this fork's "research org code".
- **`train.py`** — upstream's single-file script. Kept unmodified as a reference to diff against; not run after Round 0.

Training runs for a **per-round time budget** (120s / 300s / 900s) rather than upstream's fixed 5 minutes, so the rounds form a cheap-to-expensive ladder. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared. Every run also reports `epoch`, which must stay below 1.0: a round that repeats its data is measuring memorization, not the change under test.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Quick start

**Requirements:** A single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous
research mode. In this fork that means starting at Round 0 — see
[`program.md`](program.md). Note that `--num-shards 10` is the upstream default; Round 0
measures your platform's throughput and tells you the shard count this protocol actually
needs (32 on an H100), plus the command to fetch the held-out test shard.

## This fork: a three-round protocol with human gates

This fork replaces the single overnight loop with a staged protocol. The full
specification is in **[`research-program.md`](research-program.md)**; the agent's entry
point is [`program.md`](program.md).

| Round | Data | Budget | Purpose |
|---|---|---|---|
| R0 | 1/4 | 120 s | Baseline, throughput sizing, **σ₀** (the noise band) |
| R1 | 1/4 | 120 s | Wide screening — many cheap runs, coarse knobs |
| R2 | 1/2 | 300 s | Confirm R1 survivors at larger scale |
| R3 | full | 900 s | Final config, held-out test-shard confirmation |

Three differences from upstream, each fixing something that bites over ~100 runs:

1. **A keep must clear 2σ.** Upstream keeps any run with a lower `val_bpb`. At a
   5-minute budget much of that difference is seed noise, and greedily hill-climbing on
   noise manufactures wins that vanish on a rerun. σ is measured from three seed repeats
   of each round's baseline, and `ledger.verdict()` decides — not the agent's judgment.
2. **The time budget scales with the rounds.** Upstream fixes wall-clock at 300 s, so a
   run consumes the same token count whatever the dataset size — scaling data alone
   would give three rounds that cost the same and differ only in which documents they
   saw. An **epoch gate** (`epoch < 1.0`, enforced in every run) keeps any round from
   silently training on repeats.
3. **A held-out test shard.** Across three rounds of hill-climbing on one pinned
   validation shard you *will* overfit it. Shard 6541 stays untouched until the Round 3
   gate, and `test_bpb − val_bpb` is the headline result.

Each experiment is a folder — `exp01r1`, `exp02r1`, `exp01r2` — containing
`preprocessing.py`, `train.ipynb`, and a `report.md` with benchmarks, a score, an
assessment, and numbered evidence for every claim.

## Running the agent

Spin up your Claude/Codex in this repo and point it at the entry file:

```
Have a look at program.md and let's start Round 0.
```

The agent stops at every round gate and waits for you.

## Project structure

```
prepare.py                  constants, data prep + runtime utilities (READ-ONLY)
train.py                    upstream reference; not run after R0, kept for diffing
program.md                  agent entry point
research-program.md         the protocol: rounds, decision rule, contracts
analysis.ipynb              round-aware analysis of results.tsv
experiments/
  _shared/                  model.py, harness.py, data.py, ledger.py
  _template/                copy this to start an experiment
  exp00r0/                  baseline + noise measurement
  exp<NN>r<R>/              preprocessing.py, train.ipynb, report.md, metrics.json
rounds/                     per-round direction lists and assessments
results.tsv                 append-only ledger (untracked)
pyproject.toml              dependencies
```

The two logic modules carry their own self-checks, which need no GPU:

```bash
uv run python -m experiments._shared.data --check     # round slices, epoch sizing
uv run python -m experiments._shared.ledger --check   # the 2-sigma decision rule
```

## Design choices

- **Wall-clock budgets, not step counts.** Experiments are compared at equal time, so a change to model size, batch size or architecture is judged on what it delivers per second rather than per step. Upstream fixes this at 5 minutes; this fork scales it per round (120s / 300s / 900s) so screening is cheap and the final round is thorough. The trade-off is unchanged: results are optimal *for your platform* and not comparable with someone else's hardware.
- **Decisions are statistical, not greedy.** Three seed repeats of each round's baseline give a sigma; a change is kept only if it moves `val_bpb` by more than 2 of them. This costs runs, and it is the point — at these budgets a seed reshuffle moves the metric enough that a greedy keep-if-lower loop spends the night accumulating noise.
- **The metric is untouchable.** `prepare.py` is read-only and `evaluate_bpb` is never wrapped or reimplemented. Everything the agent can reach is on the other side of that line.
- **Validation is held out twice.** A pinned validation shard scores every run, and a separate test shard stays sealed until the Round 3 gate. Three rounds of hill-climbing on one shard will overfit it; `test_bpb - val_bpb` is how you find out by how much.
- **Every run leaves a report.** Benchmarks, a score, an assessment, and numbered evidence for each claim. A result nobody can audit later is not a result — and null results get the same treatment as wins, because the round assessments are built from them.
- **Self-contained.** PyTorch, a few small packages, and Jupyter to execute the notebooks. No distributed training, no complex configs. One GPU, one metric.

## Platform support

This code currently requires a single NVIDIA GPU. Supporting CPU, MPS and other platforms is quite possible in principle but would bloat the code, and upstream has not taken it on. People can reference (or have their agents reference) the full/parent nanochat repository that has wider platform support and shows the various solutions (e.g. a Flash Attention 3 kernels fallback implementation, generic device support, autodetection, etc.), feel free to create forks or discussions for other platforms.

The sections below (small-compute tuning advice, notable forks) are upstream's and apply to the training code, which this fork has not changed.

Seeing as there seems to be a lot of interest in tinkering with autoresearch on much smaller compute platforms than an H100, a few extra words. If you're going to try running autoresearch on smaller computers (Macbooks etc.), I'd recommend one of the forks below. On top of this, here are some recommendations for how to tune the defaults for much smaller models for aspiring forks:

1. To get half-decent results I'd use a dataset with a lot less entropy, e.g. this [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean). These are GPT-4 generated short stories. Because the data is a lot narrower in scope, you will see reasonable results with a lot smaller models (if you try to sample from them after training).
2. You might experiment with decreasing `vocab_size`, e.g. from 8192 down to 4096, 2048, 1024, or even - simply byte-level tokenizer with 256 possibly bytes after utf-8 encoding.
3. In `prepare.py`, you'll want to lower `MAX_SEQ_LEN` a lot, depending on the computer even down to 256 etc. As you lower `MAX_SEQ_LEN`, you may want to experiment with increasing `DEVICE_BATCH_SIZE` in `train.py` slightly to compensate. The number of tokens per fwd/bwd pass is the product of these two.
4. Also in `prepare.py`, you'll want to decrease `EVAL_TOKENS` so that your validation loss is evaluated on a lot less data.
5. In `train.py`, the primary single knob that controls model complexity is the `DEPTH` (default 8, here). A lot of variables are just functions of this, so e.g. lower it down to e.g. 4.
6. You'll want to most likely use `WINDOW_PATTERN` of just "L", because "SSSL" uses alternating banded attention pattern that may be very inefficient for you. Try it.
7. You'll want to lower `TOTAL_BATCH_SIZE` a lot, but keep it powers of 2, e.g. down to `2**14` (~16K) or so even, hard to tell.

I think these would be the reasonable hyperparameters to play with. Ask your favorite coding agent for help and copy paste them this guide, as well as the full source code.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (AMD)

## License

MIT
