# autoresearch — agent entry point

This project does **not** use the upstream single-agent loop. It runs a three-round
protocol with human gates between rounds.

**Read [`research-program.md`](research-program.md) before doing anything else.** It is
the whole specification: rounds, data scales, the decision rule, folder contracts, and
the report format. This file only tells you where to start.

The upstream baseline instructions this file used to hold are in git history
(`git show 228791f:program.md`).

---

## Start here

1. **Read the protocol.** `research-program.md`, all of it. The sections you will use
   every single run are §4 (the decision rule) and §5 (the report contract).
2. **Read the in-scope code.** The repo is small:
   - `prepare.py` — **read-only.** Data prep, tokenizer, dataloader, and the
     `evaluate_bpb` ground-truth metric. Never edit, never wrap, never reimplement.
   - `experiments/_shared/` — `model.py` (GPT + MuonAdamW), `harness.py` (the run
     loop), `data.py` (round → data scale), `ledger.py` (the verdict rule).
   - `train.py` — upstream reference only. Not run after R0. Kept so you can diff
     `_shared/model.py` against it.
3. **Check the data.** `uv run python experiments/exp00r0/preprocessing.py` tells you
   what is missing and the exact command to fix it.
4. **Run Round 0.** `experiments/exp00r0/train.ipynb`. It produces the baseline,
   the platform throughput that sizes `SHARD_BUDGET`, and σ₀ — the noise band every
   later verdict is measured against.
5. **Stop.** Write `experiments/exp00r0/report.md`, propose `rounds/r1-directions.md`,
   and wait for the human. Round 1 does not open on your own authority.

## One experiment, start to finish

```bash
cp -r experiments/_template experiments/exp01r1      # folder name == experiment id
# edit preprocessing.py (ROUND), train.ipynb (header, config, the one change)
uv run python experiments/exp01r1/preprocessing.py   # verify data scale
git add -A && git commit -m "exp01r1: <one-line change>"
uv run jupyter nbconvert --to notebook --execute --inplace \
  experiments/exp01r1/train.ipynb > experiments/exp01r1/run.log 2>&1
grep "^val_bpb:\|^epoch:" experiments/exp01r1/run.log
```

Then: write `report.md`, append to `results.tsv`, and either keep the commit (the
branch advances) or `git reset --hard` back to the round baseline.

If the grep comes back empty the run crashed — `tail -n 50 run.log` for the traceback.
After a few failed attempts at the same idea, record it as `crash` and move on.

## The rules you are most likely to break

These are the ones that look like harmless shortcuts in the moment:

- **A lower `val_bpb` is not a win.** It must clear 2σ. Call
  `ledger.verdict(...)`; do not decide by eye. This is the single biggest difference
  from the upstream loop, where any improvement is kept — over 100 overnight runs that
  is a greedy walk on seed noise.
- **One change per experiment.** Two changes is two experiments. Otherwise the ledger
  records an effect you cannot attribute.
- **`epoch` must be < 1.0.** Printed in every summary. Above it, the run repeated its
  data and the result is invalid regardless of `val_bpb`. R1 is the tightest round for
  this — least data, same 300 s as every other round.
- **Never look at `TEST_SHARD` (6541) before the Round 3 gate.** It is the only
  instrument that can tell you whether three rounds of hill-climbing overfit the
  validation shard.
- **Never start the next round yourself.** Every round ends with a written assessment
  and a human decision.
- **A run without `report.md` is not done.** Numbers come from `metrics.json`, never
  retyped from memory, and every assessment claim names its evidence.

## Analysis

`analysis.ipynb` reads `results.tsv` and plots per-round progress, effect sizes in σ,
and the cost side. It also runs the validity gates (epoch, sub-2σ keeps) — read those
before reading any result.
