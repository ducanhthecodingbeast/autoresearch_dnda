"""
Round -> data scale. The only place data scale is decided.

AGENT GUIDELINE
---------------
Call `prepare_round(ROUND)` once, at the top of every notebook, BEFORE touching any
dataloader or eval. It returns the data directory and repoints `prepare.DATA_DIR` at it.

How it works, and why it is safe:
  - Shards are downloaded ONCE into ~/.cache/autoresearch/data by `uv run prepare.py`.
  - This module never downloads, copies, moves or deletes a shard. It only creates a
    directory of SYMLINKS holding the round's slice, in ~/.cache/autoresearch/data_r<N>.
  - `prepare.list_parquet_files()` reads `prepare.DATA_DIR` at call time, so reassigning
    the attribute is enough. prepare.py itself is never edited.

Two rules this file enforces for you:
  1. TEST_SHARD is never symlinked into rounds 1 or 2. Round 3 only, at the final gate.
  2. Round slices are nested: R2 contains all of R1, R3 contains all of R2.

Self-check: `uv run python -m experiments._shared.data --check`
"""

import math
import os
import shutil
import sys

import prepare

# --- Project constants. Section 2 of research-program.md. Pin once, never per-run. ---

SHARD_BUDGET = 44          # "full dataset". Derive it from measured throughput (see below).
TEST_SHARD = 6541          # held out until the Round 3 gate. VAL_SHARD (6542) is prepare.py's.
SHARD_TOKENS = 61_000_000  # ~400B ClimbMix tokens / 6542 shards

ROUND_SCALE = {0: 0.25, 1: 0.25, 2: 0.5, 3: 1.0}   # section 1. the ONLY round axis.

# Every round trains for the same wall clock: prepare.TIME_BUDGET (300s, upstream's
# fixed 5 minutes). Rounds differ by data scale alone, so a result is never confounded
# by having had more training time than the run it is compared against.


def round_shards(round_idx, shard_budget=SHARD_BUDGET):
    """Train shard ids for a round. Nested: R1 subset of R2 subset of R3."""
    assert round_idx in ROUND_SCALE, f"unknown round {round_idx}"
    n = max(1, int(shard_budget * ROUND_SCALE[round_idx]))
    return list(range(n))


def required_shard_budget(throughput_tokens_per_s, headroom=1.25):
    """
    Shard count so that NO round exceeds 1.0 epoch. See research-program.md 1.2.

    A round that repeats data measures memorization, not the change under test, so its
    verdicts do not transfer upward. Every round trains for the same wall clock, so the
    binding case is the round with the LEAST data - R1 at 1/4 scale. Size for that and
    the larger rounds clear the gate automatically.

    Measure `throughput_tokens_per_s` at R0 from the baseline's own summary:
    total_tokens_M * 1e6 / training_seconds.

    Rounded up to a multiple of 4 so the 1/4 and 1/2 slices are exact and the nesting
    stays clean. The cost of the rounding is a few spare shards on disk.
    """
    consumed = throughput_tokens_per_s * prepare.TIME_BUDGET
    smallest_scale = min(ROUND_SCALE.values())
    raw = math.ceil(headroom * consumed / (SHARD_TOKENS * smallest_scale))
    return max(4, 4 * math.ceil(raw / 4))


def epochs_per_round(throughput_tokens_per_s, shard_budget=SHARD_BUDGET):
    """{round: epochs consumed}. Every value must be < 1.0 before a round opens."""
    consumed = throughput_tokens_per_s * prepare.TIME_BUDGET
    out = {}
    for r in ROUND_SCALE:
        fresh = len(round_shards(r, shard_budget)) * SHARD_TOKENS
        out[r] = consumed / fresh
    return out


def fetch_test_shard():
    """
    Download TEST_SHARD, which `prepare.py --num-shards N` never fetches.

    prepare.py pins only VAL_SHARD (6542) alongside the train slice, so the held-out
    test shard has to be fetched on its own. This reuses prepare.py's own downloader
    rather than editing it - prepare.py is read-only (research-program.md section 7).

    Run this once, before the Round 3 gate. Looking at the result earlier than that
    defeats the point of holding it out.
    """
    os.makedirs(os.path.join(prepare.CACHE_DIR, "data"), exist_ok=True)
    path = os.path.join(prepare.CACHE_DIR, "data", f"shard_{TEST_SHARD:05d}.parquet")
    if os.path.exists(path):
        print(f"TEST_SHARD {TEST_SHARD} already present")
        return path
    assert prepare.download_single_shard(TEST_SHARD), f"failed to download shard {TEST_SHARD}"
    return path


def prepare_round(round_idx, shard_budget=SHARD_BUDGET, verbose=True):
    """
    Point `prepare.DATA_DIR` at this round's shard slice. Returns the directory path.

    Exits with the exact download command if shards are missing, rather than training
    on a silently smaller dataset - which would be an invalid run that still produces a
    plausible-looking val_bpb.
    """
    real_dir = os.path.join(prepare.CACHE_DIR, "data")
    train_ids = round_shards(round_idx, shard_budget)
    needed = train_ids + [prepare.VAL_SHARD]
    if round_idx == 3:
        needed.append(TEST_SHARD)

    missing = [i for i in needed
               if not os.path.exists(os.path.join(real_dir, f"shard_{i:05d}.parquet"))]
    if missing:
        print(f"ERROR: {len(missing)} shards missing from {real_dir}", file=sys.stderr)
        print(f"  missing ids: {missing[:8]}{' ...' if len(missing) > 8 else ''}", file=sys.stderr)
        print(f"  fix: uv run prepare.py --num-shards {shard_budget}", file=sys.stderr)
        if TEST_SHARD in missing:
            print(f"  note: shard {TEST_SHARD} (TEST_SHARD) is not fetched by prepare.py, "
                  f"which only pins {prepare.VAL_SHARD}. Use:", file=sys.stderr)
            print(f"        uv run python -m experiments._shared.data --fetch-test-shard",
                  file=sys.stderr)
        sys.exit(1)

    # Rebuild the symlink view from scratch so a stale round dir can never leak shards
    # into a round that should not see them.
    round_dir = os.path.join(prepare.CACHE_DIR, f"data_r{round_idx}")
    if os.path.isdir(round_dir):
        shutil.rmtree(round_dir)       # only ever a directory of symlinks, never shard data
    os.makedirs(round_dir)

    link_ids = train_ids + [prepare.VAL_SHARD]
    if round_idx == 3:
        link_ids.append(TEST_SHARD)
    for i in link_ids:
        name = f"shard_{i:05d}.parquet"
        os.symlink(os.path.join(real_dir, name), os.path.join(round_dir, name))

    assert TEST_SHARD in link_ids or not os.path.exists(
        os.path.join(round_dir, f"shard_{TEST_SHARD:05d}.parquet")
    ), "TEST_SHARD leaked into a pre-R3 round"

    prepare.DATA_DIR = round_dir
    if verbose:
        scale = ROUND_SCALE[round_idx]
        tok = len(train_ids) * SHARD_TOKENS / 1e9
        print(f"Round {round_idx}: scale {scale:g} | {len(train_ids)} train shards "
              f"(~{tok:.2f}B tokens) | budget {prepare.TIME_BUDGET}s (fixed, all rounds)")
        print(f"  DATA_DIR -> {round_dir}")
        print(f"  val shard {prepare.VAL_SHARD}"
              + (f" + TEST shard {TEST_SHARD}" if round_idx == 3 else " (TEST shard withheld)"))
    return round_dir


def _check():
    # nesting
    q = SHARD_BUDGET // 4
    assert round_shards(1) == list(range(q)), round_shards(1)
    assert round_shards(2)[:q] == round_shards(1), "R2 must contain all of R1"
    assert round_shards(3)[:2 * q] == round_shards(2), "R3 must contain all of R2"
    assert len(round_shards(3)) == SHARD_BUDGET
    # a tiny budget must still yield at least one shard, not zero
    assert round_shards(1, shard_budget=2) == [0]
    # sizing: H100 reference throughput from the upstream summary (499.6M tokens / 300s)
    h100 = 499.6e6 / 300
    assert required_shard_budget(h100) == 44, required_shard_budget(h100)
    eps = epochs_per_round(h100, shard_budget=44)
    assert all(v < 1.0 for v in eps.values()), eps
    # R1 is the binding round now: least data, same budget as every other round
    assert max(eps, key=eps.get) in (0, 1), "the smallest-data round must be the tightest"
    # an undersized budget must be caught, not silently accepted
    assert any(v >= 1.0 for v in epochs_per_round(h100, shard_budget=8).values())
    print("data.py self-check passed")
    print("  H100 reference:", {k: round(v, 3) for k, v in eps.items()}, "epochs/round")


if __name__ == "__main__":
    if "--check" in sys.argv:
        _check()
    elif "--fetch-test-shard" in sys.argv:
        print(fetch_test_shard())
    else:
        print(prepare_round(int(sys.argv[1]) if len(sys.argv) > 1 else 1))
