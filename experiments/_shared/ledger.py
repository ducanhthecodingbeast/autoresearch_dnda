"""
The decision rule and the results ledger.

AGENT GUIDELINE
---------------
You do NOT decide keeps by eye. Call `verdict(...)` and record what it returns.

The upstream loop keeps any run with a lower val_bpb. Over ~100 overnight runs that is
a greedy walk on seed noise, and it manufactures wins that vanish on a rerun. Here a
keep must clear 2 sigma, where sigma is measured from the round baseline's own seed
repeats (see `harness.run_seeds`).

Usage:
    from experiments._shared.ledger import verdict, append_result
    v = verdict(val_bpb=0.9940, baseline_bpb=1.0120, sigma=0.0055, simpler=False)
    append_result(exp_id="exp01r1", round_idx=1, commit="a1b2c3d", metrics=m, verdict=v,
                  description="matrix_lr 0.04 -> 0.02")

Self-check: `uv run python -m experiments._shared.ledger --check`
"""

import os
import subprocess
import sys
from dataclasses import dataclass

LEDGER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results.tsv")

HEADER = ["exp_id", "round", "commit", "val_bpb", "sigma_mult", "memory_gb",
          "tok_per_s", "epoch", "status", "description"]


@dataclass
class Verdict:
    status: str        # keep | discard | retest | crash
    sigma_mult: float  # delta / sigma - the number the decision was actually made on
    delta: float
    reason: str

    def __str__(self):
        return f"{self.status.upper()} (delta={self.delta:+.6f}, {self.sigma_mult:+.1f} sigma) - {self.reason}"


def verdict(val_bpb, baseline_bpb, sigma, simpler=False, crashed=False):
    """
    Section 4 of research-program.md, as code.

    `simpler` means the change made the code strictly smaller or removed a concept.
    A result inside the noise band with simpler code is a KEEP - that is the
    simplification win, and it is the only case where a non-improvement advances.
    """
    if crashed:
        return Verdict("crash", 0.0, 0.0, "run crashed or OOMed")
    assert sigma > 0, "sigma must be measured before any verdict (3 seed repeats of the round baseline)"
    delta = val_bpb - baseline_bpb
    mult = delta / sigma

    if mult < -2.0:
        return Verdict("keep", mult, delta, "clears the 2 sigma band")
    if mult < -1.0:
        return Verdict("retest", mult, delta,
                       "between 1 and 2 sigma - rerun at 2 more seeds, keep only if mean < -1.5 sigma")
    if mult <= 1.0:
        if simpler:
            return Verdict("keep", mult, delta, "inside the noise band but the code is simpler")
        return Verdict("discard", mult, delta, "inside the noise band - null result")
    return Verdict("discard", mult, delta, "worse than baseline")


def retest_verdict(mean_bpb, baseline_bpb, sigma):
    """Resolve a `retest` after the 2 extra seeds. Threshold is -1.5 sigma on the mean."""
    delta = mean_bpb - baseline_bpb
    mult = delta / sigma
    if mult < -1.5:
        return Verdict("keep", mult, delta, "retest mean clears -1.5 sigma over 5 seeds")
    return Verdict("discard", mult, delta, "retest mean did not clear -1.5 sigma")


def git_commit_short():
    try:
        return subprocess.run(["git", "rev-parse", "--short=7", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def append_result(exp_id, round_idx, metrics, verdict, description, commit=None,
                  path=LEDGER_PATH):
    """
    Append one row to results.tsv. Tab-separated: descriptions contain commas.

    results.tsv stays UNTRACKED by git (see .gitignore). It is the cross-round ledger
    and would otherwise conflict on every branch reset.
    """
    assert "\t" not in description, "description cannot contain a tab"
    if commit is None:
        commit = git_commit_short()
    row = [
        exp_id,
        str(round_idx),
        commit,
        f"{metrics.get('val_bpb', 0.0):.6f}",
        f"{verdict.sigma_mult:+.2f}",
        f"{metrics.get('peak_vram_mb', 0.0) / 1024:.1f}",
        str(int(metrics.get("tok_per_s", 0))),
        f"{metrics.get('epoch', 0):.3f}",
        verdict.status,
        description,
    ]
    assert len(row) == len(HEADER)
    exists = os.path.exists(path)
    with open(path, "a") as f:
        if not exists:
            f.write("\t".join(HEADER) + "\n")
        f.write("\t".join(row) + "\n")
    return row


def _check():
    s = 0.0055
    base = 1.012000
    # clear win
    assert verdict(base - 3.3 * s, base, s).status == "keep"
    # borderline -> retest, never a silent keep
    assert verdict(base - 1.5 * s, base, s).status == "retest"
    # inside the band
    assert verdict(base - 0.5 * s, base, s).status == "discard"
    # ...unless the code got simpler
    assert verdict(base - 0.5 * s, base, s, simpler=True).status == "keep"
    # a tiny improvement is NOT a keep just because it is negative (the upstream bug)
    assert verdict(base - 0.000001, base, s).status == "discard"
    # worse
    assert verdict(base + 3 * s, base, s).status == "discard"
    # crash short-circuits before sigma is needed
    assert verdict(0.0, base, 0.0, crashed=True).status == "crash"
    # boundaries. tested just inside/outside rather than exactly on, because an exact
    # multiple of sigma lands on either side depending on float rounding.
    assert verdict(base - 2.01 * s, base, s).status == "keep"
    assert verdict(base - 1.99 * s, base, s).status == "retest"
    assert verdict(base - 1.01 * s, base, s).status == "retest"
    assert verdict(base - 0.99 * s, base, s).status == "discard"
    # retest resolution
    assert retest_verdict(base - 1.6 * s, base, s).status == "keep"
    assert retest_verdict(base - 1.4 * s, base, s).status == "discard"
    # sigma must exist
    try:
        verdict(base, base, 0.0)
        raise AssertionError("sigma=0 must be rejected")
    except AssertionError as e:
        assert "sigma must be measured" in str(e)
    print("ledger.py self-check passed")


if __name__ == "__main__":
    if "--check" in sys.argv:
        _check()
    else:
        print(__doc__)
