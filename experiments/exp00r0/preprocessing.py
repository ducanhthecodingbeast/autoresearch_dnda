"""
exp00r0 — data prep. Round 0 = 1/4 scale (same slice as Round 1).

AGENT GUIDELINE
---------------
Every experiment folder has one of these. It does exactly one job: pin the data scale
for this experiment's round. All the logic lives in `experiments/_shared/data.py` — this
file only names the round, so that the scale is visible in the folder and cannot drift
from what the report claims.

Round 0 uses the Round 1 slice deliberately: sigma_0 must be measured at the same data
scale the Round 1 verdicts will be judged against, or the noise band is wrong.

Run standalone to check the data is in place:
    uv run python experiments/exp00r0/preprocessing.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments._shared.data import prepare_round

ROUND = 0

if __name__ == "__main__":
    prepare_round(ROUND)
