"""
exp<NN>r<R> — data prep. Round <R> = <scale> of the dataset.

AGENT GUIDELINE
---------------
Every experiment folder has one of these. It does exactly one job: pin the data scale
for this experiment's round. All the logic lives in `experiments/_shared/data.py` — this
file only names the round, so the scale is visible in the folder and cannot drift from
what the report claims.

Set ROUND to this experiment's round and change nothing else. If you find yourself
wanting to add data logic here, it belongs in _shared/data.py, where it applies to
every experiment in the round instead of just this one.

Run standalone to check the data is in place before launching the notebook:
    uv run python experiments/exp<NN>r<R>/preprocessing.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments._shared.data import prepare_round

ROUND = 1          # <-- set to this experiment's round

if __name__ == "__main__":
    prepare_round(ROUND)
