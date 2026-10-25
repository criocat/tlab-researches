"""Frozen learned graph-feature replication on the three existing earlier weeks."""
import argparse
import subprocess
import sys
import pandas as pd
from shared.data import ROOT, save_json


def main():
    out = ROOT/"artifacts/task1_relational_temporal"; out.mkdir(exist_ok=True)
    dates = ["2021-07-26", "2021-08-02", "2021-08-09"]
    save_json(out/"study_protocol.json", {"group": "all", "dates": dates, "seeds": [42,43,44],
        "purpose": "Exploratory temporal replication of frozen graph features",
        "no_reselection": True, "warning": "Overlapping users and already inspected earlier windows; not new confirmatory data"})
    for date in dates:
        subprocess.run([sys.executable, "-m", "task1.relational", "--fixed-group", "all", "--reuse-baseline",
            "--source", f"artifacts/task1_temporal/test_{date}", "--out", f"artifacts/task1_relational_temporal/test_{date}"],
            cwd=ROOT, check=True)


if __name__ == "__main__": main()
