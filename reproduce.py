"""Run one seeded primary experiment in an isolated output directory."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task",choices=["task1","task2"])
    parser.add_argument("--out",default="artifacts/reproduced")
    parser.add_argument("--dry-run",action="store_true",help="Print commands without downloads, training or writes")
    parser.add_argument("--device",choices=["cpu","auto"],default="cpu",help="CPU matches the primary reference; auto allows CUDA")
    args=parser.parse_args()
    output=(ROOT/args.out).resolve()
    if not output.is_relative_to(ROOT):parser.error("Output must stay inside this repository")
    relative=output.relative_to(ROOT).as_posix()
    if args.task=="task1":
        modules=[
            ["shared.download"],
            ["task1.run","--users","50000","--queries","1500","--iterations","350","--seeds","42","--out",relative+"/task1"],
            ["task1.promptfe_run","--seed","42","--syntax","infix","--proposals","12","--iterations","350",
             "--feature-dir",relative+"/task1","--replay","replay/feature_search_cpu.json","--out",relative+"/task1_promptfe"],
            ["shared.analyze","--task","task1","--root",relative],
            ["shared.report","--task","task1","--root",relative,"--output",relative+"/task1_results.md"],
        ]
    else:
        modules=[
            ["shared.download"],
            ["task2.run","--users","50000","--max-train","100000","--max-eval","2000","--epochs","8",
             "--reward-epochs","5","--rl-steps","200","--seed","42","--out",relative+"/task2"],
            ["task2.control","--out",relative+"/task2"],
            ["shared.analyze","--task","task2","--root",relative],
            ["shared.report","--task","task2","--root",relative,"--output",relative+"/task2_results.md"],
        ]
    environment=dict(os.environ,PYTHONHASHSEED="42")
    if args.device=="cpu":environment["CUDA_VISIBLE_DEVICES"]=""
    for module in modules:
        command=[sys.executable,"-m",*module]
        print(shlex.join(command),flush=True)
        if not args.dry_run:subprocess.run(command,cwd=ROOT,env=environment,check=True)


if __name__=="__main__":main()
