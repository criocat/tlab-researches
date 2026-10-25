"""Public, pinned Qwen snapshot for the keyless CPU/GPU PromptFE backend.

Downloads only into the project cache and verifies the published LFS SHA-256.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import time
import requests

ROOT=Path(__file__).resolve().parents[1]
REPO="Qwen/Qwen2.5-Coder-1.5B-Instruct"
REVISION="2e1fd397ee46e1388853d2af2c993145b0f1098a"
WEIGHT_SHA256="c1b9b30e907950516ba3c646bdf570d8084c25a6410a0cdca80cf04b11bc13a8"
WEIGHT_SIZE=3087467144


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        while chunk:=f.read(1024*1024): h.update(chunk)
    return h.hexdigest()


def main():
    dest=ROOT/".cache/models/qwen2.5-coder-1.5b-instruct"; dest.mkdir(parents=True,exist_ok=True)
    base=f"https://huggingface.co/{REPO}/resolve/{REVISION}/"
    for name in ["config.json","generation_config.json","tokenizer.json","tokenizer_config.json","vocab.json","merges.txt","LICENSE"]:
        p=dest/name
        if not p.exists():
            r=requests.get(base+name,timeout=120); r.raise_for_status(); p.write_bytes(r.content)
    target=dest/"model.safetensors"
    if target.exists():
        if target.stat().st_size!=WEIGHT_SIZE or digest(target)!=WEIGHT_SHA256:
            raise ValueError("Existing proposer weights fail checksum")
    else:
        parts=dest/"parts"; parts.mkdir(exist_ok=True)
        size=64*1024*1024
        jobs=[(lo,min(lo+size,WEIGHT_SIZE)-1,parts/f"{lo:012d}.part") for lo in range(0,WEIGHT_SIZE,size)]
        def fetch(job):
            lo,hi,path=job
            if path.exists() and path.stat().st_size==hi-lo+1:return path
            for attempt in range(3):
                try:
                    with requests.get(base+"model.safetensors",headers={"Range":f"bytes={lo}-{hi}"},stream=True,timeout=120) as r:
                        r.raise_for_status()
                        if r.status_code!=206:raise RuntimeError("Expected partial-content response")
                        count=0
                        with path.open("wb") as f:
                            for chunk in r.iter_content(1024*1024):
                                count+=len(chunk)
                                if count>hi-lo+1:raise RuntimeError("Server returned excessive range")
                                f.write(chunk)
                    if path.stat().st_size!=hi-lo+1:raise RuntimeError("Incomplete range")
                    return path
                except Exception:
                    if attempt==2:raise RuntimeError(f"Public model range {lo}-{hi} failed") from None
                    time.sleep(2**attempt)
        with ThreadPoolExecutor(12) as executor:
            futures=[executor.submit(fetch,job) for job in jobs]
            for n,future in enumerate(as_completed(futures),1):
                future.result(); print("Proposer ranges",n,"/",len(jobs),flush=True)
        tmp=dest/"model.safetensors.tmp"; h=hashlib.sha256()
        with tmp.open("wb") as out:
            for _,_,part in jobs:
                with part.open("rb") as f:
                    while chunk:=f.read(1024*1024):h.update(chunk);out.write(chunk)
        if h.hexdigest()!=WEIGHT_SHA256:raise ValueError("Proposer SHA-256 mismatch")
        tmp.replace(target)
        # Only scratch ranges created above are deleted after the final hash.
        for _,_,part in jobs:part.unlink()
    manifest={"repo":REPO,"revision":REVISION,"weight_sha256":WEIGHT_SHA256,"weight_bytes":WEIGHT_SIZE,
              "license":"Apache-2.0","files":{p.name:digest(p) for p in dest.iterdir() if p.is_file() and p.name!="source.json"}}
    (dest/"source.json").write_text(json.dumps(manifest,indent=2))
    print("Verified proposer snapshot",dest,flush=True)


if __name__=="__main__":main()
