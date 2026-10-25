"""Download the original public KION files, with hashes and no credentials."""
import hashlib
import json
from pathlib import Path
import zipfile
import shutil
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://raw.githubusercontent.com/irsafilo/KION_DATASET/main/"
EXPECTED_ARCHIVE_SHA256 = "50fc5b7db78cbda4c3d0ee5bb5bea4c416cff41e8af58aab0b661d0d3e9dfa9b"


def download():
    dest = ROOT / "data/raw"
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {}
    # The separate interactions.csv.zip in upstream main is a truncated 50 MiB
    # CSV. data_original.zip contains the complete 180,032,258-byte CSV instead.
    for name, remote in [("data_original.zip", "data_original.zip")]:
        path = dest / name
        if not path.exists():
            tmp = path.with_suffix(path.suffix + ".part")
            with requests.get(BASE + remote, stream=True, timeout=120) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        f.write(chunk)
            tmp.rename(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != EXPECTED_ARCHIVE_SHA256:
            raise ValueError("KION archive differs from the audited source version")
        manifest[name] = {"url": BASE + remote, "sha256": digest, "bytes": path.stat().st_size}
        print(name, path.stat().st_size, flush=True)
    with zipfile.ZipFile(dest / "data_original.zip") as z:
        for name in ["interactions.csv", "items.csv", "users.csv"]:
            member = "data_original/"+name
            info = z.getinfo(member)
            path = dest/name
            if not path.exists() or path.stat().st_size != info.file_size:
                tmp = path.with_suffix(".csv.part")
                with z.open(member) as src, tmp.open("wb") as dst:
                    shutil.copyfileobj(src,dst)
                tmp.replace(path)
            manifest[name] = {"archive_member":member,"bytes":path.stat().st_size,
                              "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    if manifest["interactions.csv"]["bytes"] != 180032258:
        raise ValueError("Unexpected KION CSV version; audit before sampling users")
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    download()
