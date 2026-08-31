from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


MODEL_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
MODEL_MD5 = "e3aa4aaa15225a33ec84f9f4bc47e500"


def md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - upstream checksum used for artifact identity
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the IOPaint big-LaMa TorchScript model.")
    parser.add_argument("--output", type=Path, default=Path("models/big-lama.pt"))
    args = parser.parse_args()
    output = args.output
    if output.is_file() and md5(output) == MODEL_MD5:
        print(f"LaMa model already verified: {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    print(f"Downloading {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, partial)  # noqa: S310 - fixed upstream URL
    actual = md5(partial)
    if actual != MODEL_MD5:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"LaMa checksum mismatch: expected {MODEL_MD5}, got {actual}")
    partial.replace(output)
    print(f"Verified LaMa model: {output} ({actual})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
