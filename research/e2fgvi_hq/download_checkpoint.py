from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import gdown

from research.e2fgvi_hq.smoke_test import DEFAULT_CHECKPOINT


FILE_ID = "10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3"
EXPECTED_SHA256 = "afff989d41205598a79ce24630b9c83af4b0a06f45b137979a25937d94c121a5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the official E2FGVI-HQ checkpoint")
    parser.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.is_file():
        result = gdown.download(id=FILE_ID, output=str(args.output), quiet=False)
        if result is None:
            raise RuntimeError("gdown did not return a downloaded checkpoint path")
    actual = sha256(args.output)
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"Checkpoint checksum mismatch: {actual}")
    print(f"Verified {args.output} sha256={actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
