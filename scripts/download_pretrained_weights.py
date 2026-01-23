import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import requests
from tqdm import tqdm

from backbones import DEFAULT_ENCODER_NAME, default_weights_dir, get_encoder_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download pretrained encoder weights.")
    parser.add_argument("--encoder-name", type=str, default=DEFAULT_ENCODER_NAME)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_valid_file(path: Path, hash_prefix: str) -> bool:
    if not path.is_file():
        return False
    digest = sha256_file(path)
    return digest.startswith(hash_prefix)


def download_with_progress(url: str, dest: Path, hash_prefix: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total = response.headers.get("Content-Length")
    total_size = int(total) if total and total.isdigit() else None

    hasher = hashlib.sha256()
    with tmp_path.open("wb") as handle, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=dest.name,
    ) as progress:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            handle.write(chunk)
            hasher.update(chunk)
            progress.update(len(chunk))

    digest = hasher.hexdigest()
    if hash_prefix and not digest.startswith(hash_prefix):
        tmp_path.unlink()
        raise ValueError(
            f"Hash mismatch for {dest.name}. Expected prefix {hash_prefix}, got {digest}."
        )

    tmp_path.replace(dest)


def main() -> None:
    args = parse_args()
    spec = get_encoder_spec(args.encoder_name)
    output_dir = Path(args.output_dir) if args.output_dir else default_weights_dir()
    dest = output_dir / Path(spec.url).name

    if dest.exists() and not args.force:
        if is_valid_file(dest, spec.hash_prefix):
            print(f"Found valid weights at {dest}. Skipping download.")
            return
        print(f"Existing file failed validation: {dest}. Re-downloading.")

    download_with_progress(spec.url, dest, spec.hash_prefix)
    print(f"Saved weights to {dest}")


if __name__ == "__main__":
    main()
