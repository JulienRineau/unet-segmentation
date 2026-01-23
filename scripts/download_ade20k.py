import argparse
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset import discover_ade20k_test_root, discover_ade20k_trainval_root


TRAINVAL_URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
TEST_URL = "http://data.csail.mit.edu/places/ADEchallenge/release_test.zip"
TRAINVAL_ROOT = "ADEChallengeData2016"
TEST_ROOT = "release_test"


def download_with_progress(url: str, dest_path: Path, force: bool) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and not force:
        print(f"Skip download (exists): {dest_path}")
        return

    tmp_path = dest_path.with_suffix(dest_path.suffix + ".partial")
    if tmp_path.exists():
        tmp_path.unlink()

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        with open(tmp_path, "wb") as f, tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            desc=dest_path.name,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))

    tmp_path.replace(dest_path)


def extract_zip(zip_path: Path, dest_dir: Path, expected_root: str, force: bool) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted_root = dest_dir / expected_root
    if extracted_root.exists() and not force:
        print(f"Skip extract (exists): {extracted_root}")
        return

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        with tqdm(total=len(members), desc=f"extract:{zip_path.name}") as pbar:
            for member in members:
                zf.extract(member, path=dest_dir)
                pbar.update(1)


def validate_layout(data_dir: Path, want_trainval: bool, want_test: bool) -> None:
    if want_trainval:
        discover_ade20k_trainval_root(data_dir)
    if want_test:
        discover_ade20k_test_root(data_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract ADE20K official train/val and test zips.",
        epilog=(
            "Expected layout under --data-dir:\n"
            "  ADEChallengeData2016/images/{training,validation}\n"
            "  ADEChallengeData2016/annotations/{training,validation}\n"
            "  release_test/testing"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--trainval", action="store_true", help="Download ADEChallengeData2016.zip")
    parser.add_argument("--test", action="store_true", help="Download release_test.zip")
    parser.add_argument("--all", action="store_true", help="Download both train/val and test")
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    want_trainval = args.trainval or args.all
    want_test = args.test or args.all
    if not want_trainval and not want_test:
        want_trainval = True
        want_test = True

    data_dir = Path(args.data_dir)
    zip_dir = data_dir / "zips"
    trainval_zip = zip_dir / "ADEChallengeData2016.zip"
    test_zip = zip_dir / "release_test.zip"

    if want_trainval:
        download_with_progress(TRAINVAL_URL, trainval_zip, args.force)
        extract_zip(trainval_zip, data_dir, TRAINVAL_ROOT, args.force)

    if want_test:
        download_with_progress(TEST_URL, test_zip, args.force)
        extract_zip(test_zip, data_dir, TEST_ROOT, args.force)

    validate_layout(data_dir, want_trainval=want_trainval, want_test=want_test)
    print("ADE20K download/extract complete.")


if __name__ == "__main__":
    main()
