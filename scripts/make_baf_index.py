import argparse
import glob
import hashlib
import json
import os
from pathlib import Path

BAF_INDEX_PATH = "../mirdata/datasets/indexes/baf_index_1.0.json"


def md5(file_path: str) -> str:
    """Get md5 hash of a file.

    Args:
        file_path (str): File path.
    Returns:
        md5_hash (str): md5 hash of data in file_path
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as fhandle:
        for chunk in iter(lambda: fhandle.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def make_baf_index(data_path: str) -> None:
    """Create baf index.

    Args:
        data_path (str): Dataset path.
    Returns:
        None
    """
    metadata = {}
    tracks = {}

    queries_dir = os.path.join(data_path, "queries/*.wav")
    for filepath in sorted(glob.glob(queries_dir)):
        file_id = Path(filepath).stem
        tracks[file_id] = {"audio": (os.path.join(*filepath.split("/")[-2:]), md5(filepath))}

    queries_info_path = os.path.join(data_path, "queries_info.csv")
    xannotations_path = os.path.join(data_path, "cross_annotations.csv")
    metadata = {
        "queries_info": (
            os.path.join(*queries_info_path.split("/")[-1:]),
            md5(queries_info_path),
        ),
        "cross_annotations": (
            os.path.join(*xannotations_path.split("/")[-1:]),
            md5(xannotations_path),
        ),
    }
    baf_index = {"version": "1.0", "tracks": tracks, "metadata": metadata}
    with open(BAF_INDEX_PATH, "w") as fhandle:
        json.dump(baf_index, fhandle, indent=2)


def main(args):
    make_baf_index(args.baf_data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make baf index file.")
    parser.add_argument("baf_data_path", type=str, help="Path to baf data folder.")
    main(parser.parse_args())
