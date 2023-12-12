import argparse
import hashlib
import json
import os

haydn_quartets_INDEX_PATH = "../mirdata/datasets/indexes/haydn_op20_index.json"


def md5(file_path):
    """Get md5 hash of a file.

    Parameters
    ----------
    file_path: str
        File path.
    Returns
    -------
    md5_hash: str
        md5 hash of data in file_path
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as fhandle:
        for chunk in iter(lambda: fhandle.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def make_haydn_op20_index(data_path):
    movements = [
        os.path.join(data_path, "op20n" + str(number) + "-0" + str(movement))
        for number in range(1, 7)
        for movement in range(1, 5)
    ]
    haydn_op20_tracks = {}
    for track_id, m in enumerate(movements):
        annotations = os.path.join(m + ".hrm")
        haydn_op20_tracks[track_id] = {
            "annotations": (annotations.replace(data_path + "/", ""), md5(annotations))
        }
        track_id += 1
    haydn_op20_index = {"version": "1.3", "tracks": haydn_op20_tracks}
    with open(haydn_quartets_INDEX_PATH, "w") as fhandle:
        json.dump(haydn_op20_index, fhandle, indent=2)


def main(args):
    make_haydn_op20_index(args.haydn_op20_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make haydn op20 index file.")
    PARSER.add_argument("haydn_op20_data_path", type=str, help="Path to haydn op20 data folder.")
    main(PARSER.parse_args())
