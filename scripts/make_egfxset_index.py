import argparse
import json
import os

import pandas as pd

from mirdata.validate import md5

EGFXSET_INDEX_PATH = "mirdata/datasets/indexes/egfxset_index_{}.json"


def make_egfxset_index(egfxset_data_path: str, version: str) -> None:
    assert version == "1"

    fx_dirs = [
        os.path.join(d)
        for d in os.listdir(egfxset_data_path)
        if os.path.isdir(os.path.join(egfxset_data_path, d))
    ]

    fx_track_ids = {
        fx: sorted(
            [
                os.path.join(pickup, t)
                for pickup in os.listdir(os.path.join(egfxset_data_path, fx))
                if pickup != ".DS_Store"
                for t in os.listdir(os.path.join(egfxset_data_path, fx, pickup))
                if t != ".DS_Store"
            ]
        )
        for fx in fx_dirs
    }

    metadata_path = os.path.join(egfxset_data_path, f"egfxset_metadata.csv")

    # top-key level tracks
    index_tracks = {
        "{}_{}".format(fx, track_id[:-4]): {
            "audio": (
                f"{fx}/{track_id}",
                md5(os.path.join(egfxset_data_path, f"{fx}", f"{track_id}")),
            )
        }
        for fx, track_ids in fx_track_ids.items()
        for track_id in track_ids
    }

    # top-key level version
    egfxset_index = {
        "version": version,
        "tracks": index_tracks,
        "metadata": {"egfxset_metadata": ("egfxset_metadata.csv", md5(metadata_path))},
    }

    with open(EGFXSET_INDEX_PATH.format(version), "w") as fhandle:
        json.dump(egfxset_index, fhandle, indent=2)


def main(args):
    make_egfxset_index(args.egfxset_data_path, args.version)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make EGFxSet index file.")
    PARSER.add_argument("egfxset_data_path", type=str, help="Path to EGFxSet data folder.")
    PARSER.add_argument("version", type=str, help="index version")

    main(PARSER.parse_args())
