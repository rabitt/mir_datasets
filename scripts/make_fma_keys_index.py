import argparse
import glob
import json
import os
import csv
from mirdata.validate import md5

FMA_KEYS_INDEX_PATH = "mirdata/datasets/indexes/fma_keys_index_1.0.json"


def track_to_dict(t, fma_keys_data_path):
    track_id = "{:06d}".format(int(t["track_id"]))
    audio_path = track_id[:3] + "/" + track_id + ".mp3"

    return {
        "audio": [audio_path, md5(os.path.join(fma_keys_data_path, audio_path))],
    }


def make_fma_keys_index(fma_keys_data_path):
    metadata_file = os.path.join(fma_keys_data_path, "fma_keys_metadata.csv")

    with open(metadata_file) as f:
        tracks = {
            t["track_id"]: track_to_dict(t, fma_keys_data_path)
            for t in csv.DictReader(f)
        }

    # top-key level version
    fma_keys_index = {
        "version": "1.0",
        "tracks": tracks,
        "metadata": {
            "fma_keys_metadata": ("fma_keys_metadata.csv", md5(metadata_file))
        },
    }

    with open(FMA_KEYS_INDEX_PATH, "w") as fhandle:
        json.dump(fma_keys_index, fhandle, indent=2)


def main(args):
    make_fma_keys_index(args.fma_keys_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make fma_keys index file.")
    PARSER.add_argument(
        "fma_keys_data_path", type=str, help="Path to fma_keys data folder."
    )

    main(PARSER.parse_args())