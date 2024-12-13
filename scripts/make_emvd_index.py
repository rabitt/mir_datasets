import argparse
import json
import os
import csv
from mirdata.validate import md5

EMVD_PATH = "mirdata/datasets/indexes/emvd_index_{}.json"


def make_emvd_index(emvd_data_path: str, version: str) -> None:
    assert version == "1.0"

    audio_dir = "audio"
    audio_path = os.path.join(emvd_data_path, audio_dir)

    track_ids = sorted(
        [
            os.path.splitext(row["file_name"])[0]
            for row in csv.DictReader(
                open(os.path.join(emvd_data_path, "metadata_files.csv"))
            )
        ]
    )

    index_tracks = {
        track_id: {
            "audio": (
                f"{audio_dir}/{track_id}.wav",
                md5(os.path.join(audio_path, f"{track_id}.wav")),
            ),
        }
        for track_id in track_ids
    }

    emvd_index = {
        "version": version,
        "tracks": index_tracks,
        "metadata": {
            "metadata_files": (
                "metadata_files.csv",
                md5(os.path.join(emvd_data_path, "metadata_files.csv")),
            ),
            "metadata_singers": (
                "metadata_singers.csv",
                md5(os.path.join(emvd_data_path, "metadata_singers.csv")),
            ),
            "split_kfolds": (
                "split_kfolds.csv",
                md5(os.path.join(emvd_data_path, "split_kfolds.csv")),
            ),
        },
    }

    with open(EMVD_PATH.format(version), "w") as fhandle:
        json.dump(emvd_index, fhandle, indent=2)


def main(args):
    make_emvd_index(args.emvd_data_path, args.version)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make EMVD index file.")
    PARSER.add_argument(
        "emvd_data_path",
        type=str,
        help="Path to EMVD dataset folder.",
    )
    PARSER.add_argument("version", type=str, help="index version")

    main(PARSER.parse_args())
