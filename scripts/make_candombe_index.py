import argparse
import glob
import json
import os

from mirdata.validate import md5

DATASET_INDEX_PATH = "../mirdata/datasets/indexes/candombe_index_1.0.json"


def make_dataset_index(dataset_data_path):
    annotation_files = glob.glob(
        os.path.join(dataset_data_path + "candombe_annotations/with_bar_number", "*.csv")
    )

    track_ids = sorted([".".join(os.path.basename(f).split(".")[:-1]) for f in annotation_files])

    # top-key level tracks
    index_tracks = {}
    for track_id in track_ids:
        audio_checksum = md5(
            os.path.join(dataset_data_path, "candombe_audio/{}.flac".format(track_id))
        )
        annotation_checksum = md5(
            os.path.join(
                dataset_data_path, "candombe_annotations/with_bar_number/{}.csv".format(track_id)
            )
        )

        index_tracks[track_id] = {
            "audio": ("candombe_audio/{}.flac".format(track_id), audio_checksum),
            "beats": (
                "candombe_annotations/with_bar_number/{}.csv".format(track_id),
                annotation_checksum,
            ),
        }

    # top-key level version
    dataset_index = {"version": 1.0}

    # combine all in dataset index
    dataset_index.update({"tracks": index_tracks})

    with open(DATASET_INDEX_PATH, "w") as fhandle:
        json.dump(dataset_index, fhandle, indent=2)


def main(args):
    make_dataset_index(args.dataset_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make dataset index file.")
    PARSER.add_argument("dataset_data_path", type=str, help="Path to dataset data folder.")

    main(PARSER.parse_args())
