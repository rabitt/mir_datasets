import argparse
import json
import os

from mirdata.validate import md5

TONAS_INDEX_PATH = "../mirdata/datasets/indexes/tonas_index.json"


def make_tonas_index(dataset_data_path):
    tonas_index = {"version": "1.0", "tracks": {}}

    for style in os.listdir(os.path.join(dataset_data_path)):
        if "." not in style:
            for track in os.listdir(os.path.join(dataset_data_path, style)):
                if ".wav" in track:
                    # Declare track attributes
                    index = track.replace(".wav", "")
                    f0_path = index + ".f0.Corrected"
                    notes_path = index + ".notes.Corrected"

                    tonas_index["tracks"][index] = {
                        "audio": [
                            os.path.join(style, track),
                            md5(os.path.join(dataset_data_path, style, track)),
                        ],
                        "f0": [
                            os.path.join(style, f0_path),
                            md5(os.path.join(dataset_data_path, style, f0_path)),
                        ],
                        "notes": [
                            os.path.join(style, notes_path),
                            md5(os.path.join(dataset_data_path, style, notes_path)),
                        ],
                    }
    tonas_index["metadata"] = {
        "TONAS-Metadata": [
            "TONAS-Metadata.txt",
            md5(os.path.join(dataset_data_path, "TONAS-Metadata.txt")),
        ]
    }

    with open(TONAS_INDEX_PATH, "w") as fhandle:
        json.dump(tonas_index, fhandle, indent=2)


def main(args):
    print("creating index...")
    make_tonas_index(args.dataset_data_path)
    print("done!")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make TONAS index file.")
    PARSER.add_argument(
        "dataset_data_path",
        type=str,
        help="Path to TONAS data folder.",
    )

    main(PARSER.parse_args())
