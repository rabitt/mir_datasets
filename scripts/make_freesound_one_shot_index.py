import argparse
import json
import os

from mirdata.validate import md5

ONE_SHOT_DATASET_PATH = (
    "../mirdata/datasets/indexes/freesound_one_shot_percussive_sounds_index.json"
)


def make_one_shot_index(dataset_data_path):
    dataset_index = {"version": "1.0", "tracks": {}, "metadata": {}}

    for subfolder in os.listdir(dataset_data_path):
        if "one_shot_percussive_sounds" in subfolder:
            for subsubfolder in os.listdir(os.path.join(dataset_data_path, subfolder)):
                # avoid MACOS files
                if "." not in subsubfolder:
                    for track in os.listdir(
                        os.path.join(dataset_data_path, subfolder, subsubfolder)
                    ):
                        dataset_index["tracks"][track.replace(".wav", "")] = {
                            "audio": [
                                os.path.join(subfolder, subsubfolder, track),
                                md5(
                                    os.path.join(dataset_data_path, subfolder, subsubfolder, track)
                                ),
                            ],
                            "analysis": [
                                os.path.join(
                                    "analysis",
                                    subsubfolder,
                                    track.replace(".wav", "_analysis.json"),
                                ),
                                md5(
                                    os.path.join(
                                        dataset_data_path,
                                        "analysis",
                                        subsubfolder,
                                        track.replace(".wav", "_analysis.json"),
                                    )
                                ),
                            ],
                        }

    dataset_index["metadata"] = {
        "one-shot_metadata": [
            "licenses.txt",
            md5(os.path.join(dataset_data_path, "licenses.txt")),
        ]
    }

    with open(ONE_SHOT_DATASET_PATH, "w") as fhandle:
        json.dump(dataset_index, fhandle, indent=2)


def main(args):
    print("creating index...")
    make_one_shot_index(args.dataset_data_path)
    print("done!")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Make Freesound one-shot percussive sounds index file."
    )
    PARSER.add_argument(
        "dataset_data_path",
        type=str,
        help="Path to Freesound one-shot percussive data folder.",
    )

    main(PARSER.parse_args())
