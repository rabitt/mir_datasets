import argparse
import glob
import json
import os

from mirdata.validate import md5

HINDUSTANI_RHYTHM_INDEX_PATH = (
    "../mirdata/datasets/indexes/compmusic_hindustani_rhythm_full_index_1.0.json"
)


def make_compmusic_hindustani_rhythm_index(dataset_data_path):
    cmr_index = {
        "version": "1.0",
        "tracks": {},
    }
    idx = 0
    dataset_folder_name = "HMR_1.0"
    for rec in glob.glob(os.path.join(dataset_data_path, "audio", "*.wav")):
        filename = rec.split("/")[-1]
        idx = filename.split("_")[1]
        cmr_index["tracks"][idx] = {
            "audio": (
                os.path.join(dataset_folder_name, "audio", filename),
                md5(os.path.join(dataset_data_path, "audio", filename)),
            ),
            "beats": (
                os.path.join(
                    dataset_folder_name, "annotations", "beats", filename.replace(".wav", ".beats")
                ),
                md5(
                    os.path.join(
                        dataset_data_path,
                        "annotations",
                        "beats",
                        filename.replace(".wav", ".beats"),
                    )
                ),
            ),
            "meter": (
                os.path.join(
                    dataset_folder_name, "annotations", "meter", filename.replace(".wav", ".meter")
                ),
                md5(
                    os.path.join(
                        dataset_data_path,
                        "annotations",
                        "meter",
                        filename.replace(".wav", ".meter"),
                    )
                ),
            ),
        }

    with open(HINDUSTANI_RHYTHM_INDEX_PATH, "w") as fhandle:
        json.dump(cmr_index, fhandle, indent=2)


def main(args):
    print("creating index...")
    make_compmusic_hindustani_rhythm_index(args.dataset_data_path)
    print("done!")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make CompMusic Hindustani Rhythm index file.")
    PARSER.add_argument(
        "dataset_data_path", type=str, help="Path to CompMusic Hindustani Rhythm data folder."
    )

    main(PARSER.parse_args())
