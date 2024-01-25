import argparse
import hashlib
import json
import os
import glob
from mirdata.validate import md5

ROCK_RHYTHM_INDEX_PATH = "../mirdata/datasets/indexes/rock_full_index_1.0.json"

def make_rock_rhythm_index(dataset_data_path):
    cmr_index = {
        "version": "1.0",
        "tracks": {},
    }
    
    dataset_folder_name = "R_1.0"  # Update this folder name if needed
    for root, dirs, files in os.walk(dataset_data_path):
        for filename in files:
            if filename.endswith(".flac"):
                subfolder = os.path.basename(root)
                idx = filename.split(".")[0]
                cmr_index["tracks"][idx] = {
                    "audio": (
                        os.path.join(dataset_folder_name, "audio", filename),
                        md5(os.path.join(root, filename)),
                    ),
                    "beats": (
                        os.path.join(dataset_folder_name, "annotations", "beats", filename.replace(".flac", ".beats")),
                        md5(os.path.join(dataset_data_path, "annotations", "beats",  filename.replace(".flac", ".beats"))),
                    ),
                }

    with open(ROCK_RHYTHM_INDEX_PATH, "w") as fhandle:
        json.dump(cmr_index, fhandle, indent=2)

def main(args):
    print("creating index...")
    make_rock_rhythm_index(args.dataset_data_path)
    print("done!")

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make Rock Rhythm index file.")
    PARSER.add_argument(
        "dataset_data_path", type=str, help="Path to Rock Rhythm data folder."
    )

    main(PARSER.parse_args())