import argparse
import json
import os

from mirdata.validate import md5

IDMT_SMT_AUDIO_EFFECTS_INDEX_PATH = "../mirdata/datasets/indexes/idmt_smt_audio_effects_index.json"

idmt_smt_audio_effects_index = {"version": "1.0", "tracks": {}, "metadata": {}}


def make_isafx_index(idmt_smt_audio_effects_data_path):
    # Get the audio files
    for dirpath, dirnames, filenames in os.walk(idmt_smt_audio_effects_data_path):
        for filename in filenames:
            if filename.endswith(".wav"):
                file_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(file_path, idmt_smt_audio_effects_data_path)
                track_key = filename.replace(".wav", "")

                idmt_smt_audio_effects_index["tracks"][track_key] = {
                    "audio": [rel_path, md5(file_path)]
                }

    # Get the metadata from the XML files
    for dirpath, dirnames, filenames in os.walk(idmt_smt_audio_effects_data_path):
        for filename in filenames:
            if filename.endswith(".xml"):
                file_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(file_path, idmt_smt_audio_effects_data_path)

                metadata_key = os.path.basename(filename.replace(".xml", ""))
                idmt_smt_audio_effects_index["metadata"][metadata_key] = [
                    rel_path,
                    md5(file_path),
                ]

    # Save the index to file
    if os.path.exists(IDMT_SMT_AUDIO_EFFECTS_INDEX_PATH):
        print("Index file already exists. Overwriting...")

    with open(IDMT_SMT_AUDIO_EFFECTS_INDEX_PATH, "w") as fhandle:
        json.dump(idmt_smt_audio_effects_index, fhandle, indent=2)


def main(args):
    print("creating index...")
    make_isafx_index(args.idmt_smt_audio_effects_data_path)
    print("done!")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make IDMT SMT AUDIO EFFECTS index file.")
    PARSER.add_argument(
        "idmt_smt_audio_effects_data_path",
        type=str,
        help="Path to dataset data folder.",
    )

    main(PARSER.parse_args())
