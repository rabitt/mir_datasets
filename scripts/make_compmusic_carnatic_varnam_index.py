import argparse
import hashlib
import json
import os

COMPMUSIC_VARNAM_INDEX_PATH = "./mirdata/datasets/indexes/compmusic_carnatic_varnam_index_1.1.json"


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


def make_compmusic_varnam_index(dataset_data_path):
    dataset_index = {"version": 1.1, "tracks": {}, "metadata": {"annotation_metadata": []}}

    annotations_path = "Notations_Annotations/annotations"
    notations_path = "Notations_Annotations/notations"

    for top_level in os.listdir(dataset_data_path):
        if "Audio" in top_level:
            for song in os.listdir(os.path.join(dataset_data_path, top_level)):
                if ".mp3" in song:
                    idx = song.split("-")[3] + "_" + song.split("-")[5]  # Get index
                    notation_file = song.split("-")[5] + ".yaml"  # Get notation file
                    structure_file = (
                        song.split("-")[5] + "/" + song.split("-")[3] + ".yaml"
                    )  # Get structure file
                    taala_path = os.path.join(
                        "taalas", song.split("-")[5], song.split("-")[3] + ".svl"
                    )  # Get taala file

                    dataset_index["tracks"][idx] = {
                        "audio": [
                            os.path.join("carnatic_varnam_1.1", top_level, song),
                            md5(os.path.join(dataset_data_path, top_level, song)),
                        ],
                        "notation": [
                            os.path.join("carnatic_varnam_1.1", notations_path, notation_file),
                            md5(os.path.join(dataset_data_path, notations_path, notation_file)),
                        ],
                        "structure": [
                            os.path.join("carnatic_varnam_1.1", notations_path, structure_file),
                            md5(os.path.join(dataset_data_path, notations_path, structure_file)),
                        ],
                        "taala": [
                            os.path.join("carnatic_varnam_1.1", annotations_path, taala_path),
                            md5(os.path.join(dataset_data_path, annotations_path, taala_path)),
                        ],
                    }

    dataset_index["metadata"]["annotation_metadata"] = [
        os.path.join("carnatic_varnam_1.1", annotations_path, "tonics.yaml"),
        md5(os.path.join(dataset_data_path, annotations_path, "tonics.yaml")),
    ]

    with open(COMPMUSIC_VARNAM_INDEX_PATH, "w") as fhandle:
        json.dump(dataset_index, fhandle, indent=2)


def main(args):
    print("creating index...")
    make_compmusic_varnam_index(args.dataset_data_path)
    print("done!")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make Compmusic Carnatic Varnam index file.")
    PARSER.add_argument(
        "dataset_data_path", type=str, help="Path to Compmusic Carnatic Varnam data folder."
    )

    main(PARSER.parse_args())
