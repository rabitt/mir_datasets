import argparse
import hashlib
import json
import os

MRIDANGAM_INDEX_PATH = "../mirdata/indexes/mridangam_stroke_index.json"


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


def strip_first_dir(full_path):
    return os.path.join(*(full_path.split(os.path.sep)[1:]))


def make_mridangam_index(mridangam_data_path):
    stroke_dict = dict()
    for root, dirs, files in os.walk(mridangam_data_path):
        for directory in dirs:
            for root_, dirs_, files_ in os.walk(os.path.join(mridangam_data_path, directory)):
                for file in files_:
                    if file.endswith(".wav"):
                        if "XXX" not in file:
                            stroke_id = file.split("__")[0]  # Obtain stroke id
                            stroke_dict[stroke_id] = os.path.join(directory, file)

    stroke_id_list = sorted(stroke_dict.items())  # Sort strokes by id

    mridangam_index = {}
    for inst in stroke_id_list:
        rel_path = os.path.join("mridangam_stroke_1.5", inst[1])
        audio_checksum = md5(os.path.join(mridangam_data_path, inst[1]))

        mridangam_index[inst[0]] = {
            "audio": (rel_path, audio_checksum),
        }

    with open(MRIDANGAM_INDEX_PATH, "w") as fhandle:
        json.dump(mridangam_index, fhandle, indent=2)


def main(args):
    make_mridangam_index(args.mridangam_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make Maridangam Stroke index file.")
    PARSER.add_argument(
        "mridangam_data_path", type=str, help="Path to Mridangam Stroke data folder."
    )

    main(PARSER.parse_args())
