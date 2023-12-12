import argparse
import hashlib
import json
import os

GOOD_SOUND_INDEX_PATH = "../mirdata/datasets/indexes/good_sounds_index.json"


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


# I have created four json files from the sqlite database. One for each SQL table.
def make_good_sounds_index(data_path):
    takes_path = os.path.join(data_path, "takes.json")
    packs_path = os.path.join(data_path, "packs.json")
    ratings_path = os.path.join(data_path, "ratings.json")
    sounds_path = os.path.join(data_path, "sounds.json")
    with open(takes_path) as json_file:
        takes = json.load(json_file)

    with open(packs_path) as json_file:
        packs = json.load(json_file)

    with open(ratings_path) as json_file:
        ratings = json.load(json_file)

    with open(sounds_path) as json_file:
        sounds = json.load(json_file)

    good_sounds_index = {
        "version": "1.0",
        "tracks": {
            t["id"]: {
                "audio": [
                    os.path.join("good-sounds", t["filename"]),
                    md5(os.path.join(data_path, "good-sounds", t["filename"])),
                ]
            }
            for t in takes
        },
        "takes": {take["id"]: take for take in takes},
        "packs": {pack["id"]: pack for pack in packs},
        "ratings": {rating["id"]: rating for rating in ratings},
        "sounds": {sound["id"]: sound for sound in sounds},
    }
    with open(GOOD_SOUND_INDEX_PATH, "w") as fhandle:
        json.dump(good_sounds_index, fhandle, indent=2)


def main(args):
    make_good_sounds_index(args.good_sounds_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make Good-Sounds index file.")
    PARSER.add_argument("good_sounds_data_path", type=str, help="Path to Good-Sounds data folder.")

    main(PARSER.parse_args())
