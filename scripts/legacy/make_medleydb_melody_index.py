import argparse
import hashlib
import json
import os

MEDLEYDB_MELODY_INDEX_PATH = "../mirdata/indexes/medleydb_melody_index.json"


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


def make_medleydb_melody_index(data_path):
    metadata_path = os.path.join(data_path, "MedleyDB-Melody", "medleydb_melody_metadata.json")
    with open(metadata_path, "r") as fhandle:
        metadata = json.load(fhandle)

    melody_index = {}
    for trackid in metadata.keys():
        audio_path = os.path.join(data_path, metadata[trackid]["audio_path"])
        audio_checksum = md5(audio_path)
        local_mel1_path = os.path.join(data_path, metadata[trackid]["melody1_path"])
        mel1_checksum = md5(local_mel1_path)
        local_mel2_path = os.path.join(data_path, metadata[trackid]["melody2_path"])
        mel2_checksum = md5(local_mel2_path)
        local_mel3_path = os.path.join(data_path, metadata[trackid]["melody3_path"])
        mel3_checksum = md5(local_mel3_path)

        melody_index[trackid] = {
            "audio": (strip_first_dir(metadata[trackid]["audio_path"]), audio_checksum),
            "melody1": (
                strip_first_dir(metadata[trackid]["melody1_path"]),
                mel1_checksum,
            ),
            "melody2": (
                strip_first_dir(metadata[trackid]["melody2_path"]),
                mel2_checksum,
            ),
            "melody3": (
                strip_first_dir(metadata[trackid]["melody3_path"]),
                mel3_checksum,
            ),
        }

    with open(MEDLEYDB_MELODY_INDEX_PATH, "w") as fhandle:
        json.dump(melody_index, fhandle, indent=2)


def main(args):
    make_medleydb_melody_index(args.mdb_melody_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make MedleyDB-Melody index file.")
    PARSER.add_argument(
        "mdb_melody_data_path", type=str, help="Path to MedleyDB-Melody data folder."
    )

    main(PARSER.parse_args())
