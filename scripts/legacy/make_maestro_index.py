import argparse
import hashlib
import json
import os

MAESTRO_INDEX_PATH = "../mirdata/indexes/maestro_index.json"


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


def make_maestro_index(data_path):
    metadata_path = os.path.join(data_path, "maestro-v2.0.0.json")
    print(metadata_path)

    maestro_index = {}
    with open(metadata_path, "r") as fhandle:
        metadata = json.load(fhandle)

        for i, row in enumerate(metadata):
            print(i)
            trackid = row["midi_filename"].split(".")[0]
            maestro_index[trackid] = {}

            midi_path = os.path.join(data_path, row["midi_filename"])
            midi_checksum = md5(midi_path)
            maestro_index[trackid]["midi"] = [row["midi_filename"], midi_checksum]

            audio_path = os.path.join(data_path, row["audio_filename"])
            audio_checksum = md5(audio_path)
            maestro_index[trackid]["audio"] = [row["audio_filename"], audio_checksum]

    with open(MAESTRO_INDEX_PATH, "w") as fhandle:
        json.dump(maestro_index, fhandle, indent=2)


def main(args):
    print("creating index...")
    make_maestro_index(args.maestro_data_path)
    print("done!")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make MAESTRO index file.")
    PARSER.add_argument("maestro_data_path", type=str, help="Path to MAESTRO data folder.")

    main(PARSER.parse_args())
