import argparse
import csv
import hashlib
import json
import os

TINYSOL_INDEX_PATH = "../mirdata/datasets/indexes/tinysol_index.json"


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


def make_tinysol_index(tinysol_data_path):
    tinysol_index = {}

    anno_path = os.path.join(tinysol_data_path, "annotation", "TinySOL_metadata.csv")
    audio_dir = os.path.join(tinysol_data_path, "audio")

    with open(anno_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            local_path = row[0]
            audio_path = os.path.join(audio_dir, local_path)
            audio_checksum = md5(audio_path)
            key = os.path.splitext(os.path.split(local_path)[1])[0]
            tinysol_index[key] = {"audio": (os.path.join("audio", local_path), audio_checksum)}

    with open(TINYSOL_INDEX_PATH, "w") as fhandle:
        json.dump(tinysol_index, fhandle, indent=2)


def main(args):
    make_tinysol_index(args.tinysol_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make TinySOL index file.")
    PARSER.add_argument("tinysol_data_path", type=str, help="Path to TinySOL data folder.")

    main(PARSER.parse_args())
