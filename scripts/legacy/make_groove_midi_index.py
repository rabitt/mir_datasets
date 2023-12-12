import argparse
import csv
import hashlib
import json
import os

GROOVE_MIDI_INDEX_PATH = "../mirdata/indexes/groove_midi_index.json"


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


def make_groove_midi_index(data_path):
    metadata_path = os.path.join(data_path, "info.csv")

    groove_index = {}
    with open(metadata_path, "r") as fhandle:
        metadata = csv.DictReader(fhandle)

        for row in metadata:
            trackid = row["id"]
            groove_index[trackid] = {}

            midi_path = os.path.join(data_path, row["midi_filename"])
            midi_checksum = md5(midi_path)
            groove_index[trackid]["midi"] = [row["midi_filename"], midi_checksum]

            if row["audio_filename"]:
                audio_path = os.path.join(data_path, row["audio_filename"])
                audio_checksum = md5(audio_path)
                groove_index[trackid]["audio"] = [row["audio_filename"], audio_checksum]

            else:
                groove_index[trackid]["audio"] = [None, None]

    with open(GROOVE_MIDI_INDEX_PATH, "w") as fhandle:
        json.dump(groove_index, fhandle, indent=2)


def main(args):
    make_groove_midi_index(args.groove_midi_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make Groove MIDI index file.")
    PARSER.add_argument("groove_midi_data_path", type=str, help="Path to Groove MIDI data folder.")

    main(PARSER.parse_args())
