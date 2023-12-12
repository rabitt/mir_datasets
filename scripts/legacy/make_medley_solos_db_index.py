import argparse
import csv
import hashlib
import json
import os

MEDLEY_SOLOS_DB_INDEX_PATH = "../mirdata/indexes/medley_solos_db_index.json"


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


def make_medley_solos_db_index(medley_solos_db_data_path):
    instrument_dict = {
        "0": "clarinet",
        "1": "distorted electric guitar",
        "2": "female singer",
        "3": "flute",
        "4": "piano",
        "5": "tenor saxophone",
        "6": "trumpet",
        "7": "violin",
    }
    anno_path = os.path.join(medley_solos_db_data_path, "Medley-solos-DB_metadata.csv")

    medley_solos_db_index = {}

    with open(anno_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            subset, _, instrument_id, _, uuid4 = row
            wav_name = (
                "_".join(["Medley-solos-DB", subset + "-" + str(instrument_id), uuid4]) + ".wav"
            )
            audio_path = os.path.join(medley_solos_db_data_path, wav_name)
            audio_checksum = md5(audio_path)
            medley_solos_db_index[uuid4] = {
                "audio": (wav_name, audio_checksum),
            }

    with open(MEDLEY_SOLOS_DB_INDEX_PATH, "w") as fhandle:
        json.dump(medley_solos_db_index, fhandle, indent=2)


def main(args):
    make_medley_solos_db_index(args.medley_solos_db_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make Medley-solos-DB index file.")
    PARSER.add_argument(
        "medley_solos_db_data_path",
        type=str,
        help="Path to Medley-solos-DB data folder.",
    )

    main(PARSER.parse_args())
