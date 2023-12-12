import argparse
import hashlib
import json
import os

DALI_INDEX_PATH = "../mirdata/indexes/dali_index.json"


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


def make_dali_index(data_path):
    metadata_path = os.path.join(data_path, "dali_metadata.json")
    with open(metadata_path, "r") as fhandle:
        dali_metadata = json.load(fhandle)
    dali_index = {}
    for trackid in dali_metadata.keys():
        dali_index[trackid] = {}
        dali_index[trackid]["audio"] = ["audio/" + trackid + ".mp3"]
        audio_checksum = md5(os.path.join(data_path, "audio/", "{}.mp3".format(trackid)))
        dali_index[trackid]["audio"].append(audio_checksum)
        dali_index[trackid]["annot"] = ["annotations/" + trackid + ".gz"]
        annot_checksum = md5(os.path.join(data_path, "annotations/", "{}.gz".format(trackid)))
        dali_index[trackid]["annot"].append(annot_checksum)

    with open(DALI_INDEX_PATH, "w") as fhandle:
        json.dump(dali_index, fhandle, indent=2)


def main(args):
    make_dali_index(args.dali_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make DALI index file.")
    PARSER.add_argument("dali_data_path", type=str, help="Path to DALI data folder.")
    main(PARSER.parse_args())
