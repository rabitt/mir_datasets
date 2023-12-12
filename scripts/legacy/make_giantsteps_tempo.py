import argparse
import hashlib
import json
import os

giantsteps_tempo_INDEX_PATH = "../mirdata/indexes/giantsteps_tempo_index.json"
BEATLES_ANNOTATION_SCHEMA = ["JAMS"]


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


def make_giantsteps_tempo_index(data_path):
    dir_github = "giantsteps-tempo-dataset-0b7d47ba8cae59d3535a02e3db69e2cf6d0af5bb"
    meta1_dir = os.path.join(data_path, dir_github, "annotations", "jams")
    meta2_dir = os.path.join(data_path, dir_github, "annotations_v2", "jams")
    audio_dir = os.path.join(data_path, "audio")
    giantsteps_tempo_index = {}
    print(sorted(os.listdir(meta1_dir)))
    for track_id, ann_dir in enumerate(sorted(os.listdir(meta1_dir))):
        os.path.join(meta1_dir, ann_dir)
        if ".jams" in ann_dir:
            codec = ".mp3"
            audio_path = os.path.join(audio_dir, ann_dir.replace(".jams", codec))
            ann1_path = os.path.join(meta1_dir, ann_dir)
            ann2_path = os.path.join(meta2_dir, ann_dir)
            giantsteps_tempo_index[track_id] = {
                "audio": (audio_path.replace(data_path + "/", ""), md5(audio_path)),
                "annotation_v1": (ann1_path.replace(data_path + "/", ""), md5(ann1_path)),
                "annotation_v2": (ann2_path.replace(data_path + "/", ""), md5(ann2_path)),
            }
    with open(giantsteps_tempo_INDEX_PATH, "w") as fhandle:
        json.dump(giantsteps_tempo_index, fhandle, indent=2)


def main(args):
    make_giantsteps_tempo_index(args.giantsteps_tempo_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make giantsteps_tempo index file.")
    PARSER.add_argument(
        "giantsteps_tempo_data_path", type=str, help="Path to giantsteps_tempo data folder."
    )
    main(PARSER.parse_args())
