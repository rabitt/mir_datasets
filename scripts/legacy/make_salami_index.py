import argparse
import hashlib
import json
import os

SALAMI_INDEX_PATH = "../mirdata/indexes/salami_index.json"


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


def make_salami_index(data_path):
    annotations_dir = os.path.join(
        data_path, "Salami", "salami-data-public-hierarchy-corrections", "annotations"
    )
    audio_dir = os.path.join(data_path, "Salami", "audio")
    annotations_files = os.listdir(annotations_dir)
    track_ids = sorted([os.path.basename(f).split(".")[0] for f in annotations_files])

    salami_index = {}
    for track_id in track_ids:
        # audio
        audio_checksum = md5(os.path.join(audio_dir, "{}.mp3".format(track_id)))
        annot_checksum, annot_rels = [], []

        # using existing annotations (version 2.0)
        for f in ["uppercase.txt", "lowercase.txt"]:
            for a in ["1", "2"]:
                if os.path.exists(
                    os.path.join(
                        annotations_dir,
                        track_id,
                        "parsed",
                        "textfile{}_{}".format(a, f),
                    )
                ):
                    annot_checksum.append(
                        md5(
                            os.path.join(
                                annotations_dir,
                                track_id,
                                "parsed",
                                "textfile" + a + "_" + f,
                            )
                        )
                    )
                    annot_rels.append(
                        os.path.join(
                            "salami-data-public-hierarchy-corrections",
                            "annotations",
                            track_id,
                            "parsed",
                            "textfile{}_{}".format(a, f),
                        )
                    )
                else:
                    annot_checksum.append(None)
                    annot_rels.append(None)

        salami_index[track_id] = {
            "audio": (os.path.join("audio", "{}.mp3".format(track_id)), audio_checksum),
            "annotator_1_uppercase": (annot_rels[0], annot_checksum[0]),
            "annotator_1_lowercase": (annot_rels[2], annot_checksum[2]),
            "annotator_2_uppercase": (annot_rels[1], annot_checksum[1]),
            "annotator_2_lowercase": (annot_rels[3], annot_checksum[3]),
        }

    with open(SALAMI_INDEX_PATH, "w") as fhandle:
        json.dump(salami_index, fhandle, indent=2)


def main(args):
    make_salami_index(args.salami_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make Salami index file.")
    PARSER.add_argument("salami_data_path", type=str, help="Path to Salami data folder.")

    main(PARSER.parse_args())
