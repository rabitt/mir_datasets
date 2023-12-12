import argparse
import hashlib
import json
import os

BEATLES_INDEX_PATH = "../mir_dataset_loaders/indexes/beatles_index.json"
BEATLES_ANNOTATION_SCHEMA = ["beat", "chordlab", "keylab", "seglab"]


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


def make_beatles_index(data_path):
    annotations_dir = os.path.join(data_path, "Beatles", "annotations")
    cds_dir = os.path.join(annotations_dir, "all", "The Beatles")
    audio_dir = os.path.join(data_path, "Beatles", "audio")
    cds = os.listdir(cds_dir)
    track_ids = []
    beatles_index = {}
    totfiles = []
    for c in cds:
        for t in sorted(os.listdir(os.path.join(cds_dir, c))):
            if "ttl" in t:
                totfiles.append(t)

                track_id = "{}{}".format(
                    os.path.basename(c).split("_")[0][:2],
                    os.path.basename(t).split("_")[0],
                )
                if "CD" in t:
                    track_id = "10{}{}".format(
                        os.path.basename(c).split("_")[0][-1],
                        os.path.basename(t).split("_")[2][:2],
                    )
                track_ids.append(track_id)

                # checksum
                audio_checksum = md5(os.path.join(audio_dir, c, "{}.wav".format(t[:-4])))
                audio_path = "{}/{}".format("audio", os.path.join(c, "{}.wav".format(t[:-4])))

                annot_checksum, annot_rels = [], []

                for annot_type in BEATLES_ANNOTATION_SCHEMA:
                    cds_dir = os.path.join(annotations_dir, annot_type, "The Beatles")
                    annot_path = os.path.join(cds_dir, c)
                    if annot_type is "beat":
                        annot_file = "{}.txt".format(t[:-4])
                    else:
                        annot_file = "{}.lab".format(t[:-4])

                    if os.path.exists(os.path.join(annot_path, annot_file)):
                        annot_checksum.append(md5(os.path.join(annot_path, annot_file)))
                        annot_rels.append(
                            os.path.join("annotations", annot_type, "The Beatles", c, annot_file)
                        )
                    else:
                        annot_checksum.append(None)
                        annot_rels.append(None)

                beatles_index[track_id] = {
                    "audio": (audio_path, audio_checksum),
                    "beat": (annot_rels[0], annot_checksum[0]),
                    "chords": (annot_rels[1], annot_checksum[1]),
                    "keys": (annot_rels[2], annot_checksum[2]),
                    "sections": (annot_rels[3], annot_checksum[3]),
                }
    with open(BEATLES_INDEX_PATH, "w") as fhandle:
        json.dump(beatles_index, fhandle, indent=2)


def main(args):
    make_beatles_index(args.beatles_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make Beatles index file.")
    PARSER.add_argument("beatles_data_path", type=str, help="Path to Beatles data folder.")

    main(PARSER.parse_args())
