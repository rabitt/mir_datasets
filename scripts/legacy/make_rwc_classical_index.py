import argparse
import csv
import hashlib
import json
import os

RWC_CLASSICAL_INDEX_PATH = "../mirdata/datasets/indexes/rwc_classical_index.json"


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


def make_rwc_classical_index(data_path):
    annotations_dir = os.path.join(data_path, "rwc_classical", "annotations")
    metadata_dir = os.path.join(data_path, "rwc_classical", "metadata-master")
    audio_dir = os.path.join(data_path, "rwc_classical", "audio")
    annotations_files = os.listdir(os.path.join(annotations_dir, "AIST.RWC-MDB-C-2001.CHORUS"))
    metadata_file = os.path.join(metadata_dir, "rwc-c.csv")
    with open(metadata_file, "r", encoding="utf-8") as fhandle:
        dialect = csv.Sniffer().sniff(fhandle.read(1024))
        fhandle.seek(0)
        reader = csv.reader(fhandle, dialect)
        piece = []
        suffix = []
        track = []
        for line in reader:
            if not line[0] == "Piece No.":
                p = "00" + line[0].split(".")[1][1:]
                piece.append(p[len(p) - 3 :])
                suffix.append(line[1][1:])
                track.append(line[2][-2:])

    mapping_track = {p: t for p, t in zip(piece, track)}
    mapping_folder = {p: s for p, s in zip(piece, suffix)}

    track_ids = sorted(
        [os.path.basename(f).split(".")[0] for f in annotations_files if not f == "README.TXT"]
    )

    rwc_classical_index = {}
    for track_id in track_ids:
        # audio
        audio_folder = "rwc-c-m{}".format(mapping_folder[track_id[4:7]])
        audio_path = os.path.join(audio_dir, audio_folder)
        audio_track = str(int(mapping_track[track_id[4:7]]))
        audio_checksum = md5(os.path.join(audio_path, "{}.wav".format(audio_track)))
        annot_checksum = []
        annot_rels = []

        for f in ["CHORUS", "BEAT"]:
            if os.path.exists(
                os.path.join(
                    annotations_dir,
                    "AIST.RWC-MDB-C-2001.{}".format(f),
                    "{}.{}.TXT".format(track_id, f),
                )
            ):
                annot_checksum.append(
                    md5(
                        os.path.join(
                            annotations_dir,
                            "AIST.RWC-MDB-C-2001.{}".format(f),
                            "{}.{}.TXT".format(track_id, f),
                        )
                    )
                )
                annot_rels.append(
                    os.path.join(
                        "annotations",
                        "AIST.RWC-MDB-C-2001.{}".format(f),
                        "{}.{}.TXT".format(track_id, f),
                    )
                )
            else:
                annot_checksum.append(None)
                annot_rels.append(None)

        rwc_classical_index[track_id] = {
            "audio": (
                os.path.join("audio", audio_folder, "{}.wav".format(audio_track)),
                audio_checksum,
            ),
            "sections": (annot_rels[0], annot_checksum[0]),
            "beats": (annot_rels[1], annot_checksum[1]),
        }

    with open(RWC_CLASSICAL_INDEX_PATH, "w") as fhandle:
        json.dump(rwc_classical_index, fhandle, indent=2)


def main(args):
    make_rwc_classical_index(args.rwc_classical_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make RWC-Classical index file.")
    PARSER.add_argument(
        "rwc_classical_data_path", type=str, help="Path to RWC-Classical data folder."
    )

    main(PARSER.parse_args())
