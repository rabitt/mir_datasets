import argparse
import json
import os

from mirdata.validate import md5

QUEEN_INDEX_PATH = "../mirdata/datasets/indexes/queen_index.json"
QUEEN_ANNOTATION_SCHEMA = ["chordlab", "keylab", "seglab"]


def make_queen_index(data_path):
    annotations_dir = os.path.join(data_path, "Queen", "annotations")
    cds_dir = os.path.join(annotations_dir, "all", "Queen")
    audio_dir = os.path.join(data_path, "Queen", "audio")
    cds = sorted(os.listdir(cds_dir))
    track_ids = []
    queen_tracks = {}
    totfiles = []
    track_id = 0
    for c in cds:
        for t in sorted(os.listdir(os.path.join(cds_dir, c))):
            if "ttl" in t:
                totfiles.append(t)

                if "CD" in t:
                    track_id = "10{}{}".format(
                        os.path.basename(c).split("_")[0][-1],
                        os.path.basename(t).split("_")[2][:2],
                    )
                track_ids.append(track_id)

                # checksum
                audio_checksum = md5(os.path.join(audio_dir, c, "{}.flac".format(t[:-4])))
                audio_path = "{}/{}".format("audio", os.path.join(c, "{}.flac".format(t[:-4])))

                annot_checksum, annot_rels = [], []

                for annot_type in QUEEN_ANNOTATION_SCHEMA:
                    cds_dir = os.path.join(annotations_dir, annot_type, "Queen")
                    annot_path = os.path.join(cds_dir, c)

                    annot_file = "{}.lab".format(t[:-4])

                    if os.path.exists(os.path.join(annot_path, annot_file)):
                        annot_checksum.append(md5(os.path.join(annot_path, annot_file)))
                        annot_rels.append(
                            os.path.join("annotations", annot_type, "Queen", c, annot_file)
                        )
                    else:
                        annot_checksum.append(None)
                        annot_rels.append(None)

                queen_tracks[track_id] = {
                    "audio": (audio_path, audio_checksum),
                    "chords": (annot_rels[0], annot_checksum[0]),
                    "keys": (annot_rels[1], annot_checksum[1]),
                    "sections": (annot_rels[2], annot_checksum[2]),
                }
                track_id += 1
    queen_index = {"version": "1.0", "tracks": queen_tracks, "metadata": None}
    with open(QUEEN_INDEX_PATH, "w") as fhandle:
        json.dump(queen_index, fhandle, indent=2)


def main(args):
    make_queen_index(args.queen_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make Queen index file.")
    PARSER.add_argument("queen_data_path", type=str, help="Path to Queen data folder.")

    main(PARSER.parse_args())
