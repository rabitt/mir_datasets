import argparse
import csv
import json
import os

from mirdata.validate import md5

INDEX_PATH = "../mirdata/datasets/indexes/billboard_index.json"


def make_index(data_path):
    _index = {}
    index_file = csv.reader(open(os.path.join(data_path, "billboard-2.0-index.csv")))
    for row in index_file:
        k = row[0]
        _index[k] = row[1:]

    annotations_dir = os.path.join(data_path, "McGill-Billboard")
    audio_dir = os.path.join(data_path, "audio")
    anns = sorted(os.listdir(annotations_dir))

    track_ids = []
    index = {}
    index = {
        "version": "2.0",
        "tracks": {},
        "metadata": None,
    }

    txtfiles = []
    for a in anns:
        for t in os.listdir(os.path.join(annotations_dir, a)):
            if t == "salami_chords.txt":
                fp = os.path.join(annotations_dir, a, t)
                track_id = "{}".format(os.path.basename(a.lstrip("0")))

                if track_id in _index.keys():
                    txtfiles.append(t)
                    track_ids.append(track_id)

                    release_date = _index[track_id][0]
                    track_name = _index[track_id][3]
                    artist = _index[track_id][4]

                    _release_date = "{}s".format(round(int(release_date.split("-")[0]), -1))

                    audio_path = os.path.join(
                        audio_dir, _release_date, artist, track_name, "audio.flac"
                    )
                    audio_checksum = None
                    if os.path.exists(audio_path):
                        audio_checksum = md5(audio_path)
                    else:
                        audio_path = None

                    annot_rel = os.path.join("annotation", a, t)
                    audio_rel = os.path.join(
                        "audio", _release_date, artist, track_name, "audio.flac"
                    )
                    annot_checksum = md5(fp)

                    full_fp = os.path.join(annotations_dir, a, "full.lab")
                    majmin7 = os.path.join(annotations_dir, a, "majmin7.lab")
                    majmin7inv = os.path.join(annotations_dir, a, "majmin7inv.lab")
                    majmin = os.path.join(annotations_dir, a, "majmin.lab")
                    majmininv = os.path.join(annotations_dir, a, "majmininv.lab")

                    bothchroma = os.path.join(annotations_dir, a, "bothchroma.csv")
                    tuning = os.path.join(annotations_dir, a, "tuning.csv")

                    index["tracks"][track_id] = {
                        "audio": (audio_rel, audio_checksum),
                        "salami": (annot_rel, annot_checksum),
                        "bothchroma": (
                            os.path.join("McGill-Billboard", a, "bothchroma.csv"),
                            md5(bothchroma),
                        ),
                        "tuning": (
                            os.path.join("McGill-Billboard", a, "tuning.csv"),
                            md5(tuning),
                        ),
                        "lab_full": (
                            os.path.join("McGill-Billboard", a, "full.lab"),
                            md5(full_fp),
                        ),
                        "lab_majmin7": (
                            os.path.join("McGill-Billboard", a, "majmin7.lab"),
                            md5(majmin7),
                        ),
                        "lab_majmin7inv": (
                            os.path.join("McGill-Billboard", a, "majmin7inv.lab"),
                            md5(majmin7inv),
                        ),
                        "lab_majmin": (
                            os.path.join("McGill-Billboard", a, "majmin.lab"),
                            md5(majmin),
                        ),
                        "lab_majmininv": (
                            os.path.join("McGill-Billboard", a, "majmininv.lab"),
                            md5(majmininv),
                        ),
                    }

    with open(INDEX_PATH, "w") as fhandle:
        json.dump(index, fhandle, indent=2)


def main(args):
    make_index(args.data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make index file.")
    PARSER.add_argument("data_path", type=str, help="Path to data folder.")

    main(PARSER.parse_args())
