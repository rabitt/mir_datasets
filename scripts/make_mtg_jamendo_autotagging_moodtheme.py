import argparse
import csv
import json
import os

from mirdata.validate import md5

mtg_jamendo_autotagging_moodtheme_INDEX_PATH = (
    "../mirdata/datasets/indexes/mtg_jamendo_autotagging_moodtheme_index_1.0.json"
)


def make_mtg_jamendo_autotagging_moodtheme_index(data_path):
    mtg_jamendo_autotagging_moodtheme_index = {"version": "1.0", "tracks": {}}
    meta_path = os.path.join(data_path, "metadata", "data", "autotagging_moodtheme.tsv")

    with open(meta_path, "r") as fhandle:
        d = list(
            csv.DictReader(
                fhandle,
                delimiter="\t",
                fieldnames=["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION", "TAGS"],
            )
        )
        meta = {m["TRACK_ID"]: m["PATH"] for m in d[1:]}

    for track_id, path in meta.items():
        audio_path = os.path.join(data_path, "audios", path)
        mtg_jamendo_autotagging_moodtheme_index["tracks"][track_id] = {
            "audio": (audio_path.replace(data_path + "/", ""), md5(audio_path)),
        }
    with open(mtg_jamendo_autotagging_moodtheme_INDEX_PATH, "w") as fhandle:
        json.dump(mtg_jamendo_autotagging_moodtheme_index, fhandle, indent=2)


def main(args):
    make_mtg_jamendo_autotagging_moodtheme_index(args.mtg_jamendo_autotagging_moodtheme_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Make mtg_jamendo_autotagging_moodtheme index file."
    )
    PARSER.add_argument(
        "mtg_jamendo_autotagging_moodtheme_data_path",
        type=str,
        help="Path to mtg_jamendo_autotagging_moodtheme data folder.",
    )
    main(PARSER.parse_args())
