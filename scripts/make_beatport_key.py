import argparse
import json
import os

from mirdata.validate import md5

beatport_key_INDEX_PATH = "../mirdata/datasets/indexes/beatport_key_index.json"


def make_beatport_key_index(data_path):
    meta_dir = os.path.join(data_path, "meta")
    audio_dir = os.path.join(data_path, "audio")
    key_dir = os.path.join(data_path, "keys")
    beatport_key_index = {
        "version": "1.0.0",
        "tracks": {},
        "metadata": None,
    }
    for track_id, ann_dir in enumerate(sorted(os.listdir(key_dir))):
        if ".txt" in ann_dir:
            codec = ".mp3"
            audio_path = os.path.join(audio_dir, ann_dir.replace(".txt", codec))
            chord_path = os.path.join(key_dir, ann_dir)
            meta_path = os.path.join(meta_dir, ann_dir.replace(".txt", ".json"))
            if not os.path.exists(meta_path):
                meta = (None, None)
            else:
                meta = (meta_path.replace(data_path + "/", ""), md5(meta_path))

            beatport_key_index["tracks"][track_id] = {
                "audio": (audio_path.replace(data_path + "/", ""), md5(audio_path)),
                "meta": meta,
                "key": (chord_path.replace(data_path + "/", ""), md5(chord_path)),
            }
    with open(beatport_key_INDEX_PATH, "w") as fhandle:
        json.dump(beatport_key_index, fhandle, indent=2)


def main(args):
    make_beatport_key_index(args.beatport_key_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make beatport_key index file.")
    PARSER.add_argument(
        "beatport_key_data_path", type=str, help="Path to beatport_key data folder."
    )
    main(PARSER.parse_args())
