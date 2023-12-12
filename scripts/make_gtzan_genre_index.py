import argparse
import json
import os

from mirdata.validate import md5

GTZAN_GENRE_INDEX_PATH = "../mirdata/datasets/indexes/gtzan_genre_index_1.0.json"


def make_gtzan_genre_index(data_path):
    index = {
        "version": "1.0",
        "tracks": {},
    }
    audiodata_path = os.path.join(data_path, "gtzan_genre", "genres")
    for track_key, path in iter_paths(audiodata_path):
        abspath = os.path.join(audiodata_path, path)
        if not os.path.exists(abspath):
            print("Missing file: {}".format(abspath))
            continue

        audio_checksum = md5(abspath)
        audio_path = os.path.join("gtzan_genre", "genres", path)
        try:
            genre, id = path.split("/")[-1].split(".")[:-1]
            beats_path = os.path.join("gtzan_tempo_beat-main", "beats", f"gtzan_{genre}_{id}.beats")
            beats_checksum = md5(os.path.join(data_path, beats_path))
        except:
            beats_path, beats_checksum = None, None

        try:
            tempo_path = os.path.join("gtzan_tempo_beat-main", "tempo", f"gtzan_{genre}_{id}.bpm")
            tempo_checksum = md5(os.path.join(data_path, tempo_path))
        except:
            tempo_path, tempo_checksum = None, None
        index["tracks"][track_key] = {
            "audio": [audio_path, audio_checksum],
            "beats": [beats_path, beats_checksum],
            "tempo": [tempo_path, tempo_checksum],
        }

    with open(GTZAN_GENRE_INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)


def iter_paths(data_path):
    with open(os.path.join(data_path, "bextract_single.mf")) as f:
        for line in f:
            if not line.strip():  # blank space
                continue

            au_path, _ = line.split("\t")
            _, folder, au_filename = au_path.rsplit("/", 2)
            track_key, _ = os.path.splitext(au_filename)
            wav_filename = track_key + ".wav"
            path = os.path.join(folder, wav_filename)
            yield track_key, path


def main():
    parser = argparse.ArgumentParser(description="Make GTZAN-Genre sample index file.")
    parser.add_argument(
        "gtzan_genre_data_path", type=str, help="Path to the GTZAN-Genre data folder."
    )
    args = parser.parse_args()
    make_gtzan_genre_index(os.path.expanduser(args.gtzan_genre_data_path))


if __name__ == "__main__":
    main()
