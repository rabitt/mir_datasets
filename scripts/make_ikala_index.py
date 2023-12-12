import argparse
import glob
import json
import os

from mirdata.validate import md5

IKALA_INDEX_PATH = "mirdata/datasets/indexes/ikala_index_{}.json"


def make_ikala_index(ikala_data_path, version):
    assert version in ["1.0", "2.0"]
    lyrics_dir = os.path.join(ikala_data_path, "Lyrics")
    lyrics_files = glob.glob(os.path.join(lyrics_dir, "*.lab"))
    track_ids = sorted([os.path.basename(f).split(".")[0] for f in lyrics_files])

    # top-key level metadata
    metadata_checksum = md5(os.path.join(ikala_data_path, "id_mapping.txt"))
    index_metadata = {"metadata": {"id_mapping": ("id_mapping.txt", metadata_checksum)}}

    # top-key level tracks
    index_tracks = {}
    for track_id in track_ids:
        audio_checksum = md5(os.path.join(ikala_data_path, "Wavfile/{}.wav".format(track_id)))
        pitch_checksum = md5(os.path.join(ikala_data_path, "PitchLabel/{}.pv".format(track_id)))
        lyrics_checksum = md5(os.path.join(ikala_data_path, "Lyrics/{}.lab".format(track_id)))
        notes_relative = "ikala-pyin-notes/{}_vamp_pyin_pyin_notes.csv".format(track_id)
        notes_path = os.path.join(ikala_data_path, notes_relative)

        index_tracks[track_id] = {
            "audio": ("Wavfile/{}.wav".format(track_id), audio_checksum),
            "pitch": ("PitchLabel/{}.pv".format(track_id), pitch_checksum),
            "lyrics": ("Lyrics/{}.lab".format(track_id), lyrics_checksum),
        }
        if version == "2.0":
            index_tracks[track_id]["notes_pyin"] = (notes_relative, md5(notes_path))

    # top-key level version
    ikala_index = {
        "version": version,
        "tracks": index_tracks,
        "metadata": {
            "id_mapping": (
                "id_mapping.txt",
                md5(os.path.join(ikala_data_path, "id_mapping.txt")),
            ),
        },
    }

    with open(IKALA_INDEX_PATH.format(version), "w") as fhandle:
        json.dump(ikala_index, fhandle, indent=2)


def main(args):
    make_ikala_index(args.ikala_data_path, args.version)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make IKala index file.")
    PARSER.add_argument("ikala_data_path", type=str, help="Path to IKala data folder.")
    PARSER.add_argument("version", type=str, help="index version")

    main(PARSER.parse_args())
