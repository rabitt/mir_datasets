import argparse
import csv
import json
import os

from mirdata.validate import md5

VOCADITO_INDEX_PATH = "mirdata/datasets/indexes/vocadito_index_{}.json"


def make_vocadito_index(vocadito_data_path: str, version: str) -> None:
    assert version == "1"
    annotations_dir = os.path.join(vocadito_data_path, "Annotations")
    f0_dir = os.path.join(annotations_dir, "F0")
    lyrics_dir = os.path.join(annotations_dir, "Lyrics")
    notes_dir = os.path.join(annotations_dir, "Notes")

    audio_dir = os.path.join(vocadito_data_path, "Audio")

    metadata_path = os.path.join(vocadito_data_path, "vocadito_metadata.csv")
    with open(metadata_path) as f:
        track_ids = sorted([row["track_id"] for row in csv.DictReader(f)])

    # top-key level tracks
    index_tracks = {
        track_id: {
            "audio": (
                f"Audio/vocadito_{track_id}.wav",
                md5(os.path.join(audio_dir, f"vocadito_{track_id}.wav")),
            ),
            "f0": (
                f"Annotations/F0/vocadito_{track_id}_f0.csv",
                md5(os.path.join(f0_dir, f"vocadito_{track_id}_f0.csv")),
            ),
            "notesA1": (
                f"Annotations/Notes/vocadito_{track_id}_notesA1.csv",
                md5(os.path.join(notes_dir, f"vocadito_{track_id}_notesA1.csv")),
            ),
            "notesA2": (
                f"Annotations/Notes/vocadito_{track_id}_notesA2.csv",
                md5(os.path.join(notes_dir, f"vocadito_{track_id}_notesA2.csv")),
            ),
            "lyrics": (
                f"Annotations/Lyrics/vocadito_{track_id}_lyrics.txt",
                md5(os.path.join(lyrics_dir, f"vocadito_{track_id}_lyrics.txt")),
            ),
        }
        for track_id in track_ids
    }

    # top-key level version
    vocadito_index = {
        "version": version,
        "tracks": index_tracks,
        "metadata": {"vocadito_metadata": ("vocadito_metadata.csv", md5(metadata_path))},
    }

    with open(VOCADITO_INDEX_PATH.format(version), "w") as fhandle:
        json.dump(vocadito_index, fhandle, indent=2)


def main(args):
    make_vocadito_index(args.vocadito_data_path, args.version)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make Vocadito index file.")
    PARSER.add_argument("vocadito_data_path", type=str, help="Path to Vocadito data folder.")
    PARSER.add_argument("version", type=str, help="index version")

    main(PARSER.parse_args())
