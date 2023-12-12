#!/usr/bin/env python

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from mirdata.validate import md5

DATASET_INDEX_PATH = "../mirdata/datasets/indexes/openmic2018_index.json"


def make_dataset_index(dataset_data_path):
    path = Path(dataset_data_path)

    # top-key level metadata
    metadata_checksum = md5(path / Path("openmic-2018-metadata.csv"))
    classmap_checksum = md5(path / Path("class-map.json"))
    label_checksum = md5(path / Path("openmic-2018-aggregated-labels.csv"))
    response_checksum = md5(path / Path("openmic-2018-individual-responses.csv"))
    train_split = path / Path("partitions") / Path("split01_train.csv")
    test_split = path / Path("partitions") / Path("split01_test.csv")
    train_checksum = md5(train_split)
    test_checksum = md5(test_split)

    index_metadata = {
        "metadata": {
            "openmic-metadata": ("openmic-2018-metadata.csv", metadata_checksum),
            "openmic-classmap": ("class-map.json", classmap_checksum),
            "openmic-labels": ("openmic-2018-aggregated-labels.csv", label_checksum),
            "openmic-responses": ("openmic-2018-individual-responses.csv", response_checksum),
            "openmic-train": (str(train_split.relative_to(path)), train_checksum),
            "openmic-test": (str(test_split.relative_to(path)), test_checksum),
        }
    }

    # top-key level tracks
    index_tracks = {}
    for audio_file in tqdm(sorted(path.rglob("*.ogg"))):
        audio_checksum = md5(audio_file)
        arelpath = audio_file.relative_to(path)
        track_id = audio_file.stem

        vggish_file = (path / Path("vggish") / arelpath.parent.stem / track_id).with_suffix(".json")
        vggish_checksum = md5(vggish_file)
        vrelpath = vggish_file.relative_to(path)

        index_tracks[track_id] = {
            "audio": (str(arelpath), audio_checksum),
            "vggish": (str(vrelpath), vggish_checksum),
        }

    # top-key level version
    dataset_index = {"version": "1.0.0"}

    # combine all in dataset index
    dataset_index.update(index_metadata)
    dataset_index.update({"tracks": index_tracks})

    with open(DATASET_INDEX_PATH, "w") as fhandle:
        json.dump(dataset_index, fhandle, indent=2)


def main(args):
    make_dataset_index(args.dataset_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make dataset index file.")
    PARSER.add_argument("dataset_data_path", type=str, help="Path to dataset data folder.")

    main(PARSER.parse_args())
