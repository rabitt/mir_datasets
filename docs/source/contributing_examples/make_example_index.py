import argparse
import glob
import json
import os

from mirdata.validate import md5

DATASET_INDEX_PATH = "../mirdata/datasets/indexes/dataset_index.json"


def make_dataset_index(dataset_data_path):
    annotation_dir = os.path.join(dataset_data_path, "annotation")
    annotation_files = glob.glob(os.path.join(annotation_dir, "*.lab"))
    track_ids = sorted([os.path.basename(f).split(".")[0] for f in annotation_files])

    # top-key level metadata
    metadata_checksum = md5(os.path.join(dataset_data_path, "id_mapping.txt"))
    index_metadata = {"metadata": {"id_mapping": ("id_mapping.txt", metadata_checksum)}}

    # top-key level tracks
    index_tracks = {}
    for track_id in track_ids:
        audio_checksum = md5(os.path.join(dataset_data_path, "Wavfile/{}.wav".format(track_id)))
        annotation_checksum = md5(
            os.path.join(dataset_data_path, "annotation/{}.lab".format(track_id))
        )

        index_tracks[track_id] = {
            "audio": ("Wavfile/{}.wav".format(track_id), audio_checksum),
            "annotation": ("annotation/{}.lab".format(track_id), annotation_checksum),
        }

    # top-key level version
    dataset_index = {"version": None}

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
