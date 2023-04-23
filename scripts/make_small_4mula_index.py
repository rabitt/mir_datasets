import argparse
import json
import os
from mirdata.validate import md5
from mirdata.download_utils import RemoteFileMetadata, download_from_remote
from numpy import save
import pyarrow.parquet as pq
from tqdm import tqdm

DATASET_INDEX_PATH = "mirdata/datasets/indexes/small_4mula_index.json"


def make_small_4mula_index(dataset_data_path):
    annotation_dir = os.path.join(dataset_data_path["dataset_folder"], "annotation")
    melspectrogram_dir = os.path.join(dataset_data_path["dataset_folder"], "melspectrogram")

    if not os.path.exists(annotation_dir):
        os.mkdir(annotation_dir)
    if not os.path.exists(melspectrogram_dir):
        os.mkdir(melspectrogram_dir)

    batch_size = 1
    dataset = pq.ParquetFile(f"{dataset_data_path['dataset_folder']}{dataset_data_path['filename']}")
    batches = dataset.iter_batches(batch_size)  # batches will be a generator

    # top-key level tracks
    index_tracks = {}
    for track in tqdm(batches, total=9661):
        with open(f"{annotation_dir}/{track.column('music_id')[0]}.tsv", 'w') as f:
            f.write(
                'music_id\tmusic_name\tmusic_lang\t'
                'music_lyrics\tart_id\tart_name\t'
                'art_rank\tmain_genre\trelated_genre\t'
                'related_art\trelated_music\tmusicnn_tags\n')
            f.write(
                f"{track.column('music_id')[0]}\t{track.column('music_name')[0]}\t{track.column('music_lang')[0]}\t"
                f"{track.column('music_lyrics')[0]}\t{track.column('art_id')[0]}\t{track.column('art_name')[0]}\t"
                f"{track.column('art_rank')[0]}\t{track.column('main_genre')[0]}\t{track.column('related_genre')[0]}\t"
                f"{track.column('related_art')[0]}\t{track.column('related_music')[0]}"
                f"\t{track.column('musicnn_tags')[0]}")

        with open(f"{melspectrogram_dir}/{track.column('music_id')[0]}.npy", 'wb') as f:
            save(f, track.column('melspectrogram')[0].as_py())

        audio_checksum = md5(
            os.path.join(dataset_data_path["dataset_folder"], f"melspectrogram/{track.column('music_id')[0]}.npy")
        )
        annotation_checksum = md5(
            os.path.join(dataset_data_path["dataset_folder"], f"annotation/{track.column('music_id')[0]}.tsv")
        )

        index_tracks[str(track.column('music_id')[0])] = {
            "melspectrogram": (f"melspectrogram/{track.column('music_id')[0]}.npy", audio_checksum),
            "annotation": (f"annotation/{track.column('music_id')[0]}.tsv", annotation_checksum)
        }

    # top-key level version
    dataset_index = {"version": "1.0"}

    # combine all in dataset index
    dataset_index.update({"tracks": index_tracks})

    with open(DATASET_INDEX_PATH, "w") as fhandle:
        json.dump(dataset_index, fhandle, indent=2)


def main(dataset_data_path: dict):
    make_small_4mula_index(dataset_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make 4mula small index file.")
    PARSER.add_argument(
        "dataset_data_path", type=str, help="Path to 4mula small data folder.", default="4mula"
    )
    args = PARSER.parse_args()

    filename = "4mula_small.parquet"
    destination_dir = "4mula" if len(args.dataset_data_path) == 0 else args.dataset_data_path

    remote = RemoteFileMetadata(url="https://zenodo.org/record/4636802/files/4mula_small.parquet?download=1",
                                filename=f"{filename}", destination_dir=destination_dir,
                                checksum="30210cf6f52449c8d0670fc0942410c4")

    dataset_data_path = download_from_remote(remote, save_dir=destination_dir, force_overwrite=False)
    args = {"dataset_folder": dataset_data_path.split(filename)[0], "filename": filename}
    main(args)
