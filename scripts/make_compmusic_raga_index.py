import argparse
import glob
import json
import os

from mirdata.validate import md5

RAGA_DATASET_INDEX_PATH = "../mirdata/datasets/indexes/compmusic_raga_index_1.0.json"


def make_compmusic_raga_index(dataset_data_path):
    raga_index = {"version": "1.0", "tracks": {}}
    dataset_folder = "RagaDataset"

    traditions = ["Carnatic", "Hindustani"]
    for trad in traditions:
        tradition_name = trad
        for raga in glob.glob(os.path.join(dataset_data_path, trad, "audio", "*/")):
            raga_id = raga.split("/")[-2]
            if "." not in raga:
                for artist in glob.glob(os.path.join(raga, "*/")):
                    if "." not in artist:
                        artist_name = artist.split("/")[-2]
                        for concert in glob.glob(os.path.join(artist, "*/")):
                            if trad == "Hindustani":
                                print(concert)
                            if "." not in concert:
                                concert_name = concert.split("/")[-2]
                                for audio_basefile in glob.glob(os.path.join(concert, "*/")):
                                    song_name = audio_basefile.split("/")[-2]
                                    id = artist_name + "." + song_name
                                    audio_file = os.path.join(audio_basefile, song_name + ".mp3")
                                    feat_basefile = audio_basefile.replace("audio", "features")
                                    raga_index["tracks"][id] = {
                                        "audio": (None, None),
                                        "tonic": (None, None),
                                        "tonic_fine_tuned": (None, None),
                                        "pitch": (None, None),
                                        "pitch_post_processed": (None, None),
                                        "nyas_segments": (None, None),
                                        "tani_segments": (None, None),
                                    }
                                    # audio
                                    raga_index["tracks"][id]["audio"] = (
                                        os.path.join(
                                            dataset_folder,
                                            tradition_name,
                                            "audio",
                                            raga_id,
                                            artist_name,
                                            concert_name,
                                            song_name,
                                            song_name + ".mp3",
                                        ),
                                        md5(audio_file),
                                    )
                                    # tonic
                                    raga_index["tracks"][id]["tonic"] = (
                                        os.path.join(
                                            dataset_folder,
                                            tradition_name,
                                            "features",
                                            raga_id,
                                            artist_name,
                                            concert_name,
                                            song_name,
                                            song_name + ".tonic",
                                        ),
                                        md5(os.path.join(feat_basefile, song_name + ".tonic")),
                                    )
                                    # tonic fine
                                    raga_index["tracks"][id]["tonic_fine_tuned"] = (
                                        os.path.join(
                                            dataset_folder,
                                            tradition_name,
                                            "features",
                                            raga_id,
                                            artist_name,
                                            concert_name,
                                            song_name,
                                            song_name + ".tonicFine",
                                        ),
                                        md5(os.path.join(feat_basefile, song_name + ".tonicFine")),
                                    )
                                    # pitch
                                    raga_index["tracks"][id]["pitch"] = (
                                        os.path.join(
                                            dataset_folder,
                                            tradition_name,
                                            "features",
                                            raga_id,
                                            artist_name,
                                            concert_name,
                                            song_name,
                                            song_name + ".pitch",
                                        ),
                                        md5(os.path.join(feat_basefile, song_name + ".pitch")),
                                    )
                                    # pitch postprocessed
                                    raga_index["tracks"][id]["pitch_post_processed"] = (
                                        os.path.join(
                                            dataset_folder,
                                            tradition_name,
                                            "features",
                                            raga_id,
                                            artist_name,
                                            concert_name,
                                            song_name,
                                            song_name + ".pitchSilIntrpPP",
                                        ),
                                        md5(
                                            os.path.join(
                                                feat_basefile, song_name + ".pitchSilIntrpPP"
                                            )
                                        ),
                                    )
                                    # nyas segments
                                    raga_index["tracks"][id]["nyas_segments"] = (
                                        os.path.join(
                                            dataset_folder,
                                            tradition_name,
                                            "features",
                                            raga_id,
                                            artist_name,
                                            concert_name,
                                            song_name,
                                            song_name + ".flatSegNyas",
                                        ),
                                        md5(
                                            os.path.join(feat_basefile, song_name + ".flatSegNyas")
                                        ),
                                    )
                                    # tani segments
                                    raga_index["tracks"][id]["tani_segments"] = (
                                        os.path.join(
                                            dataset_folder,
                                            tradition_name,
                                            "features",
                                            raga_id,
                                            artist_name,
                                            concert_name,
                                            song_name,
                                            song_name + ".taniSegKNN",
                                        ),
                                        md5(os.path.join(feat_basefile, song_name + ".taniSegKNN")),
                                    )

    with open(RAGA_DATASET_INDEX_PATH, "w") as fhandle:
        json.dump(raga_index, fhandle, indent=2)


def main(args):
    print("creating index...")
    make_compmusic_raga_index(args.dataset_data_path)
    print("done!")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make CompMusic RagaDataset index file.")
    PARSER.add_argument(
        "dataset_data_path",
        type=str,
        help="Path to CompMusic RagaDataset data folder.",
    )

    main(PARSER.parse_args())
