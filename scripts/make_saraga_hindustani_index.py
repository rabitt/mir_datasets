import argparse
import json
import os

from mirdata.validate import md5

SARAGA_HINDUSTANI_INDEX_PATH = "../mirdata/datasets/indexes/saraga_hindustani_index.json"


def make_saraga_hindustani_index(dataset_data_path):
    saraga_index = {
        "version": 1.5,
        "tracks": {},
    }
    idx = 0
    dataset_data_path_prev = dataset_data_path.split("saraga1.5_hindustani")[0]
    for concert in os.listdir(dataset_data_path):
        if "." not in concert:
            for song in os.listdir(os.path.join(dataset_data_path, concert)):
                if "." not in song:
                    # Declare track attributes
                    index = str(idx) + "_" + song.replace(" ", "_")
                    print(index)
                    audio = (None, None)
                    ctonic = (None, None)
                    pitch = (None, None)
                    tempo = (None, None)
                    sama = (None, None)
                    sections = (None, None)
                    phrases = (None, None)
                    metadata = (None, None)

                    for file in os.listdir(os.path.join(dataset_data_path, concert, song)):
                        if ".mp3" in file:
                            audio_path = os.path.join("saraga1.5_hindustani/", concert, song, file)
                            audio_checksum = md5(os.path.join(dataset_data_path_prev, audio_path))
                            audio = (audio_path, audio_checksum)
                        if "ctonic" in file:
                            ctonic_path = os.path.join("saraga1.5_hindustani/", concert, song, file)
                            ctonic_checksum = md5(os.path.join(dataset_data_path_prev, ctonic_path))
                            ctonic = (ctonic_path, ctonic_checksum)
                        if "pitch." in file:
                            pitch_path = os.path.join("saraga1.5_hindustani/", concert, song, file)
                            pitch_checksum = md5(os.path.join(dataset_data_path_prev, pitch_path))
                            pitch = (pitch_path, pitch_checksum)
                        if "tempo-manual" in file:
                            tempo_path = os.path.join("saraga1.5_hindustani/", concert, song, file)
                            tempo_checksum = md5(os.path.join(dataset_data_path_prev, tempo_path))
                            tempo = (tempo_path, tempo_checksum)
                        if "sama-manual" in file:
                            sama_path = os.path.join("saraga1.5_hindustani/", concert, song, file)
                            sama_checksum = md5(os.path.join(dataset_data_path_prev, sama_path))
                            sama = (sama_path, sama_checksum)
                        if "sections-manual-p" in file:
                            sections_path = os.path.join(
                                "saraga1.5_hindustani/", concert, song, file
                            )
                            sections_checksum = md5(
                                os.path.join(dataset_data_path_prev, sections_path)
                            )
                            sections = (sections_path, sections_checksum)
                        if "mphrase" in file:
                            phrases_path = os.path.join(
                                "saraga1.5_hindustani/", concert, song, file
                            )
                            phrases_checksum = md5(
                                os.path.join(dataset_data_path_prev, phrases_path)
                            )
                            phrases = (phrases_path, phrases_checksum)
                        if ".json" in file:
                            metadata_path = os.path.join(
                                "saraga1.5_hindustani/", concert, song, file
                            )
                            metadata_checksum = md5(
                                os.path.join(dataset_data_path_prev, metadata_path)
                            )
                            metadata = (metadata_path, metadata_checksum)

                        saraga_index["tracks"][index] = {
                            "audio": audio,
                            "ctonic": ctonic,
                            "pitch": pitch,
                            "tempo": tempo,
                            "sama": sama,
                            "sections": sections,
                            "phrases": phrases,
                            "metadata": metadata,
                        }

                    idx = idx + 1

    with open(SARAGA_HINDUSTANI_INDEX_PATH, "w") as fhandle:
        json.dump(saraga_index, fhandle, indent=2)


def main(args):
    print("creating index...")
    make_saraga_hindustani_index(args.dataset_data_path)
    print("done!")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make Saraga Hindustani index file.")
    PARSER.add_argument(
        "dataset_data_path", type=str, help="Path to Saraga Hindustani data folder."
    )

    main(PARSER.parse_args())
