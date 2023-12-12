import argparse
import json
import os

from mirdata.validate import md5

JINJGU_ACAPPELLA_INDEX_PATH = "../mirdata/datasets/indexes/compmusic_jingju_acappella_index.json"


def make_jingju_acappella_index(dataset_data_path):
    jingju_index = {"version": 7.0, "tracks": {}, "metadata": {}}

    # Building the index while parsing the audio path
    for folder in os.listdir(dataset_data_path):
        if "wav" in folder:
            for folder_ in os.listdir(os.path.join(dataset_data_path, folder)):
                if "." not in folder_:
                    for song in os.listdir(os.path.join(dataset_data_path, folder, folder_)):
                        if ".DS" not in song:
                            index = song.replace(".wav", "").replace(".WAV", "")
                            jingju_index["tracks"][index] = {
                                "audio": (None, None),
                                "phoneme": (None, None),
                                "phrase_char": (None, None),
                                "phrase": (None, None),
                                "syllable": (None, None),
                            }
                            jingju_index["tracks"][index]["audio"] = (
                                os.path.join(folder, folder_, song),
                                md5(os.path.join(dataset_data_path, folder, folder_, song)),
                            )

    # Parsing annotations and textgrid
    for folder in os.listdir(dataset_data_path):
        if "annotation_txt" in folder:
            for folder_ in os.listdir(os.path.join(dataset_data_path, folder)):
                if "." not in folder_:
                    for file in os.listdir(os.path.join(dataset_data_path, folder, folder_)):
                        if "phoneme" in file:
                            index = file.replace("_phoneme.txt", "")
                            jingju_index["tracks"][index]["phoneme"] = (
                                os.path.join(folder, folder_, file),
                                md5(os.path.join(dataset_data_path, folder, folder_, file)),
                            )
                        if "phrase_char" in file:
                            index = file.replace("_phrase_char.txt", "")
                            jingju_index["tracks"][index]["phrase_char"] = (
                                os.path.join(folder, folder_, file),
                                md5(os.path.join(dataset_data_path, folder, folder_, file)),
                            )
                        if "phrase.txt" in file:
                            index = file.replace("_phrase.txt", "")
                            jingju_index["tracks"][index]["phrase"] = (
                                os.path.join(folder, folder_, file),
                                md5(os.path.join(dataset_data_path, folder, folder_, file)),
                            )
                        if "syllable" in file:
                            index = file.replace("_syllable.txt", "")
                            jingju_index["tracks"][index]["syllable"] = (
                                os.path.join(folder, folder_, file),
                                md5(os.path.join(dataset_data_path, folder, folder_, file)),
                            )

    # Parsing metadata
    for file in os.listdir(dataset_data_path):
        if "catalogue" in file:
            if "dan" in file:
                jingju_index["metadata"]["dan_metadata"] = (
                    file,
                    md5(os.path.join(dataset_data_path, file)),
                )
            else:
                jingju_index["metadata"]["laosheng_metadata"] = (
                    file,
                    md5(os.path.join(dataset_data_path, file)),
                )

    with open(JINJGU_ACAPPELLA_INDEX_PATH, "w") as fhandle:
        json.dump(jingju_index, fhandle, indent=2)


def main(args):
    print("creating index...")
    make_jingju_acappella_index(args.dataset_data_path)
    print("done!")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make CompMusic Jingju A Cappella index file.")
    PARSER.add_argument(
        "dataset_data_path",
        type=str,
        help="Path to CompMusic Jingju A Cappella data folder.",
    )

    main(PARSER.parse_args())
