import argparse
import glob
import json
import os

from mirdata.validate import md5

FOUR_WAY_TABLA_INDEX_PATH = "../mirdata/datasets/indexes/four_way_tabla_index.json"


def make_four_way_tabla_index(dataset_data_path):
    tabla_index = {"version": 1.0, "tracks": {}, "metadata": {}}
    subsets = ["train", "test"]
    srokes = ["b", "d", "rb", "rt"]
    # Building the index while parsing the audio path
    for subset in subsets:
        for sample in glob.glob(os.path.join(dataset_data_path, subset, "audios", "*.wav")):
            index = sample.split("/")[-1].replace(".wav", "")
            tabla_index["tracks"][index] = {
                "audio": (None, None),
                "onsets_b": (None, None),
                "onsets_d": (None, None),
                "onsets_rb": (None, None),
                "onsets_rt": (None, None),
            }
            tabla_index["tracks"][index]["audio"] = (
                os.path.join("4way-tabla-ismir21-dataset", subset, "audios", index + ".wav"),
                md5(os.path.join(dataset_data_path, subset, "audios", index + ".wav")),
            )
            for stroke in srokes:
                tabla_index["tracks"][index]["onsets_" + stroke] = (
                    os.path.join(
                        "4way-tabla-ismir21-dataset", subset, "onsets", stroke, index + ".onsets"
                    ),
                    md5(
                        os.path.join(dataset_data_path, subset, "onsets", stroke, index + ".onsets")
                    ),
                )

    with open(FOUR_WAY_TABLA_INDEX_PATH, "w") as fhandle:
        json.dump(tabla_index, fhandle, indent=2)


def main(args):
    print("creating index...")
    make_four_way_tabla_index(args.dataset_data_path)
    print("done!")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Make Four-Way Table Stroke Classification index file."
    )
    PARSER.add_argument(
        "dataset_data_path",
        type=str,
        help="Path to Four-Way Table Stroke Classification data folder.",
    )

    main(PARSER.parse_args())
