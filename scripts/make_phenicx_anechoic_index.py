import argparse
import glob
import json
import os
import string

from mirdata.validate import md5

DATASET_INDEX_PATH = "../mirdata/datasets/indexes/phenicx_anechoic_index.json"


def make_dataset_index(data_path):
    pieces = ["beethoven", "bruckner", "mahler", "mozart"]
    families = {
        "doublebass": "strings",
        "cello": "strings",
        "clarinet": "woodwinds",
        "viola": "strings",
        "violin": "strings",
        "oboe": "woodwinds",
        "flute": "woodwinds",
        "trumpet": "brass",
        "bassoon": "woodwinds",
        "horn": "brass",
    }
    totalinstruments = [20, 39, 30, 10]
    ninstruments = [10, 10, 10, 8]
    index = {"version": 1}

    index["tracks"] = {}
    index["multitracks"] = {}

    for ip, piece in enumerate(pieces):
        index["multitracks"][piece] = {}

        audio_files = sorted(glob.glob(os.path.join(data_path, "audio", piece, "*.wav")))
        instruments = [
            os.path.basename(audio_path).split(".")[0].rstrip(string.digits)
            for audio_path in audio_files
        ]
        set_instruments = list(set(instruments))

        assert (
            len(instruments) == totalinstruments[ip]
        ), "audio files for some instruments are missing"
        assert (
            len(set_instruments) == ninstruments[ip]
        ), "some instruments are missing from the dataset"

        index["multitracks"][piece]["tracks"] = []
        for instrument in set_instruments:
            assert (
                instrument in families.keys()
            ), "instrument {} is not in the list of dataset instruments".format(instrument)
            index["tracks"][piece + "-" + instrument] = {}
            index["multitracks"][piece]["tracks"].append(piece + "-" + instrument)

            #### add audios
            instrument_audio_files = sorted(
                glob.glob(os.path.join(data_path, "audio", piece, instrument + "*.wav"))
            )
            assert len(instrument_audio_files) > 0, "no audio has been found for {}".format(
                instrument
            )

            for i, audio_file in enumerate(instrument_audio_files):
                audio_checksum = md5(
                    os.path.join(data_path, "audio", piece, os.path.basename(audio_file))
                )
                source = os.path.basename(audio_file).replace(".wav", "")

                index["tracks"][piece + "-" + instrument]["audio_" + source] = (
                    "audio/{}/{}".format(piece, os.path.basename(audio_file)),
                    audio_checksum,
                )

            #### add scores
            assert os.path.exists(
                os.path.join(data_path, "annotations", piece, "{}.txt".format(instrument))
            ), "cannot find score file {}".formatos.path.join(
                data_path, "annotations", piece, "{}.txt".format(instrument)
            )
            assert os.path.exists(
                os.path.join(data_path, "annotations", piece, "{}_o.txt".format(instrument))
            ), "cannot find score file {}".formatos.path.join(
                data_path, "annotations", piece, "{}_o.txt".format(instrument)
            )

            score_checksum = md5(
                os.path.join(data_path, "annotations", piece, "{}.txt".format(instrument))
            )
            score_original_checksum = md5(
                os.path.join(data_path, "annotations", piece, "{}_o.txt".format(instrument))
            )

            index["tracks"][piece + "-" + instrument]["notes"] = (
                "annotations/{}/{}.txt".format(piece, instrument),
                score_checksum,
            )
            index["tracks"][piece + "-" + instrument]["notes_original"] = (
                "annotations/{}/{}_o.txt".format(piece, instrument),
                score_original_checksum,
            )

    with open(DATASET_INDEX_PATH, "w") as fhandle:
        json.dump(index, fhandle, indent=2)


def main(args):
    make_dataset_index(args.data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make Phenicx-anechoic index file.")
    PARSER.add_argument("data_path", type=str, help="Path to Phenicx-anechoic data folder.")

    main(PARSER.parse_args())
