import argparse
import glob
import json
import os

from mirdata.validate import md5

COMPMUSIC_TONIC_INDEX_PATH = "../mirdata/datasets/indexes/compmusic_indian_tonic_1.0.json"


def make_compmusic_indian_tonic(dataset_data_path):
    tonic_index = {"version": "1.0", "tracks": {}, "metadata": {}}

    for center_fold in glob.glob(os.path.join(dataset_data_path, "*/")):
        center = center_fold.split("/")[-2]
        for metafile in glob.glob(os.path.join(center_fold, "annotations", center + "*.json")):
            if "IITM1" not in metafile:
                with open(metafile) as fhandle:
                    meta = json.load(fhandle)
                    files = list(meta.keys())
                    wrongly_annotated = [
                        "05-saamajavara-hindolam.mp3",
                        "08-aajaa-sindhubhairavi.mp3",
                        "01-varnam-nayaki.mp3",
                    ]
                    for fil in files:
                        if any(
                            fil.split("/")[-1].replace(".mp3", "") in s
                            for s in wrongly_annotated
                            if len(fil.split("/")[-1].replace(".mp3", "")) > 8
                        ):
                            idx = fil.split("/")[-1].replace(".mp3", "")
                            print(idx)
                            tonic_index["tracks"][idx] = {
                                "audio": [
                                    os.path.join(
                                        "indian_art_music_tonic_1.0",
                                        "IITM",
                                        "audio",
                                        "Tonic_Data2",
                                        "TMKKamboji1folder",
                                        idx + ".mp3",
                                    ),
                                    md5(
                                        os.path.join(
                                            dataset_data_path,
                                            "IITM",
                                            "audio",
                                            "Tonic_Data2",
                                            "TMKKamboji1folder",
                                            idx + ".mp3",
                                        )
                                    ),
                                ]
                            }
                        else:
                            remove_ampersans = [
                                "11a-begada-alapana&tanam",
                                "11b-pallavi&ragamalika",
                                "15-EmayyaBelaga&Melukovayya-Bauli",
                            ]
                            if any(
                                fil.split("/")[-1].replace(".mp3", "") in s
                                for s in remove_ampersans
                            ):
                                fil = fil.replace("&", "_")
                            idx = fil.split("/")[-1].replace(".mp3", "")
                            tonic_index["tracks"][idx] = {
                                "audio": [
                                    fil,
                                    md5(
                                        os.path.join(
                                            dataset_data_path.replace(
                                                "/indian_art_music_tonic_1.0", ""
                                            ),
                                            fil,
                                        )
                                    ),
                                ]
                            }

    tonic_index["metadata"]["CM1"] = [
        os.path.join("indian_art_music_tonic_1.0", "CM", "annotations", "CM1.json"),
        md5(os.path.join(dataset_data_path, "CM", "annotations", "CM1.json")),
    ]
    tonic_index["metadata"]["CM2"] = [
        os.path.join("indian_art_music_tonic_1.0", "CM", "annotations", "CM2.json"),
        md5(os.path.join(dataset_data_path, "CM", "annotations", "CM2.json")),
    ]
    tonic_index["metadata"]["CM3"] = [
        os.path.join("indian_art_music_tonic_1.0", "CM", "annotations", "CM3.json"),
        md5(os.path.join(dataset_data_path, "CM", "annotations", "CM3.json")),
    ]
    tonic_index["metadata"]["IISc"] = [
        os.path.join("indian_art_music_tonic_1.0", "IISc", "annotations", "IISc.json"),
        md5(os.path.join(dataset_data_path, "IISc", "annotations", "IISc.json")),
    ]
    tonic_index["metadata"]["IITM2"] = [
        os.path.join("indian_art_music_tonic_1.0", "IITM", "annotations", "IITM2.json"),
        md5(os.path.join(dataset_data_path, "IITM", "annotations", "IITM2.json")),
    ]

    with open(COMPMUSIC_TONIC_INDEX_PATH, "w") as fhandle:
        json.dump(tonic_index, fhandle, indent=2)


def main(args):
    print("creating index...")
    make_compmusic_indian_tonic(args.dataset_data_path)
    print("done!")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make CompMusic Tonic Dataset index file.")
    PARSER.add_argument(
        "dataset_data_path",
        type=str,
        help="Path to CompMusic Tonic Dataset data folder.",
    )

    main(PARSER.parse_args())
