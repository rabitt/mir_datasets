import argparse
import collections
import hashlib
import json
import os

CANTE100_INDEX_PATH = "../mirdata/datasets/indexes/cante100_index.json"


def md5(file_path):
    """Get md5 hash of a file.

    Parameters
    ----------
    file_path: str
        File path.
    Returns
    -------
    md5_hash: str
        md5 hash of data in file_path
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as fhandle:
        for chunk in iter(lambda: fhandle.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def strip_first_dir(full_path):
    return os.path.join(*(full_path.split(os.path.sep)[1:]))


def make_cante100_index(cante100_data_path):
    cante100_dict = dict()
    for root, dirs, files in os.walk(cante100_data_path):
        for directory in dirs:
            if "spectrum" in directory:
                for root_, dirs_, files_ in os.walk(os.path.join(cante100_data_path, directory)):
                    for file in files_:
                        file_id = file.split("_")[0]
                        file_name = file.split(".")[0]

                        # data
                        audio_name = str(file_name) + ".mp3"
                        spectrum_name = str(file_name) + ".spectrum.csv"
                        f0_name = str(file_name) + ".f0.csv"
                        notes_name = str(file_name) + ".notes.csv"

                        cante100_dict[file_id] = [
                            os.path.join("cante100audio", audio_name),
                            os.path.join(directory, spectrum_name),
                            os.path.join("cante100midi_f0", f0_name),
                            os.path.join("cante100_automaticTranscription", notes_name),
                        ]

    cante100_dict = fix_audio_notation_errors(cante100_dict)
    cante100_dict_ord = collections.OrderedDict(sorted(cante100_dict.items()))
    print(cante100_dict_ord.items())

    cante100_index = {}
    for index, inst in cante100_dict_ord.items():
        print(index)
        audio_checksum = md5(os.path.join(cante100_data_path, inst[0]))
        spectrum_checksum = md5(os.path.join(cante100_data_path, inst[1]))
        f0_checksum = md5(os.path.join(cante100_data_path, inst[2]))
        notes_checksum = md5(os.path.join(cante100_data_path, inst[3]))

        cante100_index[index] = {
            "audio": (inst[0], audio_checksum),
            "spectrum": (inst[1], spectrum_checksum),
            "f0": (inst[2], f0_checksum),
            "notes": (inst[3], notes_checksum),
        }

    with open(CANTE100_INDEX_PATH, "w") as fhandle:
        json.dump(cante100_index, fhandle, indent=2)


def fix_audio_notation_errors(cante100_dict):
    """Patch to fix cante100 audio notation errors to make index work well.

    Known issues:
        - Wrong named files: 17, 80, 90.

    Parameters
    ----------
    cante100_dict

    Returns
    -------
    cante100_dict (corrected)
    """

    cante100_dict["017"][0] = "cante100audio/017_ManuelSotoSorder_TientosTangos.mp3"
    cante100_dict["080"][0] = "cante100audio/080_PericonDeCadiz_CantesAmericanos.mp3"
    cante100_dict["091"][0] = "cante100audio/090_PepeDeLaMatrona_Tonas.mp3"

    return cante100_dict


def main(args):
    make_cante100_index(args.cante100_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make cante100 index file.")
    PARSER.add_argument("cante100_data_path", type=str, help="Path to cante100 data folder.")

    main(PARSER.parse_args())
