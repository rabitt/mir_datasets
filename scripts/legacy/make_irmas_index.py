import argparse
import hashlib
import json
import os

IRMAS_INDEX_PATH = "../mirdata/indexes/irmas_index.json"


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


def make_irmas_index(irmas_data_path):
    count = 0
    irmas_dict = dict()
    for root, dirs, files in os.walk(irmas_data_path):
        for directory in dirs:
            if "Train" in directory:
                for root_, dirs_, files_ in os.walk(os.path.join(irmas_data_path, directory)):
                    for directory_ in dirs_:
                        for root__, dirs__, files__ in os.walk(
                            os.path.join(irmas_data_path, directory, directory_)
                        ):
                            for file in files__:
                                if file.endswith(".wav"):
                                    if "dru" in file:
                                        irmas_id_dru = file.split("]")[3]  # Obtain id
                                        irmas_id_dru_no_wav = irmas_id_dru.split(".")[
                                            0
                                        ]  # Obtain id without '.wav'
                                        irmas_dict[irmas_id_dru_no_wav] = os.path.join(
                                            directory, directory_, file
                                        )
                                    if "nod" in file:
                                        irmas_id_nod = file.split("]")[3]  # Obtain id
                                        irmas_id_nod_no_wav = irmas_id_nod.split(".")[
                                            0
                                        ]  # Obtain id without '.wav'
                                        irmas_dict[irmas_id_nod_no_wav] = os.path.join(
                                            directory, directory_, file
                                        )
                                    else:
                                        irmas_id = file.split("]")[2]  # Obtain id
                                        irmas_id_no_wav = irmas_id.split(".")[
                                            0
                                        ]  # Obtain id without '.wav'
                                        irmas_dict[irmas_id_no_wav] = os.path.join(
                                            directory, directory_, file
                                        )

    irmas_test_dict = dict()
    for root, dirs, files in os.walk(irmas_data_path):
        for directory in dirs:
            if "Test" in directory:
                for root_, dirs_, files_ in os.walk(os.path.join(irmas_data_path, directory)):
                    for directory_ in dirs_:
                        for root__, dirs__, files__ in os.walk(
                            os.path.join(irmas_data_path, directory, directory_)
                        ):
                            for file in files__:
                                if file.endswith(".wav"):
                                    file_name = os.path.join(directory, directory_, file)
                                    track_name = str(file_name.split(".wa")[0]) + ".txt"
                                    irmas_test_dict[count] = [file_name, track_name]
                                    count += 1

    irmas_id_list = sorted(irmas_dict.items())  # Sort strokes by id

    irmas_index = {}
    for inst in irmas_id_list:
        print(inst[1])
        audio_checksum = md5(os.path.join(irmas_data_path, inst[1]))

        irmas_index[inst[0]] = {
            "audio": (inst[1], audio_checksum),
            "annotation": (inst[1], audio_checksum),
        }

    index = 1
    for inst in irmas_test_dict.values():
        audio_checksum = md5(os.path.join(irmas_data_path, inst[0]))
        annotation_checksum = md5(os.path.join(irmas_data_path, inst[1]))

        irmas_index[index] = {
            "audio": (inst[0], audio_checksum),
            "annotation": (inst[1], annotation_checksum),
        }
        index += 1

    with open(IRMAS_INDEX_PATH, "w") as fhandle:
        json.dump(irmas_index, fhandle, indent=2)


def make_irmas_test_index(irmas_data_path):
    count = 1
    irmas_dict = dict()
    for root, dirs, files in os.walk(irmas_data_path):
        for directory in dirs:
            if "Test" in directory:
                for root_, dirs_, files_ in os.walk(os.path.join(irmas_data_path, directory)):
                    for directory_ in dirs_:
                        for root__, dirs__, files__ in os.walk(
                            os.path.join(irmas_data_path, directory, directory_)
                        ):
                            for file in files__:
                                if file.endswith(".wav"):
                                    file_name = os.path.join(directory, directory_, file)
                                    track_name = str(file_name.split(".wa")[0]) + ".txt"
                                    irmas_dict[count] = [file_name, track_name]
                                    count += 1

    irmas_index = {}
    index = 1
    for inst in irmas_dict.values():
        audio_checksum = md5(os.path.join(irmas_data_path, inst[0]))
        annotation_checksum = md5(os.path.join(irmas_data_path, inst[1]))

        irmas_index[index] = {
            "audio": (inst[0], audio_checksum),
            "annotation": (inst[1], annotation_checksum),
        }
        index += 1

    with open(IRMAS_TEST_INDEX_PATH, "w") as fhandle:
        json.dump(irmas_index, fhandle, indent=2)


def main(args):
    make_irmas_index(args.irmas_data_path)
    # make_irmas_test_index(args.irmas_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make IRMAS index file.")
    PARSER.add_argument("irmas_data_path", type=str, help="Path to IRMAS data folder.")

    main(PARSER.parse_args())
