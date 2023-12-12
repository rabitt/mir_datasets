import argparse
import glob
import hashlib
import json
import os

ORCHSET_INDEX_PATH = "../mirdata/indexes/orchset_index.json"


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


def make_orchset_index(data_path):
    mono_audio_files = sorted(glob.glob(os.path.join(data_path, "audio", "mono", "*.wav")))

    index = {}
    for audio_path in mono_audio_files:
        track_id = os.path.basename(audio_path).split(".")[0]

        audio_stereo_checksum = md5(
            os.path.join(data_path, "audio", "stereo", "{}.wav".format(track_id))
        )
        audio_mono_checksum = md5(
            os.path.join(data_path, "audio", "mono", "{}.wav".format(track_id))
        )
        melody_checksum = md5(os.path.join(data_path, "GT", "{}.mel".format(track_id)))

        index[track_id] = {
            "audio_stereo": (
                "audio/stereo/{}.wav".format(track_id),
                audio_stereo_checksum,
            ),
            "audio_mono": ("audio/mono/{}.wav".format(track_id), audio_mono_checksum),
            "melody": ("GT/{}.mel".format(track_id), melody_checksum),
        }

    with open(ORCHSET_INDEX_PATH, "w") as fhandle:
        json.dump(index, fhandle, indent=2)


def main(args):
    make_orchset_index(args.orchset_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make Orchset index file.")
    PARSER.add_argument("orchset_data_path", type=str, help="Path to Orchset data folder.")

    main(PARSER.parse_args())
