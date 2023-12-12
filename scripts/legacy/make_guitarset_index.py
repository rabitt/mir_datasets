import argparse
import glob
import hashlib
import json
import os

GUITARSET_INDEX_PATH = "../mirdata/indexes/guitarset_index.json"


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


def make_guitarset_index(guitarset_data_path):
    anno_dir = os.path.join(guitarset_data_path, "annotation")
    audio_hex_cln_dir = os.path.join(guitarset_data_path, "audio_hex-pickup_debleeded")
    audio_hex_dir = os.path.join(guitarset_data_path, "audio_hex-pickup_original")
    audio_mic_dir = os.path.join(guitarset_data_path, "audio_mono-mic")
    audio_mix_dir = os.path.join(guitarset_data_path, "audio_mono-pickup_mix")

    jams_files = glob.glob(os.path.join(anno_dir, "*.jams"))
    track_ids = sorted([os.path.basename(f).split(".")[0] for f in jams_files])

    guitarset_index = {}
    for track_id in track_ids:
        annotation_checksum = md5(os.path.join(anno_dir, "{}.jams".format(track_id)))
        audio_hex_cln_checksum = md5(
            os.path.join(audio_hex_cln_dir, "{}_hex_cln.wav".format(track_id))
        )
        audio_hex_checksum = md5(os.path.join(audio_hex_dir, "{}_hex.wav".format(track_id)))
        audio_mic_checksum = md5(os.path.join(audio_mic_dir, "{}_mic.wav".format(track_id)))
        audio_mix_checksum = md5(os.path.join(audio_mix_dir, "{}_mix.wav".format(track_id)))

        guitarset_index[track_id] = {
            "audio_hex_cln": (
                "audio_hex-pickup_debleeded/{}_hex_cln.wav".format(track_id),
                audio_hex_cln_checksum,
            ),
            "audio_hex": (
                "audio_hex-pickup_original/{}_hex.wav".format(track_id),
                audio_hex_checksum,
            ),
            "audio_mic": (
                "audio_mono-mic/{}_mic.wav".format(track_id),
                audio_mic_checksum,
            ),
            "audio_mix": (
                "audio_mono-pickup_mix/{}_mix.wav".format(track_id),
                audio_mix_checksum,
            ),
            "jams": ("annotation/{}.jams".format(track_id), annotation_checksum),
        }

    with open(GUITARSET_INDEX_PATH, "w") as fhandle:
        json.dump(guitarset_index, fhandle, indent=2)


def main(args):
    make_guitarset_index(args.guitarset_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make GuitarSet index file.")
    PARSER.add_argument("guitarset_data_path", type=str, help="Path to GuitarSet data folder.")

    main(PARSER.parse_args())
