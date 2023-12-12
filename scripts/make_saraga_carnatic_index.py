import argparse
import json
import os

from mirdata.validate import md5

SARAGA_CARNATIC_INDEX_PATH = "../mirdata/datasets/indexes/saraga_carnatic_index.json"


def make_saraga_carnatic_index(dataset_data_path):
    saraga_index = {
        "version": 1.5,
        "tracks": {},
    }
    idx = 0
    dataset_data_path_prev = dataset_data_path.split("saraga1.5_carnatic/")[0]
    for concert in os.listdir(dataset_data_path):
        if "." not in concert:
            for song in os.listdir(os.path.join(dataset_data_path, concert)):
                if "." not in song:
                    # Declare track attributes
                    index = str(idx) + "_" + song.replace(" ", "_")
                    print(index)
                    audio = (None, None)
                    audio_ghatam = (None, None)
                    audio_mridangam_left = (None, None)
                    audio_mridangam_right = (None, None)
                    audio_violin = (None, None)
                    audio_vocal = (None, None)
                    audio_vocal_s = (None, None)
                    ctonic = (None, None)
                    pitch = (None, None)
                    pitch_v = (None, None)
                    tempo = (None, None)
                    sama = (None, None)
                    sections = (None, None)
                    phrases = (None, None)
                    metadata = (None, None)

                    for file in os.listdir(os.path.join(dataset_data_path, concert, song)):
                        if ".mp3" in file:
                            if "multitrack" in file:
                                if "ghatam" in file:
                                    audio_ghatam_path = os.path.join(
                                        "saraga1.5_carnatic", concert, song, file
                                    )
                                    audio_ghatam_checksum = md5(
                                        os.path.join(dataset_data_path_prev, audio_ghatam_path)
                                    )
                                    audio_ghatam = (audio_ghatam_path, audio_ghatam_checksum)
                                if "mridangam-left" in file:
                                    audio_mridangam_left_path = os.path.join(
                                        "saraga1.5_carnatic", concert, song, file
                                    )
                                    audio_mridangam_left_checksum = md5(
                                        os.path.join(
                                            dataset_data_path_prev, audio_mridangam_left_path
                                        )
                                    )
                                    audio_mridangam_left = (
                                        audio_mridangam_left_path,
                                        audio_mridangam_left_checksum,
                                    )
                                if "mridangam-right" in file:
                                    mridangam_right_path = os.path.join(
                                        "saraga1.5_carnatic", concert, song, file
                                    )
                                    mridangam_right_checksum = md5(
                                        os.path.join(dataset_data_path_prev, mridangam_right_path)
                                    )
                                    audio_mridangam_right = (
                                        mridangam_right_path,
                                        mridangam_right_checksum,
                                    )
                                if "violin" in file:
                                    audio_violin_path = os.path.join(
                                        "saraga1.5_carnatic", concert, song, file
                                    )
                                    audio_violin_checksum = md5(
                                        os.path.join(dataset_data_path_prev, audio_violin_path)
                                    )
                                    audio_violin = (audio_violin_path, audio_violin_checksum)
                                if "vocal-s" in file:
                                    audio_vocal_s_path = os.path.join(
                                        "saraga1.5_carnatic", concert, song, file
                                    )
                                    audio_vocal_s_checksum = md5(
                                        os.path.join(dataset_data_path_prev, audio_vocal_s_path)
                                    )
                                    audio_vocal_s = (audio_vocal_s_path, audio_vocal_s_checksum)
                                if "vocal." in file:
                                    audio_vocal_path = os.path.join(
                                        "saraga1.5_carnatic", concert, song, file
                                    )
                                    audio_vocal_checksum = md5(
                                        os.path.join(dataset_data_path_prev, audio_vocal_path)
                                    )
                                    audio_vocal = (audio_vocal_path, audio_vocal_checksum)

                            else:
                                audio_path = os.path.join("saraga1.5_carnatic", concert, song, file)
                                audio_checksum = md5(
                                    os.path.join(dataset_data_path_prev, audio_path)
                                )
                                audio = (audio_path, audio_checksum)

                        if "ctonic." in file:
                            ctonic_path = os.path.join("saraga1.5_carnatic", concert, song, file)
                            ctonic_checksum = md5(os.path.join(dataset_data_path_prev, ctonic_path))
                            ctonic = (ctonic_path, ctonic_checksum)
                        if "pitch." in file:
                            pitch_path = os.path.join("saraga1.5_carnatic", concert, song, file)
                            pitch_checksum = md5(os.path.join(dataset_data_path_prev, pitch_path))
                            pitch = (pitch_path, pitch_checksum)
                        if "pitch-vocal" in file:
                            pitch_v_path = os.path.join("saraga1.5_carnatic", concert, song, file)
                            pitch_v_checksum = md5(
                                os.path.join(dataset_data_path_prev, pitch_v_path)
                            )
                            pitch_v = (pitch_v_path, pitch_v_checksum)
                        if "tempo-manual" in file:
                            tempo_path = os.path.join("saraga1.5_carnatic", concert, song, file)
                            tempo_checksum = md5(os.path.join(dataset_data_path_prev, tempo_path))
                            tempo = (tempo_path, tempo_checksum)
                        if "sama-manual" in file:
                            sama_path = os.path.join("saraga1.5_carnatic", concert, song, file)
                            sama_checksum = md5(os.path.join(dataset_data_path_prev, sama_path))
                            sama = (sama_path, sama_checksum)
                        if "sections-manual-p.txt" in file:
                            sections_path = os.path.join("saraga1.5_carnatic", concert, song, file)
                            sections_checksum = md5(
                                os.path.join(dataset_data_path_prev, sections_path)
                            )
                            sections = (sections_path, sections_checksum)
                        if "mphrase" in file:
                            phrases_path = os.path.join("saraga1.5_carnatic", concert, song, file)
                            phrases_checksum = md5(
                                os.path.join(dataset_data_path_prev, phrases_path)
                            )
                            phrases = (phrases_path, phrases_checksum)
                        if ".json" in file:
                            metadata_path = os.path.join("saraga1.5_carnatic", concert, song, file)
                            metadata_checksum = md5(
                                os.path.join(dataset_data_path_prev, metadata_path)
                            )
                            metadata = (metadata_path, metadata_checksum)

                        saraga_index["tracks"][index] = {
                            "audio-mix": audio,
                            "audio-ghatam": audio_ghatam,
                            "audio-mridangam-left": audio_mridangam_left,
                            "audio-mridangam-right": audio_mridangam_right,
                            "audio-violin": audio_violin,
                            "audio-vocal-s": audio_vocal_s,
                            "audio-vocal": audio_vocal,
                            "ctonic": ctonic,
                            "pitch": pitch,
                            "pitch-vocal": pitch_v,
                            "tempo": tempo,
                            "sama": sama,
                            "sections": sections,
                            "phrases": phrases,
                            "metadata": metadata,
                        }

                    idx = idx + 1

    with open(SARAGA_CARNATIC_INDEX_PATH, "w") as fhandle:
        json.dump(saraga_index, fhandle, indent=2)


def main(args):
    print("creating index...")
    make_saraga_carnatic_index(args.dataset_data_path)
    print("done!")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make Saraga Carnatic index file.")
    PARSER.add_argument("dataset_data_path", type=str, help="Path to Saraga Carnatic data folder.")

    main(PARSER.parse_args())
