import argparse
import glob
import json
import os

from mirdata.validate import md5

DATASET_INDEX_PATH = "../mirdata/datasets/indexes/dagstuhl_choirset_index.json"

NO_SCORE = ["Basses", "Overacted", "SE", "Solo", "Outtake"]


def make_dataset_index(data_path):
    audio_dir = os.path.join(data_path, "audio_wav_22050_mono")

    index = {"version": "1.2.3", "tracks": {}, "multitracks": {}}

    # define pieces directly from data directory
    pieces = sorted(
        list(
            set(
                [
                    "_".join(filename.split("/")[-1].split("_")[:4])
                    for filename in glob.glob(os.path.join(audio_dir, "*.wav"))
                ]
            )
        )
    )

    for ip, piece in enumerate(pieces):
        index["multitracks"][piece] = {}

        ## add mixture audios

        # STM
        audio_mix_dir = os.path.join(
            data_path, "audio_wav_22050_mono", "{}_Stereo_STM.wav".format(piece)
        )
        audio_checksum = md5(audio_mix_dir)
        index["multitracks"][piece]["audio_stm"] = (
            "audio_wav_22050_mono/{}_Stereo_STM.wav".format(piece),
            audio_checksum,
        )

        # STR
        audio_mix_dir = os.path.join(
            data_path, "audio_wav_22050_mono", "{}_Stereo_STR.wav".format(piece)
        )
        audio_checksum = md5(audio_mix_dir)
        index["multitracks"][piece]["audio_str"] = (
            "audio_wav_22050_mono/{}_Stereo_STR.wav".format(piece),
            audio_checksum,
        )

        # STL
        audio_mix_dir = os.path.join(
            data_path, "audio_wav_22050_mono", "{}_Stereo_STL.wav".format(piece)
        )
        audio_checksum = md5(audio_mix_dir)
        index["multitracks"][piece]["audio_stl"] = (
            "audio_wav_22050_mono/{}_Stereo_STL.wav".format(piece),
            audio_checksum,
        )

        # STRev
        audio_mix_dir = os.path.join(
            data_path, "audio_wav_22050_mono", "{}_StereoReverb_STM.wav".format(piece)
        )
        audio_checksum = md5(audio_mix_dir)
        index["multitracks"][piece]["audio_rev"] = (
            "audio_wav_22050_mono/{}_StereoReverb_STM.wav".format(piece),
            audio_checksum,
        )

        # beats
        index["multitracks"][piece]["beat"] = (None, None)

        # piano
        index["multitracks"][piece]["audio_spl"] = (None, None)
        index["multitracks"][piece]["audio_spr"] = (None, None)

        ## add each track inside the multitrack

        audio_files = sorted(glob.glob(os.path.join(audio_dir, "{}*.wav".format(piece))))

        singers = [singer.split("_")[-2] for singer in audio_files if not "Stereo" in singer]

        # second step to remove piano from singers
        singers = [singer for singer in singers if "Piano" not in singer]

        # mics = [singer.split('_')[-1].split('.')[0] for singer in audio_files if not 'Stereo' in singer]
        # assert len(singers) == len(mics), "number of mics does not match number of singers for {}".format(piece)
        # set_singers = set(singers)

        index["multitracks"][piece]["tracks"] = []

        for sidx, singer in enumerate(sorted(singers)):
            track_name = "{}_{}".format(piece, singer)

            # define fields as None
            index["tracks"][track_name] = {
                "audio_dyn": (None, None),
                "audio_hsm": (None, None),
                "audio_lrx": (None, None),
                "f0_crepe_dyn": (None, None),
                "f0_crepe_hsm": (None, None),
                "f0_crepe_lrx": (None, None),
                "f0_pyin_dyn": (None, None),
                "f0_pyin_hsm": (None, None),
                "f0_pyin_lrx": (None, None),
                "f0_manual_lrx": (None, None),
                "score": (None, None),
            }

            index["multitracks"][piece]["tracks"].append(track_name)

            mics = [
                mic.split("_")[-1].split(".")[0]
                for mic in glob.glob(os.path.join(audio_dir, "{}_{}*.wav".format(piece, singer)))
                if mic not in ["SPL", "SPR"]
            ]

            ### add all fields for each track

            for mic in mics:
                ## add audio
                audio_stem_dir = os.path.join(
                    data_path,
                    "audio_wav_22050_mono",
                    "{}_{}_{}.wav".format(piece, singer, mic),
                )
                audio_checksum = md5(audio_stem_dir)

                index["tracks"][track_name]["audio_{}".format(mic.lower())] = (
                    "audio_wav_22050_mono/{}_{}_{}.wav".format(piece, singer, mic),
                    audio_checksum,
                )

                ## add crepe f0s
                crepe_dir = os.path.join(
                    data_path,
                    "annotations_csv_F0_CREPE",
                    "{}_{}_{}.csv".format(piece, singer, mic),
                )
                crepe_checksum = md5(crepe_dir)

                index["tracks"][track_name]["f0_crepe_{}".format(mic.lower())] = (
                    "annotations_csv_F0_CREPE/{}_{}_{}.csv".format(piece, singer, mic),
                    crepe_checksum,
                )

                ## add pyin f0s
                pyin_dir = os.path.join(
                    data_path,
                    "annotations_csv_F0_PYIN",
                    "{}_{}_{}.csv".format(piece, singer, mic),
                )
                pyin_checksum = md5(pyin_dir)

                index["tracks"][track_name]["f0_pyin_{}".format(mic.lower())] = (
                    "annotations_csv_F0_PYIN/{}_{}_{}.csv".format(piece, singer, mic),
                    pyin_checksum,
                )

                ## add score when it exists

                # some have no associated score
                if not any(x in piece for x in NO_SCORE):
                    score_dir = os.path.join(
                        data_path,
                        "annotations_csv_scorerepresentation",
                        "{}_Stereo_STM_{}.csv".format(piece, singer[0]),
                    )
                    score_checksum = md5(score_dir)

                    index["tracks"][track_name]["score"] = (
                        "annotations_csv_scorerepresentation/{}_Stereo_STM_{}.csv".format(
                            piece, singer[0]
                        ),
                        score_checksum,
                    )

            ## add beats for the full songs when available

            if not any(x in piece for x in NO_SCORE):
                ## add beats
                beats_dir = os.path.join(
                    data_path, "annotations_csv_beat", "{}_Stereo_STM.csv".format(piece)
                )
                beats_checksum = md5(beats_dir)

                index["multitracks"][piece]["beat"] = (
                    "annotations_csv_beat/{}_Stereo_STM.csv".format(piece),
                    beats_checksum,
                )

        ## check if piano track exists and add it to the mtrack if so

        audio_pianoL_dir = os.path.join(
            data_path, "audio_wav_22050_mono", "{}_Piano_SPL.wav".format(piece)
        )
        if os.path.exists(audio_pianoL_dir):
            # add piano SPL
            audio_checksum = md5(audio_pianoL_dir)
            index["multitracks"][piece]["audio_spl"] = (
                "audio_wav_22050_mono/{}_Piano_SPL.wav".format(piece),
                audio_checksum,
            )

            # add piano SPR
            audio_checksum = md5(audio_pianoL_dir.replace("SPL", "SPR"))
            index["multitracks"][piece]["audio_spr"] = (
                "audio_wav_22050_mono/{}_Piano_SPR.wav".format(piece),
                audio_checksum,
            )

        # tracks should not be repeated
        index["multitracks"][piece]["tracks"] = sorted(
            list(set(index["multitracks"][piece]["tracks"]))
        )

    ## add the manual annotations to their corresponding tracks
    manual_files = sorted(glob.glob(os.path.join(data_path, "annotations_csv_F0_manual", "*.csv")))
    for mf in manual_files:
        track_name = "_".join(os.path.basename(mf).split("_")[:-1])

        manual_checksum = md5(mf)

        index["tracks"][track_name]["f0_manual_lrx"] = (
            "annotations_csv_F0_manual/{}".format(os.path.basename(mf)),
            manual_checksum,
        )

    with open(DATASET_INDEX_PATH, "w") as fhandle:
        json.dump(index, fhandle, indent=2)


def main(args):
    make_dataset_index(args.data_path)


#
if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make Dagstuhl ChoirSet index file.")
    PARSER.add_argument("data_path", type=str, help="Path to Dagstuhl ChoirSet data folder.")

    main(PARSER.parse_args())
