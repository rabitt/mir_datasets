import argparse
import csv
import glob
import json
import os
from mirdata.validate import md5

MIR_1K_INDEX_PATH = "mirdata/datasets/indexes/mir_1k_index_{}.json"


def make_mir_1k_index(mir_1k_data_path: str, version: str) -> None:
    assert version == "1.0"

    lyrics_dir_name = "Lyrics"
    lyrics_wav_dir_name = "LyricsWav"
    f0_dir_name = "PitchLabel"
    full_audio_dir_name = "UndividedWavfile"
    unvoiced_dir_name = "UnvoicedFrameLabel"
    vocal_dir_name = "vocal-nonvocalLabel"
    clipped_audio_dir_name = "Wavfile"

    lyrics_dir = os.path.join(mir_1k_data_path, lyrics_dir_name)
    lyrics_wav_dir = os.path.join(mir_1k_data_path, lyrics_wav_dir_name)
    f0_dir = os.path.join(mir_1k_data_path, f0_dir_name)
    full_audio_dir = os.path.join(mir_1k_data_path, full_audio_dir_name)
    unvoiced_dir = os.path.join(mir_1k_data_path, unvoiced_dir_name)
    vocal_dir = os.path.join(mir_1k_data_path, vocal_dir_name)
    clipped_audio_dir = os.path.join(mir_1k_data_path, clipped_audio_dir_name)

    track_ids = sorted(
        [
            os.path.basename(f).replace(".wav", "")
            for f in glob.glob(os.path.join(full_audio_dir, "*.wav"))
        ]
    )

    # undivided tracks: these are the full song recordings, not the clips.
    # They don't have their own f0/lyrics/voicing annotations.
    # The "spoken lyrics" recordings correspond to the full recordings, not the clips.
    # Currently not included in the index.
    undivided_track_index = {}

    track_index = {}

    for track_id in track_ids:
        clip_ids = sorted(
            [
                os.path.basename(f).replace(".wav", "")
                for f in glob.glob(os.path.join(clipped_audio_dir, f"{track_id}_*.wav"))
            ]
        )

        # need to account for some filename inconsistencies here
        spoken_lyrics_id = track_id
        if spoken_lyrics_id.startswith("abjones"):
            spoken_lyrics_id = spoken_lyrics_id.replace("abjones", "ABJones")

        if spoken_lyrics_id.startswith("stool"):
            spoken_lyrics_id += "_lyric"
        else:
            spoken_lyrics_id += "_lyrics"

        undivided_track_index[track_id] = {
            "tracks": clip_ids,
            "audio": (
                f"{full_audio_dir_name}/{track_id}.wav",
                md5(os.path.join(full_audio_dir, f"{track_id}.wav")),
            ),
            "spoken_lyrics": (
                f"{lyrics_wav_dir_name}/{spoken_lyrics_id}.wav",
                md5(os.path.join(lyrics_wav_dir, f"{spoken_lyrics_id}.wav")),
            ),
        }

        track_index.update(
            {
                clip_id: {
                    "audio": (
                        f"{clipped_audio_dir_name}/{clip_id}.wav",
                        md5(os.path.join(clipped_audio_dir, f"{clip_id}.wav")),
                    ),
                    "lyrics": (
                        f"{lyrics_dir_name}/{clip_id}.txt",
                        md5(os.path.join(lyrics_dir, f"{clip_id}.txt")),
                    ),
                    "f0": (
                        f"{f0_dir_name}/{clip_id}.pv",
                        md5(os.path.join(f0_dir, f"{clip_id}.pv")),
                    ),
                    "unvoiced-category": (
                        f"{unvoiced_dir_name}/{clip_id}.unv",
                        md5(os.path.join(unvoiced_dir, f"{clip_id}.unv")),
                    ),
                    "vocal-flag": (
                        f"{vocal_dir_name}/{clip_id}.vocal",
                        md5(os.path.join(vocal_dir, f"{clip_id}.vocal")),
                    ),
                }
                for clip_id in clip_ids
            }
        )

    mir_1k_index = {
        "version": version,
        "tracks": track_index,
    }

    with open(MIR_1K_INDEX_PATH.format(version), "w") as fhandle:
        json.dump(mir_1k_index, fhandle, indent=2)


def main(args):
    make_mir_1k_index(args.mir_1k_data_path, args.version)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make MIR-1K index file.")
    PARSER.add_argument(
        "mir_1k_data_path", type=str, help="Path to MIR-1K data folder."
    )
    PARSER.add_argument("version", type=str, help="index version")

    main(PARSER.parse_args())
