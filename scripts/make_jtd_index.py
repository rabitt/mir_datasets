import argparse
import json
import os
from mirdata.validate import md5
from tqdm import tqdm

STEMS = ["piano", "bass", "drums"]    # every recording has annotations for these instruments
CHANNELS = ["", "-lchan", "-rchan"]    # for some recordings (but not all), we have audio for individual channels

DATASET_INDEX_PATH = "../mirdata/datasets/indexes/jtd_index_2.0.json"


def get_stem_audio_filepath(directory: str, recording_name: str, stem: str = "") -> str:
    for chan in CHANNELS:
        audio = os.path.join(directory, f'{recording_name}{chan}{stem}.wav')
        # Return the first correct match (we only have one audio file per stem for each recording)
        if os.path.isfile(audio):
            return audio
    raise FileNotFoundError(f"Couldn't find audio for {recording_name} and stem {stem}, does it exist?")


def get_stem_files(data_path: str, recording: str, instrument: str) -> dict:
    # Get filepaths for annotations and processed audio
    annotation_dir = os.path.join(data_path, "annotations")
    processed_audio_dir = os.path.join(data_path, "processed")
    # Dictionary where we'll store everything
    stem_dict = {}
    # Get metadata annotations
    metadata_file = os.path.join(annotation_dir, recording, "metadata.json")
    stem_dict["metadata"] = (metadata_file.replace(data_path, "").lstrip("/"), md5(metadata_file))
    # Getting audio path for stem
    stem_audio = get_stem_audio_filepath(processed_audio_dir, recording, f'_{instrument}')
    stem_dict["audio"] = (stem_audio.replace(data_path, "").lstrip("/"), md5(stem_audio))
    # Getting onsets for stem
    stem_onsets = os.path.join(annotation_dir, recording, f"{instrument}_onsets.csv")
    stem_dict["onsets"] = (stem_onsets.replace(data_path, "").lstrip("/"), md5(stem_onsets))
    # Get beats
    beats_file = os.path.join(annotation_dir, recording, "beats.csv")
    stem_dict["beats"] = (beats_file.replace(data_path, "").lstrip("/"), md5(beats_file))
    # Getting MIDI for stem
    if instrument == "piano":
        stem_midi = os.path.join(annotation_dir, recording, f"{instrument}_midi.mid")
        midi_out = (stem_midi.replace(data_path, "").lstrip("/"), md5(stem_midi))
    # We do not currently include MIDI for bass or drums
    else:
        midi_out = (None, None)
    stem_dict["midi"] = midi_out
    return stem_dict


def get_mixed_files(data_path: str, recording: str) -> dict:
    # Get filepaths for annotations and mixed audio
    annotation_dir = os.path.join(data_path, "annotations")
    audio_dir = os.path.join(data_path, "raw")
    # Dictionary where we'll store everything
    recording_dict = {"tracks": [f'{recording}_{st}' for st in STEMS]}
    # Get metadata annotations
    metadata_file = os.path.join(annotation_dir, recording, "metadata.json")
    recording_dict["metadata"] = (metadata_file.replace(data_path, "").lstrip("/"), md5(metadata_file))
    # Get beats
    beats_file = os.path.join(annotation_dir, recording, "beats.csv")
    recording_dict["beats"] = (beats_file.replace(data_path, "").lstrip("/"), md5(beats_file))
    # Get all raw audio files (can be multiple, if we have multiple channels)
    for channel in CHANNELS:
        aud_path = os.path.join(audio_dir, f'{recording}{channel}.wav')
        if os.path.exists(aud_path):
            aud_output = (aud_path.replace(data_path, "").lstrip("/"), md5(aud_path))
        else:
            aud_output = [None, None]
        recording_dict[f"audio{channel}"] = aud_output
    return recording_dict


def make_jtd_index(dataset_data_path: str):
    annotation_dir = os.path.join(dataset_data_path, "annotations")
    # Dictionaries to hold indexes for tracks and multitracks
    index_tracks = {}
    index_multitracks = {}
    # Iterate over each recording (a separate directory inside `annotation_dir`)`
    for recording in tqdm(sorted(os.listdir(annotation_dir)), desc='Making JTD indexes'):
        # Create multitrack index for this recording
        index_multitracks[recording] = get_mixed_files(dataset_data_path, recording)
        # Create stem index for each part (piano, bass, drums)
        for stem in STEMS:
            index_tracks[f"{recording}_{stem}"] = get_stem_files(dataset_data_path, recording, stem)
    # Combine everything together in dataset index
    dataset_index = {"version": 2, "tracks": index_tracks, "multitracks": index_multitracks}
    with open(DATASET_INDEX_PATH, "w") as fhandle:
        json.dump(dataset_index, fhandle, indent=2)


def main(args):
    make_jtd_index(args.jtd_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make JTD index file.")
    PARSER.add_argument(
        "jtd_path",
        type=str,
        help="Path to JTD, should contain `raw`, `processed`, and `annotations` directories."
    )

    main(PARSER.parse_args())
