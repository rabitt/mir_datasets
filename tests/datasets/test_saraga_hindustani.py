import os

import numpy as np

from mirdata import annotations
from mirdata.datasets import saraga_hindustani
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "50_Irani_Bhairavi_Thumri"
    data_home = os.path.normpath("tests/resources/mir_datasets/saraga_hindustani")
    dataset = saraga_hindustani.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "50_Irani_Bhairavi_Thumri",
        "title": "Irani Bhairavi Thumri",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_hindustani/"),
            "saraga1.5_hindustani/New Signature by Brajeshwar Mukherjee/Irani Bhairavi Thumri/Irani Bhairavi Thumri.mp3.mp3",
        ),
        "ctonic_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_hindustani/"),
            "saraga1.5_hindustani/New Signature by Brajeshwar Mukherjee/Irani Bhairavi Thumri/Irani Bhairavi Thumri.ctonic.txt",
        ),
        "pitch_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_hindustani/"),
            "saraga1.5_hindustani/New Signature by Brajeshwar Mukherjee/Irani Bhairavi Thumri/Irani Bhairavi Thumri.pitch.txt",
        ),
        "tempo_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_hindustani/"),
            "saraga1.5_hindustani/New Signature by Brajeshwar Mukherjee/Irani Bhairavi Thumri/Irani Bhairavi Thumri.tempo-manual.txt",
        ),
        "sama_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_hindustani/"),
            "saraga1.5_hindustani/New Signature by Brajeshwar Mukherjee/Irani Bhairavi Thumri/Irani Bhairavi Thumri.sama-manual.txt",
        ),
        "sections_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_hindustani/"),
            "saraga1.5_hindustani/New Signature by Brajeshwar Mukherjee/Irani Bhairavi Thumri/Irani Bhairavi Thumri.sections-manual-p.txt",
        ),
        "phrases_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_hindustani/"),
            "saraga1.5_hindustani/New Signature by Brajeshwar Mukherjee/Irani Bhairavi Thumri/Irani Bhairavi Thumri.mphrases-manual.txt",
        ),
        "metadata_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/saraga_hindustani/"),
            "saraga1.5_hindustani/New Signature by Brajeshwar Mukherjee/Irani Bhairavi Thumri/Irani Bhairavi Thumri.json",
        ),
    }

    expected_property_types = {
        "tempo": dict,
        "phrases": annotations.EventData,
        "pitch": annotations.F0Data,
        "sama": annotations.BeatData,
        "sections": annotations.SectionData,
        "tonic": float,
        "metadata": dict,
        "audio": tuple,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    audio, sr = track.audio
    assert sr == 44100
    assert audio.shape[0] == 2


def test_to_jams():
    data_home = "tests/resources/mir_datasets/saraga_hindustani"
    dataset = saraga_hindustani.Dataset(data_home)
    track = dataset.track("50_Irani_Bhairavi_Thumri")
    jam = track.to_jams()

    assert jam["sandbox"].tonic == 138.591315

    # Tempo
    parsed_tempo = jam["sandbox"].tempo
    assert parsed_tempo == {
        "Ālāp": {
            "tempo": -1,
            "matra_interval": -1,
            "sama_interval": -1,
            "matras_per_cycle": -1,
            "start_time": 3.298,
            "duration": 58.236,
        },
        "Khyāl (vilambit ēktāl)": {
            "tempo": 13,
            "matra_interval": 4.605,
            "sama_interval": 55.265,
            "matras_per_cycle": 12,
            "start_time": 59.49,
            "duration": 678.009,
        },
        "Khyāl (dr̥t ēktāl)": {
            "tempo": 185,
            "matra_interval": 0.324,
            "sama_interval": 3.885,
            "matras_per_cycle": 12,
            "start_time": 679.834,
            "duration": 894.433,
        },
    }

    # Sections
    sections = jam.search(namespace="segment_open")[0]["data"]
    assert [section.time for section in sections] == [3.298, 59.49, 679.834]
    assert [section.duration for section in sections] == [
        56.192,
        620.344,
        218.83048979600005,
    ]
    assert [section.value for section in sections] == [
        "Ālāp-1",
        "Khyāl (vilambit ēktāl)-2",
        "Khyāl (dr̥t ēktāl)-3",
    ]
    assert [section.confidence for section in sections] == [None, None, None]


def test_load_tonic():
    data_home = "tests/resources/mir_datasets/saraga_hindustani"
    dataset = saraga_hindustani.Dataset(data_home)
    track = dataset.track("50_Irani_Bhairavi_Thumri")
    tonic_path = track.ctonic_path
    parsed_tonic = saraga_hindustani.load_tonic(tonic_path)
    assert parsed_tonic == 138.591315
    assert saraga_hindustani.load_tonic(None) is None


def test_load_pitch():
    data_home = "tests/resources/mir_datasets/saraga_hindustani"
    dataset = saraga_hindustani.Dataset(data_home)
    track = dataset.track("50_Irani_Bhairavi_Thumri")
    pitch_path = track.pitch_path
    parsed_pitch = saraga_hindustani.load_pitch(pitch_path)

    # Check types
    assert type(parsed_pitch) == annotations.F0Data
    assert type(parsed_pitch.times) is np.ndarray
    assert type(parsed_pitch.frequencies) is np.ndarray
    assert type(parsed_pitch.voicing) is np.ndarray

    # Check values
    assert np.array_equal(
        parsed_pitch.times,
        np.array(
            [
                0.0,
                0.0044444444444444444,
                0.008888888888888889,
                0.013333333333333334,
                0.017777777777777778,
                0.022222222222222223,
            ]
        ),
    )
    assert np.array_equal(
        parsed_pitch.frequencies,
        np.array(
            [
                0.0,
                11.111111111111111111,
                222.22222222222222222,
                333.333333333333333333,
                444.444444444444444444,
                0.0,
            ]
        ),
    )
    assert np.array_equal(parsed_pitch.voicing, np.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.0]))

    assert saraga_hindustani.load_pitch(None) is None


def test_load_sama():
    data_home = "tests/resources/mir_datasets/saraga_hindustani"
    dataset = saraga_hindustani.Dataset(data_home)
    track = dataset.track("50_Irani_Bhairavi_Thumri")
    sama_path = track.sama_path
    parsed_sama = saraga_hindustani.load_sama(sama_path)

    # Check types
    assert type(parsed_sama) == annotations.BeatData
    assert type(parsed_sama.times) is np.ndarray
    assert type(parsed_sama.positions) is np.ndarray

    # Check values
    assert np.array_equal(parsed_sama.times, np.array([68.385, 123.804, 179.069, 234.339]))
    assert np.array_equal(parsed_sama.positions, np.array([1, 2, 3, 4]))
    assert saraga_hindustani.load_sama(None) is None

    # Test empty sama
    track = dataset.track("51_Raag_Bairagi")
    sama_path = track.sama_path
    parsed_empty_sama = saraga_hindustani.load_sama(sama_path)
    assert parsed_empty_sama is None


def test_load_sections():
    data_home = "tests/resources/mir_datasets/saraga_hindustani"
    dataset = saraga_hindustani.Dataset(data_home)
    track = dataset.track("50_Irani_Bhairavi_Thumri")
    sections_path = track.sections_path
    parsed_sections = saraga_hindustani.load_sections(sections_path)

    # Check types
    assert type(parsed_sections) == annotations.SectionData
    assert type(parsed_sections.intervals) is np.ndarray
    assert type(parsed_sections.labels) is list

    # Check values
    assert np.array_equal(parsed_sections.intervals[:, 0], np.array([3.298, 59.49, 679.834]))
    assert np.array_equal(
        parsed_sections.intervals[:, 1],
        np.array([59.49, 679.8340000000001, 898.664489796]),
    )
    assert parsed_sections.labels == [
        "Ālāp-1",
        "Khyāl (vilambit ēktāl)-2",
        "Khyāl (dr̥t ēktāl)-3",
    ]

    assert saraga_hindustani.load_sections(None) is None

    # Test empty sections
    track = dataset.track("51_Raag_Bairagi")
    sections_path = track.sections_path
    parsed_empty_sections = saraga_hindustani.load_sections(sections_path)
    assert parsed_empty_sections is None


def test_load_phrases():
    data_home = "tests/resources/mir_datasets/saraga_hindustani"
    dataset = saraga_hindustani.Dataset(data_home)
    track = dataset.track("50_Irani_Bhairavi_Thumri")
    phrases_path = track.phrases_path
    parsed_phrases = saraga_hindustani.load_phrases(phrases_path)

    # Check types
    assert type(parsed_phrases) is annotations.EventData
    assert type(parsed_phrases.intervals) is np.ndarray
    assert type(parsed_phrases.events) is list

    # Check values
    assert np.array_equal(
        parsed_phrases.intervals,
        np.array(
            [
                [3.506213151, 10.890158729],
                [12.538775510, 18.924263038],
                [23.382494331, 32.252517006],
            ]
        ),
    )
    assert parsed_phrases.events == ["Pmr", "PnS", "rmP"]
    assert saraga_hindustani.load_phrases(None) is None

    # Test phrases with no information
    track = dataset.track("51_Raag_Bairagi")
    phrases_path = track.phrases_path
    parsed_phrases_add = saraga_hindustani.load_phrases(phrases_path)
    assert parsed_phrases_add.events == ["rg", ""]


def test_load_tempo():
    data_home = "tests/resources/mir_datasets/saraga_hindustani"
    dataset = saraga_hindustani.Dataset(data_home)
    track = dataset.track("50_Irani_Bhairavi_Thumri")
    tempo_path = track.tempo_path
    parsed_tempo = saraga_hindustani.load_tempo(tempo_path)

    print(parsed_tempo)

    assert type(parsed_tempo) == dict
    assert type(parsed_tempo["Ālāp"]) == dict
    assert type(parsed_tempo["Ālāp"]["tempo"]) == int
    assert type(parsed_tempo["Ālāp"]["duration"]) == float
    assert parsed_tempo == {
        "Ālāp": {
            "tempo": -1,
            "matra_interval": -1,
            "sama_interval": -1,
            "matras_per_cycle": -1,
            "start_time": 3.298,
            "duration": 58.236,
        },
        "Khyāl (vilambit ēktāl)": {
            "tempo": 13,
            "matra_interval": 4.605,
            "sama_interval": 55.265,
            "matras_per_cycle": 12,
            "start_time": 59.49,
            "duration": 678.009,
        },
        "Khyāl (dr̥t ēktāl)": {
            "tempo": 185,
            "matra_interval": 0.324,
            "sama_interval": 3.885,
            "matras_per_cycle": 12,
            "start_time": 679.834,
            "duration": 894.433,
        },
    }
    assert saraga_hindustani.load_tempo(None) is None


def test_load_metadata():
    data_home = "tests/resources/mir_datasets/saraga_hindustani"
    dataset = saraga_hindustani.Dataset(data_home)
    track = dataset.track("50_Irani_Bhairavi_Thumri")
    metadata_path = track.metadata_path
    parsed_metadata = saraga_hindustani.load_metadata(metadata_path)

    assert parsed_metadata["title"] == "Bairagi"
    assert parsed_metadata["raags"] == [
        {
            "common_name": "Bairagi",
            "uuid": "b143adaa-f1a6-4de4-8985-a5bd35e96279",
            "name": "Bairāgi",
        }
    ]
    assert parsed_metadata["length"] == 899469
    assert parsed_metadata["album_artists"] == [
        {"mbid": "653fa2f8-85f8-4829-871f-7c2506ea9b48", "name": "Ajoy Chakrabarty"}
    ]
    assert parsed_metadata["forms"] == [
        {
            "common_name": "Khayal",
            "uuid": "7ed81b92-aea6-4f4b-bffb-c12d80012d37",
            "name": "Khyāl",
        }
    ]
    assert parsed_metadata["mbid"] == "b71c2774-2532-4692-8761-5452e2a83118"
    assert parsed_metadata["artists"] == [
        {
            "instrument": {
                "mbid": "d92884b7-ee0c-46d5-96f3-918196ba8c5b",
                "name": "Voice",
            },
            "attributes": "lead vocals",
            "lead": True,
            "artist": {
                "mbid": "653fa2f8-85f8-4829-871f-7c2506ea9b48",
                "name": "Ajoy Chakrabarty",
            },
        },
        {
            "instrument": {
                "mbid": "c43c7647-077d-4d60-a01b-769de71b82f2",
                "name": "Harmonium",
            },
            "attributes": "",
            "lead": False,
            "artist": {
                "mbid": "afbb34e8-1f87-4dd4-81ec-b6145af4d72f",
                "name": "Paromita Mukherjee",
            },
        },
        {
            "instrument": {
                "mbid": "18e6998b-e53b-415b-b484-d3ac286da99d",
                "name": "Tabla",
            },
            "attributes": "",
            "lead": False,
            "artist": {
                "mbid": "beee80e6-aa99-451c-9edb-dcda8c2fce8a",
                "name": "Indranil Bhaduri",
            },
        },
    ]
    assert parsed_metadata["release"] == [
        {
            "mbid": "ae0f2366-9a4f-4534-9376-ac123e881f64",
            "title": "Geetinandan : Part-3",
        }
    ]
    assert parsed_metadata["works"] == [
        {
            "mbid": "b8925ff6-9c8f-4184-8fc8-d358cfdea79b",
            "title": "Mere Maname Baso Ram Abhiram Puran Ho Sab Kaam",
        },
        {"mbid": "d7a184c3-0187-4912-8708-8d12a4bd9b0a", "title": "Bar Bar Har Gai"},
    ]
    assert parsed_metadata["taals"] == [
        {
            "common_name": "Ektaal",
            "uuid": "7cb20903-5f64-4f15-8713-2fb4fcca2b5b",
            "name": "ēktāl",
        },
        {
            "common_name": "Ektaal",
            "uuid": "7cb20903-5f64-4f15-8713-2fb4fcca2b5b",
            "name": "ēktāl",
        },
    ]
    assert parsed_metadata["layas"] == [
        {
            "common_name": "Vilambit",
            "uuid": "ee58d24a-60aa-4b16-bfcf-edd105118738",
            "name": "Vilaṁbit",
        }
    ]


def test_load_audio():
    data_home = "tests/resources/mir_datasets/saraga_hindustani"
    dataset = saraga_hindustani.Dataset(data_home)
    track = dataset.track("50_Irani_Bhairavi_Thumri")
    audio_path = track.audio_path
    audio, sr = saraga_hindustani.load_audio(audio_path)

    assert sr == 44100
    assert type(audio) == np.ndarray
    assert audio.shape[0] == 2

    assert saraga_hindustani.load_audio(None) is None
