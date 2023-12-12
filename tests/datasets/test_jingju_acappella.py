import os

import numpy as np

from mirdata import annotations
from mirdata.datasets import compmusic_jingju_acappella
from tests.test_utils import run_track_tests


def test_track():
    data_home = os.path.normpath("tests/resources/mir_datasets/compmusic_jingju_acappella")
    track_id = "lseh-Tan_Yang_jia-Hong_yang_dong-qm"

    dataset = compmusic_jingju_acappella.Dataset(data_home)
    track = dataset.track(track_id)

    expected_attributes = {
        "track_id": "lseh-Tan_Yang_jia-Hong_yang_dong-qm",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/compmusic_jingju_acappella/"),
            "wav/laosheng/lseh-Tan_Yang_jia-Hong_yang_dong-qm.wav",
        ),
        "phrase_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/compmusic_jingju_acappella/"),
            "annotation_txt/laosheng/lseh-Tan_Yang_jia-Hong_yang_dong-qm_phrase.txt",
        ),
        "phrase_char_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/compmusic_jingju_acappella/"),
            "annotation_txt/laosheng/lseh-Tan_Yang_jia-Hong_yang_dong-qm_phrase_char.txt",
        ),
        "phoneme_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/compmusic_jingju_acappella/"),
            "annotation_txt/laosheng/lseh-Tan_Yang_jia-Hong_yang_dong-qm_phoneme.txt",
        ),
        "syllable_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/compmusic_jingju_acappella/"),
            "annotation_txt/laosheng/lseh-Tan_Yang_jia-Hong_yang_dong-qm_syllable.txt",
        ),
        "title": "Türk Müziğinde 75 Büyük Bestekar/ 75 Great Composers In Turkish Classical Music",
        "textgrid_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/compmusic_jingju_acappella/"),
            "textgrid/laosheng/lseh-Tan_Yang_jia-Hong_yang_dong-qm.TextGrid",
        ),
        "work": "“叹杨家投宋主心血用尽”——《洪羊洞》（杨延昭）",
        "details": None,
    }

    expected_property_types = {
        "audio": tuple,
        "phrase": annotations.LyricData,
        "phrase_char": annotations.LyricData,
        "phoneme": annotations.LyricData,
        "syllable": annotations.LyricData,
        "work": str,
        "details": None.__class__,
    }

    run_track_tests(track, expected_attributes, expected_property_types)


def test_to_jams():
    data_home = "tests/resources/mir_datasets/compmusic_jingju_acappella"
    track_id = "lseh-Tan_Yang_jia-Hong_yang_dong-qm"

    dataset = compmusic_jingju_acappella.Dataset(data_home)
    track = dataset.track(track_id)
    jam = track.to_jams()

    # Sandbox
    assert jam["sandbox"].work == "“叹杨家投宋主心血用尽”——《洪羊洞》（杨延昭）"

    # Lyrics
    phrases = jam.search(namespace="lyrics")[0]["data"]
    assert len(phrases) == 6
    assert [phrase.time for phrase in phrases] == [
        1.06,
        28.68,
        77.65,
        120.04,
        141.44,
        193.5,
    ]
    assert [phrase.duration for phrase in phrases] == [
        16.860000000000003,
        21.21,
        32.42999999999999,
        12.39,
        20.180000000000007,
        17.659999999999997,
    ]
    assert [phrase.value for phrase in phrases] == [
        "tan yang jia tou song zhu a",
        "xin xue yong a jin",
        "zhen na ke tan jiao meng e jiang ming sang fan ying",
        "zong bao er chan wei fu",
        "ruan ta kao e zhen",
        "pa zhi pa ao bu guo chi cun guang yin",
    ]
    assert [phrase.confidence for phrase in phrases] == [
        None,
        None,
        None,
        None,
        None,
        None,
    ]

    phonemes = jam.search(namespace="lyrics")[2]["data"]
    assert len(phonemes) == 6
    assert [phoneme.time for phoneme in phonemes] == [0.0, 1.06, 1.16, 2.53, 2.65, 2.94]
    assert [phoneme.duration for phoneme in phonemes] == [
        1.06,
        0.09999999999999987,
        1.3699999999999999,
        0.1200000000000001,
        0.29000000000000004,
        0.10999999999999988,
    ]
    assert [phoneme.value for phoneme in phonemes] == ["", "@", "r\\'", "?", "AU^", "9"]
    assert [phoneme.confidence for phoneme in phonemes] == [
        None,
        None,
        None,
        None,
        None,
        None,
    ]


def test_load_phrases():
    data_home = "tests/resources/mir_datasets/compmusic_jingju_acappella"
    track_id = "lseh-Tan_Yang_jia-Hong_yang_dong-qm"

    dataset = compmusic_jingju_acappella.Dataset(data_home)
    track = dataset.track(track_id)
    phrase_path = track.phrase_path
    phrase_char_path = track.phrase_char_path
    parsed_phrases = compmusic_jingju_acappella.load_phrases(phrase_path)
    parsed_phrases_char = compmusic_jingju_acappella.load_phrases(phrase_char_path)

    # Check types
    assert type(parsed_phrases) == annotations.LyricData
    assert type(parsed_phrases.intervals) is np.ndarray
    assert type(parsed_phrases.lyrics) is list

    # Check values
    assert np.array_equal(
        parsed_phrases.intervals[:, 0],
        np.array([1.06, 28.68, 77.65, 120.04, 141.44, 193.5]),
    )
    assert np.array_equal(
        parsed_phrases.intervals[:, 1],
        np.array([17.92, 49.89, 110.08, 132.43, 161.62, 211.16]),
    )
    assert np.array_equal(
        parsed_phrases.lyrics,
        np.array(
            [
                "tan yang jia tou song zhu a",
                "xin xue yong a jin",
                "zhen na ke tan jiao meng e jiang ming sang fan ying",
                "zong bao er chan wei fu",
                "ruan ta kao e zhen",
                "pa zhi pa ao bu guo chi cun guang yin",
            ]
        ),
    )

    # Check types
    assert type(parsed_phrases_char) == annotations.LyricData
    assert type(parsed_phrases_char.intervals) is np.ndarray
    assert type(parsed_phrases_char.lyrics) is list

    # Check values
    assert np.array_equal(
        parsed_phrases_char.intervals[:, 0],
        np.array([1.06, 28.68, 77.65, 120.04, 141.44, 193.5]),
    )
    assert np.array_equal(
        parsed_phrases_char.intervals[:, 1],
        np.array([17.92, 49.89, 110.08, 132.43, 161.62, 211.16]),
    )
    assert np.array_equal(
        parsed_phrases_char.lyrics,
        np.array(
            [
                "叹 杨家 投 宋主 啊",
                "心血 用 啊 尽",
                "真 可 叹 焦盂 呃 将 命丧 番营",
                "宗保儿 搀 为 父",
                "软榻 靠 呃 枕",
                "怕 只 怕 熬 不过 尺寸 光阴",
            ]
        ),
    )

    assert compmusic_jingju_acappella.load_phrases(None) is None


def test_load_phoneme():
    data_home = "tests/resources/mir_datasets/compmusic_jingju_acappella"
    track_id = "lseh-Tan_Yang_jia-Hong_yang_dong-qm"

    dataset = compmusic_jingju_acappella.Dataset(data_home)
    track = dataset.track(track_id)
    phoneme_path = track.phoneme_path
    parsed_phonemes = compmusic_jingju_acappella.load_phonemes(phoneme_path)

    # Check types
    assert type(parsed_phonemes) == annotations.LyricData
    assert type(parsed_phonemes.intervals) is np.ndarray
    assert type(parsed_phonemes.lyrics) is list

    # Check values
    assert np.array_equal(
        parsed_phonemes.intervals[:, 0], np.array([0.00, 1.06, 1.16, 2.53, 2.65, 2.94])
    )
    assert np.array_equal(
        parsed_phonemes.intervals[:, 1], np.array([1.06, 1.16, 2.53, 2.65, 2.94, 3.05])
    )
    assert parsed_phonemes.lyrics == ["", "@", "r\\'", "?", "AU^", "9"]

    assert compmusic_jingju_acappella.load_phonemes(None) is None


def test_load_syllable():
    data_home = "tests/resources/mir_datasets/compmusic_jingju_acappella"
    track_id = "lseh-Tan_Yang_jia-Hong_yang_dong-qm"

    dataset = compmusic_jingju_acappella.Dataset(data_home)
    track = dataset.track(track_id)
    syllable_path = track.syllable_path
    parsed_syllable = compmusic_jingju_acappella.load_syllable(syllable_path)

    # Check types
    assert type(parsed_syllable) == annotations.LyricData
    assert type(parsed_syllable.intervals) is np.ndarray
    assert type(parsed_syllable.lyrics) is list

    # Check values
    assert np.array_equal(parsed_syllable.intervals[:, 0], np.array([0.00, 1.06, 2.65, 2.94]))
    assert np.array_equal(parsed_syllable.intervals[:, 1], np.array([1.06, 2.65, 2.94, 3.76]))
    assert np.array_equal(parsed_syllable.lyrics, np.array(["", "tan", "", "yang"]))

    assert compmusic_jingju_acappella.load_syllable(None) is None


def test_load_metadata():
    data_home = "tests/resources/mir_datasets/compmusic_jingju_acappella"
    dataset = compmusic_jingju_acappella.Dataset(data_home)
    metadata = dataset._metadata

    track_id = list(metadata.keys())[0]
    assert track_id == "lseh-Tan_Yang_jia-Hong_yang_dong-qm"

    assert metadata[track_id]["work"] == "“叹杨家投宋主心血用尽”——《洪羊洞》（杨延昭）"
    assert metadata[track_id]["details"] is None
    assert metadata["data_home"] == "tests/resources/mir_datasets/compmusic_jingju_acappella"
