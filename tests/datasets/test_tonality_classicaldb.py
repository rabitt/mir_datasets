import os
import sys

import librosa
import numpy as np

from mirdata.datasets import tonality_classicaldb
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "0"
    data_home = os.path.normpath("tests/resources/mir_datasets/tonality_classicaldb")
    dataset = tonality_classicaldb.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/tonality_classicaldb/"),
            "audio/01-Allegro__Gloria_in_excelsis_Deo_in_D_Major - D.wav",
        ),
        "key_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/tonality_classicaldb/"),
            "keys/01-Allegro__Gloria_in_excelsis_Deo_in_D_Major - D.txt",
        ),
        "spectrum_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/tonality_classicaldb/"),
            "spectrums/01-Allegro__Gloria_in_excelsis_Deo_in_D_Major - D.json",
        ),
        "hpcp_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/tonality_classicaldb/"),
            "HPCPs/01-Allegro__Gloria_in_excelsis_Deo_in_D_Major - D.json",
        ),
        "musicbrainz_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/tonality_classicaldb/"),
            "musicbrainz_metadata/01-Allegro__Gloria_in_excelsis_Deo_in_D_Major - D.json",
        ),
        "title": "01-Allegro__Gloria_in_excelsis_Deo_in_D_Major - D",
        "track_id": "0",
    }

    expected_property_types = {
        "key": str,
        "spectrum": np.ndarray,
        "hpcp": np.ndarray,
        "musicbrainz_metadata": dict,
        "audio": tuple,
    }
    run_track_tests(track, expected_attributes, expected_property_types)

    audio, sr = track.audio
    assert sr == 44100, "sample rate {} is not 44100".format(sr)
    assert audio.shape == (88200,), "audio shape {} was not (88200,)".format(audio.shape)


def test_to_jams():
    data_home = "tests/resources/mir_datasets/tonality_classicaldb"
    dataset = tonality_classicaldb.Dataset(data_home)
    track = dataset.track("0")
    jam = track.to_jams()
    assert jam["sandbox"]["key"] == "D major", "key does not match expected"
    assert (
        jam["file_metadata"]["title"] == "01-Allegro__Gloria_in_excelsis_Deo_in_D_Major - D"
    ), "title does not match expected"

    assert "spectrum" in jam["sandbox"]
    assert "hpcp" in jam["sandbox"]
    assert "musicbrainz_metatada" in jam["sandbox"]


def test_load_key():
    key_path = "tests/resources/mir_datasets/tonality_classicaldb/keys/01-Allegro__Gloria_in_excelsis_Deo_in_D_Major - D.txt"
    key_data = tonality_classicaldb.load_key(key_path)

    assert type(key_data) == str

    assert key_data == "D major"

    assert tonality_classicaldb.load_key(None) is None


def test_load_spectrum():
    spectrum_path = "tests/resources/mir_datasets/tonality_classicaldb/spectrums/01-Allegro__Gloria_in_excelsis_Deo_in_D_Major - D.json"
    audio_path = "tests/resources/mir_datasets/tonality_classicaldb/audio/01-Allegro__Gloria_in_excelsis_Deo_in_D_Major - D.wav"
    spectrum_data = tonality_classicaldb.load_spectrum(spectrum_path)

    assert type(spectrum_data) == np.ndarray

    y, sr = librosa.load(audio_path)
    spectrum = librosa.cqt(y, sr=sr, window="blackmanharris", hop_length=4096)

    # only first 2 seconds
    spectrum_data = spectrum_data[:, : spectrum.shape[1]]

    assert spectrum.shape[0] == spectrum_data.shape[0]
    assert spectrum.shape[1] == spectrum_data.shape[1]
    assert tonality_classicaldb.load_spectrum(None) is None


np.set_printoptions(threshold=sys.maxsize)


def test_load_hpcp():
    hpcp_path = "tests/resources/mir_datasets/tonality_classicaldb/HPCPs/01-Allegro__Gloria_in_excelsis_Deo_in_D_Major - D.json"
    hpcp_data = tonality_classicaldb.load_hpcp(hpcp_path)

    assert type(hpcp_data) == np.ndarray
    assert hpcp_data.shape[0] == 2865
    assert hpcp_data.shape[1] == 12
    assert isinstance(hpcp_data[0][0], float) is True

    assert tonality_classicaldb.load_hpcp(None) is None


musicbrainz_metadata_annotated = {
    "chromaprint": "AQAEHZG4R0mMPzmO44ds4sKU0AW0E-YR4kR7uPgRhp_wHknv4uiVLWg4priUWPjh88gv_EJD7ugZHLmC4xduobeCaI9yHMqPa8jxHP4Q-sY_oDl6HLfRRngu9Hxw_B80X4MbEfGJ4jjy7JCjou2DKfJyHPuhRTtCIP9R6C_StUev4KKO4ydyaN-DhsgX6vjRPOifoD3MJ0boDcdDHA_wH79w_KhOwpNRHK_wozwaZsvxbYc75biJ88iFD13kHE2YHscx_OBzXGJyhNV0JJePXolYXA74Cz_W489R8iPMqGnQF5d3PMFl4HKCcJr2Qj1y2CkeU7hPeAvDKHCV48cz-IFPo9SLT-KEPDIS4viOH2dSlM1xXI0Ff7grhDySMyKOHyeq48d0HZd2XIqM6zF4o-aC72jMIWeOfdAS_YOOq8azVGgVfDl6EjcfmIqEn4d7nPhwnvBRLpHwYroeiD_6HTltof_RnMGTR4Rj5njSCxfe4zkYfU-x4Wl41DaO8GgueFSS43i0zAgl6Tj6Hf2RH-J1hD9O4sVwNse_odqVwd-CJKJ-5OjRH_iJ2jm8LEdOBuJx3OSQlzhcqnhjvIe_B7mHT9BsPI3R_OjRHcny43h04byC6hb64jOq6_BZ9MHfEXkUYf4gBf5jFPl0VGpg_sNzXIouUMcRURISSxpqHUd_eBKDRlR-_MfhrVkUPCKLw-dxZTHeozaHMCPDI7ly_ISvB29y_PiF80SXPCOaH8lf5Clu4ccurfhRZnjI4k2M30iKHFte9MNxKUgV6sGhqEeOHj_6wzs-_PD1IEdS4x-8ozpO5A2S7_hRwmdc9DOO5jmSiI3yI0dDLihlXMF3ojlz9McjH8qm401xDwdzNE-CI5k_ZMki_MGxl_geND_qD0eZZoT3I2ehH5F24ygPtj169Dl--OiPvEj-g-gH78eQSlxC_IFPaD-uw2eOk_iSCOHKw90shHqMH7XQfAn0TUSuHvVH-EV7XHTxD02borqCEA6Ph9CfIz1-4hem88WjDxfiw4fcLPiF60gWH6X6DP3hDz-O58Jv8Mcv-EKvIeeh68FXVF96UNehaA_CDz4e5fCNLqSF_kRzIz906sgTuJIsHMf1w0mL6_jR9vjxXHDFo0e8m9C1I3w2BT_hH_X04DtO9Bqa-viHBz2amMGPT6zQx_C5GDkDHbngj_Bxwse1Eb4m9EdOEVcg_8iP6kh3aEehBVcKPT_0XDhCIf9x8YEe3vCJ8PjxC49yXCye6-iOMIfeDHmODz-u7HjIoOnRE3cr5EGeJjFeHT2cZURYFfqFn0a-HH0cuLbxI31RHfqOnHi06OiNN8Z1BYfP40eq6NDeI-ZQG5sf1D_y5GAPPVEQpi_RC5MP80ePasxxEd6M33gX4jgx8whxHVoPU0i1cQV_7Hngm8j14As0aWeRTxbu4xRFPD56wxGP5wiVQUfeH2eP08LxKfjREz4n5MI_6If4yfjho71xHZobyvDKB8eJHD88CZfxnPhywueNHxdunvi0w-GFYzfCdDz40FBe4z3yRplQ5odPXFKK5_jRHx9-oxeN5iryBlp29Gj0GFo5nEcvpNXx4Q_O40KqHnpwCmmIvig9-MZ5vJtgK0Y9_Giqoi-Lf-jhj8cZeDxe5IkOtzl84RC1IfoLHv6x5_jxH312HH3wD7UiJKeQH3c89DmabBfQoWGUF2V3NEcOXTfyfPiOhs9RFt-GPqnw45vRG7lE6BfCZEev4RPCq4aYaim6Hvlx50P_wfmDtUWuwJHAfMeDJ4KiH7nVwDx-wxd-_IOJ8rhw3ehZOGK8Yb_xrB9eeEwSBNKRTze2wDXxo0dP-EKvI6R0qNvx6YF1aD8-ESHFvWgi-rhM1Hos7MIJzvlQHs-xo8_hiWju4Qq-fzg_XEd54cHNI_QOjYdD5Qi14tnxLSJu9Ie_4x-eaUE3LRnxBz18hDySHFfyFFcOeqjUE1N-_MelRMGjI3_x49C-EHlRScJe2biYoU-LcznxHM6p4TySKQ-i469wMcVHEk2dow-St8hRHf6COwv-wYwf_DDVHL2GHDqDPDh-5F_wQ10thFbmoEme4Mnx6AmaMUO_ID8aRk-gscTRwEV14aE-6MJ5jD8miD--4EuRHz1-nEoiXPiS48mOfDqR3MeP60Wj50G1D3Vy7DpM0LsC4-SxF7VyfEfeEIqZHnkOntOCcU-OHz_-xuipwyh74sefB6-SoaGGHpF2CkkH5pJwHj962D6eydiuwvuJl7iWC_5xFt6E2sfRJ4cRHl_x5cKhd0PPw8T7wueRC-aPXuhzuDta5Th-9GED5-jDDf3h6C-u6Md34tKOvtjz4mIKJsdFIaJ3NGoy6NKDasoFf8EtIsfhoz_G6zhIK5KEBiMXwze2I9lP5AePZ8eU3zj6FM_QIn8g5kfOFJ0T48F__MHtozn6EQ9-IY9eaMnlI8dJ9F_w4sxRq9mF5sI3M8ihBzl2NsKPj4GfHR_e40e_wz-FHu0d_DjSQ4cRojR8T8hJ6AhzHAeK9NBx5D585DsOTS9-QOOB3DgHHz-NH008JJeOWBmNNU8qjA4AAQRgHgNIoGNABKEEKUQEAQRkgghymEEACYEgUEZAAZRARhLhhLAAAGAcN4IIAyAQBBAAjIAQIU8AA0hJQiSgFEQBjBOmCgAllAgQRyAhFgBCWggaECCAIgwYoEQwwDBCFAIEGYMAghI6IAAASCiEJEOICUkBGBA5o4gxhDFKFNACiMMYIAAAIJgzCGBmACAEAWIAYIBAgAghCACAlCRCAQSKIoBJpBgiEDgBjBLGC8WYEkhQBARViAghHBIEGgGIkcQYgAQTCAkBEQBSCKQIINYYBQgABEFDAkMAAkIUAYYwJgxjBAiqjETQIGiERYod7pBwgEqLHHBQCMKEwQIxIZgEgJjDAFBNGSCIJMBZ6hSBkDhQLEEEOgEYIMRAgIk0QBlHAWBGCOKEEhQARgwxCgFCABECKSKQIkgopqwQAkHkABDIKKEgIcIZRgRUhBBCIDEMACIcUQYqIjBggEFxmBDOMCsQYwQxQIUQQAHiHBXCMLEcMhgwzZQGGAGClIhEC0gFdAAowYBhwAAiNDOQMAGIEtRCRwRxADFHgRICEGSMEkY5gZBiWhiAIOBCGO8AUIApBJABTAgtEQLCUYSUYwoRBAACxgFiCFDMAGIAQY4khoQwhAkGkDEDCAEOeEgqhYgQTBBACHBWISGEEgAAxIhSwoJBgDEOESYQchQ4BgVkCgDLGAVIiWEY4UYAQA5CgRoDJCFIGKahAYQhQSQiABAjnCOACKUBEgIJJZAxSgAAkCHGIMmYEI5I4wCBQkLhABHAIkGcsYQQRgwxgigBhCKOIQKMAMpABAAxRjkKBLRSGcGEEYAZBIADQhODBCBICEIIMoIJUCRAgghHDBNEE4GMEEgMIAACTgEgJbFEKEKkYMIYIpxhTgCFDDBIeGeQKYowx4wxABAmgCAGGUEZEcYqYQQzBBjBiABIKISIAQIQQolUSAEGgDHAIOCYEUIZB5ATRAihjDCCIGJRIcAIJIgwAggBlCAIaUYAIYKSkpgSwgQsGEPOGGAUYAYZoEgQxhEkFCFBAiEIpYJIAgAEbIgkgQFCGgMCIQooQMBlbBCRpXWeAkCYsM4IhS0FRgBiBA",
    "similar_mbids": [
        {
            "artist": "Antonio Vivaldi",
            "mbid": "6d965a63-6cdb-4e49-beae-2e1247622652",
            "score": 0.96163,
            "title": "Allegro: Gloria in excelsis Deo",
        },
        {
            "artist": "The John Alldis Choir; John Constable; Olga Hegedus; Rodney Slatford; English Chamber Orchestra; Vittorio Negri",
            "mbid": "d2a477fd-0061-4db2-b39c-26c87eead92f",
            "score": 0.96163,
            "title": 'Gloria in D, RV 589: I. Allegro "Gloria in excelsis Deo"',
        },
        {
            "artist": "Antonio Vivaldi",
            "mbid": "dc863e40-7daa-4dd0-a44c-e8f33ed07268",
            "score": 0.96163,
            "title": "Gloria RV 589, I Gloria",
        },
        {
            "artist": "The John Alldis Choir; John Constable; Olga Hegedus; Rodney Slatford; English Chamber Orchestra; Vittorio Negri",
            "mbid": "d2a477fd-0061-4db2-b39c-26c87eead92f",
            "score": 0.954246,
            "title": 'Gloria in D, RV 589: I. Allegro "Gloria in excelsis Deo"',
        },
    ],
}


def test_load_musicbrainz_metadata():
    musicbrainz_metadata_path = "tests/resources/mir_datasets/tonality_classicaldb/musicbrainz_metadata/01-Allegro__Gloria_in_excelsis_Deo_in_D_Major - D.json"
    musicbrainz_metadata_data = tonality_classicaldb.load_musicbrainz(musicbrainz_metadata_path)
    print(musicbrainz_metadata_data)
    assert type(musicbrainz_metadata_data) == dict
    assert musicbrainz_metadata_data == musicbrainz_metadata_annotated

    assert tonality_classicaldb.load_musicbrainz(None) is None
