import numpy as np

from mirdata.datasets import small_4mula
from tests.test_utils import run_track_tests

TEST_DATA_HOME = "tests/resources/mir_datasets/small_4mula"


def test_track():
    default_trackid = "3ade68b6g3429fda3"
    dataset = small_4mula.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "3ade68b6g3429fda3",
        "art_id": "3ade68b5g3558eda3",
        "art_name": "Raça Negra",
        "art_rank": 83,
        "main_genre": "Romantic",
        "music_id": "3ade68b6g3429fda3",
        "music_lang": "pt-br",
        "music_lyrics": "Cheia de manias\\nToda dengosa\\nMenina bonita"
        "\\nSabe que é gostosa\\n\\nCom esse seu jeito \\n"
        "Faz o que quer de mim\\nDomina o meu coração"
        "\\nEu fico sem saber o que fazer\\nQuero te deixar "
        "\\nVocê não quer, não quer \\n\\nEntão me ajude a segurar"
        "\\nEssa barra que é gostar de você\\n"
        "Então me ajude a segurar\\nEssa barra que é gostar de você, êh"
        "\\n\\nDidididiê\\nDidididiê ê ê"
        "\\nDidididiê\\n\\nSe estou na sua casa quero ir pro cinema"
        "\\nVocê não gosta\\nUm motelzinho "
        "você fecha a porta\\n\\nEntão me ajude a segurar"
        "\\nEssa barra que é gostar de você\\nEntão me "
        "ajude a segurar\\nEssa barra que é gostar de você, êh"
        "\\n\\nDidididiê\\nDidididiê ê ê\\nDidididiê",
        "music_name": "Cheia de Manias",
        "musicnn_tags": ["classical", "strings", "violin"],
        "related_art": [
            {"id": "3ade68b5g6a58eda3", "name": "Só Pra Contrariar"},
            {"id": "3ade68b3g9d86eda3", "name": "Art Popular"},
            {"id": "3ade68b6gb1bbeda3", "name": "Eduardo Costa"},
            {"id": "3ade68b6g1bf9eda3", "name": "Roupa Nova"},
            {"id": "3ade68b5g3758eda3", "name": "Roberto Carlos"},
            {"id": "3ade68b5g3ce8eda3", "name": "Sampa Crew"},
            {"id": "3ade68b5g5fe8eda3", "name": "Zezé Di Camargo e Luciano"},
            {"id": "3ade68b5g5458eda3", "name": "Pixote"},
            {"id": "3ade68b5g3538eda3", "name": "Alexandre Pires"},
            {"id": "3ade68b6ga27aeda3", "name": "Grupo Revelação"},
        ],
        "related_genre": [
            "Romântico",
            "Samba",
            "Pagode",
            "Axé",
            "MPB",
            "Regional",
            "House",
            "Hardcore",
            "Sertanejo",
            "Indie",
            "Electronica",
            "Bossa Nova",
            "Pop",
            "K-Pop/K-Rock",
            "Infantil",
            "Trilha Sonora",
            "Black Music",
            "Jovem Guarda",
            "Clássico",
            "Samba Enredo",
            "Dance",
            "Forró",
        ],
        "related_music": [
            {"id": "3ade68b3gcc86eda3", "name": "Que Se Chama Amor", "lang": "pt-br"},
            {"id": "3ade68b8g7786afa3", "name": "Tá Escrito", "lang": "pt-br"},
            {"id": "3ade68b8ge043bfa3", "name": "Tá Vendo Aquela Lua", "lang": "pt-br"},
            {"id": "3ade68b8gc3f3bfa3", "name": "Sissi", "lang": "pt-br"},
            {"id": "3ade68b8gcb9bdfa3", "name": "Pra Que Entender?", "lang": "pt-br"},
            {
                "id": "3ade68b8gffebdfa3",
                "name": "Balada Boa (Tchê Tchê Rere)",
                "lang": "pt-br",
            },
            {"id": "3ade68b8ga4bddfa3", "name": "Ai, Se Eu Te Pego", "lang": "pt-br"},
            {"id": "3ade68b6g94b2fda3", "name": "Essa Tal Liberdade", "lang": "pt-br"},
            {"id": "3ade68b8g0635bfa3", "name": "A Gente Faz A Festa", "lang": "pt-br"},
            {"id": "3ade68b8gb845bfa3", "name": "Um Minuto", "lang": "pt-br"},
        ],
        "annotation_path": f"{TEST_DATA_HOME}/annotation/3ade68b6g3429fda3.tsv",
        "melspectrogram_path": f"{TEST_DATA_HOME}/melspectrogram/3ade68b6g3429fda3.npy",
    }

    expected_property_types = {
        "music_id": str,
        "music_name": str,
        "music_lang": str,
        "music_lyrics": str,
        "art_id": str,
        "art_name": str,
        "art_rank": int,
        "main_genre": str,
        "related_genre": list,
        "related_artist": list,
        "related_music": list,
        "musicnn_tags": list,
        "load_spectrogram": np.ndarray,
    }

    run_track_tests(track, expected_attributes, expected_property_types)


def test_melspectrogram():
    default_trackid = "3ade68b6g3429fda3"
    dataset = small_4mula.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    melspectrogram_data = track.load_spectrogram

    # check types
    assert type(melspectrogram_data) is np.ndarray
    assert melspectrogram_data.shape[0] == 128
    # check values
    assert np.allclose(
        melspectrogram_data[0][:5],
        np.array([0.20557478, 0.08086664, 0.0562083162, 0.02781249, 0.01821307]),
    )


def test_load_melspectrogram():
    _4mula_path = f"{TEST_DATA_HOME}/4mula_small.parquet"
    melspectrogram_data = small_4mula.load_melspectrogram(_4mula_path)

    # check types
    assert type(melspectrogram_data) is np.ndarray
    assert melspectrogram_data.shape == (3,)
    assert melspectrogram_data[0].shape[0] == 128
    assert melspectrogram_data[1].shape[0] == 128
    assert melspectrogram_data[2].shape[0] == 128

    # check values
    assert np.allclose(
        melspectrogram_data[0][0][:5],
        np.array([0.20557478, 0.08086664, 0.0562083162, 0.02781249, 0.01821307]),
    )
    assert np.allclose(
        melspectrogram_data[1][0][:5],
        np.array([0.1855194, 0.12639369, 0.02548128, 0.05115534, 0.05102731]),
    )
    assert np.allclose(
        melspectrogram_data[2][0][:5],
        np.array([0.03604873, 0.01234677, 0.00791732, 0.0339014, 0.08904114]),
    )
