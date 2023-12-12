import argparse
import hashlib
import itertools
import json
import os

DA_TACOS_INDEX_PATH = "../mirdata/datasets/indexes/da_tacos_index_1.1_crema.json"
DA_TACOS_ANNOTATION_SCHEMA = ["JAMS"]


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


def make_da_tacos_index(data_path, only_crema=False):
    metadata = {}
    tracks = {}
    for subset in ["benchmark", "coveranalysis"]:
        path_subset = os.path.join(
            data_path, "da-tacos_metadata", "da-tacos_" + subset + "_subset_metadata.json"
        )
        with open(path_subset) as f:
            meta = json.load(f)
        for work_id in meta.keys():
            for performance_id in meta[work_id].keys():
                track_id = subset + "#" + work_id + "#" + performance_id
                dir = "da-tacos_" + subset + "_subset"
                ext = "_crema"
                crema_path = os.path.join(
                    data_path, dir + ext, work_id + ext, performance_id + ext + ".h5"
                )
                if not only_crema:
                    ext = "_cens"
                    cens_path = os.path.join(
                        data_path, dir + ext, work_id + ext, performance_id + ext + ".h5"
                    )
                    ext = "_hpcp"
                    hpcp_path = os.path.join(
                        data_path, dir + ext, work_id + ext, performance_id + ext + ".h5"
                    )
                    ext = "_key"
                    key_path = os.path.join(
                        data_path, dir + ext, work_id + ext, performance_id + ext + ".h5"
                    )
                    ext = "_madmom"
                    madmom_path = os.path.join(
                        data_path, dir + ext, work_id + ext, performance_id + ext + ".h5"
                    )
                    ext = "_mfcc"
                    mfcc_path = os.path.join(
                        data_path, dir + ext, work_id + ext, performance_id + ext + ".h5"
                    )
                    if subset == "coveranalysis":
                        ext = "_tags"
                        tags_path = os.path.join(
                            data_path, dir + ext, work_id + ext, performance_id + ext + ".h5"
                        )
                if only_crema:
                    tracks[track_id] = {
                        "crema": (crema_path.replace(data_path + "/", ""), md5(crema_path)),
                    }
                else:
                    tracks[track_id] = {
                        "cens": (cens_path.replace(data_path + "/", ""), md5(cens_path)),
                        "crema": (crema_path.replace(data_path + "/", ""), md5(crema_path)),
                        "hpcp": (hpcp_path.replace(data_path + "/", ""), md5(hpcp_path)),
                        "key": (key_path.replace(data_path + "/", ""), md5(key_path)),
                        "madmom": (madmom_path.replace(data_path + "/", ""), md5(madmom_path)),
                        "mfcc": (mfcc_path.replace(data_path + "/", ""), md5(mfcc_path)),
                        "tags": (tags_path.replace(data_path + "/", ""), md5(tags_path))
                        if subset == "coveranalysis"
                        else (None, None),
                    }
        metadata[subset] = (path_subset.replace(data_path + "/", ""), md5(path_subset))
    da_tacos_index = {"version": "1.0.0", "tracks": tracks, "metadata": metadata}
    with open(DA_TACOS_INDEX_PATH, "w") as fhandle:
        json.dump(da_tacos_index, fhandle, indent=2)


def main(args):
    make_da_tacos_index(args.da_tacos_data_path, True)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make da_tacos index file.")
    PARSER.add_argument("da_tacos_data_path", type=str, help="Path to da_tacos data folder.")
    main(PARSER.parse_args())
