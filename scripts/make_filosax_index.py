import argparse
import json
from pathlib import Path

from mirdata.validate import md5

BACKING_FOLDER = "Backing"
SAX_FOLDER = "Participant "
FULL_TRACKS = 48
LITE_TRACKS = 5
FULL_PARTICIPANTS = 5
LITE_PARTICIPANTS = 2

# FILOSAX_INDEX_PATH = "/mir_datasets/Filosax_Lite/Scripts"
FILOSAX_INDEX_PATH = "/mir_datasets/Filosax_Lite/Scripts"


def tuple_for_file(file_path):
    relative_path = Path(file_path.parts[-3]) / file_path.parts[-2] / file_path.parts[-1]
    return (relative_path.as_posix(), md5(file_path))


def save_json(file_name, save_file, version_num):
    # save_path = Path(FILOSAX_INDEX_PATH) / ("%s_%s.json" % (file_name, version_num))
    save_path = Path(filosax_path) / ("%s_%s.json" % (file_name, version_num))
    with open(save_path, "w") as fhandle:
        json.dump(save_file, fhandle, indent=2)


def get_backing_files(track_id, filosax_path, include_checksum=True):
    backing_path = Path(filosax_path) / BACKING_FOLDER / track_id
    a = backing_path / "Bass_Drums.wav"
    b = backing_path / "Piano_Drums.wav"
    c = backing_path / "annotations.jams"
    if include_checksum:
        return (backing_path, tuple_for_file(a), tuple_for_file(b), tuple_for_file(c))
    return (backing_path, a, b, c)


def get_sax_files(sax_num, track_id, filosax_path, include_checksum=True):
    sax_folder = SAX_FOLDER + str(sax_num + 1)
    sax_path = Path(filosax_path) / sax_folder / track_id
    a = sax_path / "Sax.wav"
    b = sax_path / "annotations.json"
    c = sax_path / "Sax.mid"
    d = sax_path / "Sax.musicxml"
    e = sax_path / "Sax.pdf"
    if include_checksum:
        return (
            sax_path,
            tuple_for_file(a),
            tuple_for_file(b),
            tuple_for_file(c),
            tuple_for_file(d),
            tuple_for_file(e),
        )
    return (sax_path, a, b, c, d, e)


def get_backing_names(multi_name):
    a_ = multi_name + "_bass_drums"
    b_ = multi_name + "_piano_drums"
    return a_, b_


def check_files(track_id, filosax_path):
    backing_path, a, b, c = get_backing_files(track_id, filosax_path, include_checksum=False)

    # Check if backing folder exists
    if not backing_path.exists():
        raise OSError(
            "Error: folder structure incomplete. '%s' folder missing from 'Backing' folder."
            % track_id
        )

    # Check if backing files exist
    if (not a.exists()) or (not b.exists()) or (not c.exists()):
        raise OSError("Error: files missing from %s folder in 'Backing'." % track_id)

    # Check if sax folders and files exist
    for sax_num in range(FULL_PARTICIPANTS):
        sax_folder = SAX_FOLDER + str(sax_num + 1)
        (sax_path, a, b, c, d, e) = get_sax_files(
            sax_num, track_id, filosax_path, include_checksum=False
        )
        if not sax_path.exists():
            raise OSError(
                "Error: folder structure incomplete. '%s' folder missing from '%s' folder."
                % (track_id, sax_folder)
            )
        if (
            (not a.exists())
            or (not b.exists())
            or (not c.exists())
            or (not d.exists())
            or (not e.exists())
        ):
            raise OSError("Error: files missing from '%s' folder in '%s'." % (track_id, sax_folder))


def make_filosax_indexes(filosax_path, full_version, lite_version):
    print("Making Filosax indexes. Warning: this can take around 30 mins!")

    # Makes 4 indexes: full, full_sax, lite, lite_sax
    full_index = {"version": full_version}
    full_sax_index = {"version": full_version}
    lite_index = {"version": lite_version}
    lite_sax_index = {"version": lite_version}

    tracks_full, tracks_full_sax, tracks_lite, tracks_lite_sax = ({} for i in range(4))
    multitracks_full, multitracks_full_sax, multitracks_lite, multitracks_lite_sax = (
        {} for i in range(4)
    )

    # Iterate over tracks
    for track_num in range(FULL_TRACKS):
        print(".", end="")
        track_id = "%.2d" % (track_num + 1)
        check_files(track_id, filosax_path)
        multi_name = "multitrack_" + track_id
        track_names_full, track_names_full_sax, track_names_lite, track_names_lite_sax = (
            [] for i in range(4)
        )

        # Process backing files
        _, a, b, backing_annot = get_backing_files(track_id, filosax_path)
        a_, b_ = get_backing_names(multi_name)
        track_names_full.extend([a_, b_])
        tuplet_none = (None, None)
        tracks_full.update(
            {
                a_: {
                    "audio": a,
                    "annotation": tuplet_none,
                    "midi": tuplet_none,
                    "musicXML": tuplet_none,
                    "pdf": tuplet_none,
                }
            }
        )
        tracks_full.update(
            {
                b_: {
                    "audio": b,
                    "annotation": tuplet_none,
                    "midi": tuplet_none,
                    "musicXML": tuplet_none,
                    "pdf": tuplet_none,
                }
            }
        )

        if track_num < LITE_TRACKS:
            track_names_lite.extend([a_, b_])
            tracks_lite.update(
                {
                    a_: {
                        "audio": a,
                        "annotation": tuplet_none,
                        "midi": tuplet_none,
                        "musicXML": tuplet_none,
                        "pdf": tuplet_none,
                    }
                }
            )
            tracks_lite.update(
                {
                    b_: {
                        "audio": b,
                        "annotation": tuplet_none,
                        "midi": tuplet_none,
                        "musicXML": tuplet_none,
                        "pdf": tuplet_none,
                    }
                }
            )

        # Iterate over participants
        for sax_num in range(FULL_PARTICIPANTS):
            (sax_path, a, b, c, d, e) = get_sax_files(sax_num, track_id, filosax_path)

            # Process sax files
            track_name = multi_name + "_sax_" + str(sax_num + 1)
            track_dict = {"audio": a, "annotation": b, "midi": c, "musicXML": d, "pdf": e}
            track_names_full.append(track_name)
            track_names_full_sax.append(track_name)
            tracks_full.update({track_name: track_dict})
            tracks_full_sax.update({track_name: track_dict})
            if (track_num < LITE_TRACKS) and (sax_num < LITE_PARTICIPANTS):
                track_names_lite.append(track_name)
                track_names_lite_sax.append(track_name)
                tracks_lite.update({track_name: track_dict})
                tracks_lite_sax.update({track_name: track_dict})

        # Make multitrack entries
        multitracks_full[multi_name] = {"tracks": track_names_full, "annotations": backing_annot}
        multitracks_full_sax[multi_name] = {
            "tracks": track_names_full_sax,
            "annotations": backing_annot,
        }

        if track_num < LITE_TRACKS:
            multitracks_lite[multi_name] = {
                "tracks": track_names_lite,
                "annotations": backing_annot,
            }
            multitracks_lite_sax[multi_name] = {
                "tracks": track_names_lite_sax,
                "annotations": backing_annot,
            }

    # Compile indexes x4
    full_index.update({"tracks": tracks_full, "multitracks": multitracks_full})
    full_sax_index.update({"tracks": tracks_full_sax, "multitracks": multitracks_full_sax})
    lite_index.update({"tracks": tracks_lite, "multitracks": multitracks_lite})
    lite_sax_index.update({"tracks": tracks_lite_sax, "multitracks": multitracks_lite_sax})

    # Save as JSON
    save_json("filosax_index_full", full_index, full_version)
    save_json("filosax_index_full_sax", full_sax_index, full_version)
    save_json("filosax_index_lite", lite_index, lite_version)
    save_json("filosax_index_lite_sax", lite_sax_index, lite_version)

    print("")
    print("Indexes created.")


def main(args):
    make_filosax_indexes(args.filosax_data_path, args.full_version, args.lite_version)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make Filosax index files.")
    PARSER.add_argument("filosax_data_path", type=str, help="Path to Filosax data folder.")
    PARSER.add_argument("full_version", type=str, help="full index version")
    PARSER.add_argument("lite_version", type=str, help="lite index version")

    main(PARSER.parse_args())
