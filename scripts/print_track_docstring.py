import argparse
import itertools
import types

import mirdata

TEST_TRACKIDS = {
    "beatles": "0111",
    "dali": "4b196e6c99574dd49ad00d56e132712b",
    "gtzan_genre": "country.00000",
    "guitarset": "03_BN3-119-G_solo",
    "ikala": "10161_chorus",
    "maestro": "2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1",
    "medley_solos_db": "d07b1fc0-567d-52c2-fef4-239f31c9d40e",
    "medleydb_melody": "MusicDelta_Beethoven",
    "medleydb_pitch": "AClassicEducation_NightOwl_STEM_08",
    "openmic2018": "000046_3840",
    "orchset": "Beethoven-S3-I-ex1",
    "rwc_classical": "RM-C003",
    "rwc_jazz": "RM-J004",
    "rwc_popular": "RM-P001",
    "salami": "2",
    "slakh": "Track00001-S01",
    "tinysol": "Fl-ord-C4-mf-N-T14d",
    "dagstuhl_choirset": "DCS_LI_QuartetB_Take04_B2",
}


def get_attributes_and_properties(class_instance):
    attributes = []
    properties = []
    cached_properties = []
    functions = []
    for val in dir(class_instance.__class__):
        if val.startswith("_"):
            continue

        attr = getattr(class_instance.__class__, val)
        if isinstance(attr, mirdata.core.cached_property):
            cached_properties.append(val)
        elif isinstance(attr, property):
            properties.append(val)
        elif isinstance(attr, types.FunctionType):
            functions.append(val)
        else:
            raise ValueError("Unknown type {}".format(attr))

    non_attributes = list(itertools.chain.from_iterable([properties, cached_properties, functions]))
    for val in dir(class_instance):
        if val.startswith("_"):
            continue
        if val not in non_attributes:
            attributes.append(val)
    return {
        "attributes": sorted(attributes),
        "properties": sorted(properties),
        "cached_properties": sorted(cached_properties),
        "functions": sorted(functions),
    }


def main(args):
    data_home = "tests/resources/mir_datasets/{}".format(args.dataset)
    print(data_home)

    dataset = mirdata.initialize(args.dataset, data_home=data_home, version="test")

    if args.dataset in TEST_TRACKIDS.keys():
        track_id = TEST_TRACKIDS[args.dataset]
    else:
        print("No test track found for {}. ".format(args.dataset))
        print("Please add a test track to the dictionary at the top of this script.")
        return

    track = dataset.track(track_id)
    data = get_attributes_and_properties(track)

    print('"""{} Track class'.format(args.dataset))
    print("")
    print("Args:")
    print("    track_id (str): track id of the track")
    print("")

    if len(data["attributes"] + data["properties"]) > 0:
        print("Attributes:")
        for attr in data["attributes"] + data["properties"]:
            if attr == "track_id":
                print("    {} ({}): track id".format(attr, type(getattr(track, attr)).__name__))
            else:
                print("    {} ({}): TODO".format(attr, type(getattr(track, attr)).__name__))
        print("")

    if len(data["cached_properties"]) > 0:
        print("Cached Properties:")
        for attr in data["cached_properties"]:
            print("    {} ({}): TODO".format(attr, type(getattr(track, attr)).__name__))
        print("")

    print('"""')


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Print an empty docstring")
    PARSER.add_argument("dataset", type=str, help="dataset module name.")

    main(PARSER.parse_args())
