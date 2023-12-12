import argparse
import csv
import itertools
import os

from mirdata.validate import md5

acousticbrainz_genre_INDEX_PATH = "../mirdata/datasets/indexes/test_acousticbrainz_genre_index.json"
ACOUSTICBRAINZ_GENRE_ANNOTATION_SCHEMA = ["JAMS"]


def make_acousticbrainz_genre_index(data_path):
    index = 0
    datasets = ["tagtraum", "allmusic", "lastfm", "discogs"]
    dataset_types = ["validation", "train"]
    f = open(acousticbrainz_genre_INDEX_PATH, "w")
    f.write("{\n")
    for dataset, dataset_type in itertools.product(datasets, dataset_types):
        tsv_file = open(
            os.path.join(
                data_path, "acousticbrainz-mediaeval-" + dataset + "-" + dataset_type + ".tsv"
            )
        )
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        next(read_tsv, None)
        read_tsv_list = list(read_tsv)
        for line, row in enumerate(read_tsv_list):
            mbid = ""
            track_id = dataset + "#" + dataset_type
            for i, r in enumerate(row):
                track_id = track_id + "#" + r
                if i == 0:
                    mbid = r
            ann_path = os.path.join(
                data_path, "acousticbrainz-mediaeval-" + dataset_type, mbid[:2], mbid + ".json"
            )
            f.write('  "%s": {\n' % (track_id,))
            f.write('    "data": [\n')
            f.write('      "%s",\n' % (ann_path.replace(data_path + "/", ""),))
            f.write('      "%s"\n' % md5(ann_path))
            f.write("    ]\n")
            is_the_last = (
                dataset == datasets[-1]
                and dataset_type == dataset_types[-1]
                and line == len(read_tsv_list) - 1
            )
            if not is_the_last:
                f.write("  },\n")
            else:
                f.write("  }\n")
            index += 1

    f.write("}")


def main(args):
    make_acousticbrainz_genre_index(args.acousticbrainz_genre_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make acousticbrainz_genre index file.")
    PARSER.add_argument(
        "acousticbrainz_genre_data_path", type=str, help="Path to acousticbrainz_genre data folder."
    )
    main(PARSER.parse_args())
    # with open(acousticbrainz_genre_INDEX_PATH, 'r') as json_file:
    #     data = json.load(json_file)
    #     for row in data:
    #         print(row)
