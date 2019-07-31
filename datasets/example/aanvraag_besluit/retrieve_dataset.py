import csv
import time
import urllib
import urllib.parse
import urllib.request
from urllib.error import HTTPError

import yaml
import os

URL_BASE = os.environ.get('URL_BASE')

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# DATASET_DIR = os.path.join(SCRIPT_DIR, 'tweede_dataset')
# INPUT_CSV = os.path.join(DATASET_DIR, 'src/alle_aanvragen_en_besluiten_na_1980_HK-annotated.csv')

DATASET_DIR = os.path.join(SCRIPT_DIR, 'dataset_3_ZO_AnB_aanvragen')
INPUT_CSV = os.path.join(DATASET_DIR, 'src/ZuidOost_aanvragen_20190616.csv')  # Don't forget to also force type
# DATASET_DIR = os.path.join(SCRIPT_DIR, 'dataset_3_ZO_AnB_other')
# INPUT_CSV = os.path.join(DATASET_DIR, 'src/ZO_eerste_paginas_AenB_openbaarr_en_openbaar_dossier_20190716.csv')  # Don't forget to also force type

# DATASET_DIR = os.path.join(SCRIPT_DIR, 'dataset_4_ZO_other_production')
# INPUT_CSV = os.path.join(DATASET_DIR, 'src/ZO_eerste_paginas_niet_AenB_openbaar_en_openbaar_dossier.csv')

OUT_LABEL_DIR = os.path.join(DATASET_DIR, 'labels/')
OUT_IMG_DIR = os.path.join(DATASET_DIR, 'images/')

TARGET_DIMS = [
    (250, 250,),
    (400, 400,),
    (800, 800,),
    # (1200, 1200,),
]

MAX_CNT = 99999


# MAX_CNT = 2000
# MAX_CNT = 10


def write_label(id, data):
    filename = f"{OUT_LABEL_DIR}/{id}.yaml"
    with open(filename, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def get_image_dir(dim):
    return os.path.join(OUT_IMG_DIR, f'{dim[0]}x{dim[1]}/')


def actually_download(url, target_file):
    total_tries = 3
    remaining_tries = 3
    sleep_seconds = 3

    while remaining_tries > 0:
        try:
            urllib.request.urlretrieve(url, target_file)
        except IOError as e:
            print(f'error downloading {url} on try: {str(total_tries - remaining_tries)}. Excpetion: {str(e)}')
            print(f'sleeping {sleep_seconds}s')
            time.sleep(sleep_seconds)
            remaining_tries -= 1
            continue
        else:
            break


def download_image(stadsdeel_code, dossier_nummer, filename):
    basename, ext = os.path.splitext(filename)
    document_part = f'{stadsdeel_code}/{str(dossier_nummer).zfill(5)}/{basename}{ext.lower()}'

    # print(document_part)
    document_encoded = urllib.parse.quote_plus(document_part)
    # print(document_encoded)

    for dim in TARGET_DIMS:
        url = f'{URL_BASE}{document_encoded}/full/{dim[0]},{dim[1]}/0/default.jpg'
        print(url)
        target_file = os.path.join(get_image_dir(dim), filename)

        exists = os.path.isfile(target_file)
        if exists:
            print(f'skipping download, file exists: {filename}')
        else:
            actually_download(url, target_file)


def retrieve_dataset(csv_path, update_label=False):
    print(f'retrieving: {csv_path}')
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_LABEL_DIR, exist_ok=True)

    for dim in TARGET_DIMS:
        path = get_image_dir(dim)
        os.makedirs(path, exist_ok=True)

    with open(csv_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        row_idx = 0
        skip_cnt = 0
        for row in csv_reader:
            print(f'{row_idx}: {row}')

            if row_idx != 0:  # skip header
                if len(row) > 7:
                    print(f'skipping invalid row: {row_idx}: {row}')
                    skip_cnt += 1
                else:
                    # type = 'aanvraag'

                    if len(row) < 7:
                        type = 'unknown'
                    else:
                        type = row[6]

                    filename = row[4]

                    basename, _ = os.path.splitext(filename)
                    # print(basename)
                    id = basename

                    # url = row[5]

                    stadsdeel_code = row[0]
                    dossier_nummer = row[1]

                    try:
                        download_image(stadsdeel_code, dossier_nummer, filename)
                    except HTTPError as e:
                        print(f'Download failed row: {row_idx}: {row}, {e}')
                        skip_cnt += 1
                    if update_label:
                        write_label(id, {
                            'reference': filename,
                            'type': type,
                            'stadsdeel_code': stadsdeel_code,
                            'dossier_nummer': dossier_nummer,
                            'dossier_type': row[2],
                            'dossier_jaar': row[3],
                        })

            row_idx += 1
            if row_idx >= MAX_CNT:
                break

        print(f'done')
        print(f'skipped {skip_cnt}')
        print(f'row count, {row_idx}')


retrieve_dataset(INPUT_CSV)