import csv
import sys
import time
import urllib
import urllib.parse
import urllib.request
from urllib.error import HTTPError

import yaml
import os

IIIF_API_ROOT = os.environ.get('IIIF_API_ROOT')
assert IIIF_API_ROOT

assert len(sys.argv) == 3
dataset_dir = sys.argv[1]
input_csv = sys.argv[2]

out_label_dir = os.path.join(dataset_dir, 'labels/')
out_img_dir = os.path.join(dataset_dir, 'images/')


TARGET_DIMS = [
    (250, 250,),
    # (400, 400,),
    (800, 800,),
    # (1200, 1200,),
]

WRITE_LABEL = True  # Write label file
OVERWRITE_LABEL = False  # Overwrite existing label files
MAX_CNT = 99999


def write_labelfile(id, data):
    filename = f"{out_label_dir}/{id}.yaml"

    exists = os.path.isfile(filename)
    print(f'{filename}, {exists}')
    if OVERWRITE_LABEL or not exists:
        print('writing')
        with open(filename, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)


def get_dimension_dir(img_dir, dim):
    return os.path.join(img_dir, f'{dim[0]}x{dim[1]}/')


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


def download_image(img_dir, stadsdeel_code, dossier_nummer, filename):
    basename, ext = os.path.splitext(filename)
    document_part = f'{stadsdeel_code}/{str(dossier_nummer).zfill(5)}/{basename}{ext.lower()}'

    # print(document_part)
    document_encoded = urllib.parse.quote_plus(document_part)
    # print(document_encoded)

    for dim in TARGET_DIMS:
        url = f'{IIIF_API_ROOT}{document_encoded}/full/{dim[0]},{dim[1]}/0/default.jpg'
        target_file = os.path.join(get_dimension_dir(img_dir, dim), filename)
        print(f'{url} -> {target_file}')

        exists = os.path.isfile(target_file)
        if exists:
            print(f'skipping download, file exists: {filename}')
        else:
            actually_download(url, target_file)


def retrieve_dataset(csv_path, out_image_dir, out_label_dir, write_label=False):
    print(f'retrieving: {csv_path}')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    for dim in TARGET_DIMS:
        path = get_dimension_dir(out_image_dir, dim)
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
                        download_image(out_image_dir, stadsdeel_code, dossier_nummer, filename)
                    except HTTPError as e:
                        print(f'Download failed row: {row_idx}: {row}, {e}')
                        skip_cnt += 1
                    if write_label:
                        write_labelfile(id, {
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


retrieve_dataset(input_csv, out_img_dir, out_label_dir, write_label=WRITE_LABEL)
