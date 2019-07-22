import logging
import os
import pandas
import sys
import time
from urllib.error import HTTPError

import pandas as pd

from objectstore_lib import upload_file
from predict.config import OUTPUT_DIR
from predict.predict import predict_single

root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
root.addHandler(handler)

log = logging.getLogger(__name__)

PROGRESS_LOG_MODULO = 100

assert os.getenv('BOUWDOSSIERS_OBJECTSTORE_PASSWORD')

assert len(sys.argv) == 2
input_csv = sys.argv[1]


def open_csv(input_path):
    return pandas.read_csv(input_path)


def write_csv(data, target_file):
    df = pd.DataFrame(data)
    df.to_csv(target_file)
    log.info(f'results written to {target_file}')
    return target_file


def perform_prediction(input_csv, do_upload):
    filename = os.path.basename(input_csv)
    basename, _ = os.path.splitext(filename)
    csv_file_name = f'{basename}_results.csv'
    csv_file_path = os.path.join(OUTPUT_DIR, csv_file_name)

    data_frame = pandas.read_csv(input_csv)

    results = []

    t1 = time.time()
    t2 = time.time()
    for index, row in data_frame.iterrows():
        prediction = None
        confidence = None

        stadsdeel_code = row.get('stadsdeel_code')
        dossier_nummer = str(row.get('dossier_nummer')).zfill(5)
        filename = row.get('file_naam')

        result = {
            'stadsdeel_code': stadsdeel_code,
            'dossier_nummer': dossier_nummer,
            'filename': filename,
            'notes': ''
        }

        try:
            prediction, confidence, url = predict_single(stadsdeel_code, dossier_nummer, filename)
        except HTTPError as e:
            if e.code == 404:
                message = f'Image not found: {e.url}'
                log.debug(message)
                result['url'] = e.url
                result['notes'] = message
        result['prediction'] = prediction
        result['confidence'] = confidence
        result['url'] = url

        results.append(result)

        if index % PROGRESS_LOG_MODULO == 0:
            difference = time.time() - t2
            t2 = time.time()
            log.info(f'row index: {index}, {round(1000 / difference * PROGRESS_LOG_MODULO, 3)} rows per second')

    difference = time.time() - t1
    log.info(f'image retrieval & model prediction time: {round(difference, 3)}ms')

    write_csv(results, csv_file_path)

    if do_upload:
        target_file = f'automation/prediction/{csv_file_name}'
        upload_file(csv_file_path, target_file)

    return results


t0 = time.time()

perform_prediction(input_csv, do_upload=False)

global_diff = time.time() - t0
log.info(f'total script time: {round(global_diff, 3)}s')
