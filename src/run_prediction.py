import json
import logging
import os
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

assert os.getenv('BOUWDOSSIERS_OBJECTSTORE_PASSWORD')

assert len(sys.argv) == 2
input_json = sys.argv[1]


def write_csv(data, target_file):
    df = pd.DataFrame(data)
    df.to_csv(target_file)
    log.info(f'results written to {target_file}')
    return target_file


def perform_prediction(input_json):
    filename = os.path.basename(input_json)
    basename, _ = os.path.splitext(filename)
    csv_file_name = f'{basename}_results.csv'
    csv_file_path = os.path.join(OUTPUT_DIR, csv_file_name)

    with open(input_json) as f:
        data = json.load(f)

    dataset = data.get('data')

    results = []

    t0 = time.time()
    for element in dataset:
        prediction = None
        confidence = None

        try:
            prediction, confidence, url = predict_single(element)
        except HTTPError as e:
            if e.code == 404:
                message = f'Image not found: {e.url}'
                log.debug(message)
                results.append({
                    **element,
                    'url': e.url,
                    'notes': message
                })
                continue

        results.append({
            **element,
            'prediction': prediction,
            'confidence': confidence,
            'url': url
        })
    difference = time.time() - t0
    log.info(f'image retrieval & model prediction time: {round(difference, 3)}ms')

    write_csv(results, csv_file_path)
    target_file = f'automation/prediction/{csv_file_name}'
    upload_file(csv_file_path, target_file)

    return results


t0 = time.time()

perform_prediction(input_json)

global_diff = time.time() - t0
log.info(f'total script time: {round(global_diff, 3)}s')
