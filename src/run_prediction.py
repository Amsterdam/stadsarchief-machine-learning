import json
import logging
import os
import sys
import time
from urllib.error import HTTPError

import pandas as pd

from predict.config import OUTPUT_DIR
from predict.predict import predict_single

root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
root.addHandler(handler)

log = logging.getLogger(__name__)

assert len(sys.argv) == 2
input_json = sys.argv[1]


def write_csv(data):
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(OUTPUT_DIR, 'results.csv'))


def perform_prediction(input_json):
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

    write_csv(results)

    return results


t0 = time.time()

perform_prediction(input_json)

difference = time.time() - t0
log.info(f'total script time: {round(difference, 3)}s')
