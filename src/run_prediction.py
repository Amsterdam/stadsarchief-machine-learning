import json
import os
import sys
import time

import pandas as pd

from predict.config import OUTPUT_DIR
from predict.predict import predict_single


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
        prediction, confidence = predict_single(element)

        results.append({
            **element,
            'prediction': prediction,
            'confidence': confidence
        })
    difference = time.time() - t0
    print(f'image retrieval & model prediction time: {round(difference, 3)}ms')

    write_csv(results)

    return results


t0 = time.time()

perform_prediction(input_json)

difference = time.time() - t0
print(f'total script time: {round(difference, 3)}s')
