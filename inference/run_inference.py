import json
import os
import time

import pandas as pd

import config  # load environment variables
from inference import run_inference_single

INPUT_JSON = os.getenv("INPUT_JSON")
assert INPUT_JSON is not None


def write_csv(data):
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(config.OUTPUT_DIR, 'results.csv'))


def run_inference():
    with open(INPUT_JSON) as f:
        data = json.load(f)

    dataset = data.get('data')

    results = []

    t0 = time.time()
    for element in dataset:
        prediction, confidence = run_inference_single(element)

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

run_inference()

difference = time.time() - t0
print(f'total script time: {round(difference, 3)}s')
