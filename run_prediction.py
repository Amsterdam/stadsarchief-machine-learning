#!/usr/bin/env python3
import asyncio
import logging
import os
from dataclasses import dataclass

import pandas
import sys
import time

import pandas as pd

from src.objectstore_lib import upload_file
from src.predict.config import OUTPUT_DIR
from src.iiif.iiif import HttpErrorCode
from src.predict.predict import predict_single, iiifClient

log_level = logging.INFO
root = logging.getLogger()
root.setLevel(log_level)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(log_level)
root.addHandler(handler)

log = logging.getLogger(__name__)

N_IMAGE_FETCHERS = int(os.getenv('N_IMAGE_FETCHERS', 4))
PROGRESS_LOG_MODULO = 10
DIMENSION = [250, 250]

assert os.getenv('BOUWDOSSIERS_OBJECTSTORE_PASSWORD')

assert len(sys.argv) == 2
input_csv = sys.argv[1]

value = os.getenv('SKIP_PREDICTION')
SKIP_PREDICTION = os.getenv('SKIP_PREDICTION') is not None
if SKIP_PREDICTION:
    print(f'SKIP_PREDICTION value is not None (value is {value}). So the prediction step will be skipped.')


results = []

@dataclass()
class abData:
    index: int
    row: pd.Series


@dataclass()
class bcData:
    index: int
    img_path: str
    intermediate_result: dict


def open_csv(input_path):
    return pandas.read_csv(input_path)


def write_csv(data, target_file):
    columns = [
        'stadsdeel_code',
        'dossier_nummer',
        'file_name',
        'prediction',
        'confidence',
        'url',
        'notes',
    ]
    df = pd.DataFrame(data)
    df.to_csv(target_file, columns=columns)
    log.info(f'results written to {target_file}')
    return target_file


def crash_coroutine(e):
    """
    Used by coroutines running without a await call,
    Exceptions in these routines would normally silently fail and result in an unknown program state.
    The exception is logged AND the program is crashed as this is an unexpected runtime error.
    :param e:
    :return:
    """
    if type(e) is not asyncio.CancelledError:
        log.error(f'exception: {type(e)}, {e}')
        log.error('intentionally stopping further program execution!')
        sys.exit('coroutine crashed')


async def csv_reader(queue, input_csv):
    data_frame = pandas.read_csv(input_csv)

    for index, row in data_frame.iterrows():
        log.debug(f'stage A, sending {row.get("file_naam")}')
        await queue.put(abData(index, row))


async def image_fetcher(queue_in, queue_out):
    try:
        while True:
            example: abData = await queue_in.get()

            row = example.row

            stadsdeel_code = row.get('stadsdeel_code')
            dossier_nummer = str(row.get('dossier_nummer')).zfill(5)
            filename = row.get('file_naam')
            log.debug(f'stage B, processing {filename}')

            result = {
                'stadsdeel_code': stadsdeel_code,
                'dossier_nummer': dossier_nummer,
                'file_name': filename,
                'notes': '',
                'url': ''
            }

            t3 = time.time()
            try:
                [path, url] = await iiifClient.get_image(stadsdeel_code, dossier_nummer, filename, DIMENSION)
            except HttpErrorCode as e:
                url = e.url
                if e.code == 404:
                    message = f'Image not found: {url}'
                else:
                    message = f'Image could not be opened, status code {e.code}: {url}'
                log.warning(message)
                result['url'] = e.url
                result['notes'] = message
                await queue_out.put(bcData(example.index, None, result))
                queue_in.task_done()
                continue

            result['url'] = url

            difference = time.time() - t3
            log.debug(f'stage B, image retrieved {filename}: {round(difference, 3)}s')

            await queue_out.put(bcData(example.index, path, result))
            queue_in.task_done()
    except Exception as e:
        crash_coroutine(e)


async def image_predictor(queue_in):
    global results
    try:
        t2 = time.time()
        count = 0
        while True:
            imageData: bcData = await queue_in.get()
            result = imageData.intermediate_result
            path = imageData.img_path

            if path is not None:
                log.debug(f'stage C, processing {path}')

                if SKIP_PREDICTION:
                    log.warning('skipping prediction for {path}')
                else:
                    prediction, confidence = predict_single(path)

                    result['prediction'] = prediction
                    result['confidence'] = confidence
            else:
                log.debug('stage C, missing path')

            results.append(result)
            queue_in.task_done()

            count += 1
            if count % PROGRESS_LOG_MODULO == 0:
                difference = time.time() - t2
                t2 = time.time()
                log.info(f'index: {imageData.index}, count: {count}, {round(difference, 3)}s, {round(PROGRESS_LOG_MODULO / difference, 3)} rows per second')
    except Exception as e:
        crash_coroutine(e)


async def perform_prediction(input_csv: str):
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(handle_exception)

    queue1 = asyncio.Queue(maxsize=500)
    queue2 = asyncio.Queue(maxsize=100)

    log.info(f'number of image fetchers: {N_IMAGE_FETCHERS}')
    producer = loop.create_task(csv_reader(queue1, input_csv))
    img_consumers = [loop.create_task(image_fetcher(queue1, queue2)) for _ in range(N_IMAGE_FETCHERS)]
    prediction_consumers = [loop.create_task(image_predictor(queue2))]

    await producer
    log.info('---- done reading csv')

    await queue1.join()
    log.info('---- done retrieving images')
    for c in img_consumers:
        c.cancel()

    await queue2.join()
    log.info('---- done predicting')
    for c in prediction_consumers:
        c.cancel()

    return results


def handle_exception(loop, context):
    # first, handle with default handler
    loop.default_exception_handler(context)

    exception = context.get('exception')
    if isinstance(exception, Exception):
        log.error(context)
        loop.stop()


async def main(do_upload=False):
    t0 = time.time()

    t1 = time.time()
    await perform_prediction(input_csv)
    difference = time.time() - t1
    log.info(f'image retrieval & prediction time: {round(difference, 3)}s')

    filename = os.path.basename(input_csv)
    basename, _ = os.path.splitext(filename)
    csv_file_name = f'{basename}_results.csv'
    csv_file_path = os.path.join(OUTPUT_DIR, csv_file_name)
    write_csv(results, csv_file_path)

    if do_upload:
        target_file = f'automation/prediction/{csv_file_name}'
        upload_file(csv_file_path, target_file)

    global_diff = time.time() - t0
    log.info(f'total script time: {round(global_diff, 3)}s')


asyncio.run(main())
