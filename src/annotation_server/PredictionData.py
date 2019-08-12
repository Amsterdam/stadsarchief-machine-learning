import logging
import pandas as pd

log = logging.getLogger(__name__)


class PredictionData:

    def __init__(self, csv_path):
        self._load_csv(csv_path)

    def _load_csv(self, path):
        self.data = pd.read_csv(path)
        log.info('loaded prediction csv')

    def list_ids(self):
        return self.ids

    def get_example(self, id: str):
        df = self.data
        selection: pd.DataFrame = df.loc[df['file_name'] == f'{id}.jpg']
        hit: pd.Series = selection.iloc[0]
        data = {
            'confidence': hit['confidence'],
            'prediction': hit['prediction'],
        }
        log.debug(f'prediction data for {id}: {data}')
        return data
