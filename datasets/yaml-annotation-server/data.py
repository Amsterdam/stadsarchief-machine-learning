import datetime
import glob
import logging
import os

import yaml

log = logging.getLogger(__name__)


class LabelData:

    def __init__(self, directory):
        self.directory = directory
        self._get_listing()

    def _get_listing(self):
        file_paths = glob.glob(f"{self.directory}/*.yaml")

        if len(file_paths) == 0:
            log.warning(f'no labels found, does {self.directory} exists?')

        file_paths_sorted = sorted(file_paths)

        self.ids = []
        for fname in file_paths_sorted:
            basename = os.path.basename(fname)
            id, _ = os.path.splitext(basename)
            self.ids.append(id)

    def list_ids(self):
        return self.ids

    def get_example(self, id: str):
        path = os.path.join(self.directory, f'{id}.yaml')
        log.info(f'opening {path}')
        with open(path, 'r') as stream:
            data = yaml.safe_load(stream)
        log.debug(f'data: {data}')
        return data

    def update_example(self, id: str, item: dict):
        path = os.path.join(self.directory, f'{id}.yaml')
        item['last_update'] = datetime.datetime.now().isoformat()
        log.info(f'updating {path}:\n{item}')

        with open(path, 'w') as outfile:
            yaml.dump(item, outfile, default_flow_style=False)

        return item
