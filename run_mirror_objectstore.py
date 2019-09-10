import os
import sys

from src.objectstore_lib import get_all_files

assert len(sys.argv) == 2
tmp_dir = sys.argv[1]

assert os.getenv('BOUWDOSSIERS_OBJECTSTORE_PASSWORD')

CONTAINER_NAME = 'automation'

target_dir = os.path.join(tmp_dir, CONTAINER_NAME)
print(f'mirroring {CONTAINER_NAME} to {target_dir}')
get_all_files(CONTAINER_NAME, target_dir)
print('mirror done')
