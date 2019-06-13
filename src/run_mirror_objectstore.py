import os

from objectstore_lib import get_all_files

TMP_DIR = os.getenv('TMP_DIR')
assert os.getenv('TMP_DIR')
assert os.getenv('BOUWDOSSIERS_OBJECTSTORE_PASSWORD')

CONTAINER_NAME='automation'

target_dir=os.path.join(TMP_DIR, CONTAINER_NAME)
get_all_files(CONTAINER_NAME, target_dir)
