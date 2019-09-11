"""
Flask app serving images from multiple directories (static file server)

e.g.:
python run_image_server.py datasets/dataset_0/images/250x250/ datasets/dataset_1_mixed_hand_annotated/resized/800x800/
"""
import logging
import os
import sys

from src.image_server.server import create_app

log_level = logging.DEBUG
root = logging.getLogger()
root.setLevel(log_level)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(log_level)
root.addHandler(handler)

log = logging.getLogger(__name__)

assert len(sys.argv) > 1, 'use <command> image_dir1 image_dir2 ... image_dirX'

image_dirs = sys.argv[2:]
nl = '\n'
log.info(f'image dirs: {nl.join(image_dirs)}')

for image_dir in image_dirs:
    if not os.path.exists(image_dir):
        log.info(f'directory does not exist: {image_dir}')


app = create_app(image_dirs)

if __name__ == '__main__':
    app.run()
