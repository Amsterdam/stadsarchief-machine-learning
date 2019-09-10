"""
Resizes all images in INPUT_DIR and stores them in TARGET_DIR
"""

import PIL
from PIL import Image
import os

# INPUT_DIR = "beeldbank-scraped_set/full/"
# INPUT_DIR = "architect_set_object_recognition/full/"
INPUT_DIR = "aanvraag_besluit/src/data/"

TARGET_DIMS = (250, 250,)
# TARGET_DIMS = (400, 400,)
TARGET_DIMS = (800, 800,)
# TARGET_DIR = f"beeldbank-scraped_set/{TARGET_DIMS[0]}x{TARGET_DIMS[1]}/"
# TARGET_DIR = f"architect_set_object_recognition/{TARGET_DIMS[0]}x{TARGET_DIMS[1]}/"
TARGET_DIR = f"aanvraag_besluit/resized/{TARGET_DIMS[0]}x{TARGET_DIMS[1]}/"


def resize_dir(input_dir, target_dir, dim):
    os.makedirs(target_dir, exist_ok=True)

    items = os.listdir(input_dir)

    for item in items:
        if os.path.isfile(input_dir + item):
            input_file = input_dir + item
            print(f"input: {input_file}")
            try:
                im = Image.open(input_dir + item)
            except OSError as e:
                print(f"os error {e}, skipping")
                continue
            except PIL.Image.DecompressionBombError:
                print('file to large, skipping')
                continue

            if im.mode != "RGB":
                rgb_img = Image.new("RGB", im.size)
                rgb_img.paste(im)
                im = rgb_img
            im.convert('RGB')

            basename = os.path.basename(input_file)
            im_resize = im.resize(dim, Image.ANTIALIAS)
            target_file = f"{target_dir}{basename}"
            print(f"output: {target_file}")
            print(im.mode)
            im_resize.save(target_file, 'JPEG', quality=90)


if __name__== "__main__":
    resize_dir(INPUT_DIR, TARGET_DIR, TARGET_DIMS)
