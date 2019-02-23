"""
File to quickly test individual parts of code
"""

from data import load_data

IMG_DIR = 'examples/0-src/beeldbank-scraped_set/300x300/'
LABEL_DIR = 'examples/0-src/beeldbank-scraped_set/labels/'

[X, Y] = load_data(IMG_DIR, LABEL_DIR)

print(f"shape X: {X.shape}")
print(f"shape Y: {Y.shape}")

print(f"Y first few: {Y[:5]}")
