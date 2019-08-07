"""
Flask app allowing annotations of yaml label files
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from .LabelData import LabelData
from predict.iiif_url import get_image_url

label_dir = os.environ.get('LABEL_DIR')
assert label_dir, 'path to label dir required (directory with <id>.yaml files)'

iiif_api_root = os.environ.get('IIIF_API_ROOT')
assert iiif_api_root


data = LabelData(label_dir)

app = Flask(__name__)
CORS(app)


@app.route('/')
def get_ids():
    ids = data.list_ids()
    return jsonify({'ids': ids})


@app.route('/<id>')
def get_single(id):
    item = data.get_example(id)
    dim = [1200, 1200]
    stadsdeel_code = item.get('stadsdeel_code')
    dossier_nummer = item.get('dossier_nummer')
    filename = f'{id}.jpg'
    url = get_image_url(iiif_api_root, stadsdeel_code, dossier_nummer, filename, dim)
    return jsonify({
        'url': url,
        'meta': item
    })


@app.route('/<id>', methods=['PUT'])
def change_annotation(id):
    content = request.json
    result = data.update_example(id, content)
    return jsonify(result)
