"""
Flask app allowing annotations of yaml label files
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import os
import sys
from data import LabelData

label_dir = os.environ.get('LABEL_DIR')
assert label_dir, 'path to label dir required (directory with <id>.yaml files)'

log_level = logging.DEBUG
root = logging.getLogger()
root.setLevel(log_level)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(log_level)
root.addHandler(handler)

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
    return jsonify(item)


@app.route('/<id>', methods=['PUT'])
def change_annotation(id):
    content = request.json
    result = data.update_example(id, content)
    return jsonify(result)
