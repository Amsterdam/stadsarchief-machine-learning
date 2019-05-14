from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from examples.aanvraag_besluit.tweede_dataset.serve_annotations.data import AnnotationData

app = Flask(__name__)
CORS(app)

annotationData = AnnotationData()
annotationData.load_source()

@app.route('/')
def get_all():
    data = annotationData.get_json()
    return Response(data, mimetype='text/json')


@app.route('/put/<index_str>', methods=['PUT'])
def change_annotation(index_str):
    content = request.json
    index = int(index_str)
    annotationData.set_row_type(index, content.get('document_type'))
    return jsonify({"done": True})


@app.route('/save', methods=['PUT'])
def save_annotations():
    annotationData.save()
    return jsonify({"done": True})
