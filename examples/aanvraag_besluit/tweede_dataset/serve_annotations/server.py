from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from examples.aanvraag_besluit.tweede_dataset.serve_annotations.data import AnnotationData

app = Flask(__name__)
CORS(app)

annotationData = AnnotationData()
annotationData.load_source()


@app.route('/<index_str>')
def get_single(index_str):
    index = int(index_str)
    item = annotationData.get_json_row(index)
    return Response(item)


@app.route('/')
def get_count():
    count = annotationData.get_count()
    return jsonify({'count': count})


@app.route('/put/<index_str>', methods=['PUT'])
def change_annotation(index_str):
    content = request.json
    index = int(index_str)
    document_type = content.get('document_type')
    print(f'"{document_type}"')
    annotationData.set_row_type(index, document_type)
    annotationData.save()
    return jsonify({"done": True})


@app.route('/save', methods=['PUT'])
def save_annotations():
    annotationData.save()
    return jsonify({"done": True})
