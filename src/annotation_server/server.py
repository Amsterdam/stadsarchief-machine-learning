"""
Flask app allowing annotations of yaml label files
"""
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import logging
from .LabelData import LabelData
from predict.iiif_url import get_image_url

log = logging.getLogger(__name__)


def create_app(iiif_api_root, label_dir, image_dir):
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

    if image_dir:
        log.info(f'Serving images from {app.root_path}/{image_dir}')

        @app.route('/<basename>.jpg')
        def get_image(basename):
            filename = f'{basename}.jpg'
            return send_from_directory(image_dir, filename)

    return app
