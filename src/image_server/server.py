"""
Flask app allowing annotations of yaml label files
"""
import os

import flask
from flask import Flask, send_from_directory
from flask_cors import CORS
import logging

log = logging.getLogger(__name__)


def create_app(image_dirs):
    app = Flask(__name__)
    CORS(app)

    @app.route('/<filename>')
    def get_image(filename):
        for image_dir in image_dirs:
            file_path = os.path.join(image_dir, filename)
            if os.path.exists(file_path):
                log.info(f'file {filename} found in {image_dir}')

                # Convert path from normal current working directory relative to Flask server relative
                flask_img_dir = os.path.join('../', image_dir)

                log.info(f'file {filename} found in flask relative dir {flask_img_dir}')

                return send_from_directory(flask_img_dir, filename)
        flask.abort(404)
    return app
