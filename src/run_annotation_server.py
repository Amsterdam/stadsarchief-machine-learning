"""
Flask app allowing annotations of yaml label files
"""
import logging
import sys

log_level = logging.DEBUG
root = logging.getLogger()
root.setLevel(log_level)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(log_level)
root.addHandler(handler)

from annotation_server.server import app
