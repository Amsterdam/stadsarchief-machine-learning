import os
import time
import logging
from calendar import timegm
from typing import List, Tuple
from objectstore import objectstore, get_full_container_list

log = logging.getLogger(__name__)


DIR_CONTENT_TYPE = 'application/directory'
DocumentList = List[Tuple[str, str]]

assert os.getenv('BOUWDOSSIERS_OBJECTSTORE_PASSWORD')

STORE_SETTINGS = dict(
    VERSION='2.0',
    AUTHURL='https://identity.stack.cloudvps.com/v2.0',
    TENANT_NAME='BGE000081_BOUWDOSSIERS',
    TENANT_ID='9d078258c1a547c09e0b5f88834554f1',
    USER=os.getenv('OBJECTSTORE_USER', 'bouwdossiers'),
    PASSWORD=os.getenv('BOUWDOSSIERS_OBJECTSTORE_PASSWORD'),
    REGION_NAME='NL',
)


def get_objectstore_connection():
    assert os.getenv('BOUWDOSSIERS_OBJECTSTORE_PASSWORD')
    connection = objectstore.get_connection(STORE_SETTINGS)
    return connection


def get_all_files(container_name: str, target_dir: str):
    connection = get_objectstore_connection()
    os.makedirs(target_dir, exist_ok=True)
    documents_meta = get_full_container_list(connection, container_name)

    for meta in documents_meta:
        if meta.get('content_type') != DIR_CONTENT_TYPE:
            name = meta.get('name')
            last_modified = meta.get('last_modified')
            dt = time.strptime(last_modified, "%Y-%m-%dT%H:%M:%S.%f")
            epoch_dt = timegm(dt)
            output_path = os.path.join(target_dir, name)
            dirname = os.path.dirname(output_path)
            os.makedirs(dirname, exist_ok=True)
            if os.path.isfile(output_path) and epoch_dt == os.path.getmtime(output_path):
                log.info(f"Using cached file: {output_path}")
            else:
                log.info(f"Fetching file: {output_path}")
                new_data = connection.get_object(container_name, name)[1]
                with open(output_path, 'wb') as file:
                    file.write(new_data)
                os.utime(output_path, (epoch_dt, epoch_dt))
