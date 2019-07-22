import logging
import os
import urllib


log = logging.getLogger(__name__)


class IIIFClient:

    def __init__(self, apiRoot, imageDir):
        self.apiRoot = apiRoot
        self.imageDir = imageDir
        os.makedirs(imageDir, exist_ok=True)

    def get_image_dir(self, dim):
        dir = os.path.join(self.imageDir, f'{dim[0]}x{dim[1]}/')
        os.makedirs(dir, exist_ok=True)
        return dir

    def get_image(self, stadsdeel_code, dossier_nummer, filename, dim) -> str:
        assert len(dim) == 2, 'dimension (dim) should be of the form [width, height]'

        target_dir = self.get_image_dir(dim)
        target_file = os.path.join(target_dir, filename)

        document_part = f'{stadsdeel_code}/{dossier_nummer}/{filename}'

        document_encoded = urllib.parse.quote_plus(document_part)
        url = f'{self.apiRoot}{document_encoded}/full/{dim[0]},{dim[1]}/0/default.jpg'
        if os.path.isfile(target_file):
            log.info(f'skipping download, file exists: {filename}')
        else:
            log.info(f'downloading: {url} -> {target_file}')
            urllib.request.urlretrieve(url, target_file)
        return [target_file, url]
