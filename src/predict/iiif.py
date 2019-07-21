import os
import urllib


class IIIFClient:

    def __init__(self, apiRoot, imageDir):
        self.apiRoot = apiRoot
        self.imageDir = imageDir
        os.makedirs(imageDir, exist_ok=True)

    def get_image_dir(self, dim):
        return os.path.join(self.imageDir, f'{dim[0]}x{dim[1]}/')

    def get_image(self, stadsdeel_code, dossier_nummer, document_id, dim) -> str:
        assert len(dim) == 2, 'dimension (dim) should be of the form [width, height]'

        filename = f'{document_id}.jpg'

        target_dir = self.get_image_dir(dim)
        target_file = os.path.join(target_dir, filename)

        if os.path.isfile(target_file):
            print(f'skipping download, file exists: {filename}')
        else:
            document_part = f'{stadsdeel_code}/{dossier_nummer}/{filename}'
            document_encoded = urllib.parse.quote_plus(document_part)
            url = f'{self.apiRoot}{document_encoded}/full/{dim[0]},{dim[1]}/0/default.jpg'
            print(f'downloading: {url} -> {target_file}')
            urllib.request.urlretrieve(url, target_file)
        return target_file
