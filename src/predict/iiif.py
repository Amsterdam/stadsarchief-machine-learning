import logging
import os
import httpx
from httpx.exceptions import HttpError

from predict.config import IIIF_TIMEOUT
from predict.iiif_url import get_image_url

log = logging.getLogger(__name__)


class HttpErrorCode(HttpError):
    """
    Extension of httpx error so 4xx and 5xx errors can be distinguished from other status code exceptions
    """
    def __init__(self, message, code, url):
        super(HttpErrorCode, self).__init__(message)
        self.code = code
        self.url = url


class IIIFClient:

    def __init__(self, apiRoot, imageDir):
        self.apiRoot = apiRoot
        self.imageDir = imageDir
        self.httpxClient = httpx.AsyncClient()
        os.makedirs(imageDir, exist_ok=True)

    def get_image_dir(self, dim):
        dir = os.path.join(self.imageDir, f'{dim[0]}x{dim[1]}/')
        os.makedirs(dir, exist_ok=True)
        return dir

    async def download_image(self, url, target_file):
        r = await self.httpxClient.get(url, timeout=IIIF_TIMEOUT)
        if 400 <= r.status_code < 600:
            # Raise 4xx or 5xx exception, e.g.: 404 exception.
            # Using httpx raise_for_status to get error message
            try:
                r.raise_for_status()
            except Exception as e:
                raise HttpErrorCode(str(e), r.status_code, url)
        if r.status_code != 200 or len(r.content) == 0:
            r.raise_for_status()  # raise exception
        open(target_file, 'wb').write(r.content)

    async def get_image(self, stadsdeel_code, dossier_nummer, filename, dim) -> str:
        assert len(dim) == 2, 'dimension (dim) should be of the form [width, height]'

        target_dir = self.get_image_dir(dim)
        target_file = os.path.join(target_dir, filename)

        url = get_image_url(self.apiRoot, stadsdeel_code, dossier_nummer, filename, dim)
        if os.path.isfile(target_file):
            log.info(f'skipping download, file exists: {filename}')
        else:
            log.info(f'downloading: {url} -> {target_file}')
            await self.download_image(url, target_file)
        return [target_file, url]
