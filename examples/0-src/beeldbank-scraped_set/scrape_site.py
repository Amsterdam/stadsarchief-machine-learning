import os
import urllib

from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup


DOWNLOAD_DIR='./full'


def simple_get(url):
    """
    Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None.
    """
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None


def is_good_response(resp):
    """
    Returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200
            and content_type is not None
            and content_type.find('html') > -1)


def log_error(e):
    print(e)



def get_image(idx):
    url_prefix = "https://beeldbank.amsterdam.nl"
    url = f"https://beeldbank.amsterdam.nl/beeldbank/indeling/detail/start/{idx}?f_sk_documenttype%5B0%5D=bouwtekening"
    raw_html = simple_get(url)
    html = BeautifulSoup(raw_html, 'html.parser')

    download_link_elements = html.select('a[title="Afbeelding opslaan"]')
    if not download_link_elements:
        log_error('no link found on url')
        return

    parent_el = html.select('li.dc_identifier')
    id = parent_el[0].findChildren('span')[1].text.strip()
    print(id)


    filename = f"{DOWNLOAD_DIR}/{id}.jpg"
    download_link = url_prefix + download_link_elements[0].attrs.get('href')
    print(download_link)
    urllib.request.urlretrieve(download_link, filename)
    # data = simple_get(download_link)
    # print(data[:100])


def load_images():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    for idx in range(30, 1000):
        print(idx)
        get_image(idx)


load_images()
