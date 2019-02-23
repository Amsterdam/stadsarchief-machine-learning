import os
import urllib

import yaml
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup


DOWNLOAD_DIR='./full'
LABEL_DIR='./labels'


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


def write_label(document_type, id):
    filename = f"{LABEL_DIR}/{id}.yaml"

    data = dict(
        type=document_type
    )

    with open(filename, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def write_image(html, id):
    url_prefix = "https://beeldbank.amsterdam.nl"

    download_link_elements = html.select('a[title="Afbeelding opslaan"]')
    if not download_link_elements:
        log_error('no link found on url')
        return

    filename = f"{DOWNLOAD_DIR}/{id}.jpg"
    download_link = url_prefix + download_link_elements[0].attrs.get('href')
    print(download_link)
    urllib.request.urlretrieve(download_link, filename)


def get_image(document_type, idx):
    url = f"https://beeldbank.amsterdam.nl/beeldbank/indeling/detail/start/{idx}?f_sk_documenttype%5B0%5D={document_type}"
    raw_html = simple_get(url)
    html = BeautifulSoup(raw_html, 'html.parser')

    parent_el = html.select('li.dc_identifier')
    id = parent_el[0].findChildren('span')[1].text.strip()
    print(id)

    write_image(html, id)
    write_label(document_type, id)


def load_images(document_type, low=1, high=300):
    os.makedirs(LABEL_DIR, exist_ok=True)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    for idx in range(low, high):
        print(f"{document_type}, {idx}")
        get_image(document_type, idx)


# load_images('bouwtekening', 500, 700)
# load_images('kaart', 500, 700)
# load_images('foto', 500, 700)
# load_images('affiche', 500, 700)
load_images('prent', 500, 700)
