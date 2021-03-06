import urllib


def get_image_url(root, stadsdeel_code, dossier_nummer, filename, dim):
    document_part = f'{stadsdeel_code}/{str(dossier_nummer).zfill(5)}/{filename}'
    document_encoded = urllib.parse.quote_plus(document_part)
    identifier = f'edepot:{document_encoded}'
    url = f'{root}{identifier}/full/{dim[0]},{dim[1]}/0/default.jpg'
    return url
