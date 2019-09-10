
def transform_aanvraag_labels(labels):
    """
    # Convert all labels to either "aanvraag" or "other"
    :param labels:
    :return:
    """
    return ['aanvraag' if x == 'aanvraag' else 'other' for x in labels]
