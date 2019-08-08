from itertools import compress


def filter_unlabeled(yaml_items: list, data_list: list):
    """
    Removes unlabeled items from lists (items without annotation)
    :param yaml_items:
    :param data_list:
    :return:
    """
    bool_arr = [item.get('type') != '' for item in yaml_items]
    yaml_items_filtered = list(compress(yaml_items, bool_arr))
    data_list_filtered = list(compress(data_list, bool_arr))
    return yaml_items_filtered, data_list_filtered


def filter_unchecked(yaml_items: list, data_list: list):
    """
    Removes items that have not been manually checked (not checked by human)
    :param yaml_items:
    :param data_list:
    :return:
    """
    bool_arr = [item.get('checked') is True for item in yaml_items]
    yaml_items_filtered = list(compress(yaml_items, bool_arr))
    data_list_filtered = list(compress(data_list, bool_arr))
    return yaml_items_filtered, data_list_filtered
