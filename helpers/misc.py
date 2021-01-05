"""
This module contains general helper functions with no single purpose
"""

import yaml

def load_yaml(yaml_file_path):
    """
    Function to load dictionary from a specified yaml file

    :param yaml_file_path: Path for yaml file (str)
    :return: yaml content (dict)
    """
    with open(yaml_file_path) as stream:
        yaml_output = yaml.safe_load(stream)
    return yaml_output

# Todo: Take out again when not needed anymore
def load_queries(filepath):
    with open(filepath, mode="r", encoding="utf-8") as f:
        queries = f.readlines()
    queries = [query[:-1].split() for query in queries]
    query_num = [query[0] for query in queries]
    query = [" ".join(query[1:]) for query in queries]
    return query_num, query