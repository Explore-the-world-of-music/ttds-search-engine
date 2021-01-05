"""
This module contains general helper functions with no single purpose
"""

import yaml

def load_yaml(yaml_file_path):
    """
    Function to load dictionary from a specified yaml file

    yaml_file_path (str): the path of the yaml file to read
    returns dictOut (Dict): the content of the yaml file as dictionary object
    """

    with open(yaml_file_path) as stream:
        yaml_output = yaml.safe_load(stream)
    return yaml_output

def save_yaml(yaml_file_path, data):
    """
    Function to save list to a yaml file

    yaml_file_path (str): the path of the yaml file to store
    data (list): Object to be stored in yaml
    """

    with open(yaml_file_path,"w") as file:
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