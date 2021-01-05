"""
This module creates the indexer class to create the index
"""

import pandas as pd
import numpy as np
import yaml
from nltk.stem import PorterStemmer
import re
from collections import defaultdict

class Indexer():
    """
    Class to create indexer given the settings in the config
    """
    def __init__(self, config):
        self.index = None
        self.all_docs_ids = None

    def build_index(self, preprocessor, doc_ids, raw_doc_texts, store_all_doc_ids = True):
        """
        Function to build the index based on given documents

        :param preprocessor: Preprocessor class instance
        :param doc_ids: List of document ids
        :param raw_doc_texts: List of document texts
        :return: index as dictionary
        """
        # Todo: Think about delta index as Ivan did
        # Initiate empty index dict
        index = defaultdict(lambda : defaultdict(list))

        # Create dictionary entry for each token and doc id
        for doc_id, raw_line in zip(doc_ids, raw_doc_texts):
            line = preprocessor.preprocess(raw_line)
            for pos, token in enumerate(line):
                index[token][doc_id].append(pos)

        self.index = index
        return index

    def add_all_doc_ids(self,doc_ids):
        self.all_doc_ids = set(doc_ids)

    def store_index(self):
        """
        Function to save final index in a defined format
        :param file_path: output path
        :return: None
        """
        with open("index.txt", "w") as text_output:
            for term in sorted(self.index.keys()):
                text_output.write(f"{term}:{len(self.index[term])}\n")
                for doc in self.index[term]:
                    items = str(self.index[term][doc])[1:-1].replace(" ", "")
                    text_output.write(f"\t{doc}: {items}\n")

    def load_index(self):
        """
        Function to load index from file

        return: loaded index
        """

        # Define a few helper functions
        def extract_word(line):
            word = re.findall(r"[\w']+", line)
            return word[0]
        def extract_document(line):
            document = line.split(":")[0]
            return document
        def extract_indices(line):
            indices = line.split(":")[1].split(",")
            return indices

        # Initialise empty dictionary to store index
        index = {}

        with open('index.txt', "r", encoding="utf8") as f:

            line_iter = iter(f.readlines())
            for line in line_iter:
                if '\t' not in line:
                    index[extract_word(line)] = {}
                    current_key = extract_word(line)
                else:
                    line = line.replace("\n", "").replace("\t", "").replace(" ", "")
                    document = extract_document(line)
                    index[current_key][document] = extract_indices(line)

        return index





