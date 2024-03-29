"""
This module creates the preprocessor class to load and preprocess the data
"""

from nltk.stem import PorterStemmer
import re
import os
import xml.etree.ElementTree as ET

# Set path as needed for Preprocessor class
path = os.path.abspath(__file__)
dname = os.path.dirname(os.path.dirname(path))
os.chdir(dname)

class Preprocessor():
    """
    Class to create preprocessor given the settings in the config
    """
    def __init__(self, config):
        self.stemmer = PorterStemmer()
        self.stopping = config["preprocessing"]["remove_stop_words"]
        self.stemming = config["preprocessing"]["use_stemming"]
        self.replacement_patterns = config["preprocessing"]["use_replacement_patterns"]
        self.stop_set = set()
        with open("data/stopping_words.txt", "r") as stop:
            for word in stop:
                self.stop_set.add(word.rstrip())
        with open("data/replacement_patterns.txt", "r") as patterns:
            self.list_replacement_patterns = [tuple(pattern.rstrip().split(",")) for pattern in patterns]

    # Todo: Take out and replace by database (that is why it is implemented that inefficient)
    def load_data(self, file_path):
        with open(file_path, mode='rt', encoding='utf-8') as f:
            xml_root = ET.fromstring(f.read())
            num_docs = len(xml_root)
            doc_ids = [None] * num_docs
            raw_doc_texts = [None] * num_docs

            for idx, doc in enumerate(xml_root):
                doc_ids[idx] = int(doc.findtext('DOCNO'))
                raw_doc_texts[idx] = doc.findtext('HEADLINE') + doc.findtext('TEXT')
        return doc_ids, raw_doc_texts

    def replace_replacement_patterns(self, line):
        for (pattern, replacement) in self.list_replacement_patterns:
            cur_replace_regex = re.compile(pattern, re.IGNORECASE)
            line = cur_replace_regex.sub(replacement, line)
        return line

    def preprocess(self, line):
        """
        Function to perform the preprocessing for one line

        :param line: Raw input line (str)
        :return: Preprocessed line (str)
        """
        if self.replacement_patterns:
            line = self.replace_replacement_patterns(line)

        # Replace \\n by \n which we need for the data loading
        line = line.replace("\\n","\n")

        tokenized = re.findall("[\w]+", line)
        line = [x.lower() for x in tokenized if x != ""]
        if self.stopping:
            line = [x for x in line if x not in self.stop_set]

        if self.stemming:
            line = [self.stemmer.stem(x) for x in line]
        return line