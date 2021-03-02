from collections import Counter, defaultdict
import numpy as np
import dill
import os
import re
import time
from nltk.stem import PorterStemmer



class Query_Completer():
    def __init__(self, n):
        """
        Initializes the Query_Completer class

        :param n: indicating the type of the ngram (n=3 --> trigram) (int)
        """
        self.n = n
        self.model = defaultdict(dict) # model where the ngrams are stored (dict)

        # Mapping Dictionaries
        self.mapping_to_int = {}    
        self.mapping_to_token = {}
        self.last_mapping = -1 # Indicator which is the last mapping assigned to a token

        self.stemmer = PorterStemmer() # Stemmer to stem identificators

    def tokenize_lyrics(self, lyrics):
        """
        Function which returns the tokenized and lowered lyrics

        :param lyrics: The lyrics to preprocess (str)
        :return line: Tokenized and lowered list of tokens (list)
        """
        tokenized = re.findall(r"[\w]+", lyrics)

        line = [x.lower() for x in tokenized if x != ""]
        return line


    def create_ngram(self, sentence):
        """
        Function which returns the ngrams of the sentence

        :param sentence: Tokens (list)
        :return ngram_list: List containing all the possible ngrams of the sentence in tuples of n items (list of tuples)
        """

        ngram_list = []
        for idx, word in enumerate(sentence):

            if idx-self.n <= -1:
                # Left padding the first n-1 observations
                existing_words = sentence[0:idx+1]

                # TODO Replace "None" with something different
                cur_list =  ["None"] * (self.n-len(existing_words)) + existing_words

                if "None" in cur_list:
                    ngram_list.append(tuple(cur_list))

            elif idx+self.n > (len(sentence)):

                # Right padding the last n-1 observations
                existing_words = sentence[idx:len(sentence)]
                cur_list = existing_words + ["None"] * (self.n-len(existing_words))
                ngram_list.append(tuple(cur_list))

            else:
                # Append n-grams for all other observations
                ngram_list.append(tuple(sentence[idx-self.n:idx]))

        return ngram_list


    def add_single_lyric(self, lyrics):
        """
        Function which adds the ngrams of the given sentence to the model

        :param lyrics: The lyrics (str)
        """
        # Tokenize the lyrics 
        token_list = self.tokenize_lyrics(lyrics)
        ngram_list = self.create_ngram(token_list)

        for tokens in ngram_list:
            # Stem the identifiers and leave the final word as is
            tokens = [self.stemmer.stem(token) for token in tokens[:-1]] + [tokens[-1]]

            mapping = [0]*len(tokens) # create mapping to int to pass to model 
            for idx, token in enumerate(tokens):

                # create a mapping for not existing tokens
                if token not in self.mapping_to_int:
                    self.last_mapping += 1
                    self.mapping_to_int[token] = self.last_mapping
                    self.mapping_to_token[self.last_mapping] = token

                # create mapping to pass to model
                mapping[idx] = self.mapping_to_int[token]

            # Add the ngram to the model
            if mapping[-1] not in self.model[tuple(mapping[:-1])]:
                self.model[tuple(mapping[:-1])][mapping[-1]] = 1
            else: 
                self.model[tuple(mapping[:-1])][mapping[-1]] += 1


    def add_lyrics(self, lyric_list):
        """
        Function which adds the ngrams of the given sentences to the model

        :param lyric_list: List of lyrics (list of strings)
        """
        for lyrics in lyric_list:
            self.add_single_lyric(lyrics)


    def predict_next_token(self, current_query):
        """
        Function which returns most probable next words based on the model

        :param current_query: The currently inserted query (str)

        :return sorted_result: Sorted list of most probable continuations + sentence (list of str)
        """
        
        query_token_list = self.tokenize_lyrics(current_query)



        if len(query_token_list) < self.n-1:
            # If not enough tokens in query
            return None

        # extracts the last m = n-1 tokens from the token list
        last_m_tokens = query_token_list[-(self.n-1):]
        last_m_tokens = [self.stemmer.stem(token) for token in last_m_tokens] 

        query_mapping = [0]*len(last_m_tokens)

        # Finds mapping for existing token but leaves 0 = None for each non existing token
        for idx, q_token in enumerate(last_m_tokens):
            try:
                query_mapping[idx] = self.mapping_to_int[q_token]
            except KeyError:
                continue

        results = dict(self.model[tuple(query_mapping)]) # returns the relevant dict of the model
        
        # Sorts the keys by the value and returns them with the most probable word in the first position
        sorted_result = [current_query + " " + self.mapping_to_token[int_map] for int_map, v in sorted(results.items(), key=lambda item: item[1],reverse = True)[0:5]]        
        return sorted_result


    def save_model(self, model_filepath = "qc_model.pkl", map_to_int_filepath = "qc_map_to_int.pkl", map_to_token_filepath = "qc_map_to_token.pkl"]):
        """
        Function which saved the models in the file

        :param model_filepath: Filepath to the model - has to be of type .pkl (str)
        :param map_to_int_filepath: Filepath to the mapping from token to int - has to be of type .pkl (str)
        :param map_to_token_filepath: Filepath to the mapping from int to token - has to be of type .pkl (str)
        """
        with open(model_filepath, 'wb') as file:
            dill.dump(self.model, file)

        with open(map_to_int_filepath, 'wb') as file:
            dill.dump(self.mapping_to_int, file)

        with open(map_to_token_filepath, 'wb') as file:
            dill.dump(self.mapping_to_token, file)

        
    def load_model(self, model_filepath = "qc_model.pkl", map_to_int_filepath = "qc_map_to_int.pkl", map_to_token_filepath = "qc_map_to_token.pkl"]):
        """
        Function which loades the model in the file

        :param model_filepath: Filepath to the model - has to be of type .pkl (str)
        :param map_to_int_filepath: Filepath to the mapping from token to int - has to be of type .pkl (str)
        :param map_to_token_filepath: Filepath to the mapping from int to token - has to be of type .pkl (str)
        """

        with open(model_filepath, 'rb') as file:
            self.model = dill.load(file)

        with open(map_to_int_filepath, 'rb') as file:
            self.mapping_to_int = dill.load(file)

        with open(map_to_token_filepath, 'rb') as file:
            self.mapping_to_token = dill.load(file)

        self.last_mapping = max(self.mapping_to_int, key = self.mapping_to_int.get)

    def reduce_model(self, cutoff):
        """
        Function which reduces the model by removing the identifiers which occured less than 'cutoff' times

        :param cutoff: Number indicating the least amount of occurrences to keep
        """
        to_delete = []
        for key, subdict in self.model.items():

            # Count how many occurences an identifier has
            count = 0
            for _, cur_count in subdict.items():
                count += cur_count

            if count < cutoff:
                to_delete.append(key)

        # Delete the relevant keys from the model
        for key in to_delete:
            del self.model[key]


'''
# Set path as needed for Query_Completer class
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)



qc = Query_Completer(n = 3)
qc.load_model("qc_new_data_model.pkl")
print(len(qc.model))

with open("keys.txt", "w", encoding = "utf-8") as out:
    for i in qc.model.keys():
        out.write(str(i) + "\n")

unique = set()


with open("unique.txt", "w", encoding = "utf-8") as out:
    for key in qc.model.keys():
        unique.update(list(key)) 
    for i in unique:
        out.write(str(i) + "\n")
print(len(unique))   


qc = Query_Completer(n = 3)

# Train on current data
import pandas as pd


qc = Query_Completer(n = 3)


data = pd.read_csv("data-song_v1.csv")

begin = time.time()
for ids, (idx, row) in enumerate(data.iterrows()):
    if ids%1000 == 0:
        print(f"{ids} out of {data.shape[0]}", end='\r')
    qc.add_single_lyric(row["SongLyrics"])

begin2 = time.time()
qc.reduce_model(5)
print(f"cutoff took: {time.time() - begin2}")

qc.save_model("qc_model.pkl")
print(f"Training and saving took: {time.time() - begin}")


# Reload the model and make predictions
qc.load_model("qc_model.pkl")

print("Predicting")
begin = time.time()
print(qc.predict_next_token("deux trois"))
print(qc.predict_next_token("Cinq six sept"))
print(qc.predict_next_token("did it"))
print(qc.predict_next_token("HALLO I BIMS"))
print(qc.predict_next_token("Oops I"))
print(qc.predict_next_token("My loneliness"))
print(qc.predict_next_token("Es ragen aus ihrem aufgeschlitzten Bauch"))
print(f"Prediction took took: {time.time() - begin}")'''



