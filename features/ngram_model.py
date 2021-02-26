from collections import Counter, defaultdict
import numpy as np
import dill
import os
import re
import time



class Query_Completer():
    def __init__(self, n):
        self.n = n
        self.model = defaultdict(lambda: defaultdict(lambda: 0))

    def preprocess_lyrics(self, lyrics):
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
                cur_list =  [None] * (self.n-len(existing_words)) + existing_words

                if None in cur_list:
                    ngram_list.append(tuple(cur_list))

            elif idx+self.n > (len(sentence)):

                # Right padding the last n-1 observations
                existing_words = sentence[idx:len(sentence)]
                cur_list = existing_words + [None] * (self.n-len(existing_words))
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
        token_list = self.preprocess_lyrics(lyrics)
        ngram_list = self.create_ngram(token_list)

        for tokens in ngram_list:
            # The last token in the context of the former tokens is more likely through this example
            self.model[tokens[:-1]][tokens[-1]] += 1


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
        
        query_token_list = self.preprocess_lyrics(current_query)

        if len(query_token_list) < self.n-1:
            # If not enough tokens in query
            return None

        # extracts the last m = n-1 tokens from the token list
        last_m_tokens = query_token_list[-(self.n-1):]
        results = dict(self.model[tuple(last_m_tokens)]) # returns the relevant dict of the model
        
        # Sorts the keys by the value and returns them with the most probable word in the first position
        sorted_result = [current_query + " " + word for word, v in sorted(results.items(), key=lambda item: item[1],reverse = True)[0:5]]        
        return sorted_result


    def save_model(self, filepath):
        """
        Function which saves the current model

        :param filepath: path to the save location - File has to be of type .pkl  (str)
        """
        with open(filepath, 'wb') as file:
            dill.dump(self.model, file)

        
    def load_model(self, filepath):
        """
        Function which loades the model in the file

        :param filepath: path to the save location - File has to be of type .pkl  (str)
        """

        with open(filepath, 'rb') as file:
            self.model = dill.load(file)


# Set path as needed for Query_Completer class
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


'''
qc = Query_Completer(n = 3)

# Train on current data
import pandas as pd

begin = time.time()
qc = Query_Completer(n = 3)
data = pd.read_csv("data-song-sample.csv")

for idx, row in data.iterrows():
    qc.add_single_lyric(row["SongLyrics"])

qc.save_model("qc_model.pkl")
print(f"Training and saving took: {time.time() - begin}")


# Reload the model and make predictions
qc.load_model("qc_model.pkl")
print("Predicting")
print(qc.predict_next_token("deux trois"))
print(qc.predict_next_token("Cinq six sept"))
print(qc.predict_next_token("did it"))
print(qc.predict_next_token("Oops I"))
print(qc.predict_next_token("My loneliness"))
print(qc.predict_next_token("Es ragen aus ihrem aufgeschlitzten Bauch"))'''



