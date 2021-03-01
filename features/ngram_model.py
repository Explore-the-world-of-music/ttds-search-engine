from collections import Counter, defaultdict
import numpy as np
import dill
import os
import re
import time
from operator import itemgetter



class Query_Completer():
    def __init__(self):
        self.bigram = defaultdict(lambda: defaultdict(lambda: 0))
        self.trigram = defaultdict(lambda: defaultdict(lambda: 0))
        self.fourgram = defaultdict(lambda: defaultdict(lambda: 0))

    def preprocess_lyrics(self, lyrics):
        """
        Function which returns the tokenized and lowered lyrics

        :param lyrics: The lyrics to preprocess (str)
        :return line: Tokenized and lowered list of tokens (list)
        """
        tokenized = re.findall(r"[\w]+", lyrics)
        line = [x.lower() for x in tokenized if x != ""]
        return line


    def create_ngram(self, sentence, n):
        """
        Function which returns the ngrams of the sentence

        :param sentence: Tokens (list)
        :param n: number of tokens (int)
        :return ngram_list: List containing all the possible ngrams of the sentence in tuples of n items (list of tuples)
        """

        ngram_list = []
        for idx, word in enumerate(sentence):

            if idx-n <= -1:
                # Left padding the first n-1 observations
                existing_words = sentence[0:idx+1]
                cur_list =  [None] * (n-len(existing_words)) + existing_words

                if None in cur_list:
                    ngram_list.append(tuple(cur_list))

            elif idx+n > (len(sentence)):

                # Right padding the last n-1 observations
                existing_words = sentence[idx:len(sentence)]
                cur_list = existing_words + [None] * (n-len(existing_words))
                ngram_list.append(tuple(cur_list))

            else:
                # Append n-grams for all other observations
                ngram_list.append(tuple(sentence[idx-n:idx]))

        return ngram_list


    def add_single_lyric(self, lyrics):
        """
        Function which adds the ngrams of the given sentence to the model

        :param lyrics: The lyrics (str)
        """
        token_list = self.preprocess_lyrics(lyrics)
        bigram_list = self.create_ngram(token_list, n = 2)
        trigram_list = self.create_ngram(token_list, n = 3)
        fourgram_list = self.create_ngram(token_list, n = 4)

        for tokens in bigram_list:
            # The last token in the context of the former tokens is more likely through this example
            self.bigram[tokens[:-1]][tokens[-1]] += 1

        for tokens in trigram_list:
            self.trigram[tokens[:-1]][tokens[-1]] += 1

        for tokens in fourgram_list:
            self.fourgram[tokens[:-1]][tokens[-1]] += 1


    def add_lyrics(self, lyric_list):
        """
        Function which adds the ngrams of the given sentences to the model

        :param lyric_list: List of lyrics (list of strings)
        """
        for lyrics in lyric_list:
            self.add_single_lyric(lyrics)


    def predict_next_token(self, current_query):
        """
        Function which returns most probable next words based on the matching ngram

        :param current_query: The currently inserted query (str)

        :return sorted_result:  Sorted list of most probable continuations + sentence (list of str)
                                empty if no results are found
        """
        
        query_token_list = self.preprocess_lyrics(current_query)
        
        # Either n or max(n, maximum_ngram)
        n = len(query_token_list)

        # extracts the last m = n-1 tokens from the token list
        last_m_tokens = query_token_list[-(n):]

        if n+1 == 4:
            results = dict(self.fourgram[tuple(last_m_tokens)]) # returns the relevant dict of the trigram
        elif n+1 == 3:
            results = dict(self.trigram[tuple(last_m_tokens)]) # returns the relevant dict of the trigram
        else:
            results = dict(self.bigram[tuple(last_m_tokens)]) # returns the relevant dict of the bigram (basic model)
        
        # Sorts the keys by the value and returns them with the most probable word in the first position
        sorted_result = [current_query + " " + word for word, v in sorted(results.items(), key=lambda item: item[1],reverse = True)[0:5]]        
        return sorted_result


    def save_model(self):
        """
        Function which saves the current models
        """
        filenames = ["qc_bigram", "qc_trigram", "qc_fourgram"]
        models = [self.bigram, self.trigram, self.fourgram]

        for filename, model in zip(filenames, models):
            with open(filename+".pkl", 'wb') as file:
                dill.dump(model, file)

        
    def load_model(self):
        """
        Function which loades the models from the files

        """
        with open("qc_bigram"+".pkl", 'rb') as file:
            self.bigram = dill.load(file)

        with open("qc_trigram"+".pkl", 'rb') as file:
            self.trigram = dill.load(file)

        with open("qc_fourgram"+".pkl", 'rb') as file:
            self.fourgram = dill.load(file)


    def reduce_size(self, n):
        """
        DEPRECTED: This function reduces the size of the current dictionary to the most observed n terms.

        :param n: number of predictions too keep for each ngram (int)
        """

        print("Function reduce_size Deprecated")
        #for key, _ in self.bigram.items():
        #    if len(self.bigram[key]) <= n:
        #        continue
        #    else:
        #        self.bigram[key] = dict(sorted(self.bigram[key].items(), key = itemgetter(1), reverse = True)[:n])



# Set path as needed for Query_Completer class
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)



# Train on current data
import pandas as pd

begin = time.time()
qc = Query_Completer()
data = pd.read_csv("data-song_v1.csv")

for idx, row in data[0:1000].iterrows():
    qc.add_single_lyric(row["SongLyrics"])

qc.save_model()
print(f"Training and saving took: {time.time() - begin}")

qc.load_model()

print(qc.predict_next_token("deux trois"))
print(qc.predict_next_token("deux trois quatre"))
print(qc.predict_next_token("deux trois quatre"))
print(qc.predict_next_token("Cinq six"))
print(qc.predict_next_token("did it"))
print(qc.predict_next_token("Oops I"))
print(qc.predict_next_token("My loneliness"))
print(qc.predict_next_token("Es ragen aus ihrem aufgeschlitzten Bauch"))