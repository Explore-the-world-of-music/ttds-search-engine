from collections import Counter, defaultdict
import numpy as np
import dill
import os

'''
# Set path as needed for Query_Completer class
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)'''

class Query_Completer():
    def __init__(self, n):
        self.n = n
        self.model = defaultdict(lambda: defaultdict(lambda: 0))


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


    def add_sentence(self, sentence):
        """
        Function which adds the ngrams of the given sentence to the model

        :param sentence: Tokens (list)
        """

        ngram_list = self.create_ngram(sentence)

        for tokens in ngram_list:
            # The last token in the context of the former tokens is more likely through this example
            self.model[tokens[:-1]][tokens[-1]] += 1


    def add_sentences(self, sentence_list):
        """
        Function which adds the ngrams of the given sentences to the model

        :param sentence_list: List of token lists (list)
        """
        for sentence in sentence_list:
            self.add_sentence(sentence)


    def predict_next_token(self, token_list):
        """
        Function which returns most probable next words based on the model

        :param token_list: Tokens (list)
        :return sorted_result: Sorted list of most probable continuations (list)
        """

        # extracts the last m = n-1 tokens from the token list
        last_m_tokens = token_list[-(self.n-1):]
        results = dict(self.model[tuple(last_m_tokens)]) # returns the relevant dict of the model
        
        # Sorts the keys by the value and returns them with the most probable word in the first position
        sorted_result = [k for k, v in sorted(results.items(), key=lambda item: item[1],reverse = True)]        
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

'''
with open("processed_lyrics.txt", "r") as f:
    lines = f.readlines()
    
lines = [line.split() for line in lines]


qc = Query_Completer(n = 3)
#qc.add_sentences(lines[0:2000])

#qc.save_model("qc_model.pkl")
qc.load_model("qc_model.pkl")
print(qc.predict_next_token(["night", "gotta"]))'''