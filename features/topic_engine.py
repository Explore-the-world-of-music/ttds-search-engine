import pandas as pd
import numpy as np
import os
import re

from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel


class TopicRecommender():
    def __init__(self, stemmer, stop_set, num_topics = 4):
        self.stemmer = stemmer
        self.stop_set = stop_set
        self.num_topics = num_topics

    
    def cut_down_to_english_songs(self, plain_df):
        '''
        Returns a version of plain_df where only english songs are included

        :param plain_df: Dataframe at least containing the lyrics and the language (DF)
        :return eng_df: Dataframe containing only the english songs (DF)
        '''

        return plain_df.copy().loc[plain_df["language"] == "en"]


    def preprocess_single_lyrics(self, unprocessed_lyrics):
        '''
        Applies preprocessing to the unprocessed lyrics and adds them to a list of tokens

        :param unprocessed_lyrics: Lyrics to preprocess (str)
        :return tokens: List the preprocessed tokens (list)
        '''

        return [self.stemmer.stem(x) for x in re.findall(r"[\w]+", unprocessed_lyrics.lower()) if (x not in self.stop_set or "")]


    def preprocess_df_lyrics(self, df):
        '''
        Applies preprocessing to all lyrics in the dataframe and adds them to a list of lists of tokens

        :param df: Dataframe at least containing the lyrics (DF)
        :return whole_doc_tokens: List of Lists containing the preprocessed tokens (list)
        '''


        whole_doc_tokens = []

        for _, row in df.iterrows():
            whole_doc_tokens.append(self.preprocess_single_lyrics(row["SongLyrics"]))

        return whole_doc_tokens


    def train_model(self, whole_doc_tokens):
        '''
        Trains the LDA model on the provided tokens

        :param whole_doc_tokens: List of Lists containing the preprocessed tokens (list)
        :param num_topics: Number of topics to predict (int)
        '''

        self.common_dictionary = Dictionary(whole_doc_tokens)
        self.common_corpus = [self.common_dictionary.doc2bow(text) for text in whole_doc_tokens]

        self.model = LdaModel(self.common_corpus, self.num_topics, random_state  = 5)

    
    def predict_topic_for_single_lyrics(self, lyrics):
        '''
        Predicts the topic ID for a single lyrics string

        :param lyrics: Unprocessed lyrics to predict the topic for (str)

        :return topic: TopicID for the most probable topic (int)
        '''
        tokens = self.preprocess_single_lyrics(lyrics)
        
        cur_corpus = [self.common_dictionary.doc2bow(text) for text in tokens]
        output = list(self.model[cur_corpus])[0]
        topic = sorted(output,key=lambda x:x[1],reverse=True)[0][0]
        return topic


    def print_top10_topics_terms(self):
        '''
        Prints the most probable tokens and the token probabilities  for each topic
        '''
        for topic_id in range(0, self.num_topics):
            df_top_10 = pd.DataFrame(columns = ["token", "token_prob"])

            top_10_array = np.array(self.model.get_topic_terms(topicid = topic_id, topn=20))

            for idx, (token_id, token_prob) in enumerate(top_10_array):
                df_top_10.loc[idx,:] = [self.common_dictionary[token_id], np.round(token_prob,4)]

            print("Topic: ", topic_id)
            print(df_top_10)


    def save_model(self, model_name= "lda_model"):
        '''
        Saves the model

        :param model_name: Name of the file to save the model in
        '''

        self.model.save(model_name)


    def load_model(self, model_name= "lda_model"):
        '''
        Loads the model

        :param model_name: Name of the file to sload the model from
        '''

        self.model = LdaModel.load(model_name)






from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stop_set = stopwords.words('english')

topic_recommender = TopicRecommender(stemmer, stop_set)