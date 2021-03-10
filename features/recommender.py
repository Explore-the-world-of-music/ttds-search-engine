from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import random
import dill
import time
from scipy import spatial
from collections import defaultdict

class RecommendationEngine():
    def __init__(self, 
                min_values,
                diff_values,
                numeric_cols,
                additional_cols,
                normalized_weight_list,
                vector_size = 100,
                learning_rate = 0.0025,
                min_learning_rate = 0.0000025,
                min_count = 1):

        self.model = Doc2Vec(vector_size = vector_size, alpha = learning_rate, min_alpha= min_learning_rate, min_count = min_count)
        self.contained_doc_ids = {}

        self.recommended = defaultdict(dict)

        self.min_values = min_values
        self.diff_values = diff_values
        self.numeric_cols = numeric_cols
        self.additional_cols = additional_cols
        self.normalized_weight_list = normalized_weight_list

    def train_doc2vec(self, tokenized_lyrics_list, song_ids, max_epochs):

        '''
        Trains the model again - based on the provided samples

        :param tokenized_lyrics_list: List of Lists of preprocessed, tokenized lyrics (str)
        :param tokenized_lyrics_list: List of corresponding song IDs (int)
        :params max_epochs: Number of maximum training epochs (int)
        '''

        # Bring Documents into relevant format for Doc2Vec
        # Can be simply a list of elements, but for larger corpora,consider an iterable that streams the documents directly from disk/network.
        tagged_data = []
        for tokenized_lyrics, song_id in zip(tokenized_lyrics_list, song_ids):
            tagged_data.append(TaggedDocument(words=tokenized_lyrics, tags=[str(song_id)]))
            self.contained_doc_ids[song_id] = True

        self.model.build_vocab(tagged_data)

        # save learning rate and determine linear decay rate
        prior_learning_rate = self.model.alpha
        learning_decay = prior_learning_rate / max_epochs

        # Train the model for max_epochs epochs
        for epoch in range(max_epochs):
            if epoch / (max_epochs/10) == 0:
                print(f"Epoch {epoch} out of {max_epochs} running.")
            self.model.train(tagged_data,
                        total_examples=self.model.corpus_count,
                        epochs=self.model.epochs)

            # decrease the learning rate
            self.model.alpha -= learning_decay

        # reset learning rate after training
        self.model.alpha = prior_learning_rate

        # reduce memory usage
        self.model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    
    def find_doc2vec_similar_songs_known_song(self, song_id, n):

        '''
        Find and return the song IDs of the n most similar songs for a document the model was trained on

        :param song_id: Song ID for the song the most similar songs should be returned (int)
        :param n: Number indicating the number of most similar songs (int)

        :return: List of the most similar Song IDs (int)
        '''
        if song_id not in self.contained_doc_ids:
            # If the song_id is not trained on --> return random sample of doc IDs
            # TODO: Replace with getting the lyrics from the database and then predict
            print("Random")
            return random.sample(list(self.contained_doc_ids), n)
        else:
            # Find and return the most similar song IDs
            similar_docs = self.model.docvecs.most_similar(str(song_id), topn = n)
            return [int(similar_doc[0]) for similar_doc in similar_docs]

    def calc_cosine_similarity(self, id1, id2):
        '''
        Returns the cosine similarity between the calculated doc2vec vectors for the given document

        :param id1: Song ID of the first song (int)
        :param id2: Song ID of the second song (int)

        :return cosine_similarity: Cosine similarity of the two doc vectors (float)
        '''
        return spatial.distance.cosine(self.model.docvecs[str(id1)], self.model.docvecs[str(id2)])

    def equality_check(self, obj1, obj2, col_name):
        return (obj1[col_name] == obj2[col_name])

    def normalized_l2_norm(self, obj1, obj2, col_name):
        return ((obj1[col_name]-self.min_values[col_name])/self.diff_values[col_name] - (obj2[col_name]-self.min_values[col_name])/self.diff_values[col_name])**2

    def calc_recommendation_score(self, current_objective, cur_eval):
        '''
        :param current_objective: currently compared song row of the dataframe
        :param cur_eval: currently evaluated song row of the dataframe
        
        :return recommendation_score: achieved score for the recommendation
        '''

        # Categorical Variables
        song_distances = [0] * len(self.normalized_weight_list)
        for idx, col_name in enumerate(["ArtistMain", "language", "time_signature", "SongTitle"]):
            song_distances[idx] = self.equality_check(current_objective, cur_eval, col_name)

        # camelot matching
        cam_dist = current_objective["camelot_number"] - cur_eval["camelot_number"] 
        if cam_dist == 0 or (cam_dist == 1 and current_objective["camelot_major"] == cur_eval["camelot_major"]):
            song_distances[4] == True

        song_distances[5] = (cur_eval["SongRating"] / 10) # popularity

        song_distances[6] = self.calc_cosine_similarity(current_objective["SongID"], cur_eval["SongID"]) # lyric similarity 

        # Calculate Distance Measures for numerical colums
        for idx, col_name in enumerate(numeric_cols):
            song_distances[idx + 7] = self.normalized_l2_norm(current_objective, cur_eval, col_name)

        # Distance of valence, danceability...
        distance = 0
        for col_name in self.additional_cols:
            distance += self.normalized_l2_norm(current_objective, cur_eval, col_name)

        song_distances[11] = distance / len(self.additional_cols)

        recommendation_score = sum(normalized_weight_list * song_distances)
        return recommendation_score

    def check_if_recommended(self, current_objective, cur_eval):
        '''
        Retrieves the recommendation score and inserts it into the recommended dict
        if the new evaluation object reaches a higher recommendation score, than the
        current minimum in the recommended dict. It also removes the minimum key from the dict.

        :param current_objective: currently compared song row of the dataframe
        :param cur_eval: currently evaluated song row of the dataframe
        '''

        objective_song_id = current_objective["SongID"]
        recommendation_score = self.calc_recommendation_score(current_objective, cur_eval)

        if len(self.recommended[objective_song_id].items()) < 3:
            self.recommended[objective_song_id][cur_eval["SongID"]] = recommendation_score

        else:
            cur_min_recommended_score = min([v for _, v in self.recommended[objective_song_id].items()])

            if recommendation_score > cur_min_recommended_score:

                cur_min_key = [k for k, v in self.recommended[objective_song_id].items() if v == cur_min_recommended_score][0]
                del self.recommended[objective_song_id][cur_min_key]

                self.recommended[objective_song_id][cur_eval["SongID"]] = recommendation_score

    
    def find_doc2vec_similar_songs_unknown_song(self, tokenized_lyric_list, n):
        '''
        Find and return the song IDs of the n most similar songs for a document the model was not trained on

        :param tokenized_lyric_list: List of preprocessed, tokenized lyrics (str)
        :param n: Number indicating the number of most similar songs (int)

        :return: List of the most similar Song IDs (int)
        '''

        # Convert the lyrics into a vector
        vector = self.model.infer_vector(tokenized_lyric_list)

        # Find and return the most similar song IDs
        similar_docs = self.model.docvecs.most_similar([vector], topn = n)
        print(similar_docs)
        return [int(similar_doc[0]) for similar_doc in similar_docs]
    


    def save_doc2vec_model(self, filepath_model, filepath_dict):
        '''
        Saves the current model and the contained_doc_ids

        :param filepath_model: path to the save location of model - File has to be of type .model  (str)
        :param filepath_dict: path to the save location of dictionary - File has to be of type .pkl  (str)
        '''
        self.model.save(filepath_model)

        with open(filepath_dict, 'wb') as file:
            dill.dump(self.contained_doc_ids, file)

    def load_doc2vec_model(self, filepath_model, filepath_dict):
        '''
        Loades a saved model and the contained_doc_ids

        :param filepath_model: path to the saved location of model - File has to be of type .model  (str)
        :param filepath_dict: path to the saved location of dictionary - File has to be of type .pkl  (str)
        '''
        self.model = Doc2Vec.load(filepath_model)

        with open(filepath_dict, 'rb') as file:
            self.contained_doc_ids = dill.load(file)

    def save_recommender(self, filepath = "rc_recommender_model.pkl"):
        with open(filepath, 'wb') as file:
            dill.dump(self.recommended, file)

    def load_recommender(self, filepath = "rc_recommender_model.pkl"):
        with open(filepath, 'rb') as file:
            self.recommended = dill.load(file)



# Set path as needed for Preprocessor class
path = os.path.abspath(__file__)
dname = os.path.dirname(os.path.dirname(path))
os.chdir(dname)

import numpy as np
import pandas as pd


os.chdir(os.path.join(dname, "features", "recommender_model"))

from datetime import datetime


#print(rec_eng.find_similar_songs_known_song(724643, 10))

merged_df = pd.read_csv("merged_song_data.csv", encoding="utf-8")
merged_df = merged_df[merged_df["camelot_number"].notna()] # removing rows where no data could be scraped

def convert_time(x):
    try:
        return datetime.strptime(x, "%Y-%m-%d %H:%M:%S").year
    except:
        return datetime.strptime(x, "%Y-%m-%d").year


merged_df["release_year"] = merged_df["release_date"].apply(lambda x: convert_time(x)) 


max_values = merged_df.max()
min_values = merged_df.min()
diff_values = {
}

numeric_cols = ["bpm", "length", "loudness_decibel", "release_year"]
additional_cols =  ["acousticness","energy","liveness","speachiness","danceability","instrumentalness","loudness","valence"]



for col in numeric_cols + additional_cols:
    # if col == "release_year":
    #     print(max_values[col].to_pydatetime(), min_values[col].to_pydatetime())
    #     diff_values[col] = datetime.strptime(str(max_values[col]), "%Y-%m-%d %H:%M:%S") - datetime.strptime(str(min_values[col]), "%Y-%m-%d %H:%M:%S") 

    # else:
    diff_values[col] = max_values[col] - min_values[col]


weight_list = np.asarray([
    2, # artist
    3, # language
    3, # time signature
    -4, # title
    4, #camelot
    2, # popularity
    3, # lyric similarity
    -1, # bpm
    -1, # length
    -1, # loudness
    -1, # release date
    -1, # additional measures
])

normalized_weight_list = weight_list/weight_list.sum(0)

rec_eng = RecommendationEngine(min_values, diff_values, numeric_cols, additional_cols, normalized_weight_list)
rec_eng.load_doc2vec_model("word2vec2.model", "rec_model.pkl")

for idx, obj in merged_df.iterrows():
    #if idx%1000 == 0:
    print(f"{idx} out of {merged_df.shape[0]}", end='\r')
    current_objective = obj

    for _, evaluator in merged_df.iterrows():
        if evaluator["SongID"] == current_objective["SongID"]:
            continue
        cur_eval = evaluator
        rec_eng.check_if_recommended(current_objective, cur_eval)

rec_eng.save_recommender()
print(rec_eng.recommended[101336])

from twilio.rest import Client
import os

account_sid = "ACb6a955fcf2a50bea609380af8b72a928"
auth_token  = "51600f32cb0cbbb690225693ffd6d285"

client = Client(account_sid, auth_token)

from_whatsapp_number = 'whatsapp:+14155238886'
to_whatsapp_number = 'whatsapp:+4917621386912'

client.messages.create(body = "Recommender Saved", from_ = from_whatsapp_number, to=to_whatsapp_number)

'''
print(os.getcwd())
import sys
sys.path.append(os.getcwd())

from helpers.misc import load_yaml
from ETL.preprocessing import Preprocessor
config = load_yaml("config/config.yaml")
preprocessor = Preprocessor(config)


# Train on current data
import pandas as pd
data = pd.read_csv(os.path.join("features", "data-song_v1.csv"))

begin = time.time()
preprocessed_lyrics = [None] * data.shape[0]
for idx, row in data.iterrows():
    preprocessed_lyrics[idx] = preprocessor.preprocess(row["SongLyrics"])

ids = list(data["SongID"])

print(f"Data preprocessed in {begin - time.time()}")
begin = time.time()

rec_eng = RecommendationEngine()
rec_eng.load_model("word2vec2.model", "rec_model.pkl")

begin = time.time()
print(rec_eng.find_similar_songs_known_song(724643, 10))
print(f"Retrieval took: {time.time() - begin}")
#print(rec_eng.find_similar_songs_known_song(20, 10))


#rec_eng.train(preprocessed_lyrics, ids, max_epochs = 50)

#print(f"Engine trained in {begin - time.time()}")



#strg = preprocessor.preprocess("I've forgotten all the rest")
#print(rec_eng.find_similar_songs_unknown_song(strg, 3))
#begin = time.time()
#rec_eng.save_model("word2vec2.model", "rec_model.pkl")
#print(f"Model saved in {begin - time.time()}")

#rec_eng.load_model("word2vec2.model", "rec_model.pkl")
#print(rec_eng.contained_doc_ids)
#print(rec_eng.find_similar_songs_known_song(3, 1))




data = [["i","love","machine","learning.","its","awesome."],["i","love","coding","in","python"],["love","building","chatbots"],["chat","amagingly","well"]]
ids = [1,2,3,4]

rec_eng = RecommendationEngine()
rec_eng.train(data, ids, max_epochs = 10)

print(rec_eng.find_similar_songs_known_song(1, 1))
print(rec_eng.find_similar_songs_unknown_song(["i", "love", "chatbots"], 1))

rec_eng.save_model("word2vec2.model")
rec_eng.load_model("word2vec2.model")
print(rec_eng.find_similar_songs_known_song(1, 1))'''