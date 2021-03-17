from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import random
import dill
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import preprocessing
from collections import defaultdict
import dill

class RecommendationEngine():
    def __init__(self, lyrics_similarity_calculator, 
    weight_vector = np.asarray([
        -4000, # title
        -5, # bpm
        -30, # length
        -100, # loudness
        -200, # release date
        -30, # additional measures
        0.3, # Lyric similarity
        ])):

        self.weight_vector = weight_vector
        self.numeric_columns = ["bpm", "length", "loudness_decibel", "release_year"]
        self.subjective_rating_columns = ["acousticness","energy","liveness","speachiness","danceability","instrumentalness","loudness","valence"]
        self.scaling_columns = self.numeric_columns + self.subjective_rating_columns
        self.LSM = lyrics_similarity_calculator
        self.recommendation = defaultdict(lambda: [])



    def equality_check(self, observation, relevant_df, col_name):
        return np.asarray(relevant_df[col_name].values == observation[col_name].values[0])

    
    def squared_distance(self, observation, relevant_df, col_name):
        return np.square(relevant_df[col_name].values - observation[col_name].values[0])


    def convert_time(self, x):
        '''
        Converts the strings into years
        '''
        try:
            return datetime.strptime(x, "%Y-%m-%d %H:%M:%S").year
        except:
            return datetime.strptime(x, "%Y-%m-%d").year

    
    def data_cleaning_and_conversion(self, df):
        '''
        Returns a DF with the only relevant observations and with correct column types
        '''
        self.plain_df = df

        # Removing irrelevant rows for recommender
        df = df[df["camelot_number"].notna()] # remove rows where additional data is not available
        #mask_rating = df["SongRating"] != 0
        #df = df.loc[mask_rating]

        df = df.drop(["released", "lyrics", "key"], axis = 1)
        
        # Converting columns to integers
        integer_columns = ["camelot_number", "bpm", "length", "time_signature",
                    "acousticness","energy","liveness", "speachiness",
                    "danceability","instrumentalness","loudness","valence"]

        for col_name in integer_columns:
            df[col_name] = df[col_name].astype(int)
            
        df["camelot_major"] = df["camelot_major"].astype(bool)

        df["release_year"] = df["release_date"].apply(lambda x: self.convert_time(x))
        df = df.drop("release_date", axis = 1)

        return df


    def scale_data(self, df, scaling_columns):
        '''
        Returns the dataframe to recommend on
        '''
        
        min_max_scaler = preprocessing.MinMaxScaler()

        for scale_col in scaling_columns:
            df.loc[:, scale_col+"_scaled"] = min_max_scaler.fit_transform(df[[scale_col]]).reshape(1,-1)[0]
        
        df = df.drop(scaling_columns, axis = 1) # Dropping original unscaled Columns
        df.reset_index(drop=True, inplace=True)
        
        return df


    def get_random_songID_from_artist(self, artist_id):
        '''
        Returns a random song of the artist
        '''

        artist_df = self.plain_df[self.plain_df["artist_id"] == artist_id]
        return artist_df["id"].sample(n = 1).values


    def train(self, to_predict_list, plain_df, n = 5):
        '''
        trains the recommendation engine for the IDs to predict

        :param to_predict_list: List of SongIDs to predict the recommendations for (list)
        :param plain_df: unprocessed Song Information dataframe including all songs (DF)
        :param n: Number of recommended songs to add (int)
        '''

        df = self.data_cleaning_and_conversion(plain_df)
        df = self.scale_data(df, self.scaling_columns)

        for idx, song_id in enumerate(to_predict_list):
            if idx % 10 == 0:
                print(f"{idx} out of {len(to_predict_list)} recommended!", end = "\r")

            if song_id not in df["id"].values:
                artist_id = self.plain_df.loc[self.plain_df["id"] == song_id]["artist_id"].values[0]
                recommended_ids = [self.get_random_songID_from_artist(artist_id) for i in range(n)]

            else:
                observation = df.loc[df["id"] == song_id]
                recommended_ids = self.calc_recommended_songs_for_one_obs(observation, df, n, threshold)
                

            self.recommendation[song_id] = recommended_ids            


    def calc_recommended_songs_for_one_obs(self, observation, df, n, threshold):
        '''
        Calculates the recommended songs for one observation

        :param observation: Row of the song to get the recommended songs for (Pandas Row)
        :param df: Dataframe to get the recommendations from (df)
        :param n: Number of recommended songs to add (int)
        '''


        # # Create Boolean Masks to reduce comparisons
        # mask_time_signature = df["time_signature"].values == observation["time_signature"].values[0]
        # #mask_camelot_major = df["camelot_major"].values == observation["camelot_major"].values[0]
        # camelot_number = observation["camelot_number"].values[0]
        # mask_camelot_major = np.logical_and(df["camelot_number"].isin([camelot_number-1, camelot_number, camelot_number+1]).values, (df["camelot_major"] == observation["camelot_major"].values[0]).values)
        # mask_language = df["language"].values == observation["language"].values[0]
        # mask = np.logical_and(np.logical_and(mask_language, mask_time_signature), mask_camelot_major)
        mask_same_artist = df["artist_id"].values == observation["artist_id"].values[0]
        mask_not_same_song = df["id"].values != observation["id"].values[0]
        mask = np.logical_and(mask_same_artist, mask_not_same_song)
        
        relevant_df = df.loc[mask]

        if relevant_df.shape[0] < n:
            needed = n - relevant_df.shape[0]
            recommended_ids = relevant_df["id"].values.tolist()
            recommended_ids += [self.get_random_songID_from_artist(observation["artist_id"].values[0]) for i in range(needed)]

        else:
            rec_df = pd.DataFrame()
            rec_df["id"] = relevant_df["id"]

            # Perform equality checks
            #for idx, col_name in enumerate(["ArtistMainID", "SongTitle"]):
            for idx, col_name in enumerate(["artist_id"]):
                rec_df[col_name] = self.equality_check(observation, relevant_df, col_name)

            # Calculate Distances between numeric attributes
            for idx, col_name in enumerate(self.numeric_columns):
                col_name += "_scaled"
                rec_df[col_name] = self.squared_distance(observation, relevant_df, col_name)

            for idx, col_name in enumerate(self.subjective_rating_columns):
                if idx == 0:
                    subjective_distance = self.squared_distance(observation, relevant_df, col_name + "_scaled")
                else:
                    subjective_distance += self.squared_distance(observation, relevant_df, col_name + "_scaled")

            rec_df["subjective_distance"] = subjective_distance/len(self.subjective_rating_columns)

            # Retrieve Lyric Similarity from LyricSimilarityCalculator
            rec_df["lyrics_similarity"] = rec_df.apply(lambda row: self.LSM.calculate_cosine_similarity(observation["id"], row["id"]), axis = 1)

            rec_df["recommendation_score"] = np.sum(np.multiply(self.weight_vector, rec_df.drop("id", axis = 1)), axis = 1)
            
            sorted_recommendations = rec_df.sort_values(by = "recommendation_score", ascending = False)
            # Normalizing the lists from list of np arrays into list of numbers
            recommended_ids = [num[0] if type(num)==np.ndarray else num for num in sorted_recommendations["id"].values[0:n]]

        return recommended_ids

    def get_recommendation(self, song_id, n):
        '''
        Returns the top n recommended songs for the given song_id

        :param song_id: SongID for the song to recommend for
        :param n: Number of recommended songs to add (int)
        '''

        return self.recommendation[song_id][0:n]

    def save_model(self, model_filepath = "recommendations.pkl"):
        """
        Function which saved the models in the file
        :param model_filepath: Filepath to the model - has to be of type .pkl (str)
        """
       

        with open(model_filepath, 'wb') as file:
            dill.dump(self.recommendation , file)


    def load_model(self, model_filepath = "recommendations.pkl"):
        """
        Function which loades the model in the file
        :param model_filepath: Filepath to the model - has to be of type .pkl (str)
        """

        with open(model_filepath, 'rb') as file:
            self.recommendation = dill.load(file)


    def save_model_to_csv(self, csv_filepath = 'recommendations.csv'):
        '''
        Saves the recommendation dictionary to a csv

        :param  csv_filepath: Filepath to save the csv (str)
        '''

        with open(csv_filepath, 'a') as f:
            for key, value in self.recommendation.items():
                value_str = ", ".join(str(x) for x in value)
                f.write(str(key)+","+value_str+"\n")



    










'''

weight_vector = np.asarray([
#    0.7, # artist
    -4000, # title
    -5, # bpm
    -30, # length
    -100, # loudness
    -200, # release date
    -30, # additional measures
    0.3, # Lyric similarity
])


# Load LSC
os.chdir("C:/Users/janni/Desktop/TTDS - Recommendation")

from lyric_similarity_calculator import LyricSimilarityCalculator

os.chdir("C:/Users/janni/Desktop/TTDS - Recommendation/recommender_model")
lyrics_similarity_calculator = LyricSimilarityCalculator()
lyrics_similarity_calculator.load_model("word2vec2.model", "rec_model.pkl")

# Initialize Recommendation Engine
rec_engine = RecommendationEngine(weight_vector = weight_vector, lyrics_similarity_calculator = lyrics_similarity_calculator)




plain_df = pd.read_csv("songs2.csv", encoding = "utf-8")
song_ids = plain_df["id"].values

begin = time.time()
rec_engine.train(to_predict_list=song_ids, plain_df=plain_df)
print(f"Training took: {time.time() - begin}")
rec_engine.save_model()

rec_engine.save_model_to_csv()

rec_engine.load_model()

print(rec_engine.find_similar_songs_known_song( 101763, 5))
print(rec_engine.find_similar_songs_known_song( 101763, 3))
'''


