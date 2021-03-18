# Feature Documentation
## Query Completion (ngram_model.py)
This module trains and stores an ngram model to predict the next word from an inputted query. It returns the whole sentence with the five most probable continuations. The pretrained model and the relevant mappings for our data is available for download [here](https://www.dropbox.com/sh/sr0ypvx2t1t31cp/AAAyQUq36VYczKIyJPeLlAQRa?dl=0) and can be loaded using the `load_model` function.

The class `Query_Completer` needs the parameter for `n` for initialization (n=3 in the current case).

### Important Functions for Connection
- `add_single_lyric(lyrics)` adds a new unprocessed lyric (str) to the model
- `add_lyrics_linewise(lyrics)` adds the lines of a new unprocessed lyric (str) to the model - additional ngrams [None, None, "New"] --> New York
- `save_model(model_filepath, map_to_int_filepath, map_to_token_filepath)` saves the models in pickle files - default values exist
- `load_model(model_filepath, map_to_int_filepath, map_to_token_filepath)` restores the models from pickle files - default values exist
- `predict_next_token(current_query)` predicts the five most probable continuations for a unprocessed query based on the last n-1 tokens of the query
- `reduce_model(cutoff)` reduces the model by removing the identifiers which occured less than 'cutoff' times

### Examples
```Py
data = pd.read_csv("data-song-sample.csv") # Load the provided data

# Instanciate the model
qc = Query_Completer(n = 3)

# Add all Lyrics to the model
for idx, row in data.iterrows():
    qc.add_single_lyric(row["SongLyrics"])

# Reducing the size of the model    
qc.reduce_model(5)

# Saving and reloading of the model
qc.save_model()
qc.load_model()

# Predicting some continuations
print(qc.predict_next_token("deux trois"))
# Output: ['deux trois quatre', 'deux trois mousseau', 'deux trois no', 'deux trois calibre', 'deux trois mots']
print(qc.predict_next_token("Oops I"))
# Output: ['Oops I did', 'Oops I mean', 'Oops I meant', 'Oops I got', 'Oops I forgot']
print(qc.predict_next_token("Es ragen aus ihrem aufgeschlitzten Bauch"))
# Output: ['Es ragen aus ihrem aufgeschlitzten Bauch rippen'] <- "aufgeschlitzten Bauch" only occured in this context and wont be part of the model after reduction
```
---

## Word Completion (word_completion.py)
This module trains and stores an ngram model to predict the next word from an inputted query. It returns the whole query with the five most probable completed tokens. The pretrained model is available for download [here](https://www.dropbox.com/s/bgifg45sbe3jbgl/wc_model.pkl?dl=0) and can be loaded using the `load_model` function.

### Important Functions for Connection
- `add_single_lyric(lyrics)` adds a new unprocessed lyric (str) to the model
- `save_model(model_filepath = "wc_model.pkl")` saves the models in a pickle file
- `load_model(model_filepath = "wc_model.pkl")` restores the models from a pickle file
- `predict_token(query, n)` predicts the `n` most probable words for the last splittable token

### Examples
```Py
data = pd.read_csv("data-song-sample.csv") # Load the provided data

# Instanciate the model
wc = Word_Completer()

# Add all Lyrics to the model
for idx, row in data.iterrows():
    wc.add_single_lyric(row["SongLyrics"])

# Saving and reloading of the model
qc.save_model()
qc.load_model()

# Predicting
n = 5
print(wc.predict_token("Hell",n))
# Output: ['hell', 'hello', 'hella', 'hells', 'hellish']
print(wc.predict_token("Oop",n))
# Output: [' oops', ' oop', ' oopsie', ' oopsy', ' oopiri']
print(wc.predict_token("World qua",n)) # In this case, only the last word is completed and appended to the former words
# Output: ['world quand', 'world quando', 'world quarter', 'world quanto', 'world qua']

```

---

## Recommendation Module (recommender.py + lyric_similarity_calculator.py)
This module `lyric_similarity_calculator.py` trains and stores a Doc2Vec model to predict the most similar lyrics of a given song.  The module `recommender.py` uses this lyric similarity and additional information about the songs to recommend songs from the same artist. The trained [model](https://www.dropbox.com/sh/p9kpfo843mftoz6/AABBaIezWkNlNshhOx6OyZkNa?dl=0) is available for download .

### Important Functions for Connection 
- `train(self, to_predict_list, plain_df, n = 5)` trains the recommendation engine for the IDs to predict
- `get_recommendation(self, song_id, n)` Returns the top n recommended songs for the given song_id
- `save_model(self, model_filepath = "recommendations.pkl")` Function which saved the models in the pkl file
- `load_model(self, model_filepath = "recommendations.pkl")` Function which loades the model from the pkl file
- `save_model_to_csv(self, csv_filepath = 'recommendations.csv')`  Saves the recommendation dictionary to a csv


```Py
from lyric_similarity_calculator import LyricSimilarityCalculator
lyrics_similarity_calculator = LyricSimilarityCalculator()
lyrics_similarity_calculator.load_model("word2vec2.model", "rec_model.pkl")

rec_engine = RecommendationEngine(lyrics_similarity_calculator)

plain_df = pd.read_csv("songs2.csv", encoding = "utf-8")
song_ids = plain_df["id"].values

rec_engine.train(to_predict_list=song_ids, plain_df=plain_df)

# Saving the model in two ways
rec_engine.save_model() #pkl dict
rec_engine.save_model_to_csv() # csv

# Load the Model
rec_engine.load_model()

# Get recommendation directly from the database

print(rec_engine.get_recommendation(song_id = 168861, n = 5))
# Output: [137760, 137761, 461554, 498408, 479446]
```





# LDA Topic Module (topic_engine.py)

The Topic Module trains a LDA Model on the available training data and creates `num_topics` number of topics. If further can predict the topic for an unseen document. The trained [model](https://www.dropbox.com/sh/ba5fdchl1p4cfi2/AAD4wQlOuXCmOM9FDhjrAkkia?dl=0) is available for download .

### Important Functions for Connection 

- `train(self, plain_df)` Trains the LDA model on the provided tokens
- `predict_topic_for_single_lyrics(self, lyrics)` Predicts the topic ID for a single lyrics string
- `print_top10_topics_terms(self, topn)` Prints the `topn` most probable tokens and the token probabilities  for each topic


```Py
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stop_set = stopwords.words('english')

plain_df = pd.read_csv("songs2.csv", encoding = "utf-8")

topic_recommender = TopicRecommender(stemmer, stop_set)

# Train and Save the model
topic_recommender.train(plain_df)
topic_recommender.save()

# Load the model and predict a topic
topic_recommender.load()
print(predict_topic_for_single_lyrics(lyrics = "Oops, I did it again!"))
# Output: 1


```