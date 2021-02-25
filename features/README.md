# Feature Documentation
## Query Completion (ngram_model.py)
This module trains and stores an ngram model to predict the next word from an inputted query. It returns the whole sentence with the five most probable continuations. The pretrained model for our data is available for download [here](https://www.dropbox.com/s/78bq8ogpdkxrtki/qc_model.pkl?dl=0) (1GB) and can be loaded using the `load_model` function.

The class `Query_Completer` needs the parameter for `n` for initialization (n=3 in the current case).

### Important Functions for Connection
- `add_single_lyric(lyric)` adds a new unprocessed lyric (str) to the model
- `save_model(filename)` saves the model in a pickle file
- `load_model(filename)` restores the model from a pickle file (might take a while)
- `predict_next_token(query)` predicts the five most probable continuations for a unprocessed query based on the last n-1 tokens of the query

### Examples
```Py
data = pd.read_csv("data-song-sample.csv") # Load the provided data

# Instanciate the model
qc = Query_Completer(n = 3)

# Add all Lyrics to the model
for idx, row in data.iterrows():
    qc.add_single_lyric(row["SongLyrics"])

# Saving and reloading of the model
qc.save_model("qc_model.pkl")
qc.load_model("qc_model.pkl")

# Predicting some continuations
print(qc.predict_next_token("deux trois"))
# Output: ['deux trois quatre', 'deux trois mousseau', 'deux trois no', 'deux trois calibre', 'deux trois mots']
print(qc.predict_next_token("Oops I"))
# Output: ['Oops I did', 'Oops I mean', 'Oops I meant', 'Oops I got', 'Oops I forgot']
print(qc.predict_next_token("Es ragen aus ihrem aufgeschlitzten Bauch"))
# Output: ['Es ragen aus ihrem aufgeschlitzten Bauch rippen'] <- "aufgeschlitzten Bauch" only occured in this context 
```

