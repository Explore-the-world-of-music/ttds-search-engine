from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class RecommendationEngine():
    def __init__(self, vector_size = 100, learning_rate = 0.0025, min_learning_rate = 0.0000025, min_count = 1):
        self.model = Doc2Vec(vector_size = vector_size, alpha = learning_rate, min_alpha= min_learning_rate, min_count = min_count)

    def train(self, tokenized_lyrics_list, song_ids, max_epochs):

        '''
        Trains the model again - based on the provided samples

        :param tokenized_lyrics_list: List of Lists of preprocessed, tokenized lyrics (str)
        :param tokenized_lyrics_list: List of corresponding song IDs (int)
        :params max_epochs: Number of maximum training epochs (int)
        '''

        # Bring Documents into relevant format for Doc2Vec
        # Can be simply a list of elements, but for larger corpora,consider an iterable that streams the documents directly from disk/network.
        tagged_data = [TaggedDocument(words=tokenized_lyrics, tags=[song_id]) for tokenized_lyrics, song_id in zip(tokenized_lyrics_list, song_ids)]
        self.model.build_vocab(tagged_data)

        # save learning rate and determine linear decay rate
        prior_learning_rate = self.model.alpha
        learning_decay = prior_learning_rate / max_epochs

        # Train the model for max_epochs epochs
        for epoch in range(max_epochs):
            self.model.train(tagged_data,
                        total_examples=self.model.corpus_count,
                        epochs=self.model.epochs)

            # decrease the learning rate
            self.model.alpha -= learning_decay

        # reset learning rate after training
        self.model.alpha = prior_learning_rate

        # reduce memory usage
        self.model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    
    def find_similar_songs_known_song(self, song_id, n):

        '''
        Find and return the song IDs of the n most similar songs for a document the model was trained on

        :param song_id: Song ID for the song the most similar songs should be returned (int)
        :param n: Number indicating the number of most similar songs (int)

        :return: List of the most similar Song IDs (int)
        '''

        # Find and return the most similar song IDs
        similar_docs = self.model.docvecs.most_similar(song_id, topn = n)
        return [similar_doc[0] for similar_doc in similar_docs]

    def find_similar_songs_unknown_song(self, tokenized_lyric_list, n):
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
        return [similar_doc[0] for similar_doc in similar_docs]
    


    def save_model(self, filepath):
        '''
        Saves the current model

        :param filepath: path to the save location - File has to be of type .model  (str)
        '''
        self.model.save(filepath)

    def load_model(self, filepath):
        '''
        Loades a saved model

        :param filepath: path to the saved location - File has to be of type .model  (str)
        '''
        self.model = Doc2Vec.load(filepath)





'''
data = [["i","love","machine","learning.","its","awesome."],["i","love","coding","in","python"],["love","building","chatbots"],["chat","amagingly","well"]]
ids = [1,2,3,4]

rec_eng = RecommendationEngine()
rec_eng.train(data, ids, max_epochs = 10)

print(rec_eng.find_similar_songs_known_song(1, 1))
print(rec_eng.find_similar_songs_unknown_song(["i", "love", "chatbots"], 1))

rec_eng.save_model("word2vec2.model")
rec_eng.load_model("word2vec2.model")
print(rec_eng.find_similar_songs_known_song(1, 1))'''