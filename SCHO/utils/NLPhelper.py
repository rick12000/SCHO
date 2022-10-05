import os

import nltk
import numpy as np
import pandas as pd

nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

import gensim
import gensim.downloader
import pickle
import warnings

from SCHO.utils.DataHandling import Filing

warnings.filterwarnings("ignore")
import time


class NLPEncoder:

    @staticmethod
    def load_twitter_pretrained_encoder():
        if not os.path.exists(Filing.parent_folder_path + "/_storage/NLP_autoencoders"):
            os.makedirs(Filing.parent_folder_path + "/_storage/NLP_autoencoders")
        cached_file_path = Filing.parent_folder_path + "/_storage/NLP_autoencoders/twitter_encoder.pkl"
        if os.path.exists(cached_file_path) and abs(
                time.time() - os.path.getmtime(cached_file_path)) < 60 * 60 * 24 * 7:
            file = open(cached_file_path, 'rb')
            gensim_model = pickle.load(file)
            file.close()
            print("Found stored NLP autoencoder that is less than 7 days old, using cached autoencoder...")
        else:
            gensim_model = gensim.downloader.load('glove-twitter-50')
            gensim_model.save(cached_file_path)

        return gensim_model

    @staticmethod
    def word_to_vec(gensim_autoencoder, X, pretrained=True):
        X_df = pd.DataFrame(X).reset_index(drop=True)
        if pretrained:
            X_df_clean = np.array(X_df.apply(gensim_autoencoder.utils.simple_preprocess))

            X_encoded = []
            for sentence in X_df_clean:
                vector_representation = np.zeros(len(gensim_autoencoder[0]))
                for word in sentence:
                    try:  # TODO if it's not in the databank pass in a custom unique vector to filter out later
                        word_vector = gensim_autoencoder[word]
                        vector_representation = vector_representation + word_vector
                    except:
                        vector_representation = np.zeros(len(gensim_autoencoder[0]))

                X_encoded.append(vector_representation)

            X_encoded = X_encoded.to_numpy()

        return X_encoded

    @staticmethod
    def fit_BOW_vectorizer(X, min_df=0.005, max_df=0.9):
        stemmer = PorterStemmer()  # lemmatizer = WordNetLemmatizer()

        # Lemmatize:
        X_lem = []
        for sentence in X:
            new_lemmatized_sentence = []
            for word in sentence.split(" "):
                word = stemmer.stem(word)  # word = lemmatizer.lemmatize(word)
                new_lemmatized_sentence.append(word)
            new_lemmatized_sentence = ' '.join(new_lemmatized_sentence)
            X_lem.append(new_lemmatized_sentence)
        X_lem = np.array(X_lem)

        # TF IDF:
        tfidf_object = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=(1, 2))
        fitted_tfidf = tfidf_object.fit(X_lem)

        return tfidf_object, fitted_tfidf

    @staticmethod
    def apply_BOW_vectorizer(vectorizer, tfidf_object, X):
        X_bow_raw = vectorizer.transform(X)
        X_bow = pd.DataFrame(X_bow_raw.todense(), columns=tfidf_object.get_feature_names())
        X_bow = X_bow.to_numpy()

        return X_bow
