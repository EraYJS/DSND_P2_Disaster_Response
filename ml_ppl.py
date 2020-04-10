import argparse as ap
import numpy as np
import pandas as pd
import pickle
import re

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion

from sqlalchemy import create_engine


def arg_parse():
    parser = ap.ArgumentParser()
    parser.add_argument("-p", action="store",
                        default="disaster_sn_msg", dest="db_path")
    parser.add_argument("-p", action="store",
                        default="model.pkl", dest="md_pth")


def load_db(path):
    engine = create_engine("sqlite:///" + path)
    data = pd.read_sql_table("data", engine)

    X = data['message']
    y = data.iloc[:, 4:]

    return X, y


def msg_tokenize(text):
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    stop_words = stopwords.words("english")

    # tokenize
    words = word_tokenize(text)

    # stemming
    stemmed = [PorterStemmer().stem(w) for w in words]

    # lemmatizing
    words_lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed if
                    w not in stop_words]

    return words_lemmed


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class

    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = pos_tag(msg_tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_ppl():
    ppl = Pipeline([
            ('features', FeatureUnion([
                    ('text_pipeline', Pipeline([
                            ('vect', CountVectorizer(tokenizer=msg_tokenize)),
                            ('tfidf', TfidfTransformer())
                    ])),

                    ('starting_verb', StartingVerbExtractor())
            ]))])

    return ppl


def model_save(model, path):
    filename = path
    pickle.dump(model, open(filename, 'wb'))
    pass


def ml_main():
    pass


if __name__ == "__main__":
    ml_main()
