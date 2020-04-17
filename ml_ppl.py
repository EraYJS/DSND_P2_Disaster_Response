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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion

from xgboost import XGBClassifier


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    stop_words = stopwords.words("english")

    words = word_tokenize(text)
    stemmed = [PorterStemmer().stem(w) for w in words]
    words_lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed if
                    w not in stop_words]

    return words_lemmed


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    @staticmethod
    def starting_verb(text):
        sentence_list = sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = pos_tag(tokenize(sentence))
            try:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
            except IndexError:
                pass
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_ppl(clf_type):
    """
    Grid Search Results:
        AdaBoost:
            {'clf__estimator__learning_rate': 1.2,
             'clf__estimator__n_estimators': 51,
             'vect__max_df': 0.4}


        XGBoost:
            {'clf__estimator__colsample_bytree': 1.0,
             'clf__estimator__gamma': 5.0,
             'clf__estimator__learning_rate': 0.5,
             'clf__estimator__min_child_weight': 1,
             'clf__estimator__subsample': 1.0,
             'vect__max_df': 0.75}
    """
    ada_ppl = Pipeline([
            ('features', FeatureUnion([
                    ('text_pipeline', Pipeline([
                            ('vect', CountVectorizer(
                                    tokenizer=tokenize,
                                    max_df=0.4
                            )),
                            ('tfidf', TfidfTransformer())
                    ])),
                    ('starting_verb', StartingVerbExtractor())
            ])),
            ('clf', MultiOutputClassifier(
                    AdaBoostClassifier(
                            n_estimators=51,
                            learning_rate=1.2
                    )))
    ])

    # XGBoost models are much more computationally expensive than AdaBoost
    xgb_ppl = Pipeline([('features', FeatureUnion([
                    ('text_pipeline', Pipeline([
                            ('vect', CountVectorizer(
                                    tokenizer=tokenize,
                                    max_df=0.75)),
                            ('tfidf', TfidfTransformer())
                    ])),
                    ('starting_verb', StartingVerbExtractor())
            ])),
            ('clf', MultiOutputClassifier(
                    XGBClassifier(
                            colsample_bytree=1.0,
                            gamma=5.0,
                            learning_rate=0.5,
                            min_child_weight=1,
                            subsample=1.0
                    )))
    ])

    if clf_type == "Ada":
        return ada_ppl
    elif clf_type == "XG":
        return xgb_ppl


def perf_eval(y_test, y_pred):
    """
    This is a customized evaluation function that measures both the label recall
    for each sample and the precision of each label across all samples.
    :param y_test: ground truth
    :param y_pred: predictions
    :return: sample_label_recall: the label recall for each sample
             label_precision: the precision of each label across all samples
             f1_score: the f1 score calculated from the mean of the above two
    """
    ytest = np.array(y_test)
    sample_label_recall = []
    label_precision = []

    for i in range(y_test.shape[0]):
        if ytest[i].sum() != 0:
            sample_label_recall.append(np.bitwise_and(
                    y_pred[i], ytest[i]).sum() / ytest[i].sum())
        elif ytest[i].sum() == 0:
            sample_label_recall.append(1)

    for j in range(y_test.shape[1]):
        label_precision.append(np.invert(
                np.logical_xor(y_pred[:, j],
                               ytest[:, j])).sum() / ytest.shape[0])

    slr_mean = np.array(sample_label_recall).mean()
    lp_mean = np.array(label_precision).mean()
    f1_score = 2 * slr_mean * lp_mean / (slr_mean + lp_mean)

    return sample_label_recall, label_precision, f1_score


def gs_eval(y_test, y_pred):
    """
    This function returns f1-score for grid search purpose.
    :param y_test: ground truth
    :param y_pred: predictions
    :return: f1-score
    """
    ytest = np.array(y_test)
    sample_label_recall = []
    label_precision = []

    for i in range(y_test.shape[0]):
        if ytest[i].sum() != 0:
            sample_label_recall.append(np.bitwise_and(
                    y_pred[i], ytest[i]).sum() / ytest[i].sum())
        elif ytest[i].sum() == 0:
            sample_label_recall.append(1)

    for j in range(y_test.shape[1]):
        label_precision.append(np.invert(
                np.logical_xor(y_pred[:, j],
                               ytest[:, j])).sum() / ytest.shape[0])

    slr_mean = np.array(sample_label_recall).mean()
    lp_mean = np.array(label_precision).mean()
    f1_score = 2 * slr_mean * lp_mean / (slr_mean + lp_mean)

    return f1_score
