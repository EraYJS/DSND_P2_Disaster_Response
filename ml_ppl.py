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
from sklearn.metrics import classification_report
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

    if clf_type == "Ada":

        ada_ppl = Pipeline([
                ('feats', FeatureUnion([
                        ('text_ppl', Pipeline([
                                ('vect', CountVectorizer(
                                        tokenizer=tokenize,
                                        max_df=0.4)),
                                ('tfidf', TfidfTransformer())
                        ])),
                        ('start_verb', StartingVerbExtractor())])),
                ('clf', MultiOutputClassifier(
                        AdaBoostClassifier(
                                n_estimators=51,
                                learning_rate=1.2
                        ))
                 )])

        return ada_ppl

    elif clf_type == "XG":

        # XGBoost models are much more computationally expensive than AdaBoost
        xgb_ppl = Pipeline([
                ('feats', FeatureUnion([
                        ('text_ppl', Pipeline([
                                ('vect', CountVectorizer(
                                        tokenizer=tokenize,
                                        max_df=0.75)),
                                ('tfidf', TfidfTransformer())
                        ])),
                        ('start_verb', StartingVerbExtractor())])),
                ('clf', MultiOutputClassifier(
                        XGBClassifier(
                                colsample_bytree=1.0,
                                gamma=5.0,
                                learning_rate=0.5,
                                min_child_weight=1,
                                subsample=1.0
                        ))
                 )])

        return xgb_ppl


def perf_eval(y_test_df, y_pred_np):
    """
    This is a customized evaluation function that measures both the label recall
    for each sample and the precision of each label across all samples.
    :param y_test_df: ground truth
    :param y_pred_np: predictions
    :return: sample_label_recall: the label recall for each sample
             label_precision: the precision of each label across all samples
             f1_score: the f1 score calculated from the mean of the above two
    """
    y_test_np = np.array(y_test_df)
    sample_label_recall = []
    label_precision = []

    for i in range(y_test_df.shape[0]):
        if y_test_np[i].sum() != 0:
            sample_label_recall.append(np.bitwise_and(
                    y_pred_np[i], y_test_np[i]).sum() / y_test_np[i].sum())
        elif y_test_np[i].sum() == 0:
            sample_label_recall.append(1)

    for j in range(y_test_np.shape[1]):
        label_precision.append(np.invert(
                np.logical_xor(y_pred_np[:, j],
                               y_test_np[:, j])).sum() / y_test_np.shape[0])

    slr_mean = np.array(sample_label_recall).mean()
    lp_mean = np.array(label_precision).mean()
    f1_score = 2 * slr_mean * lp_mean / (slr_mean + lp_mean)

    label_precision_df = pd.DataFrame(label_precision).transpose()
    label_precision_df.columns = y_test_df.columns

    report = pd.DataFrame()
    y_pred_df = pd.DataFrame(y_pred_np)
    y_pred_df.columns = y_test_df.columns

    for col in y_test_df.columns:
        class_dict = classification_report(output_dict=True,
                                           y_true=y_test_df.loc[:, col],
                                           y_pred=y_pred_df.loc[:, col])

        eval_df = pd.DataFrame(pd.DataFrame.from_dict(class_dict))

        # dropping unnecessary information
        eval_df.drop(['macro avg', 'weighted avg'], axis=1,
                     inplace=True)
        eval_df.drop(index='support', inplace=True)

        av_eval_df = pd.DataFrame(eval_df.transpose().mean()).transpose()

        report = report.append(av_eval_df, ignore_index=True)

    report.index = y_test_df.columns

    return slr_mean, lp_mean, label_precision_df, f1_score, report


def gs_eval(y_test_df, y_pred_np):
    """
    This function returns f1-score for grid search purpose.
    :param y_test: ground truth
    :param y_pred: predictions
    :return: f1-score
    """
    y_test_np = np.array(y_test_df)
    sample_label_recall = []
    label_precision = []

    for i in range(y_test_np.shape[0]):
        if y_test_np[i].sum() != 0:
            sample_label_recall.append(np.bitwise_and(
                    y_pred_np[i], y_test_np[i]).sum() / y_test_np[i].sum())
        elif y_test_np[i].sum() == 0:
            sample_label_recall.append(1)

    for j in range(y_test_np.shape[1]):
        label_precision.append(np.invert(
                np.logical_xor(y_pred_np[:, j],
                               y_test_np[:, j])).sum() / y_test_np.shape[0])

    slr_mean = np.array(sample_label_recall).mean()
    lp_mean = np.array(label_precision).mean()
    f1_score = 2 * slr_mean * lp_mean / (slr_mean + lp_mean)

    return f1_score
