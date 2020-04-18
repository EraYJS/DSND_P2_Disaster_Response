import numpy as np
import pandas as pd
import re

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, make_scorer
from sklearn.model_selection import GridSearchCV
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


def build_ppl(clf_type, gs=False):
    """
    This function construct models of desired classifier.

    NOTE: the 'gs' option is set to False by default, model parameters are
          'best_param_' acquired from grid search in ML_pipeline_whole.ipynb

    :param clf_type: 'Ada' or 'XG', desired type of classifier
    :param gs: True or False, whether to return a GridSearchCV object

    :return: a Pipeline object or a GridSearchCV object of AdaBoost or XGBoost
    """
    # Grid Search Results:
    #     AdaBoost:
    #         {'clf__estimator__learning_rate': 1.2,
    #          'clf__estimator__n_estimators': 51,
    #          'vect__max_df': 0.4}
    #
    #
    #     XGBoost:
    #         {'clf__estimator__colsample_bytree': 1.0,
    #          'clf__estimator__gamma': 5.0,
    #          'clf__estimator__learning_rate': 0.5,
    #          'clf__estimator__min_child_weight': 1,
    #          'clf__estimator__subsample': 1.0,
    #          'vect__max_df': 0.75}

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

        if gs:
            ada_params = {'feats__text_ppl__vect__max_df': (0.3, 0.4, 0.5),
                          'clf__estimator__n_estimators' : range(50, 54, 1),
                          'clf__estimator__learning_rate': np.arange(1.2, 2.0,
                                                                     0.2)
                          }

            ada_gs = GridSearchCV(
                    ada_ppl,
                    param_grid=ada_params,
                    scoring=make_scorer(gs_eval),
                    n_jobs=12,
                    verbose=7
            )

            return ada_gs

        else:
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

        if gs:
            xgb_params = {'feats__text_ppl__vect__max_df'   : (0.3, 0.4, 0.5),
                          'clf__estimator__max_depth'       : range(5, 10, 2),
                          'clf__estimator__learning_rate'   : [0.5, 0.75, 1.0],
                          'clf__estimator__min_child_weight': np.arange(1, 4,
                                                                        1),
                          'clf__estimator__gamma'           : np.arange(5, 7.5,
                                                                        0.5)
                          }

            # %%

            xgb_gs = GridSearchCV(
                    xgb_ppl,
                    param_grid=xgb_params,
                    scoring=make_scorer(gs_eval),
                    n_jobs=12,
                    verbose=7
            )
        else:
            return xgb_ppl


def perf_eval(y_test_df, y_pred_np):
    """
    This is a customized evaluation function that measures both the label recall
    for each sample and the precision of each label across all samples.
    :param y_test_df: pd.DataFrame, ground truth
    :param y_pred_np: np.array, predictions

    :return: slr_mean: mean of label recall for each sample
             lp_mean : mean of each label precision
             label_precision_df: pd.DataFrame, the precision of each label
             across all samples
             f1_score: the f1 score calculated from the mean of the above two
             report : pd.DataFrame, a scikit-learn classification report
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
    :param y_test: pd.DataFrame, ground truth
    :param y_pred: np.array, predictions

    :return: custom f1-score
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
