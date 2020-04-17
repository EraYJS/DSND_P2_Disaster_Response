import argparse as ap
import numpy as np
import pandas as pd
import pickle
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion

from sqlalchemy import create_engine


def arg_parse():
    parser = ap.ArgumentParser()
    parser.add_argument("-d", action="store", required=True,
                        default="disaster_sn_msg.db", dest="db_pth")
    parser.add_argument("-m", action="store", required=True,
                        default="model.pkl", dest="md_pth")
    parser.add_argument("-e", action="store", required=False,
                        default=True, dest="eval")
    parser.add_argument("-g", action="store", required=False,
                        default=False, dest="gs")
    parser.add_argument("-t", action="store", required=False,
                        default="XGB", dest="clf_type")

    args = parser.parse_args()

    return args


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


def build_ppl(clf_type):
    """
    AdaBoost GridSearch
    {'clf__estimator__learning_rate': 1.0,
     'clf__estimator__n_estimators': 55,
     'vect__max_df': 0.75}

    XGBoost GridSearch
    {'clf__estimator__colsample_bytree': 1.0,
     'clf__estimator__gamma': 5.0,
     'clf__estimator__learning_rate': 0.5,
     'clf__estimator__min_child_weight': 1,
     'clf__estimator__subsample': 1.0,
     'vect__max_df': 0.75}
    """
    ada_ppl = Pipeline([('vect', CountVectorizer(
                            tokenizer=msg_tokenize,
                            max_df=0.75,)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(
                            AdaBoostClassifier(
                                learning_rate=1.0,
                                n_estimators=55)))
                        ])

    xgb_ppl = Pipeline([('vect', CountVectorizer(
                            tokenizer=tokenize,
                            max_df=0.75)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(
                            XGBClassifier(
                                colsample_bytree=1.0,
                                gamma=5.0,
                                learning_rate=0.5,
                                min_child_weight=1,
                                subsample=1.0)))
                        ])

    if clf_type == "ADA":
        return ada_ppl
    elif clf_type == "XGB":
        return xgb_ppl


def model_save(model, path):
    filename = path
    pickle.dump(model, open(filename, 'wb'))
    pass


def perf_eval(model, X_test, y_test):
    y_pred = model.predict(X_test)
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



def ml_main():
    args = arg_parse()

    X, y = load_db(args.db_pth)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    pipeline = build_ppl(args.clf_type)

    pipeline.fit(X_train, y_train)

    if args.eval:
        slr_mean, lp_mean, f1_score = perf_eval(pipeline, X_test, y_test)

        # TODO: plot scores

    model_save(pipeline, args.md_pth)

    pass


if __name__ == "__main__":
    ml_main()
