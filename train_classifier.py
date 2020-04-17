import argparse as ap
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from ml_ppl import build_ppl, perf_eval


def arg_parse():
    parser = ap.ArgumentParser()
    parser.add_argument("-d", action="store", required=True,
                        default="data/disaster_sn_msg.db", dest="db_pth",
                        help='path to database file')
    parser.add_argument("-m", action="store", required=True,
                        default="model.pkl", dest="md_pth",
                        help='path to save the model')
    parser.add_argument("-e", action="store", required=False,
                        default=True, dest="eval",
                        help='whether printing evaluation stats')
    parser.add_argument("-t", action="store", required=False,
                        default="Ada", dest="clf_type",
                        help='classifier type to be built, Ada or XG')

    args = parser.parse_args()

    return args


def load_db(path):
    engine = create_engine("sqlite:///" + path)
    data = pd.read_sql_table("data", engine)

    X = data['message']
    y = data.iloc[:, 4:]

    return X, y


def model_save(model, path):
    filename = path
    pickle.dump(model, open(filename, 'wb'))
    pass


def ml_main():
    args = arg_parse()

    print(f'Loading dataset from {args.db_pth}...')
    X, y = load_db(args.db_pth)
    print('Dataset loaded.')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    print(f'Building {args.clf_type}Boost model...')
    pipeline = build_ppl(args.clf_type)

    pipeline.fit(X_train, y_train)

    if args.eval:
        y_pred = pipeline.predict(X_test)
        slr_mean, lp_mean, f1_score = perf_eval(y_test, y_pred)

        # TODO: plot scores

    sv_pth = args.clf_type + "_" + args.md_pth
    print(f'Saving model to {sv_pth}...')

    model_save(pipeline, sv_pth)
    print('Model saved. Exiting...')


if __name__ == "__main__":
    ml_main()
