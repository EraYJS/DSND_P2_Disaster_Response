import argparse as ap
import numpy as np
import pandas as pd

from sqlalchemy import create_engine


def arg_parser():
    parser = ap.ArgumentParser()
    parser.add_argument("-m", action="store",
                        default="data/messages.csv", dest="msg")
    parser.add_argument("-c", action="store",
                        default="data/categories.csv", dest="cat")
    parser.add_argument("-d", action="store",
                        default="data/disaster_sn_msg.db", dest="db")

    args = parser.parse_args()

    return args


def data_extract(msg_path, cat_path):
    messages = pd.read_csv(msg_path)
    categories = pd.read_csv(cat_path)

    return messages, categories


def data_transform(msg, cat):
    labels = cat["categories"].str.split(pat=";", expand=True)
    labels.columns = labels.iloc[0].apply(lambda x: x[:-2])

    for column in labels:
        # set each value to be the last character of the string
        labels[column] = labels[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        labels[column] = labels[column].astype(int)


    data = pd.concat([msg, labels], axis=1)
    data.drop_duplicates(inplace=True)

    return data


def data_load(data, path):
    engine = create_engine("sqlite:///" + path)
    data.to_sql("data", engine, index=False)
    pass


def etl_main():
    args = arg_parser()

    msg, cat = data_extract(args.msg, args.cat)
    data = data_transform(msg, cat)

    data_load(data, args.db)


if __name__ == "__main__":
    etl_main()
