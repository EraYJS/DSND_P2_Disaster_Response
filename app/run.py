import argparse as ap
import joblib
import json
import pandas as pd
import plotly
import sys

from flask import Flask
from flask import render_template, request
from plotly.graph_objects import Bar
from sqlalchemy import create_engine


sys.path += ['.']

def arg_parse():
    parser = ap.ArgumentParser()
    parser.add_argument("-d", action="store", required=True,
                        default="../data/disaster_sn_msg.db", dest="db_pth")
    parser.add_argument("-m", action="store", required=True,
                        default="../model.pkl", dest="md_pth")
    parser.add_argument("-p", action="store", required=False,
                        default=3001, dest="port")

    args = parser.parse_args()

    return args


args = arg_parse()

app = Flask(__name__)

# load data
engine = create_engine("sqlite:///" + args.db_pth)
df = pd.read_sql_table('data', engine)

# load model
model = joblib.load(args.md_pth)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # genre and aid_related status
    aid_rel1 = df[df['aid_related'] == 1].groupby('genre').count()['message']
    aid_rel0 = df[df['aid_related'] == 0].groupby('genre').count()['message']
    genre_names = list(aid_rel1.index)

    # let's calculate distribution of classes with 1
    class_distr1 = df.drop(['id', 'message', 'original', 'genre'],
                           axis=1).sum() / len(df)

    # sorting values in ascending
    class_distr1 = class_distr1.sort_values(ascending=False)

    # series of values that have 0 in classes
    class_distr0 = (class_distr1 - 1) * -1
    class_name = list(class_distr1.index)

    # create visuals
    graphs = [{
            'data'  : [
                    Bar(
                            x=genre_names,
                            y=aid_rel1,
                            name='Aid Related'
                    ),
                    Bar(
                            x=genre_names,
                            y=aid_rel0,
                            name='Not Aid Related'
                    )],

            'layout': {
                    'title'  : 'Message Genre and Aid Relativity',
                    'yaxis'  : {
                            'title': "Number of Messages"
                    },
                    'xaxis'  : {
                            'title': "Genre"
                    },
                    'barmode': 'group'
            }
    },
            {
                    'data'  : [
                            Bar(
                                    x=class_name,
                                    y=class_distr1,
                                    name='Class = 1'
                            ),
                            Bar(
                                    x=class_name,
                                    y=class_distr0,
                                    name='Class = 0',
                                    marker=dict(color='rgb(212, 228, 247)')
                            )
                    ],

                    'layout': {
                            'title'  : 'Label Distribution',
                            'yaxis'  : {
                                    'title': "Percentage"
                            },
                            'xaxis'  : {
                                    'title': "Label",
                            },
                            'barmode': 'stack'
                    }
            }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
            'go.html',
            query=query,
            classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=args.port, debug=True)


if __name__ == '__main__':
    main()
