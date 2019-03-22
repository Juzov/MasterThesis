import sys

import numpy as np
import pandas as pd
import random

import softsubspace as ss


def remove_irrelevant_data(data):
    my_filter = (list(data.filter(regex='iso|include|_at|_by')))
    data = data.drop(columns=my_filter)
    data = data.drop(columns=['remote_addr', 'user_agent', 'type', 'isrc'])
    return data


def embeddings_to_columns(data):
    # embeddings = pd.DataFrame(data[''].values.tolist())
    embeddings = pd.DataFrame(data.pop('embeddings').tolist())
    embeddings = embeddings.add_prefix('emb_')
    data = data.join(embeddings)
    return data


def normalize_data(X):
    X = X.astype(float)
    X_std = (X - X.min()) / (X.max() - X.min())
    return X_std


def get_numerical_data(data, only_embedding):
    emb_columns = list(data.filter(regex='emb_'))

    if only_embedding:
        numerical_filter = emb_columns
    else:
        numerical_filter = ['bpm', 'duration_ms', 'origin_year'] + emb_columns

    return data.filter(numerical_filter)


def json_parsing(location_to_json, only_embedding=True, save_json=True):
    if save_json:
        data = pd.read_json(path_or_buf=location_to_json+"clustering_dataset.json", lines=True)

        data = remove_irrelevant_data(data)

        returnable_data = data

        data.to_json(
            'data.json',
            orient='records',
            lines=True
        )

        data = embeddings_to_columns(data)

        numerical_data = get_numerical_data(data, only_embedding)

        numerical_data = numerical_data.apply(normalize_data, axis=0)

        numerical_data.to_json(
            'numerical_test.json',
            orient='records',
            lines=True
        )

    else:
        numerical_data = pd.read_json(path_or_buf="numerical_test.json", lines=True)

    return returnable_data, numerical_data


data, numerical_data = json_parsing(
    "/home/ejuzovitski/Documents/master_thesis/beam/kmeans/",
    save_json=True
)
# print(data)

x = numerical_data.values

nr, nc = x.shape

k = 50
lamb = 0.5
maxiter = 20
maxrestart = 4
delta = 0.02
init = 0
iterations = 0
restarts = 0
totiters = 100

cluster = np.zeros((nr), dtype='int32')
centers = np.zeros((k*nr))
weights = np.zeros((k*nc))


ss.ewkm(
    x.ravel(order='F'),
    nr,
    nc,
    k,
    lamb,
    maxiter,
    delta,
    maxrestart,
    init,
    iterations,
    cluster,
    centers,
    weights,
    restarts,
    totiters
)

data['clusters'] = cluster
data.to_json(
    'result.json',
    orient='records',
    lines=True
)
# data1 = data[data.clusters == 41]
# data1.to_json(
#     'data1.json',
#     orient='records',
#     lines=True
# )
# print(data1['genres'].mode())
