import sys, os

import numpy as np
import pandas as pd
import pandas_profiling as pp
import random

# def set_cluster_centers(data):
#     """Set each center to be a specific song of a genre
#     """
#     unique_genres = data[data.genres].unique()
#     for genre in unique_genres:
#         print(data[data["genres"] == genre].iloc[0])

#     return data

def remove_irrelevant_columns(data):
    my_filter = (list(data.filter(regex='iso|include|_at|_by')))
    data = data.drop(columns=my_filter)
    data = data.drop(columns=['remote_addr', 'user_agent', 'type', 'isrc'])
    return data


def embeddings_to_columns(data):
    # embeddings = pd.DataFrame(data[''].values.tolist())
    embeddings = pd.DataFrame(data.pop('embeddings').tolist())
    embeddings = embeddings.add_prefix('emb_')
    data = data.join(embeddings).copy()
    return data


def get_numerical_data(data, only_embedding):
    numerical_data = embeddings_to_columns(data)
    emb_columns = list(numerical_data.filter(regex='emb_'))

    if only_embedding:
        numerical_filter = emb_columns
    else:
        numerical_filter = ['bpm', 'duration_ms', 'origin_year'] + emb_columns

    numerical_data = numerical_data.filter(numerical_filter)
    numerical_data = numerical_data.apply(normalize_data, axis=0)

    numerical_data.to_json(
        'data/numerical.json',
        orient='records',
        lines=True
    )

    return numerical_data


def normalize_data(X):
    X = X.astype(float)
    X_std = (X - X.min()) / (X.max() - X.min())
    return X_std


def merge_column(data, column):
    """ Add cluster column to dataframe
    Copy in case we want to use the original data again
    """
    merged_data = data.copy()
    merged_data['clusters'] = column
    return merged_data


def generate_cluster_specific_json(data, k, lamb, path):
    str_lamb = str(lamb).replace('.', '-')

    path = path + "/" + 'K_' + str(k) + '_L_' + str_lamb

    os.mkdir(path)
    for i in range(0, k):
        data[data.clusters == i].to_json(
            path + '/' + str(i) + '.json',
            orient='records',
            lines=True
        )

def parse_filter_json(
        file_location="/home/ejuzovitski/Documents/" +
        "master_thesis/repos/ewkm/data/"
):

    try:
        data = pd.read_json(
            path_or_buf=file_location + "clustering_dataset.json",
            lines=True)
    except FileNotFoundError:
        print("Cannot find clustering_datset.json")

    data = remove_irrelevant_columns(data)

    data.to_json(
        'data/mixed.json',
        orient='records',
        lines=True
    )

    return data


def read_datasets(file_location="/home/ejuzovitski/Documents/" +
                  "master_thesis/repos/ewkm/data/"):


    try:
        returnable_data = pd.read_json(
            path_or_buf=file_location+"mixed.json", lines=True)
    except FileNotFoundError:
        print("Cannot find mixed.json")
    try:
        numerical_data = pd.read_json(
            path_or_buf=file_location+"numerical.json", lines=True)
    except FileNotFoundError:
        print("Cannot find numerical.json")

    return returnable_data, numerical_data
