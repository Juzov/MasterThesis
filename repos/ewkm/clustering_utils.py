import sys
import os
import json

from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyclustering.cluster.kmeans import kmeans
# import pandas_profiling as pp

# for kmeans only
def k_clustering(data, k, convergance_delta):
    """Run Kmeans from pyclustering
    prepare initial_centers
    flatten the cluster array
    """
    # # create K-Means algorithm instance.
    initial_centers = data.sample(n=k, random_state=1)
    # observer = kmeans_observer()
    kmeans_instance = kmeans(
        data=data,
        initial_centers=initial_centers,
        tolerance=convergance_delta
    )
    # # start processing.
    kmeans_instance.process()
    # # obtain clusters
    clusters = kmeans_instance.get_clusters()

    cluster_arr = cluster_arr = np.zeros(len(data.index), dtype=int)

    for i, cluster in enumerate(clusters):
        for j in cluster:
            # i cluster number
            # set datapoint of arr to cluster number
            cluster_arr[j] = i

    return cluster_arr


def remove_irrelevant_columns(data):
    """Removes labels not used
    or not of interest
    """
    my_filter = (list(data.filter(regex='iso|include|_at|_by')))
    data = data.drop(columns=my_filter)
    data = data.drop(columns=['remote_addr', 'user_agent', 'type', 'isrc'])
    return data


def embeddings_to_columns(data):
    """Extract embedding to seperate features"""
    # embeddings = pd.DataFrame(data[''].values.tolist())
    embeddings = pd.DataFrame(data.pop('embeddings').tolist())
    embeddings = embeddings.add_prefix('emb_')
    data = data.join(embeddings).copy()
    return data


def get_numerical_data(data, only_embedding):
    """Get the numerical part of given data"""

    numerical_data = embeddings_to_columns(data)
    emb_columns = list(numerical_data.filter(regex='emb_'))

    if only_embedding:
        numerical_filter = emb_columns
    else:
        numerical_filter = ['bpm', 'duration_ms', 'origin_year'] + emb_columns

    numerical_data = numerical_data.filter(numerical_filter)
    # numerical_data = numerical_data.agg(min_max_scaling, axis=0)
    numerical_data = numerical_data.agg(z_score, axis=0)

    numerical_data.to_json(
        'data/numerical.json',
        orient='records',
        lines=True
    )

    return numerical_data


def min_max_scaling(series):
    """min max normalization per column
    or row depending on axis on agg/apply
    """
    series = series.astype(float)
    series_std = (series - series.min()) / (series.max() - series.min())
    return series_std

def z_score(series):
    """z-score standardization per column
    or row depending on axis on agg/apply
    """
    # print(series.shape)
    series = series.astype(float)
    Z = (series - series.mean()) / series.std()
    # print("hello {0}".format(Z.mean()))
    return Z


def merge_columns(data, column, distances):
    """ Add cluster column to dataframe
    Copy in case we want to use the original data again
    """
    merged_data = data.copy()
    merged_data['cluster'] = column
    merged_data['distance'] = distances
    merged_data = merged_data.sort_values(by='distance')
    return merged_data


def genre_ratio(series):
    """ Get genre ratio
    returns a dict
    """
    d = defaultdict(float)
    values = series.tolist()
    count = 0

    for sublist in values:
        for x in sublist:
            d[x] += 1
        count += 1

    for k, v in d.items():
        v /= count

        d[k] = v * 100
    d = OrderedDict(sorted(d.items(), key=lambda x: x[1], reverse=True))
    return d


def generate_run_specific_plot(lambdas, costs, scores, iterations, restarts, path):
    '''Generate matplotlib plots
    lambdas/costs
    lambdas/scores
    '''
    plot_path = '{0}/plots'.format(path)
    os.mkdir(plot_path)

    plt.plot(lambdas, scores, marker='D', linestyle='-')
    plt.ylabel(
        r'Average Silhouette ($\overline{s}_{co}(\mathcal{D})$) (1000 Point Sample)')
    plt.xlabel(r'Gamma ($\gamma$)')
    plt.savefig(plot_path + "/gamma-sil.png")
    plt.clf()

    plt.plot(lambdas, costs, color='r', marker='D', linestyle='-')
    plt.ylabel(r'Cost Function ($P(U,C,W)$)')
    plt.xlabel(r'Gamma ($\gamma$)')
    plt.savefig(plot_path + "/gamma-costs.png")
    plt.clf()

    plt.plot(lambdas, restarts, color='g', marker='D', linestyle='-')
    plt.ylabel(r'Restarts (Re-sampling of start centers)')
    plt.xlabel(r'Gamma ($\gamma$)')
    plt.savefig(plot_path + "/gamma-restarts.png")
    plt.clf()

def generate_run_specific_json(lambdas, costs, scores, path):
    """Generate a json file on parameter scores
    Cost-Function
    Silhouette-score
    """
    json_list = [
        {'gamma': lambdas[i],
         'scores': {'cost': costs[i], 'silhouette': scores[i]}
        }
        for i in range(0, lambdas.size)
    ]

    with open(f'{path}/run.json', 'w') as fp:
        json.dump(json_list, fp)

def generate_cluster_specific_json(data, k, lamb, weights, path):
    """Generate a nested summary json
    And a json for each cluster containing the songs
    sorted by the average distance to cluster center
    """

    # group by cluster
    # No need to keep the embeddings or cluster tag per song
    to_keep_filter = list(data.filter(regex="^(?!(embeddings|cluster|version|fully_tagged))"))
    grouped = data.groupby(['cluster'], as_index=False)
    grouped_mean = grouped.mean()
    grouped_size = grouped.size()
    grouped_size = grouped_size.rename("count")
    genres_count = grouped['genres'].agg(genre_ratio)

    # create a nested dataframe
    j = (grouped.apply(lambda x: x[to_keep_filter].to_dict('r'))
         .reset_index()
         .rename(columns={'index': 'cluster'})
         .merge(grouped_mean, left_on='cluster', right_on='cluster', how='inner')
         .merge(genres_count, left_on='cluster', right_on='cluster', how='inner')
         .merge(grouped_size, left_on='cluster', right_on='cluster', how='inner')
         .rename(columns={0: 'songs'})
         )

    rows, columns = weights.shape

    # Only show the highest weights
    w_list = weights.tolist()
    w = [
        dict(
            sorted(
                zip(range(0, columns), t),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        )
        for t in w_list
    ]

    try:
        j["hi_w"] = w
        j = j.query('count > 1').sort_values(by='distance')

        # reorder the columns
        cols = ["cluster", "distance", "count"] + list(j.filter(regex="^(?!(cluster|distance|count|songs))")) + ["songs"]
        j = j[cols]

        # summary json
        j_summary = j.drop(columns=['songs'])

        j_summary.to_json(
            path + '/summary.json',
            orient='records'
            # lines=True
        )
    except:
        return

    # json for each specific cluster
    for i in range(0, k):
        try:
            j[j['cluster'] == i].to_json(
                "{0}/{1}.json".format(path,i),
                orient='records',
                lines=True
            )
        except:
            print(f'skipped writing {i}.json due to no members')
            continue


def generate_parameter_specific_plots(weights, k, lamb, path):
    """Generate matplotlib plots
    Value of Weights/No. Dimensions
    Variance/No. Dimensions
    """
    path_plots = path + "/plots"

    os.mkdir(path + "/plots")
    # generate variance plot
    # var = np.var(weights, axis=1)
    # var_max = int(np.amax(var)) + 1
    # print(var)
    # print("toasda")
    # print(var_max)
    # bins = np.linspace(0, var_max, 1)
    # plt.hist(var.reshape, bins=bins,
    #          color='white', edgecolor='black', linewidth=0.5)
    # plt.ylabel(r'Number of points')
    # plt.xlabel(r'Variance $\sigma^2$')
    # plt.savefig(path + "/variance.png")
    # plt.clf()

    # generate weights plot
    rows, columns = weights.shape
    # low_bin = columns / 160
    bins = np.linspace(0, 1, 100)
    # bins = np.insert(bins, 1, low_bin)
    plt.title(r'Weight Distribution at $\gamma = {0}, k = {1}$'.format(lamb, k))
    plt.yscale('log')
    plt.hist(np.hstack(weights), bins=bins,
             color='white', edgecolor='black', linewidth=0.5)
    plt.ylabel(r'Number Of Dimensions')
    plt.xlabel(r'Value of Weight')
    plt.savefig(path_plots + "/weights.png")
    plt.clf()


def parse_filter_json(
        file_location="/home/ejuzovitski/Documents/" +
        "master_thesis/repos/ewkm/data/"
):
    """ Reads the raw datset
    Removes unecessary columns
    """

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
    """Read already parsed and seperted data"""
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
