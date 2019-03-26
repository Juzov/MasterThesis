#!/usr/bin/python3.7

import click
import numpy as np
import pandas as pd
from EWKM import EWKM

import clustering_utils as cs


# argv

@click.command()
@click.option('-e', "--only-embedding/--all", default=True, help="Only use audio embedding ignore other numerical data")
@click.option('-p', "--re-parse", is_flag=True, default=False, help="Reparse clustering_data.json")
@click.option('-k', default=50, help="Amount of clusters")
@click.option('-l', "--lamb", default=1.0, help="Lambda, 0<l<inf, likeliness to represent clusters with more attributes")
@click.option('-i', "--max-iter", default=100, help="Maximum number of iterations")
@click.option('-c', "--convergance-delta", default=0.5, help="convergance delta")
@click.option('-r', "--max-restart", default=0, help="Allow restarts value determines amount of max. restarts")
# @click.option('-d', "--directory", is_flag=True, help="decide directory")
#kmeans++?

def main(only_embedding, re_parse, k, lamb, max_iter, convergance_delta, max_restart):
    if re_parse is False:
        mixed_data, numerical_data = cs.read_datasets()
    else:
        mixed_data = cs.parse_filter_json()
        # normalized
        numerical_data = cs.get_numerical_data(mixed_data, only_embedding)

    ewkm = EWKM(numerical_data)

    clusters, centers, weights = ewkm.predict(k=k, lamb=lamb, max_iter=max_iter, delta=convergance_delta, max_restart=max_restart)

    merged_data = cs.merge_column(mixed_data, clusters)

    # maybe let it also add some statistics
    cs.generate_cluster_specific_json(merged_data, k, lamb)

main()
